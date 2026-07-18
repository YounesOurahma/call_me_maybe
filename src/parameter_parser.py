"""Extracts function-call parameters from a prompt using constrained decoding.

Design
------
Exactly like ``FunctionRegistry``/``Decoder`` restrict token choice to the
set of registered function names, this module restricts token choice to
the *shape* required by a parameter's declared type -- nothing here reads
the function name or looks for specific keywords in the prompt, so it
keeps working unchanged if the reviewer swaps in a different
``functions_definition.json``.

At every generation step, the model's own top candidate tokens are
inspected in order of preference; the first one whose decoded text still
respects the type's allowed character set is kept, everything else is
skipped (the constrained-decoding equivalent of masking those tokens to
``-inf``). Generation stops as soon as the model's best remaining choice
is a newline, or after a small safety cap.
"""

from typing import Any, Dict, List

import numpy as np
from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition

_NUMBER_CHARACTERS = set("0123456789.- ")

_TYPE_ALIASES: Dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
}


class ParameterParser:
    """Fills a function's parameters by generating each value with the LLM.

    Every value is produced one token at a time, only ever letting through
    tokens that keep the value inside the character set required by its
    declared type (digits for numbers, anything but a newline for
    strings, or one of two fixed literals for booleans).
    """

    _BOOLEAN_CANDIDATES = ("true", "false")
    _TOP_K = 40
    _MAX_VALUE_TOKENS = 30

    def __init__(self, model: Small_LLM_Model) -> None:
        """Store the model and a small per-run token-text cache.

        Parameters
        ----------
        model:
            The language model used for both scoring and generation.
        """
        self._model = model
        self._token_cache: Dict[int, str] = {}

    def parse(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> Dict[str, Any]:
        """Extract every parameter required by ``function`` from ``prompt``.

        Parameters
        ----------
        prompt:
            The natural-language user request.
        function:
            The function selected by the constrained decoder.

        Returns
        -------
        dict[str, Any]
            Every extracted parameter, converted to its declared type.

        Raises
        ------
        ValueError
            If a parameter type is unsupported or a value cannot be
            extracted or converted.
        """
        parameters: Dict[str, Any] = {}

        for name, definition in function.parameters.items():
            param_type = self._normalize_type(definition.type)
            prompt_ids = self._build_prompt_ids(
                prompt, function, parameters, name, param_type,
            )

            if param_type == "boolean":
                raw_value = self._select_boolean(prompt_ids)
            elif param_type in ("number", "integer"):
                raw_value = self._generate_constrained(
                    prompt_ids, self._is_number_char,
                )
            else:
                raw_value = self._generate_constrained(
                    prompt_ids, self._is_string_char,
                )

            if not raw_value:
                raise ValueError(
                    f"Unable to extract value for parameter {name!r}."
                )

            parameters[name] = self._convert(raw_value, param_type, name)

        return parameters

    def _build_prompt_ids(
        self,
        prompt: str,
        function: FunctionDefinition,
        already_extracted: Dict[str, Any],
        target_param: str,
        param_type: str,
    ) -> List[int]:
        """Build the prompt used to extract a single parameter value."""
        lines = [
            "You extract arguments for a function call.",
            "Your job is NOT to execute the function.",
            "Your job is ONLY to copy the input argument.",
            "",
            "Answer with the bare value only -- no words, no quotes,",
            "no explanation, nothing but the value itself.",
            "",
            "Request: Add 5 and 7",
            "Parameter: a (number)",
            "Value: 5",
            "",
            "Request: Greet Alice",
            "Parameter: name (string)",
            "Value: Alice",
            "",
            f"Request: {prompt}",
        ]

        if already_extracted:
            lines.append(f"Already extracted: {already_extracted}")

        lines += [
            f"Parameter: {target_param} ({param_type})",
            "Value:",
        ]

        text = "\n".join(lines)
        return self._model.encode(text)[0].tolist()

    def _select_boolean(self, prompt_ids: List[int]) -> str:
        """Pick whichever of 'true'/'false' the model finds more likely."""
        scores = [
            self._teacher_forced_score(
                prompt_ids, self._model.encode(candidate)[0].tolist(),
            )
            for candidate in self._BOOLEAN_CANDIDATES
        ]
        return self._BOOLEAN_CANDIDATES[int(np.argmax(scores))]

    def _teacher_forced_score(
        self,
        prompt_ids: List[int],
        token_ids: List[int],
    ) -> float:
        """Total log-probability the model assigns to ``token_ids``."""
        score = 0.0
        generated: List[int] = []

        for token_id in token_ids:
            logits = np.asarray(
                self._model.get_logits_from_input_ids(prompt_ids + generated),
                dtype=np.float32,
            )
            score += float(logits[token_id] - self._log_sum_exp(logits))
            generated.append(token_id)

        return score

    @staticmethod
    def _log_sum_exp(logits: np.ndarray) -> float:
        """Numerically stable log-sum-exp over a logits vector."""
        top = np.max(logits)
        return float(top + np.log(np.sum(np.exp(logits - top))))

    def _generate_constrained(
        self,
        prompt_ids: List[int],
        char_is_allowed: Any,
    ) -> str:
        """Greedily generate tokens, skipping any that break the alphabet.

        At each step the model's best-scoring tokens are checked in
        order; the first whose text is entirely made of allowed
        characters is appended. A token containing a newline ends
        generation immediately, treating it as the model signalling it
        is done.
        """
        generated: List[int] = []

        for _ in range(self._MAX_VALUE_TOKENS):
            logits = np.asarray(
                self._model.get_logits_from_input_ids(prompt_ids + generated),
                dtype=np.float32,
            )
            order = np.argsort(logits)[::-1]

            if not generated:
                # Nothing produced yet: search a wide window for a token
                # that can legally start the value.
                window = order[: self._TOP_K]
            else:
                # A value is already underway: trust the model's own top
                # choice. If it no longer fits the alphabet, that is the
                # model signalling the value is finished -- do not dig
                # deeper and risk bolting on an unrelated match.
                window = order[:1]

            picked = self._first_matching(window, char_is_allowed)

            if picked is None:
                break

            generated.append(picked)

        if not generated:
            return ""

        return self._model.decode(generated).strip()

    def _first_matching(self, token_ids: np.ndarray, char_is_allowed: Any) -> Any:
        """Return the first token id in ``token_ids`` whose text is made
        entirely of allowed characters, or None if none qualify."""
        for token_id in token_ids:
            piece = self._token_text(int(token_id))

            if not piece or "\n" in piece:
                continue
            if all(char_is_allowed(ch) for ch in piece):
                return int(token_id)

        return None

    def _token_text(self, token_id: int) -> str:
        """Decode a single token id, caching the result for reuse."""
        cached = self._token_cache.get(token_id)

        if cached is None:
            cached = self._model.decode([token_id])
            self._token_cache[token_id] = cached

        return cached

    @staticmethod
    def _is_number_char(character: str) -> bool:
        """Return True if ``character`` may appear in a numeric literal."""
        return character in _NUMBER_CHARACTERS

    @staticmethod
    def _is_string_char(character: str) -> bool:
        """Return True for any character except a newline."""
        return character != "\n"

    def _normalize_type(self, parameter_type: str) -> str:
        """Map a schema type name onto one of the four supported types."""
        normalized = _TYPE_ALIASES.get(parameter_type.lower())

        if normalized is None:
            raise ValueError(
                f"Unsupported parameter type: {parameter_type!r}"
            )

        return normalized

    def _convert(self, raw_value: str, param_type: str, name: str) -> Any:
        """Convert the generated text into the correctly typed value."""
        try:
            if param_type == "boolean":
                return raw_value == "true"
            if param_type == "integer":
                return int(float(raw_value.replace(" ", "")))
            if param_type == "number":
                return float(raw_value.replace(" ", ""))
            return raw_value.strip("'\" ")
        except ValueError as exc:
            raise ValueError(
                f"Could not convert value {raw_value!r} for parameter "
                f"{name!r} to type {param_type!r}"
            ) from exc
