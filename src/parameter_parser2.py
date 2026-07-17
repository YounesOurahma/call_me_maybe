from typing import Any, Dict, List
import re
import numpy as np
from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition
from .token_trie import TokenTrie


class ParameterParser:
    """
    Extracts parameter values from a prompt after the correct function
    has already been selected, using constrained decoding.

    This mirrors exactly how Generator/FunctionRegistry select the function
    name: known candidates are encoded once with model.encode() and stored
    in a TokenTrie, then generation masks logits down to only the tokens
    that keep the sequence inside the trie.

    Numbers and strings can't be enumerated into a trie the same way (there
    are infinitely many), so:
      - numbers are constrained to a small set of "digit-ish" tokens
        (0-9, '.', '-') discovered the same encode()-based way.
      - strings fall back to unconstrained generation, stopped on a newline,
        since their content genuinely can't be restricted to a fixed set of
        tokens without the model's vocabulary.
    """

    _BOOLEAN_VALUES = ["true", "false"]
    _NUMBER_CHARACTERS = "0123456789.-+"
    _NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
    _TYPE_ALIASES: dict[str, str] = {
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

    def __init__(self, model: Small_LLM_Model) -> None:
        self._model = model
        self._boolean_trie = self._build_trie(self._BOOLEAN_VALUES)
        self._allowed_number_tokens: set[int] = set()
        for character in self._NUMBER_CHARACTERS:
            token_ids = self._model.encode(character)[0].tolist()
            self._allowed_number_tokens.update(token_ids)

    def _normalize_type(self, parameter_type: str) -> str:
        """
        Normalize schema type names.

        Examples
        --------
        int -> integer
        float -> number
        str -> string
        bool -> boolean
        """
        normalized = self._TYPE_ALIASES.get(parameter_type.lower())

        if normalized is None:
            raise ValueError(
                f"Unsupported parameter type: {parameter_type!r}"
            )

        return normalized

    def parse(self, prompt: str, function: FunctionDefinition) -> Dict[str, Any]:
        """
        Extract every parameter required by ``function`` from ``prompt``.

        Parameters
        ----------
        prompt:
            User prompt.
        function:
            Function selected by the constrained decoder.

        Returns
        -------
        dict[str, Any]
            Dictionary containing every extracted parameter, correctly typed.

        Raises
        ------
        ValueError
            If a parameter type is unsupported or a value cannot be
            extracted/converted.
        """
        parameters: Dict[str, Any] = {}

        for name, definition in function.parameters.items():

            param_type = self._normalize_type(definition.type)
            prompt_ids = self._build_prompt_ids(prompt, function, name, param_type)

            if param_type == "boolean":
                raw_value = self._generate_from_trie(
                    prompt_ids,
                    self._boolean_trie,
                )
            elif param_type == "integer":
                raw_value = self._generate_number(
                    prompt_ids,
                    integer=True,
                )
            elif param_type == "number":
                raw_value = self._generate_number(
                    prompt_ids,
                )

            elif param_type == "string":
                raw_value = self._generate_string(prompt_ids)
            else:
                raise ValueError(f"Unsupported parameter type: {param_type!r}")

            parameters[name] = self._convert(raw_value, param_type, name)

        return parameters

    def _build_prompt_ids(
        self,
        prompt: str,
        function: FunctionDefinition,
        target_param: str,
        param_type: str,
    ) -> List[int]:
        """
        Build the prompt used to extract a single parameter.

        The prompt provides the complete function schema so the model
        understands the role of the requested parameter and returns
        only its value.
        """

        lines: List[str] = []

        lines.append(
            "You are an assistant that extracts ONE parameter for a function call."
        )
        lines.append("")
        lines.append("User request:")
        lines.append(prompt)
        lines.append("")

        lines.append("Function parameters:")

        for name, parameter in function.parameters.items():
            lines.append(f"- {name}: {parameter.type}")

        lines.append("")
        lines.append(f"Parameter name: {target_param}")
        lines.append(f"Expected type: {param_type}")
        lines.append("")
        lines.append("IMPORTANT:")
        lines.append("- Do NOT solve the user's request.")
        lines.append("- Do NOT compute any result.")
        lines.append("- Copy the requested value exactly as it appears in the user request.")
        lines.append("- Your answer MUST already exist inside the user request.")
        lines.append("- Return only the value.")

        lines.append("")

        lines.append("Instructions:")
        lines.append("- Do NOT explain your answer.")
        lines.append("- Do NOT output JSON.")
        lines.append("")

        lines.append("Answer:")

        text = "\n".join(lines)

        # print("=" * 80)
        # print(text)
        # print("=" * 80)

        return self._model.encode(text)[0].tolist()

    def _build_trie(self, candidates: List[str]) -> TokenTrie:
        """Encode each candidate once and store its token sequence in a trie."""
        trie: TokenTrie = TokenTrie()

        for candidate in candidates:
            token_ids = tuple(self._model.encode(candidate)[0].tolist())
            trie.insert(token_ids, candidate)

        return trie

    def _encode_single_tokens(self, chars: List[str]) -> Dict[str, int]:
        """
        Encode each character on its own and keep its first token id.

        This is a simplification: it assumes each character in `chars`
        encodes to (or starts with) a single stable token id in isolation.
        That can be untrue for some tokenizers/contexts -- if digits merge
        into multi-character tokens differently depending on what precedes
        them, this mapping won't catch every valid continuation. It's a
        best-effort constraint, not a perfect one, given no vocabulary file
        is used anywhere else in this project either.
        """
        ids: Dict[str, int] = {}

        for char in chars:
            token_ids = self._model.encode(char)[0].tolist()
            if token_ids:
                ids[char] = token_ids[0]

        return ids

    def _mask(self, logits: List[float], allowed_ids) -> np.ndarray:
        """Same masking approach as Decoder.apply: -inf everywhere except
        the allowed token ids."""
        logits_array = np.asarray(logits, dtype=np.float32)
        masked = np.full(logits_array.shape, -np.inf, dtype=np.float32)

        if allowed_ids:
            indices = np.fromiter(allowed_ids, dtype=np.int64)
            masked[indices] = logits_array[indices]

        return masked

    def _generate_number(
        self,
        prompt_ids: List[int],
        *,
        integer: bool = False,
        max_tokens: int = 20,
    ) -> str:
        """
        Generate a numeric value using constrained decoding.

        Only numeric tokens (0-9, '.', '-', '+') and end-of-line tokens
        are allowed to be generated.
        """

        generated: List[int] = []

        newline_token = self._model.encode("\n")[0].tolist()[0]

        allowed_tokens = set(self._allowed_number_tokens)
        allowed_tokens.add(newline_token)

        for _ in range(max_tokens):

            input_ids = prompt_ids + generated

            logits = self._model.get_logits_from_input_ids(input_ids)

            masked = self._mask(logits, allowed_tokens)

            if np.all(np.isneginf(masked)):
                break

            next_token = int(np.argmax(masked))

            if next_token == newline_token:
                break

            generated.append(next_token)

        text = self._model.decode(generated).strip()

        if text.endswith("."):
            text = text[:-1]

        if not text:
            raise ValueError(
                "Could not generate a number."
            )

        if integer:
            return str(int(float(text)))

        return text

    def _clean_string(self, value: str) -> str:
        """
        Normalize a generated string value.
        """

        value = value.strip()

        if len(value) >= 2:
            if value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]

        return value.strip()

    def _generate_string(self, prompt_ids: List[int], max_tokens: int = 40) -> str:
        """
        Strings can't be constrained to a fixed token set, so this generates
        unconstrained (same argmax loop as Generator, no masking) and stops
        at the first newline or the token cap.
        """
        generated: List[int] = []

        for _ in range(max_tokens):
            input_ids = prompt_ids + generated
            logits = self._model.get_logits_from_input_ids(input_ids)
            logits_array = np.asarray(logits, dtype=np.float32)
            next_token = int(np.argmax(logits_array))
            generated.append(next_token)

            if "\n" in self._model.decode(generated):
                break

        text = self._model.decode(generated)
        return text.split("\n")[0].strip()

    def _convert(self, raw_value: str, param_type: str, name: str) -> Any:
        """Convert the generated text into the correctly-typed Python value."""
        try:
            if param_type == "boolean":
                return raw_value == "true"
            if param_type == "integer":
                return int(raw_value)
            if param_type == "number":
                return float(raw_value)
            if param_type == "string":
                return self._clean_string(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Could not convert value {raw_value!r} for parameter "
                f"{name!r} to type {param_type!r}"
            ) from exc

        raise ValueError(f"Unsupported parameter type: {param_type!r}")
