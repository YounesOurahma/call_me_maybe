import re
from typing import Any, Dict, List

import numpy as np
from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition

# Generic candidate discovery -- this only ever finds CANDIDATES for the
# model to choose from via constrained decoding. It never decides the
# answer itself and it isn't tied to any specific function name.
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
_QUOTED_PATTERN = re.compile(r'"([^"]*)"|\'([^\']*)\'')
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")

# Small generic fallback pool for string parameters whose value can't be a
# literal substring of the prompt (e.g. a regex pattern implied by the word
# "vowels", or a replacement symbol implied by "asterisks"). These are
# offered as extra candidates -- the model still has to pick one via
# constrained decoding, this list doesn't pick for it.
_COMMON_REGEX_CANDIDATES = [r"\d+", r"[aeiouAEIOU]", r"\s+", r"[a-zA-Z]+"]
_COMMON_SYMBOL_CANDIDATES = ["*", "_", "#", ""]


class ParameterParser:
    """
    Extracts parameter values from a prompt after the correct function has
    already been selected, using constrained decoding.

    Every value the model can produce comes from a small set of KNOWN
    candidates -- it never free-generates content. Each candidate is encoded
    once with model.encode(), then scored by its teacher-forced
    log-likelihood under the model (see _score_candidate); whichever
    candidate the model finds most likely is returned:
      - booleans: candidates are always {"true", "false"}.
      - numbers/integers: candidates are every numeric literal that
        actually appears in the prompt.
      - strings: candidates are quoted substrings and standalone words from
        the prompt, plus a small generic fallback pool (common regex
        patterns / replacement symbols) for values that must be inferred
        rather than copied verbatim.

    Earlier versions generated numbers digit-by-digit and strings
    token-by-token with no anchor to the prompt's actual content, which
    produced hallucinated/zero-padded numbers and fabricated strings.
    Scoring full known candidates against the model's own likelihood
    guarantees every extracted value is something that genuinely exists in
    (or is a plausible generic completion of) the prompt, without the
    early-commitment bias a step-by-step trie walk has.
    """

    _BOOLEAN_VALUES = ["true", "false"]
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

    def __init__(self, model: Small_LLM_Model) -> None:
        self._model = model
        self._encode_cache: Dict[str, tuple] = {}

    def _encode(self, text: str) -> tuple:
        """Cache encode() results -- the same candidate (e.g. 'cat') often
        gets scored across multiple parameters within one prompt."""
        if text not in self._encode_cache:
            self._encode_cache[text] = tuple(self._model.encode(text)[0].tolist())
        return self._encode_cache[text]

    def _normalize_type(self, parameter_type: str) -> str:
        normalized = self._TYPE_ALIASES.get(parameter_type.lower())
        if normalized is None:
            raise ValueError(f"Unsupported parameter type: {parameter_type!r}")
        return normalized

    def parse(self, prompt: str, function: FunctionDefinition) -> Dict[str, Any]:
        """
        Extract every parameter required by ``function`` from ``prompt``.

        Raises
        ------
        ValueError
            If a parameter type is unsupported or a value cannot be
            extracted/converted.
        """
        numeric_pool = self._numeric_candidates(prompt)
        string_pool = self._string_candidates(prompt)

        # Track which candidates have already been assigned to an earlier
        # parameter in this same call, so a second same-type parameter
        # can't just pick the same value again -- it's forced to choose
        # among whatever's left.
        used_numeric: List[str] = []
        used_string: List[str] = []

        parameters: Dict[str, Any] = {}

        for name, definition in function.parameters.items():
            param_type = self._normalize_type(definition.type)
            prompt_ids = self._build_prompt_ids(prompt, function, name, param_type)

            if param_type == "boolean":
                candidates = self._BOOLEAN_VALUES
            elif param_type in ("number", "integer"):
                candidates = self._unused(numeric_pool, used_numeric) or numeric_pool or ["0"]
            elif param_type == "string":
                candidates = self._unused(string_pool, used_string) or string_pool or [""]
            else:
                raise ValueError(f"Unsupported parameter type: {param_type!r}")

            raw_value = self._select_best_candidate(prompt_ids, candidates)

            if param_type in ("number", "integer"):
                used_numeric.append(raw_value)
            elif param_type == "string":
                used_string.append(raw_value)

            parameters[name] = self._convert(raw_value, param_type, name)

        return parameters

    def _unused(self, pool: List[str], used: List[str]) -> List[str]:
        """Candidates from `pool` that haven't already been assigned to an
        earlier parameter (falls back to the full pool if everything's
        been used, e.g. more same-type parameters than distinct values)."""
        return [c for c in pool if c not in used]

    # -- candidate discovery ------------------------------------------- #

    def _numeric_candidates(self, prompt: str) -> List[str]:
        """Every numeric literal that appears in the prompt, deduplicated,
        in order of appearance."""
        return list(dict.fromkeys(_NUMBER_PATTERN.findall(prompt)))

    def _string_candidates(self, prompt: str) -> List[str]:
        """Quoted substrings + standalone content words from the prompt
        (common stopwords dropped to keep the candidate pool -- and scoring
        cost -- small), plus a small generic fallback pool for inferred
        (non-literal) values."""
        quoted = [
            group1 or group2
            for group1, group2 in _QUOTED_PATTERN.findall(prompt)
        ]
        words = [
            word for word in _WORD_PATTERN.findall(prompt)
        ]

        candidates = quoted + words + _COMMON_REGEX_CANDIDATES + _COMMON_SYMBOL_CANDIDATES
        return list(dict.fromkeys(c for c in candidates if c))

    # -- prompt construction --------------------------------------------- #

    def _build_prompt_ids(
        self,
        prompt: str,
        function: FunctionDefinition,
        target_param: str,
        param_type: str,
    ) -> List[int]:
        lines: List[str] = [
            "You are an assistant that extracts ONE parameter for a function call.",
            "",
            "User request:",
            prompt,
            "",
            "Selected function:",
            function.name,
            "",
            "Function description:",
            function.description,
            "",
            "Function parameters:",
        ]

        for name, parameter in function.parameters.items():
            lines.append(f"- {name}: {parameter.type}")

        lines += [
            "",
            f"Parameter to extract: {target_param}",
            f"Expected type: {param_type}",
            "",
            "Instructions:",
            "- Return ONLY the requested parameter's value, copied from the request.",
            "- Do NOT compute or transform anything -- copy the raw value as-is.",
            "- Do NOT explain your answer.",
            "- Do NOT output JSON.",
            "- Do NOT include the parameter name.",
            "- Output exactly one value.",
            "",
            "Answer:",
        ]

        text = "\n".join(lines)
        return self._model.encode(text)[0].tolist()

    # -- constrained decoding core ---------------------------------------- #
    #
    # Rather than walking a TokenTrie step-by-step (which commits to a
    # branch at the first token where two candidates diverge, based on a
    # single step's logits -- observed in practice to collapse onto
    # whichever candidate happened to be inserted/tokenized first,
    # regardless of which parameter was actually being asked for), each
    # full candidate is scored by its teacher-forced log-likelihood under
    # the model, and the highest-scoring one is returned. This still only
    # ever returns a value from the known candidate set (real constrained
    # decoding, not free generation) but compares whole sequences against
    # the parameter-specific prompt instead of committing early.

    def _encode_candidates(self, candidates: List[str]) -> Dict[str, tuple]:
        """Encode each candidate, reusing cached encodings where possible."""
        return {candidate: self._encode(candidate) for candidate in candidates}

    def _log_prob(self, logits_array: np.ndarray, token_id: int) -> float:
        """Numerically-stable log-softmax probability of one token id."""
        max_logit = np.max(logits_array)
        log_sum_exp = max_logit + np.log(np.sum(np.exp(logits_array - max_logit)))
        return float(logits_array[token_id] - log_sum_exp)

    def _score_candidate(self, prompt_ids: List[int], token_ids: tuple) -> float:
        """Sum of log P(token_i | prompt, token_<i) for the candidate's
        exact token sequence (teacher forcing -- the real candidate tokens
        are fed back in, never the model's own argmax choice)."""
        score = 0.0
        generated: List[int] = []

        for token_id in token_ids:
            input_ids = prompt_ids + generated
            logits = self._model.get_logits_from_input_ids(input_ids)
            logits_array = np.asarray(logits, dtype=np.float32)
            score += self._log_prob(logits_array, token_id)
            generated.append(token_id)

        return score

    def _select_best_candidate(self, prompt_ids: List[int], candidates: List[str]) -> str:
        """Return whichever candidate the model assigns the highest total
        likelihood to, given this parameter's specific extraction prompt."""
        encoded = self._encode_candidates(candidates)

        best_candidate = candidates[0]
        best_score = -np.inf

        for candidate, token_ids in encoded.items():
            score = self._score_candidate(prompt_ids, token_ids)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    # -- value conversion --------------------------------------------------- #

    def _convert(self, raw_value: str, param_type: str, name: str) -> Any:
        try:
            if param_type == "boolean":
                return raw_value == "true"
            if param_type == "integer":
                return int(float(raw_value))
            if param_type == "number":
                return float(raw_value)
            if param_type == "string":
                return raw_value
        except ValueError as exc:
            raise ValueError(
                f"Could not convert value {raw_value!r} for parameter "
                f"{name!r} to type {param_type!r}"
            ) from exc

        raise ValueError(f"Unsupported parameter type: {param_type!r}")
