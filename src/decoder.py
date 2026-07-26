from typing import List, Optional, Set, Tuple

import numpy as np

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition


class Decoder:
    """
    Picks the registered function that best matches a prompt.
    """

    def __init__(
        self,
        functions: List[FunctionDefinition],
        model: Small_LLM_Model,
    ) -> None:
        """Store the function catalog and pre-encode everything that
        does not depend on the prompt.
        """
        if not functions:
            raise ValueError(
                "Decoder requires at least one registered function."
            )

        self._functions = list(functions)
        self._model = model

        self._candidates: List[Tuple[int, ...]] = [
            tuple(model.encode(f'"{function.name}",')[0].tolist())
            for function in self._functions
        ]

        self._instruction_ids = self._build_instructions()

    def select(self, prompt: str) -> FunctionDefinition:
        """Select the function that best matches ``prompt``.
        """
        prompt_ids = (
            self._model.encode(f"Request: {prompt}")[0].tolist()
            + self._instruction_ids
            + self._model.encode(
                f'\nRequest: {prompt}\n{{"name": "')[0].tolist()
        )

        generated: List[int] = []

        while True:
            logits = self._model.get_logits_from_input_ids(
                prompt_ids + generated
            )

            masked = self.apply(generated, logits)

            if np.all(np.isneginf(masked)):
                raise RuntimeError(
                    "No registered function matches the generated "
                    f"prefix: {self._model.decode(generated)!r}"
                )

            next_token = int(np.argmax(masked))
            generated.append(next_token)

            selected = self._match(generated)
            if selected is not None:
                return selected

    def apply(self, generated: List[int], logits: List[float]) -> np.ndarray:
        """
        Mask every token that cannot continue ``generated`` as a
        prefix of at least one registered function name.
        """
        allowed = self._allowed_next_tokens(generated)

        logits_array = np.array(logits)
        masked = np.full(logits_array.shape, -np.inf)

        if allowed:
            indices = np.array(list(allowed))
            masked[indices] = logits_array[indices]

        return masked

    def _allowed_next_tokens(self, generated: List[int]) -> Set[int]:
        """Every token id that keeps ``generated`` a valid, not-yet-
        complete prefix of at least one candidate function name.
        """
        depth = len(generated)
        prefix = tuple(generated)

        return {
            candidate[depth]
            for candidate in self._candidates
            if len(candidate) > depth and candidate[:depth] == prefix
        }

    def _match(self, generated: List[int]) -> Optional[FunctionDefinition]:
        """Return the function whose full (token-encoded) name exactly
        equals ``generated``, if any.
        """
        sequence = tuple(generated)

        for function, candidate in zip(self._functions, self._candidates):
            if candidate == sequence:
                return function

        return None

    def _build_instructions(self) -> List[int]:
        """
        Encode the static part of the prompt: instructions plus the
        full catalog of available functions. Computed once and reused
        for every call to :py:meth:`select`.
        """
        lines = [
            "",
            "Select the best matching function.",
            "",
            "Available functions:"
        ]

        for function in self._functions:
            lines.append(function.name)
            lines.append(f"Description: {function.description}")

            if function.parameters:
                lines.append("Parameters:")
                for name, parameter in function.parameters.items():
                    lines.append(f"- {name}: {parameter.type}")

            lines.append("")

        text = "\n".join(lines)

        return self._model.encode(text)[0].tolist()
