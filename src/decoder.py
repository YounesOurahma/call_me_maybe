from typing import List
import numpy as np
from .function_registry import FunctionRegistry
from .models import FunctionDefinition


class Decoder:
    """
    Performs constrained decoding using the registered function names.

    Given the current generated token sequence and the model logits,
    the decoder masks every token that cannot legally continue the
    sequence according to the FunctionRegistry.
    """

    def __init__(self, registry: FunctionRegistry) -> None:
        self._registry = registry

    def apply(self, generated: List[int], logits: List[float]) -> np.ndarray:
        """
        Apply token constraints to the model logits.

        Parameters
        ----------
        generated:
            Tokens generated so far.

        logits:
            Raw logits returned by the language model.

        Returns
        -------
        np.ndarray
            A masked logits vector where every forbidden token
            has value -np.inf.
        """

        allowed = self._registry.allowed_next_tokens(generated)

        logits_array = np.asarray(logits, dtype=np.float32)

        masked = np.full(logits_array.shape, -np.inf, dtype=np.float32)

        if allowed:
            indices = np.fromiter(allowed, dtype=np.int64)

            masked[indices] = logits_array[indices]

        return masked

    def is_complete(self, generated: List[int]) -> bool:
        """
        Return True if the generated sequence exactly matches
        one registered function.
        """

        return self._registry.is_complete(generated)

    def selected_function(self, generated: List[int]) -> FunctionDefinition:
        """
        Return the FunctionDefinition corresponding to the
        generated sequence.

        This method should only be called after is_complete()
        returns True.
        """

        return self._registry.selected_function(generated)
