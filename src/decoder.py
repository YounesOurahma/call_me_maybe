import numpy as np
from .vocab import Vocabulary
from .state_machine import JSONStateMachine
from typing import List


class JSONConstraintDecoder:
    """
    Goes into the LLM generation loop to mask out invalid tokens
    based on the strict rules of the JSON state machine.
    """

    def __init__(self, vocab: Vocabulary, state_machine: JSONStateMachine):
        self.vocab = vocab
        self.state_machine = state_machine

    def apply_mask(self, logits: np.ndarray) -> np.ndarray:
        """
        Takes the raw logits from the LLM, applies a -infinity mask to
        invalid tokens, and returns the strictly compliant logits.

        Args:
            logits: A 1D numpy array of shape representing
                    the model's raw token predictions.
        """
        allowed_strings: List[str] = self.state_machine.get_allowed_strings()

        if not allowed_strings:
            return logits

        masked_logits = np.full_like(logits, fill_value=-np.inf)

        for string in allowed_strings:
            token_id = self.vocab.get_id(string)
            if token_id != -1:
                masked_logits[token_id] = logits[token_id]

        return masked_logits
