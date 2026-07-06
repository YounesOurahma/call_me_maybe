from __future__ import annotations
from typing import List
import numpy as np
from llm_sdk import Small_LLM_Model
from .decoder import Decoder
from .models import FunctionDefinition


class Generator:
    """
    Generates the function name using constrained decoding.
    """

    def __init__(self, model: Small_LLM_Model, decoder: Decoder) -> None:
        self._model = model
        self._decoder = decoder

    def generate(self, prompt: str) -> FunctionDefinition:
        """
        Select the function that best matches the prompt.

        Parameters
        ----------
        prompt:
            Natural language user request.

        Returns
        -------
        FunctionDefinition
            The selected function.
        """

        prompt_ids = self._encode_prompt(prompt)

        generated: List[int] = []

        while not self._decoder.is_complete(generated):

            input_ids = prompt_ids + generated

            logits = self._model.get_logits_from_input_ids(input_ids)

            masked_logits = self._decoder.apply(generated, logits)

            if np.all(np.isneginf(masked_logits)):
                raise RuntimeError(
                    f"No valid continuation found after generating: "
                    f"{self._model.decode(generated)!r}"
                )
            next_token = int(np.argmax(masked_logits))

            generated.append(next_token)

        return self._decoder.selected_function(generated)

    def _encode_prompt(self, prompt: str) -> List[int]:
        """
        Convert the prompt into a list of token ids.
        """

        encoded = self._model.encode(prompt)

        return encoded.squeeze(0).tolist()
