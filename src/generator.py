from typing import List
import numpy as np
from llm_sdk import Small_LLM_Model
from .decoder import Decoder
from .models import FunctionDefinition
from .function_registry import FunctionRegistry


class Generator:
    """
    Generates the function name using constrained decoding.
    """

    def __init__(self, model: Small_LLM_Model, decoder: Decoder,
                 registry: FunctionRegistry) -> None:

        self._model = model
        self._decoder = decoder
        self._registry = registry
        self._function_prompt_ids = self._encode_function_prompt()

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

    def _build_prompt(self, prompt: str) -> str:
        """
        Build the prompt sent to the language model.
        """

        lines: List[str] = []

        lines.append(
            "Select the best matching function for the user request."
        )
        lines.append("")
        lines.append("Available functions:")
        lines.append("")

        for function in self._registry.functions:

            lines.append(function.name)
            lines.append(f"Description: {function.description}")

            if function.parameters:
                lines.append("Parameters:")

                for name, parameter in function.parameters.items():
                    lines.append(
                        f"- {name}: {parameter.type}"
                    )

            lines.append("")

        lines.append("User request:")
        lines.append(prompt)
        lines.append("")
        lines.append("Selected function:")

        return "\n".join(lines)

    def _encode_function_prompt(self) -> List[int]:
        """
        Encode the static part containing available functions.
        """

        lines = []

        lines.append(
            "Select the best matching function."
        )
        lines.append("")
        lines.append("Available functions:")

        for function in self._registry.functions:

            lines.append(function.name)
            lines.append(
                f"Description: {function.description}"
            )

            if function.parameters:
                lines.append("Parameters:")

                for name, parameter in function.parameters.items():
                    lines.append(
                        f"- {name}: {parameter.type}"
                    )

            lines.append("")

        lines.append("")
        lines.append("User request:")

        text = "\n".join(lines)

        encoded = self._model.encode(text)

        return encoded.squeeze(0).tolist()

    def _encode_prompt(self, prompt: str) -> List[int]:
        """
        Encode only the dynamic user request.
        """

        encoded = self._model.encode(prompt)

        return (
            self._function_prompt_ids
            +
            encoded.squeeze(0).tolist()
            +
            self._model.encode(
                "\nSelected function:"
            ).squeeze(0).tolist()
        )
