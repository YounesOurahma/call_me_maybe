from typing import Iterable

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition
from .token_trie import TokenTrie


class FunctionRegistry:
    """
    Registry of all available functions.

    Each function name is encoded exactly once and stored
    inside a TokenTrie.

    During constrained decoding, this class answers:

        - Which tokens are allowed next?
        - Is the current sequence complete?
        - Which function was selected?
    """

    def __init__(self, functions: Iterable[FunctionDefinition],
                 model: Small_LLM_Model) -> None:

        self._functions = list(functions)
        self._trie: TokenTrie[FunctionDefinition] = TokenTrie()

        for function in functions:

            token_ids = tuple(model.encode(function.name)[0].tolist())

            self._trie.insert(token_ids, function)

    def allowed_next_tokens(self, generated: list[int]) -> set[int]:
        """
        Return every token that may legally follow the
        generated sequence.
        """

        return self._trie.allowed_next_tokens(tuple(generated))

    def is_complete(self, generated: list[int]) -> bool:
        """
        Return True iff the generated sequence exactly
        matches one registered function.
        """

        return self._trie.contains(tuple(generated))

    def selected_function(self, generated: list[int]) -> FunctionDefinition:
        """
        Return the FunctionDefinition corresponding to the
        generated token sequence.

        Raises
        ------
        ValueError
            If the sequence does not correspond to any
            registered function.
        """

        return self._trie.get(tuple(generated))

    @property
    def functions(self) -> list[FunctionDefinition]:
        """
        Return every registered function.
        """

        return self._functions
