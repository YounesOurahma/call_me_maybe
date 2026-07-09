from typing import Any, Optional


class TrieNode:
    """
    A node inside a TokenTrie.
    """
    def __init__(self) -> None:
        self.children: dict[int, "TrieNode"] = {}
        self.terminal: bool = False
        self.value: Optional[Any] = None


class TokenTrie:
    """
    Trie that stores sequences of token ids.

    Each complete sequence can be linked to any object
    (in this project, a FunctionDefinition).
    """

    def __init__(self) -> None:
        self._root = TrieNode()

    def _follow(self, sequence: tuple[int, ...]) -> Optional[TrieNode]:
        """
        Follow a sequence inside the trie.

        Returns
        -------
        TrieNode | None
            The node reached after following the sequence,
            or None if the sequence does not exist.
        """

        node = self._root

        for token_id in sequence:
            if token_id not in node.children:
                return None
            node = node.children[token_id]

        return node

    def insert(self, sequence: tuple[int, ...], value: Any) -> None:
        """
        Insert a token sequence together with its associated value.
        """

        node = self._root

        for token_id in sequence:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]

        node.terminal = True
        node.value = value

    def allowed_next_tokens(self, sequence: tuple[int, ...]) -> set[int]:
        """
        Return every valid next token after the given sequence.
        """

        node = self._follow(sequence)

        if node is None:
            return set()

        return set(node.children.keys())

    def contains(self, sequence: tuple[int, ...]) -> bool:
        """
        Return True if the sequence is a complete entry.
        """

        node = self._follow(sequence)

        return node is not None and node.terminal

    def get(self, sequence: tuple[int, ...]) -> Any:
        """
        Return the object associated with a complete sequence.

        Raises
        ------
        ValueError
            If the sequence does not exist.
        """

        node = self._follow(sequence)

        if node is None or not node.terminal:
            raise ValueError(
                "Unknown token sequence."
            )

        return node.value
