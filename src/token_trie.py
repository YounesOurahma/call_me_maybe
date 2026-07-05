from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class TrieNode(Generic[T]):
    """
    A node inside a TokenTrie.
    """

    children: Dict[int, "TrieNode[T]"] = field(default_factory=dict)
    terminal: bool = False
    value: Optional[T] = None


class TokenTrie(Generic[T]):
    """
    Generic trie storing token-id sequences.

    The trie is independent of the project domain.
    Any object can be associated with a sequence.
    """

    def __init__(self) -> None:
        self._root: TrieNode[T] = TrieNode()

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _follow(
        self,
        sequence: tuple[int, ...],
    ) -> Optional[TrieNode[T]]:
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
            node = node.children.get(token_id)

            if node is None:
                return None

        return node

    # ==========================================================
    # Public API
    # ==========================================================

    def insert(
        self,
        sequence: tuple[int, ...],
        value: T,
    ) -> None:
        """
        Insert a token sequence together with its associated value.
        """

        node = self._root

        for token_id in sequence:

            child = node.children.get(token_id)

            if child is None:
                child = TrieNode()
                node.children[token_id] = child

            node = child

        node.terminal = True
        node.value = value

    def allowed_next_tokens(
        self,
        sequence: tuple[int, ...],
    ) -> set[int]:
        """
        Return every valid next token after the given sequence.
        """

        node = self._follow(sequence)

        if node is None:
            return set()

        return set(node.children.keys())

    def contains(
        self,
        sequence: tuple[int, ...],
    ) -> bool:
        """
        Return True if the sequence is a complete entry.
        """

        node = self._follow(sequence)

        return node is not None and node.terminal

    def get(
        self,
        sequence: tuple[int, ...],
    ) -> T:
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

        assert node.value is not None

        return node.value
