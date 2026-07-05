import json
from typing import Dict, List


class Vocabulary:
    def __init__(self, path: str):
        """Loads the vocabulary JSON file into memory."""
        with open(path, 'r', encoding='utf-8') as f:
            self.token_to_id: Dict[str, int] = json.load(f)
        self.id_to_token: Dict[int, str] = {
            v: k
            for k, v in self.token_to_id.items()
        }
        self.tokens: List[str] = list(self.token_to_id.keys())

    def get_id(self, token: str) -> int:
        """
        Returns the integer token ID for a given string.
        Returns -1 if the token is not found in the vocabulary.
        """
        return self.token_to_id.get(token, -1)

    def get_token(self, token_id: int) -> str:
        """
        Return the string token ID for a given integer.
        """
        return self.id_to_token[token_id]

    def continuation_tokens(
            self,
            generated: str,
            candidates: List[str],
            ) -> List[str]:
        """
            Return every vocabulary token that can legally continue
            one of the candidate strings.

            Returns every token such that

                generated + token

            is still a prefix of at least one candidate.
        """

        allowed = []

        for token in self.tokens:

            new_text = generated + token

            for candidate in candidates:

                if candidate.startswith(new_text):
                    allowed.append(token)
                    break

        return allowed

    def continuation_ids(
        self,
        generated: str,
        candidates: List[str],
    ) -> List[int]:
        """
        Same as continuation_tokens(),
        but directly returns token ids.
        """

        return [
            self.get_id(token)
            for token in self.continuation_tokens(
                generated,
                candidates,
            )
        ]
