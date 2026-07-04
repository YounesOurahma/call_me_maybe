import json
from typing import Dict, List


class Vocabulary:
    def __init__(self, vocab_path: str):
        """Loads the vocabulary JSON file into memory."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.token_to_id: Dict[str, int] = json.load(f)

    def get_id(self, token_str: str) -> int:
        """
        Returns the integer token ID for a given string.
        Returns -1 if the token is not found in the vocabulary.
        """
        return self.token_to_id.get(token_str, -1)

    def get_allowed_ids(self, allowed_strings: List[str]) -> List[int]:
        """
        Takes a list of allowed strings and returns their corresponding IDs.
        This will be heavily used by your mask array.
        """
        allowed_ids: List = []
        for s in allowed_strings:
            token_id = self.get_id(s)
            if token_id != -1:
                allowed_ids.append(token_id)
        return allowed_ids
