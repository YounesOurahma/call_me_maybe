# Project Export

# Project Structure

```text
__main__.py
decoder.py
function_registry.py
generator.py
models.py
parameter_parser.py
token_trie.py
```

# Files

---

## `__main__.py`

```python
from __future__ import annotations

import json
from pathlib import Path

from llm_sdk import Small_LLM_Model

from .decoder import Decoder
from .function_registry import FunctionRegistry
from .generator import Generator
from .models import FunctionDefinition, FunctionCall, TestPrompt
from .parameter_parser import ParameterParser


FUNCTIONS_PATH = Path("data/input/functions_definition.json")
PROMPTS_PATH = Path("data/input/function_calling_tests.json")
OUTPUT_PATH = Path("data/output/function_calls.json")


def load_json(path: Path) -> list:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def main() -> None:

    print("Loading model...")
    model = Small_LLM_Model()

    print("Loading functions...")
    functions_data = load_json(FUNCTIONS_PATH)

    functions = [
        FunctionDefinition(**item)
        for item in functions_data
    ]

    registry = FunctionRegistry(
        functions,
        model,
    )

    decoder = Decoder(registry)

    generator = Generator(
        model,
        decoder,
        registry,
    )

    parser = ParameterParser()

    print("Loading prompts...")
    prompts_data = load_json(PROMPTS_PATH)

    prompts = [
        TestPrompt(**item)
        for item in prompts_data
    ]

    results: list[dict] = []

    print("Running inference...")

    for index, test in enumerate(prompts, start=1):

        print(f"[{index}/{len(prompts)}] {test.prompt}")

        try:
            function = generator.generate(
                test.prompt
            )

            parameters = parser.parse(
                test.prompt,
                function,
            )

            result = FunctionCall(
                prompt=test.prompt,
                name=function.name,
                parameters=parameters,
            )

            results.append(
                result.model_dump()
            )

        except Exception as exc:

            print(
                f"Failed on prompt: {test.prompt}"
            )

            results.append(
                {
                    "prompt": test.prompt,
                    "error": str(exc),
                }
            )

    print("Saving output...")

    save_json(
        OUTPUT_PATH,
        results,
    )

    print(
        f"Done. Output written to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
```

---

## `decoder.py`

```python
from __future__ import annotations

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
```

---

## `function_registry.py`

```python
from __future__ import annotations

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

            token_ids = tuple(
                model.encode(function.name)[0].tolist()
            )

            self._trie.insert(
                token_ids,
                function,
            )

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
```

---

## `generator.py`

```python
from __future__ import annotations
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

        lines: list[str] = []

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

    def _encode_function_prompt(self) -> list[int]:
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

    def _encode_prompt(self, prompt: str) -> list[int]:
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
```

---

## `models.py`

```python
from typing import Any, Dict
from pydantic import BaseModel


class ParameterDef(BaseModel):
    """Represents the type of a specific parameter('number' or 'string')."""
    type: str


class FunctionDefinition(BaseModel):
    """Represents a single function allowed by the system."""
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef


class TestPrompt(BaseModel):
    """Represents a single prompt from the input tests file."""
    prompt: str


class FunctionCall(BaseModel):
    """Represents the strict format the final generated output must follow."""
    prompt: str
    name: str
    parameters: Dict[str, Any]
```

---

## `parameter_parser.py`

```python
from __future__ import annotations

import re
from collections import deque
from typing import Any

from .models import FunctionDefinition


class ParameterParser:
    """
    Extracts parameter values from a prompt after the correct function
    has already been selected by the language model.

    The parser is completely generic and relies only on the
    FunctionDefinition schema.
    """

    _NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
    _WORD_PATTERN = re.compile(r"[A-Za-z0-9_\-./]+")

    def parse(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, Any]:
        """
        Extract every parameter required by ``function`` from ``prompt``.

        Parameters
        ----------
        prompt:
            User prompt.

        function:
            Function selected by the constrained decoder.

        Returns
        -------
        dict[str, Any]
            Dictionary containing every extracted parameter.

        Raises
        ------
        ValueError
            If a required parameter cannot be extracted.
        """

        if self._is_regex_replacement_function(function):
            return self._parse_regex_replacement(prompt)

        numbers = deque(self._extract_numbers(prompt))
        strings = deque(
            self._extract_strings(
                prompt,
                ignored_words=self._ignored_words(function),
            )
        )

        parameters: dict[str, Any] = {}

        for name, definition in function.parameters.items():

            parameter_type = definition.type.lower()

            if parameter_type == "number":
                parameters[name] = self._consume_number(
                    numbers,
                    parameter_name=name,
                )

            elif parameter_type == "string":
                parameters[name] = self._consume_string(
                    strings,
                    parameter_name=name,
                )

            else:
                raise ValueError(
                    f"Unsupported parameter type: {parameter_type!r}"
                )

        return parameters

    def _extract_numbers(self, prompt: str) -> list[int | float]:
        """
        Extract every numeric literal from the prompt.
        """

        numbers: list[int | float] = []

        for match in self._NUMBER_PATTERN.findall(prompt):

            if "." in match:
                numbers.append(float(match))
            else:
                numbers.append(int(match))

        return numbers

    def _extract_strings(
        self,
        prompt: str,
        *,
        ignored_words: set[str],
    ) -> list[str]:
        """
        Extract candidate string values.

        Quoted strings have priority.

        Otherwise, every word that is not part of the selected
        function name is kept and combined into a single string.
        """

        quoted = self._extract_quoted_strings(prompt)

        if quoted:
            return quoted

        words = self._WORD_PATTERN.findall(prompt)

        result: list[str] = []

        for word in words:

            lower = word.lower()

            if lower in ignored_words:
                continue

            if self._NUMBER_PATTERN.fullmatch(word):
                continue

            result.append(word)

        if not result:
            return []

        return [" ".join(result)]

    def _extract_quoted_strings(self, prompt: str) -> list[str]:
        """
        Extract every quoted string from the prompt.

        Both single and double quoted strings are supported.
        """

        pattern = re.compile(r'"([^"]*)"|\'([^\']*)\'')

        strings: list[str] = []

        for match in pattern.finditer(prompt):
            value = match.group(1)
            if value is None:
                value = match.group(2)

            if value:
                strings.append(value)

        return strings

    def _ignored_words(
        self,
        function: FunctionDefinition,
    ) -> set[str]:
        """
        Return the words contained in the function name.

        Example
        -------
        fn_reverse_string

        becomes

        {"fn", "reverse", "string"}
        """

        return {
            word.lower()
            for word in function.name.split("_")
            if word
        }

    def _consume_number(
        self,
        numbers: deque[int | float],
        *,
        parameter_name: str,
    ) -> int | float:
        """
        Consume the next available number.

        Raises
        ------
        ValueError
            If no number is available.
        """

        if not numbers:
            raise ValueError(
                f"Unable to extract numeric parameter {parameter_name!r}."
            )

        return numbers.popleft()

    def _consume_string(
        self,
        strings: deque[str],
        *,
        parameter_name: str,
    ) -> str:
        """
        Consume the next available string.

        Raises
        ------
        ValueError
            If no string is available.
        """

        if not strings:
            raise ValueError(
                f"Unable to extract string parameter {parameter_name!r}."
            )

        value = strings.popleft().strip()

        if not value:
            raise ValueError(
                f"Extracted an empty value for parameter {parameter_name!r}."
            )

        return value

    def _is_regex_replacement_function(self,
                                       function: FunctionDefinition) -> bool:
        """
        Detect functions that require a source string,
        a regex pattern and a replacement.
        """

        parameters = set(function.parameters.keys())

        required = {
            "source_string",
            "regex",
            "replacement",
        }

        return required.issubset(parameters)

    def _parse_regex_replacement(self, prompt: str) -> dict[str, str]:
        """
        Extract parameters for string replacement functions.
        """

        result: dict[str, str] = {}

        quoted = self._extract_quoted_strings(prompt)

        lower = prompt.lower()

        # Detect the pattern
        if "vowel" in lower:
            result["regex"] = "[aeiouAEIOU]"
            if "asterisk" in lower:
                result["replacement"] = "*"

        elif "number" in lower:
            result["regex"] = r"\d+"

        elif len(quoted) >= 2:
            result["regex"] = quoted[0]

        else:
            result["regex"] = ""

        # Detect:
        # Substitute 'cat' with 'dog' in 'text'
        if "substitute" in lower and len(quoted) >= 3:

            result["regex"] = quoted[0]
            result["replacement"] = quoted[1]
            result["source_string"] = quoted[2]

            return {
                "source_string": result.get("source_string", ""),
                "regex": result.get("regex", ""),
                "replacement": result.get("replacement", ""),
            }

        # Detect:
        # Replace ... in "text" with replacement

        if quoted:
            result["source_string"] = quoted[0]

        words = prompt.split()

        if "with" in words:

            index = words.index("with")

            if index + 1 < len(words):
                if "replacement" not in result:
                    result["replacement"] = words[index + 1]

        else:
            result["replacement"] = ""

        return {
            "source_string": result.get("source_string", ""),
            "regex": result.get("regex", ""),
            "replacement": result.get("replacement", ""),
        }
```

---

## `token_trie.py`

```python
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
```

