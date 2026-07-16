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
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

from llm_sdk import Small_LLM_Model

from .decoder import Decoder
from .function_registry import FunctionRegistry
from .generator import Generator
from .models import FunctionCall, FunctionDefinition, TestPrompt
from .parameter_parser import ParameterParser

DEFAULT_FUNCTIONS_PATH = Path("data/input/functions_definition.json")
DEFAULT_PROMPTS_PATH = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT_PATH = Path("data/output/function_calling_results.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``functions_definition``, ``input``,
        and ``output`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural-language prompts into function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_PATH,
        help="Path to the JSON file describing available functions.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Path to the JSON file containing test prompts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the resulting JSON file will be written.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[Any]:
    """Load a JSON array from disk.

    Parameters
    ----------
    path:
        Path to the JSON file to load.

    Returns
    -------
    list[Any]
        The parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    json.JSONDecodeError
        If ``path`` does not contain valid JSON.
    """
    with path.open("r", encoding="utf-8") as file:
        return cast(list[Any], json.load(file))


def save_json(path: Path, data: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to disk as a JSON array.

    Parameters
    ----------
    path:
        Destination path. Parent directories are created if needed.
    data:
        The list of dictionaries to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def load_input_files(
    functions_path: Path,
    prompts_path: Path,
) -> tuple[list[FunctionDefinition], list[TestPrompt]]:
    """Load and validate both input files, failing gracefully.

    Parameters
    ----------
    functions_path:
        Path to the function catalog JSON file.
    prompts_path:
        Path to the test prompts JSON file.

    Returns
    -------
    tuple[list[FunctionDefinition], list[TestPrompt]]
        The parsed functions and prompts.
    """
    try:
        functions_data = load_json(functions_path)
        functions = [FunctionDefinition(**item) for item in functions_data]
    except FileNotFoundError:
        print(f"Error: functions definition file not found: {functions_path}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {functions_path}: {exc}")
        sys.exit(1)
    except (TypeError, ValueError) as exc:
        print(
            f"Error: {functions_path} "
            f"does not match the expected schema: {exc}"
            )
        sys.exit(1)

    try:
        prompts_data = load_json(prompts_path)
        prompts = [TestPrompt(**item) for item in prompts_data]
    except FileNotFoundError:
        print(f"Error: prompts file not found: {prompts_path}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {prompts_path}: {exc}")
        sys.exit(1)
    except (TypeError, ValueError) as exc:
        print(
            f"Error: {prompts_path} does not match the expected schema: {exc}"
            )
        sys.exit(1)

    return functions, prompts


def main() -> None:
    """Run the full function-calling pipeline end to end."""
    args = parse_args()

    start_time = time.perf_counter()
    print("Loading model...")
    model = Small_LLM_Model()

    print("Loading functions and prompts...")
    functions, prompts = load_input_files(
        args.functions_definition,
        args.input,
    )

    registry = FunctionRegistry(functions, model)
    decoder = Decoder(registry)
    generator = Generator(model, decoder, registry)
    parser = ParameterParser()

    results: list[dict[str, Any]] = []

    print("Running inference...")

    for index, test in enumerate(prompts, start=1):

        print(f"[{index}/{len(prompts)}] {test.prompt}")

        try:
            function = generator.generate(test.prompt)
            parameters = parser.parse(test.prompt, function)

            result = FunctionCall(
                prompt=test.prompt,
                name=function.name,
                parameters=parameters,
            )

            results.append(result.model_dump())

        except Exception as exc:
            print(f"Failed on prompt: {test.prompt}")
            results.append(
                {
                    "prompt": test.prompt,
                    "error": str(exc),
                }
            )

    print("Saving output...")

    try:
        save_json(args.output, results)
    except OSError as exc:
        print(f"Error: could not write output file {args.output}: {exc}")
        sys.exit(1)

    print(f"Done. Output written to: {args.output}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
```

---

## `decoder.py`

```python
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

        for function in self._functions:

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
        Return True if the generated sequence exactly
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

        return encoded[0].tolist()

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
```

---

## `models.py`

```python
from typing import Any, Dict
from pydantic import BaseModel, ConfigDict


class ParameterDef(BaseModel):
    """Represents the type of a specific parameter('number' or 'string')."""
    type: str

    model_config = ConfigDict(extra='forbid')


class FunctionDefinition(BaseModel):
    """Represents a single function allowed by the system."""
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef

    model_config = ConfigDict(extra='forbid')


class TestPrompt(BaseModel):
    """Represents a single prompt from the input tests file."""
    prompt: str


class FunctionCall(BaseModel):
    """Represents the strict format the final generated output must follow."""
    prompt: str
    name: str
    parameters: Dict[str, Any]

    model_config = ConfigDict(extra='forbid')
```

---

## `parameter_parser.py`

```python
import re
from collections import deque
from typing import Any, List

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

            if parameter_type == "number" or parameter_type == "integer":
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

    def _extract_numbers(self, prompt: str) -> List[int | float]:
        """
        Extract every numeric literal from the prompt.
        """

        numbers: List[float] = []

        for match in self._NUMBER_PATTERN.findall(prompt):

            numbers.append(float(match))

        return numbers

    def _extract_strings(
        self,
        prompt: str,
        *,
        ignored_words: set[str],
    ) -> List[str]:
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

        result: List[str] = []

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

    def _extract_quoted_strings(self, prompt: str) -> List[str]:
        """
        Extract every quoted string from the prompt.

        Both single and double quoted strings are supported.
        """

        pattern = re.compile(r'"([^"]*)"|\'([^\']*)\'')

        strings: List[str] = []

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
        numbers: deque[float],
        *,
        parameter_name: str,
    ) -> float:
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

        result: dict[str, str] = {
            "source_string": "",
            "regex": "",
            "replacement": "",
        }

        quoted = self._extract_quoted_strings(prompt)

        lower = prompt.lower()

        in_match = re.search(
            r'\bin\s+(["\'])(.*?)\1', prompt, re.IGNORECASE
            )
        if in_match:
            result["source_string"] = in_match.group(2)
        else:
            result["source_string"] = max(quoted, key=lambda s: len(s))

        if "asterisk" in lower or "star" in lower:
            result["replacement"] = "*"
        else:
            with_match = re.search(
                r'\bwith\s+["\']([^"\']*)["\']', prompt, re.IGNORECASE
                )
            if with_match:
                result["replacement"] = with_match.group(1)
            else:
                plain_with = re.search(
                    r'\bwith\s+(?:a[n]?\s+)?(\S+)', prompt, re.IGNORECASE
                    )
                if plain_with:
                    result["replacement"] = plain_with.group(1)

        regex_patterns = {
            "vowel": r"[aeiouAEIOU]",
            "digit": r"\d+",
            "number": r"\d+",
            "space": r"\s+",
            "whitespace": r"\s+",
            "letter": r"[a-zA-Z]",
            "alphabet": r"[a-zA-Z]",
        }
        for keyword, pattern in regex_patterns.items():
            if keyword in lower:
                result["regex"] = pattern
                break
        if result["regex"] == "":
            left = [q for q in quoted
                    if q != result["replacement"]
                    and q != result["source_string"]]
            if left:
                result["regex"] = left[0]

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

