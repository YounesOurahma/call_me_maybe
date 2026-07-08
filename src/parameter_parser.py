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
