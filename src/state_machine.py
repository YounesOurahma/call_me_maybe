from enum import Enum, auto
from typing import List, Optional
from .models import FunctionDefinition


class GenerationState(Enum):
    """The discrete states of our JSON generation system."""
    START = auto()
    EXPECTING_KEY = auto()
    EXPECTING_COLON = auto()
    EXPECTING_OPEN_QUOTE = auto()
    EXPECTING_VALUE_PROMPT = auto()
    EXPECTING_VALUE_NAME = auto()
    EXPECTING_VALUE_PARAMS = auto()
    INSIDE_PARAMS = auto()
    EXPECTING_PARAM_COLON = auto()
    EXPECTING_PARAM_VALUE = auto()
    EXPECTING_COMMA_OR_END = auto()
    EXPECTING_OUTER_COMMA_OR_END = auto()
    DONE = auto()


class JSONStateMachine:
    def __init__(self, allowed_functions: List[FunctionDefinition]):
        self.state = GenerationState.START
        self.allowed_functions = allowed_functions
        self.selected_function: Optional[FunctionDefinition] = None
        self._next_value_type: str = ""
        self._filled_params: set = set()
        self._current_param: str = ""
        self._token_step: int = 0
        self._pending: str = ""

    def _get_name_tokens(self, name: str) -> List[str]:
        """Split function name into sub-tokens based on underscore boundaries.

        Args:
            name: Function name e.g. 'fn_add_numbers'

        Returns:
            List of tokens e.g. ['fn', '_add', '_numbers']
        """
        parts = name.split('_')
        result = [parts[0]]
        for part in parts[1:]:
            result.append('_' + part)
        return result

    def _get_param_tokens(self, param: str) -> List[str]:
        """Split param name into tokens including surrounding quotes.

        Args:
            param: Parameter name e.g. 'a', 'b', 's'

        Returns:
            List of tokens e.g. ['"a', '"'] or ['"', 'b', '"']
        """
        if len(param) == 1:
            return [f'"{param}', '"']
        else:
            return ['"', param, '"']

    def get_allowed_strings(self) -> List[str]:
        """Return the list of allowed next tokens given the current state."""
        if self.state == GenerationState.START:
            return ['{']

        elif self.state == GenerationState.EXPECTING_KEY:
            # '"prompt"' -> ['"', 'prompt', '"']
            # '"name"'   -> ['"name', '"']
            # '"parameters"' -> ['"', 'parameters', '"']
            if self._token_step == 0:
                return ['"', '"name']
            elif self._token_step == 1:
                if self._pending == '"name':
                    return ['"']
                else:
                    return ['prompt', 'parameters']
            elif self._token_step == 2:
                return ['"']
            return []

        elif self.state == GenerationState.EXPECTING_COLON:
            return [' :']

        elif self.state == GenerationState.EXPECTING_OPEN_QUOTE:
            if self._next_value_type == "parameters":
                return [' {', ' {"', ' {\n']
            else:
                return [' "']

        elif self.state == GenerationState.EXPECTING_VALUE_PROMPT:
            # free generation until closing '"'
            return []

        elif self.state == GenerationState.EXPECTING_VALUE_NAME:
            # '"fn_add_numbers"' -> ['"', 'fn', '_add', '_numbers', '"']
            # step 0: first token after ' "' is already consumed
            # we need: 'fn' first, then name-specific tokens, then '"'
            if self._token_step == 0:
                return ['fn']
            else:
                # find which functions still match _pending
                candidates = []
                for func in self.allowed_functions:
                    tokens = self._get_name_tokens(func.name)
                    if self._token_step <= len(tokens):
                        prefix = ''.join(tokens[:self._token_step])
                        if self._pending == prefix:
                            if self._token_step < len(tokens):
                                candidates.append(tokens[self._token_step])
                            else:
                                candidates.append('"')
                return list(set(candidates)) if candidates else ['"']

        elif self.state == GenerationState.EXPECTING_VALUE_PARAMS:
            return [' {', ' {"', ' {\n']

        elif self.state == GenerationState.INSIDE_PARAMS:
            if not self.selected_function:
                return ['}', ' }']
            unfilled = [
                p for p in self.selected_function.parameters
                if p not in self._filled_params
            ]
            if not unfilled:
                return [' }', '}']
            if self._token_step == 0:
                first_tokens: set = set()
                for p in unfilled:
                    tokens = self._get_param_tokens(p)
                    first_tokens.add(tokens[0])
                result = list(first_tokens)
                result.extend([' }', '}'])
                return result
            else:
                candidates = []
                for p in unfilled:
                    tokens = self._get_param_tokens(p)
                    prefix = ''.join(tokens[:self._token_step])
                    if self._pending == prefix and \
                       self._token_step < len(tokens):
                        candidates.append(tokens[self._token_step])
                return list(set(candidates)) if candidates else ['"']

        elif self.state == GenerationState.EXPECTING_PARAM_COLON:
            return [' :']

        elif self.state == GenerationState.EXPECTING_PARAM_VALUE:
            # free generation for param value
            return []

        elif self.state == GenerationState.EXPECTING_COMMA_OR_END:
            # inside parameters object
            return [',', ' ,', ' }', '}']

        elif self.state == GenerationState.EXPECTING_OUTER_COMMA_OR_END:
            # outside, in the main object
            return [' ,', ' }\n', ' }', '}']

        return []

    def update_state(self, generated_text: str) -> None:
        """Advance the state machine based on the generated token.

        Args:
            generated_text: The most recently generated token string.
        """
        cleaned = generated_text.strip()

        if self.state == GenerationState.START:
            if '{' in generated_text:
                self.state = GenerationState.EXPECTING_KEY
                self._token_step = 0
                self._pending = ''

        elif self.state == GenerationState.EXPECTING_KEY:
            if self._token_step == 0:
                self._pending = cleaned
                self._token_step = 1
            elif self._token_step == 1:
                if self._pending == '"name' and cleaned == '"':
                    self._next_value_type = "name"
                    self._token_step = 0
                    self._pending = ''
                    self.state = GenerationState.EXPECTING_COLON
                elif self._pending == '"':
                    self._pending += cleaned
                    self._token_step = 2
            elif self._token_step == 2:
                if 'prompt' in self._pending:
                    self._next_value_type = "prompt"
                elif 'parameters' in self._pending:
                    self._next_value_type = "parameters"
                self._token_step = 0
                self._pending = ''
                self.state = GenerationState.EXPECTING_COLON

        elif self.state == GenerationState.EXPECTING_COLON:
            if ':' in generated_text:
                self.state = GenerationState.EXPECTING_OPEN_QUOTE

        elif self.state == GenerationState.EXPECTING_OPEN_QUOTE:
            if self._next_value_type == "prompt":
                self.state = GenerationState.EXPECTING_VALUE_PROMPT
            elif self._next_value_type == "name":
                self.state = GenerationState.EXPECTING_VALUE_NAME
                self._token_step = 0
                self._pending = ''
            elif self._next_value_type == "parameters":
                self.state = GenerationState.EXPECTING_VALUE_PARAMS

        elif self.state == GenerationState.EXPECTING_VALUE_PROMPT:
            if cleaned.endswith('"') and len(cleaned) > 1:
                self.state = GenerationState.EXPECTING_OUTER_COMMA_OR_END

        elif self.state == GenerationState.EXPECTING_VALUE_NAME:
            self._pending += cleaned
            # check if closing '"' and a full name is assembled
            if cleaned == '"':
                for func in self.allowed_functions:
                    if func.name in self._pending:
                        self.selected_function = func
                        self._token_step = 0
                        self._pending = ''
                        self.state = GenerationState.EXPECTING_OUTER_COMMA_OR_END
                        break
            else:
                self._token_step += 1

        elif self.state == GenerationState.EXPECTING_VALUE_PARAMS:
            if '{' in generated_text:
                self.state = GenerationState.INSIDE_PARAMS
                self._token_step = 0
                self._pending = ''

        elif self.state == GenerationState.INSIDE_PARAMS:
            if '}' in cleaned:
                self.state = GenerationState.EXPECTING_OUTER_COMMA_OR_END
                self._token_step = 0
                self._pending = ''
            else:
                self._pending += cleaned
                if self.selected_function:
                    for p_name in self.selected_function.parameters:
                        if p_name not in self._filled_params:
                            tokens = self._get_param_tokens(p_name)
                            full = ''.join(tokens)
                            if self._pending.endswith(full):
                                self._current_param = p_name
                                self._next_value_type = "param_value"
                                self._token_step = 0
                                self._pending = ''
                                self.state = GenerationState.EXPECTING_PARAM_COLON
                                break
                    else:
                        self._token_step += 1

        elif self.state == GenerationState.EXPECTING_PARAM_COLON:
            if ':' in generated_text:
                self.state = GenerationState.EXPECTING_PARAM_VALUE

        elif self.state == GenerationState.EXPECTING_PARAM_VALUE:
            # detect end of value: closing '"' for strings, or ',' or '}'
            # for numbers the value ends when we see ',' or '}'
            if cleaned.endswith('"') and len(cleaned) > 1:
                self._filled_params.add(self._current_param)
                self.state = GenerationState.EXPECTING_COMMA_OR_END
            elif cleaned in [',', ' ,']:
                self._filled_params.add(self._current_param)
                self.state = GenerationState.INSIDE_PARAMS
                self._token_step = 0
                self._pending = ''
            elif '}' in cleaned:
                self._filled_params.add(self._current_param)
                self.state = GenerationState.EXPECTING_OUTER_COMMA_OR_END

        elif self.state == GenerationState.EXPECTING_COMMA_OR_END:
            if ',' in generated_text:
                self.state = GenerationState.INSIDE_PARAMS
                self._token_step = 0
                self._pending = ''
            elif '}' in generated_text:
                self.state = GenerationState.EXPECTING_OUTER_COMMA_OR_END

        elif self.state == GenerationState.EXPECTING_OUTER_COMMA_OR_END:
            if ',' in generated_text:
                self.state = GenerationState.EXPECTING_KEY
                self._token_step = 0
                self._pending = ''
            elif '}' in generated_text:
                self.state = GenerationState.DONE
