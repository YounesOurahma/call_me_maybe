from typing import Any, Dict, List, Literal
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


_FUNCTION_DEFINITION_REQUIRED_KEYS = {
    "name",
    "description",
    "parameters",
    "returns",
}


class ParameterDef(BaseModel):
    """Represents the declared type of a single parameter or return value.
    """

    type: Literal["string", "number", "boolean", "integer"]

    model_config = ConfigDict(extra="forbid")


class FunctionDefinition(BaseModel):
    """Represents a single function the system is allowed to call.
    """

    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def check_required_keys(cls, data: Any) -> Any:
        """Reject anything that isn't a JSON object with exactly the
        four required keys, before pydantic even tries to coerce
        individual fields.
        """
        if not isinstance(data, dict):
            raise ValueError(
                "Each function definition must be a JSON object with "
                "'name', 'description', 'parameters' and 'returns'."
            )

        keys = set(data.keys())
        missing = _FUNCTION_DEFINITION_REQUIRED_KEYS - keys
        extra = keys - _FUNCTION_DEFINITION_REQUIRED_KEYS

        if missing:
            raise ValueError(
                "Function definition is missing required key(s): "
                f"{sorted(missing)}"
            )

        if extra:
            raise ValueError(
                "Function definition contains unexpected key(s): "
                f"{sorted(extra)}"
            )

        return data

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Function 'name' must not be empty.")
        return value

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Function 'description' must not be empty.")
        return value

    @model_validator(mode="after")
    def check_parameters(self) -> "FunctionDefinition":
        """Every declared parameter must have a non-empty name."""
        for parameter_name in self.parameters:
            if not parameter_name.strip():
                raise ValueError(
                    f"Function {self.name!r} has a parameter with an "
                    "empty name."
                )
        return self


class TestPrompt(BaseModel):
    """Represents a single prompt from the input tests file."""

    prompt: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("'prompt' must not be empty.")
        return value


class FunctionCall(BaseModel):
    """Represents the strict format the final generated output must follow."""

    prompt: str
    name: str
    parameters: Dict[str, Any]


def parse_function_definitions(
        data: Any, source: str) -> List[FunctionDefinition]:
    """Validate and build every ``FunctionDefinition`` found in ``data``.
    """
    if not isinstance(data, list):
        raise ValueError(
            f"{source} must contain a JSON array of function definitions."
        )

    if not data:
        raise ValueError(
            f"{source} must contain at least one function definition."
        )

    functions = [FunctionDefinition.model_validate(item) for item in data]
    functions_names = [function.name for function in functions]
    if len(set(functions_names)) < len(functions_names):
        raise ValueError(
            "Can't accept duplicated functions."
        )
    return functions


def parse_test_prompts(data: Any, source: str) -> List[TestPrompt]:
    """Validate and build every ``TestPrompt`` found in ``data``.
    """
    if not isinstance(data, list):
        raise ValueError(f"{source} must contain a JSON array of prompts.")

    if not data:
        raise ValueError(f"{source} must contain at least one prompt.")

    return [TestPrompt.model_validate(item) for item in data]
