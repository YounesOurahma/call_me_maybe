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
    """Represents the strict format your final generated output must follow."""
    prompt: str
    name: str
    parameters: Dict[str, Any]
