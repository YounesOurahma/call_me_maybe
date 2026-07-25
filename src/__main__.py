import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

from llm_sdk import Small_LLM_Model
from pydantic import ValidationError

from .decoder import Decoder
from .models import (
    FunctionCall,
    FunctionDefinition,
    TestPrompt,
    parse_function_definitions,
    parse_test_prompts,
)
from .parameter_parser import ParameterParser

DEFAULT_FUNCTIONS_PATH = Path("data/input/functions_definition.json")
DEFAULT_PROMPTS_PATH = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT_PATH = Path("data/output/function_calling_results.json")


def my_parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``functions_definition``, ``input``,
        and ``output`` attributes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_PATH,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
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


def _format_error(exc: ValueError) -> str:
    """Turn an exception into a short, user-facing message.

    ``pydantic.ValidationError`` prints a verbose multi-line block
    (including a link to its docs) by default, which is not something
    an end user running this CLI needs to see. This collapses it down
    to ``"<field>: <reason>"`` pairs; any other ``ValueError`` is
    returned as-is.
    """
    if isinstance(exc, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: "
            f"{error['msg']}"
            for error in exc.errors()
        )
        return details

    return str(exc)


def load_input_files(
    functions_path: Path,
    prompts_path: Path,
) -> tuple[list[FunctionDefinition], list[TestPrompt]]:
    """Load and validate both input files, failing gracefully.

    All schema validation (missing/extra keys, disallowed parameter
    types, empty prompts, empty function names, ...) is delegated to
    ``models.parse_function_definitions`` and
    ``models.parse_test_prompts``, both of which only ever raise
    ``ValueError`` (``pydantic.ValidationError`` is a ``ValueError``
    subclass) for anything that doesn't match the expected schema.
    This function therefore only needs to worry about file-system and
    JSON-syntax errors on top of that single exception type.

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
        functions = parse_function_definitions(
            functions_data, str(functions_path)
            )
    except FileNotFoundError:
        print(f"Error: functions definition file not found: {functions_path}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {functions_path}: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(
            f"Error: {functions_path} "
            f"does not match the expected schema: {_format_error(exc)}"
        )
        sys.exit(1)

    try:
        prompts_data = load_json(prompts_path)
        prompts = parse_test_prompts(prompts_data, str(prompts_path))
    except FileNotFoundError:
        print(f"Error: prompts file not found: {prompts_path}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {prompts_path}: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(
            f"Error: {prompts_path} does not match the expected schema: "
            f"{_format_error(exc)}"
        )
        sys.exit(1)

    return functions, prompts


def main() -> None:
    """Run the full function-calling pipeline end to end."""
    args = my_parse_args()

    print("Loading functions and prompts...")
    functions, prompts = load_input_files(
        args.functions_definition,
        args.input,
    )

    print("Loading model...")
    model = Small_LLM_Model()
    start_time = time.perf_counter()

    decoder = Decoder(functions, model)
    parser = ParameterParser(model)

    results: list[dict[str, Any]] = []

    print("Running inference...")

    for index, test in enumerate(prompts, start=1):

        print(f"[{index}/{len(prompts)}] {test.prompt}")

        try:
            function = decoder.select(test.prompt)
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
