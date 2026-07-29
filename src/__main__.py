import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast, Tuple, List, Dict

from llm_sdk.llm_sdk import Small_LLM_Model
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


def error_on_duplicates(
        pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Validate the duplication of paramaters"""
    seen = set()
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"Duplicate key detected: {key}: {value}")
        seen.add(key)
    return dict(pairs)


def load_json(path: Path) -> List[Any]:
    """Load a JSON array from disk.
    """
    with path.open("r", encoding="utf-8") as file:
        return cast(
            List[Any], json.load(
                file, object_pairs_hook=error_on_duplicates)
                )


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries to disk as a JSON array.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4
        )


def _format_error(exc: ValueError) -> str:
    """Turn an exception into a short, user-facing message.
    """
    if isinstance(exc, ValidationError):
        details = "\n".join(
            f"{error['msg']}"
            for error in exc.errors()
        )
        return details

    return str(exc)


def load_input_files(
    functions_path: Path,
    prompts_path: Path,
) -> Tuple[List[FunctionDefinition], List[TestPrompt]]:
    """Load and validate both input files, failing gracefully.
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
    except OSError as exc:
        print(f"Error: could not read {functions_path}: {exc}")
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
    except OSError as exc:
        print(f"Error: could not read {prompts_path}: {exc}")
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

    results: List[Dict[str, Any]] = []

    print("Running inference...")

    for test in prompts:
        try:
            function = decoder.select(test.prompt)
            parameters = parser.parse(test.prompt, function)

            result = FunctionCall(
                prompt=test.prompt,
                name=function.name,
                parameters=parameters,
            )
            print("Generated: {name:",
                  f"{function.name}",
                  "},",
                  "{Parameters:",
                  f"{parameters}",
                  "}.")
            results.append(result.model_dump())

        except Exception:
            default_values = parser.default_parameters(function)
            result = FunctionCall(
                prompt=test.prompt,
                name=function.name,
                parameters=default_values,
            )
            print("Generated: {name:",
                  f"{function.name}",
                  "},",
                  "{Parameters:",
                  f"{default_values}",
                  "}.")
            results.append(result.model_dump())

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
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
