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
