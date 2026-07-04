import json
import argparse
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional

from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition, TestPrompt, FunctionCall
from .vocab import Vocabulary
from .state_machine import JSONStateMachine, GenerationState
from .decoder import JSONConstraintDecoder


def load_functions(path: str) -> Optional[List[FunctionDefinition]]:
    """Load and parse function definitions from a JSON file.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        A list of FunctionDefinition objects, or None on error.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [FunctionDefinition(**item) for item in data]
    except FileNotFoundError:
        print(
            f"Error: functions definition file not found: {path}",
            file=sys.stderr
            )
        return None
    except json.JSONDecodeError as e:
        print(
            f"Error: invalid JSON in functions definition file: {e}",
            file=sys.stderr
            )
        return None
    except Exception as e:
        print(
            f"Error: failed to load functions definition: {e}",
            file=sys.stderr
              )
        return None


def load_test_prompts(path: str) -> Optional[List[TestPrompt]]:
    """Load and parse test prompts from a JSON file.

    Args:
        path: Path to the test prompts JSON file.

    Returns:
        A list of TestPrompt objects, or None on error.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [TestPrompt(**item) for item in data]
    except FileNotFoundError:
        print(f"Error: input file not found: {path}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in input file: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: failed to load input file: {e}", file=sys.stderr)
        return None


def generate(
    prompt: str,
    model: Small_LLM_Model,
    vocab: Vocabulary,
    allowed_functions: List[FunctionDefinition]
) -> Optional[FunctionCall]:
    """Generate a structured function call from a natural language prompt.

    Args:
        prompt: The natural language input.
        model: The loaded LLM model.
        vocab: The vocabulary mapping tokens to IDs.
        allowed_functions: The list of available function definitions.

    Returns:
        A FunctionCall object, or None on error.
    """
    try:
        state_machine = JSONStateMachine(allowed_functions)
        decoder = JSONConstraintDecoder(vocab, state_machine)

        input_ids: List[int] = model.encode(prompt).squeeze().tolist()
        generated_ids: List[int] = []
        max_tokens = 100 #debug
        while state_machine.state != GenerationState.DONE:
            if len(generated_ids) > max_tokens: #DEBUG
                print("  → Warning: max tokens reached", file=sys.stderr)
                break
            logits = model.get_logits_from_input_ids(input_ids)
            logits_np = np.array(logits)
            if logits_np.ndim > 1:
                logits_np = logits_np[-1, :]
            masked_logits = decoder.apply_mask(logits_np)
            if np.max(masked_logits) == float('-inf'):
                print(f"\n[DEBUG] FATAL: All tokens masked out!", file=sys.stderr)
                print(f"[DEBUG] State machine stuck at state: {state_machine.state}", file=sys.stderr)
                
                # See what the top 5 tokens the model ACTUALLY wanted to output were
                top_5_raw = np.argsort(logits_np)[-5:][::-1]
                print(f"[DEBUG] Model originally wanted to output (top 5):", file=sys.stderr)
                for t in top_5_raw:
                    # Decode each token to see its raw string representation
                    print(f"   Token ID {t} -> {repr(model.decode([t]))}", file=sys.stderr)
                    
                break
            next_token_id = int(np.argmax(masked_logits))
            next_token_str = model.decode([next_token_id])
            state_machine.update_state(next_token_str)
            input_ids.append(next_token_id)
            generated_ids.append(next_token_id)

        generated_text = model.decode(generated_ids)
        print(f"  → generated text: {repr(generated_text)}", flush=True) #debub
        result = json.loads(generated_text)
        return FunctionCall(**result)

    except json.JSONDecodeError as e:
        print(
            f"Error: generated output is not valid JSON: {e}",
            file=sys.stderr
              )
        return None
    except Exception as e:
        print(
            f"Error: generation failed for prompt '{prompt}': {e}",
            file=sys.stderr
            )
        return None


def main() -> None:
    """Entry point: parse arguments, run generation, save results."""
    base_dir = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Function calling tool using constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default=str(base_dir / "data" / "input" / "functions_definition.json"),
        help="Path to the functions definition JSON file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(
            base_dir / "data" / "input" / "function_calling_tests.json"
                    ),
        help="Path to the input prompts JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            base_dir / "data" / "output" / "function_calling_results.json"
            ),
        help="Path to the output JSON file."
    )
    args = parser.parse_args()

    allowed_functions = load_functions(args.functions_definition)
    if allowed_functions is None:
        sys.exit(1)

    test_prompts = load_test_prompts(args.input)
    if test_prompts is None:
        sys.exit(1)

    try:
        print("Loading model...")
        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()
        vocab = Vocabulary(vocab_path)
    except Exception as e:
        print(f"Error: failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    results: List[dict] = []
    for i, test in enumerate(test_prompts):
        print(f"[{i + 1}/{len(test_prompts)}] {test.prompt}")
        function_call = generate(test.prompt, model, vocab, allowed_functions)
        if function_call is not None:
            print(f"  → {function_call.name}({function_call.parameters})")
            results.append(function_call.model_dump())
        else:
            print("  → skipped due to error", file=sys.stderr)

    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"Error: failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
