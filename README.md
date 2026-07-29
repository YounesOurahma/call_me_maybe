*This project has been created as part of the 42 curriculum by yourahma.*

# Description

Call Me Maybe is a lightweight function-calling system built around a small language model. Given a natural language prompt and a list of available function definitions, the program predicts the most appropriate function to invoke and extracts its parameters.

The project demonstrates how constrained decoding can significantly improve the reliability of structured outputs by restricting the language model to produce only valid function names and correctly formatted parameter values.

The implementation follows the project constraints and uses only the Python standard library together with the provided SDK.


# Instructions

-> Prerequisites:

    -Python 3.10 or later
    -uv package manager

-> Installation:

    To set up the virtual environment and install all dependencies (including numpy, pydantic, and the local llm-sdk package):

make install
or directly via
uv sync
```
```bash
uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
```
```
Or execute default tasks using the included `Makefile`:

# Run the default pipeline (reads data/input/ and outputs to data/output/)
make run

# Run with interactive debugging (PDB)
make debug

# Clean bytecode caches and virtual environments
make clean

# Run static analysis and linting
make lint
make lint-strict

```

# Algorithm explanation

The program processes each input prompt in two main phases: function selection and parameter extraction.

### 1. Function Selection

When the program starts, all function definitions are loaded and validated using Pydantic. Each function name is then encoded into its corresponding sequence of LLM tokens. These encoded sequences are stored once and reused throughout the execution.

For every prompt, the decoder builds a context containing:

* the user request,
* the list of available functions,
* each function's description,
* the expected parameter types.

Generation then begins one token at a time. At each iteration, the language model produces logits for the entire vocabulary. Instead of allowing every token, the decoder computes the set of valid next tokens by comparing the already generated prefix against every registered function name.

Only tokens that keep the generated sequence as a valid prefix of at least one function are kept. Every other token receives a score of negative infinity (`-∞`), preventing the model from selecting it.

The highest-scoring remaining token is appended to the generated sequence, and the process repeats until one complete encoded function name is matched.

This constrained decoding approach guarantees that the generated function always belongs to the registered function catalog.

### 2. Parameter Extraction

After selecting the function, the program generates every parameter individually according to its declared type.

For **numbers** and **integers**, generation is constrained to tokens containing only digits, decimal points, minus signs and spaces. The allowed vocabulary is computed only once by scanning every token in the model vocabulary and caching the valid token IDs.

For **strings**, generation is constrained by allowing every token except those containing a double quote (`"`) or a newline (`\n`). This prevents prematurely closing the JSON string or generating invalid JSON.

For **booleans**, constrained decoding is simplified because only two valid outputs exist: `true` and `false`. Instead of masking the entire vocabulary, the implementation compares the logits of these two candidates directly and selects the one with the highest score.

Generated values are finally converted to their declared Python types before being inserted into the output JSON.

---

# Design Decisions

Several implementation choices were made to improve both correctness and efficiency.

### Prefix-based constrained decoding

Instead of allowing the language model to freely generate text, every generation step computes which tokens can legally continue at least one registered function name. Invalid tokens are masked before selecting the next token.

This guarantees that the final generated function always exists in the function definitions and eliminates invalid function names.

### Cached vocabulary filtering

Building the list of valid token IDs for numbers and strings requires scanning the entire vocabulary. Since the vocabulary never changes during execution, these lists are computed only once and cached.

This avoids repeatedly scanning thousands of vocabulary entries for every parameter generation.

### Type-specific generation

Different parameter types use different generation strategies.

* Numbers are restricted to numeric characters.
* Strings forbid only quotes and newlines to preserve valid JSON while keeping generation flexible.
* Booleans are handled separately by directly comparing the logits of `true` and `false`, avoiding unnecessary masking of the entire vocabulary.

Using specialized generators makes parameter extraction both simpler and faster.

### Pydantic validation

All input files are validated before execution using Pydantic models.

Model validators verify the JSON structure before field validation, field validators ensure individual values are valid, and post-validation checks verify relationships between fields.

This catches malformed input files before inference begins and keeps the rest of the code simpler.

### Modular architecture

The implementation separates responsibilities into independent components.

* `Decoder` performs constrained function selection.
* `ParameterParser` generates parameter values.
* `models.py` validates input and output schemas.
* `__main__.py` coordinates the complete pipeline.

This separation makes each component easier to understand, maintain and test.

---

# Performance Analysis

The implementation prioritizes correctness and deterministic outputs over unrestricted language generation.

Function selection is highly reliable because constrained decoding prevents the model from generating any function outside the registered catalog. Every generated function is therefore guaranteed to exist in the input definitions.

Parameter extraction also benefits from constrained decoding. Numeric parameters cannot contain arbitrary text, string generation preserves valid JSON formatting, and boolean generation always returns either `true` or `false`.

From a performance perspective, vocabulary scanning is performed only once. The resulting allowed token IDs are cached and reused for every subsequent generation, significantly reducing repeated computation.

Boolean generation is particularly efficient because only two candidate logits are compared instead of masking the complete vocabulary.

Overall, the program trades a small amount of preprocessing for faster repeated inference and substantially improved output reliability.

---

# Challenges Faced

The main challenge was understanding how to implement constrained decoding using only the provided SDK.

Unlike modern LLM libraries that provide built-in constrained generation, the SDK exposes only token encoding, decoding and logits. Therefore, token masking had to be implemented manually.

Another difficulty was parameter extraction. Simply allowing unrestricted generation often produced values with invalid characters or malformed JSON. This was solved by creating separate constrained generators for numbers, strings and booleans, each using token-level restrictions appropriate for its type.

Determining where generation should stop was another challenge. The implementation uses different stopping conditions depending on the parameter type while also handling nested brackets correctly during constrained generation.

Finally, extensive input validation was required to guarantee robust behaviour. Pydantic validators were introduced to detect malformed JSON files, duplicate function names, invalid identifiers and missing required fields before inference starts.

---

# Testing Strategy

Validation was performed at several levels.

**Schema validation**

Input JSON files were tested with missing fields, extra fields, duplicate function names, invalid identifiers and incorrect parameter definitions to verify that Pydantic rejected malformed inputs.

**Function selection**

Prompts were executed against different function catalogs to verify that constrained decoding always selected one of the registered functions and never generated an unknown function name.

**Parameter extraction**

Each supported parameter type (string, number, integer and boolean) was tested individually to ensure correct generation and type conversion.

**Failure handling**

Malformed generations and conversion failures were intentionally triggered to verify that default parameter values were generated instead of terminating execution.

**End-to-end testing**

The provided prompt dataset was executed from beginning to end, and every generated JSON output was manually verified against the expected function and parameter values.

---

# Example Usage

Run the program using the default input files:

```bash
uv run python -m src
```

or specify custom files:

```bash
python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

Example console output:

```text
Loading functions and prompts...
Loading model...
Running inference...

Generated: {name: fn_add_numbers}, {Parameters: {'a': 15, 'b': 27}}.

Saving output...
Done. Output written to: data/output/function_calling_results.json
Total execution time: 0.42 seconds
```

The generated output file contains one JSON object per prompt with the selected function name and the extracted parameters.


# Resources

youtube video:

https://www.youtube.com/watch?v=LPZh9BOjkQs

articles:

https://www.geeksforgeeks.org/nlp/what-is-tokenization/
https://www.geeksforgeeks.org/nlp/byte-pair-encoding-bpe-in-nlp/

# Ai usage

Artificial Intelligence was used as a development assistant throughout this project.

Its contributions included:
```
Explaining Python concepts.
Explaining constrained decoding techniques.
Reviewing code.
Suggesting refactoring ideas.
Helping understand error messages.
Assisting with documentation.
```