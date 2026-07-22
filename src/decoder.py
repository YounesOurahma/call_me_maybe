"""Selects which registered function best matches a prompt.

Every candidate function name is scored directly: the model is
teacher-forced through each candidate's exact token sequence, and the
average per-token log-probability it assigns to that sequence becomes
the candidate's score. The function with the highest score wins.

This is deliberately *not* a token-by-token greedy walk. A greedy walk
has to commit to one branch every time two candidate names diverge,
using only the logits available at that one step -- if the model's
confidence at that specific token is noisy (very plausible for a
600M-parameter model), it can permanently commit to the wrong
function even though the *rest* of the correct function's name would
have been a much better fit than continuing down the wrong branch.

Scoring whole candidates instead means every function is judged on
how well its entire name fits the request, and the decision is made
only once all the evidence is in -- while the model is still never
able to produce anything other than one of the registered function
names.
"""

from typing import List, Tuple

import numpy as np

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition


class Decoder:
    """
    Picks the registered function that best matches a prompt.
    """

    _HEADER_INSTRUCTIONS = "Select the best matching function."

    def __init__(
        self,
        functions: List[FunctionDefinition],
        model: Small_LLM_Model,
    ) -> None:
        """Store the function catalog and pre-encode everything that
        does not depend on the prompt.

        Parameters
        ----------
        functions:
            Every function the model is allowed to select from.
        model:
            The language model used for scoring.

        Raises
        ------
        ValueError
            If ``functions`` is empty.
        """
        if not functions:
            raise ValueError(
                "Decoder requires at least one registered function."
            )

        self._functions = list(functions)
        self._model = model

        # Every candidate function name, pre-encoded exactly once.
        self._candidates: List[Tuple[int, ...]] = [
            tuple(model.encode(function.name)[0].tolist())
            for function in self._functions
        ]

        self._header_ids = self._build_header()

    def select(self, prompt: str) -> FunctionDefinition:
        """Select the function that best matches ``prompt``.

        Parameters
        ----------
        prompt:
            Natural language user request.

        Returns
        -------
        FunctionDefinition
            The registered function whose name the model assigns the
            highest average per-token log-probability to.
        """
        context_ids = (
            self._header_ids
            + self._model.encode(prompt)[0].tolist()
            + self._model.encode('\n{"name": "').squeeze(0).tolist()
        )

        best_function = self._functions[0]
        best_score = -np.inf

        for function, candidate in zip(self._functions, self._candidates):
            score = self._score_candidate(context_ids, candidate)

            if score > best_score:
                best_score = score
                best_function = function

        return best_function

    def _score_candidate(
        self,
        context_ids: List[int],
        candidate: Tuple[int, ...],
    ) -> float:
        """Return the average per-token log-probability the model
        assigns to generating ``candidate`` (teacher-forced), token by
        token, right after ``context_ids``.
        """
        log_prob_total = 0.0

        for index, token_id in enumerate(candidate):
            logits = np.asarray(
                self._model.get_logits_from_input_ids(
                    context_ids + list(candidate[:index])
                ),
                dtype=np.float64,
            )
            log_probs = self._log_softmax(logits)
            log_prob_total += float(log_probs[token_id])

        return log_prob_total / len(candidate)

    @staticmethod
    def _log_softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable log-softmax."""
        shifted = logits - np.max(logits)
        return shifted - np.log(np.sum(np.exp(shifted)))

    def _build_header(self) -> List[int]:
        """
        Encode the static part of the prompt: instructions plus the
        full catalog of available functions. Computed once and reused
        for every call to :py:meth:`select`.
        """
        lines = [self._HEADER_INSTRUCTIONS, "", "Available functions:"]

        for function in self._functions:
            lines.append(function.name)
            lines.append(f"Description: {function.description}")

            if function.parameters:
                lines.append("Parameters:")
                for name, parameter in function.parameters.items():
                    lines.append(f"- {name}: {parameter.type}")

            lines.append("")

        lines.append("Request: ")

        text = "\n".join(lines)

        return self._model.encode(text)[0].tolist()
