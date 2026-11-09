"""Optional LLM-as-judge faithfulness scoring with Claude (needs ``ANTHROPIC_API_KEY``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from rag_eval_kit.answer.base import format_contexts
from rag_eval_kit.llm import MODEL, LLMError, call_claude, check_stop_reason, make_client, request_options
from rag_eval_kit.models import ScoredChunk

JUDGE_PROMPT = """You are grading a retrieval-augmented answer for faithfulness.
An answer is faithful when every claim in it is directly supported by the numbered sources.
Do not judge whether the answer is complete or helpful, only whether it is supported.
Return a score from 1 (mostly unsupported) to 5 (every claim supported) and a one-sentence rationale."""


class Verdict(BaseModel):
    """Structured judge output."""

    score: int = Field(ge=1, le=5, description="1 = mostly unsupported, 5 = fully supported.")
    faithful: bool = Field(description="True when every claim is supported (score 5 or 4).")
    rationale: str


class FaithfulnessJudge:
    """Scores whether an answer is supported by its retrieved contexts."""

    def __init__(self, client: Any | None = None, model: str = MODEL) -> None:
        self.client: Any = client if client is not None else make_client()
        self.model = model

    def judge(self, question: str, answer: str, contexts: Sequence[ScoredChunk]) -> Verdict:
        """Return the judge's verdict for one answer.

        Raises:
            LLMError: on API errors, refusals, truncation or a missing parsed output.
        """
        prompt = f"Sources:\n\n{format_contexts(contexts)}\n\nQuestion: {question}\n\nAnswer to grade:\n{answer}"
        response = call_claude(
            lambda: self.client.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=JUDGE_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=Verdict,
                **request_options(),
            )
        )
        check_stop_reason(response.stop_reason)
        verdict: Verdict | None = response.parsed_output
        if verdict is None:
            raise LLMError("judge returned no parseable verdict")
        return verdict
