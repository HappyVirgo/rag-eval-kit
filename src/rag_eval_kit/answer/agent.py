"""A small tool-calling research agent that can do multi-hop retrieval.

Claude gets two tools, ``search_docs`` and ``get_chunk``, and decides how many searches
to run (e.g. first find which plan includes SSO, then look up that plan's price). The
loop is written by hand so the trace of tool calls is easy to inspect and test.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_eval_kit.answer.base import Answer, refusal
from rag_eval_kit.llm import MODEL, LLMError, call_claude, check_stop_reason, make_client, request_options
from rag_eval_kit.models import Chunk, ScoredChunk
from rag_eval_kit.retrieval.base import Retriever

NOT_FOUND = "NOT_FOUND"
_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.\-]+#\d+)\]")

SYSTEM_PROMPT = f"""You are a documentation research assistant for a product.
Use the search_docs tool to find relevant passages; search again with refined queries when
a question needs facts from more than one place (for example, first find which plan has a
feature, then look up that plan's limits). Use get_chunk to read a passage in full.
Answer only from passages you retrieved, in one to three sentences, and cite each claim with
the chunk id in square brackets, e.g. [billing-plans#0].
If the documents do not contain the answer, reply with exactly {NOT_FOUND}."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_docs",
        "description": "Search the product documentation. Returns the most relevant chunks "
        "with their chunk ids, headings and text.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_chunk",
        "description": "Fetch the full text of one chunk by its id, e.g. billing-plans#0.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"chunk_id": {"type": "string", "description": "Chunk id."}},
            "required": ["chunk_id"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class AgentResult:
    """The agent's final answer plus the trace of tool calls it made."""

    answer: Answer
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)


class ResearchAgent:
    """Manual tool-use loop over ``search_docs`` / ``get_chunk``.

    Args:
        retriever: Retriever backing ``search_docs``.
        chunks: All chunks by id, backing ``get_chunk``.
        client: An ``anthropic.Anthropic`` client (created from the environment if omitted).
        k: Results returned per search.
        max_turns: Upper bound on model calls before giving up.
    """

    def __init__(
        self,
        retriever: Retriever,
        chunks: Mapping[str, Chunk],
        client: Any | None = None,
        k: int = 5,
        max_turns: int = 8,
        model: str = MODEL,
    ) -> None:
        self.retriever = retriever
        self.chunks = chunks
        self.client: Any = client if client is not None else make_client()
        self.k = k
        self.max_turns = max_turns
        self.model = model

    def run_tool(self, name: str, args: Mapping[str, Any], result: AgentResult) -> str:
        """Execute one tool call and return its JSON result string."""
        if name == "search_docs":
            hits = self.retriever.search(str(args.get("query", "")), self.k)
            result.retrieved.extend(h.chunk.chunk_id for h in hits if h.chunk.chunk_id not in result.retrieved)
            return json.dumps(
                [{"chunk_id": h.chunk.chunk_id, "heading": h.chunk.heading, "text": h.chunk.text} for h in hits]
            )
        if name == "get_chunk":
            chunk = self.chunks.get(str(args.get("chunk_id", "")))
            if chunk is None:
                raise KeyError(f"unknown chunk id: {args.get('chunk_id')!r}")
            return json.dumps({"chunk_id": chunk.chunk_id, "heading": chunk.heading, "text": chunk.text})
        raise KeyError(f"unknown tool: {name!r}")

    def ask(self, question: str) -> AgentResult:
        """Run the agent loop until Claude produces a final answer.

        Raises:
            LLMError: on API errors, refusals, truncation, or exceeding ``max_turns``.
        """
        result = AgentResult(answer=refusal())
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        for _ in range(self.max_turns):
            response = call_claude(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                    **request_options(),
                )
            )
            check_stop_reason(response.stop_reason)
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                result.answer = self._to_answer(text)
                return result
            results: list[dict[str, Any]] = []
            for block in tool_uses:
                result.tool_calls.append((block.name, dict(block.input)))
                try:
                    content, is_error = self.run_tool(block.name, block.input, result), False
                except KeyError as exc:
                    content, is_error = str(exc), True
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}
                )
            messages.append({"role": "user", "content": results})
        raise LLMError(f"agent did not finish within {self.max_turns} turns")

    def answer(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        """:class:`Answerer` adapter: ignores ``contexts`` and retrieves on its own."""
        return self.ask(question).answer

    def _to_answer(self, text: str) -> Answer:
        if not text or text.startswith(NOT_FOUND):
            return refusal()
        cited = [cid for cid in _CITATION_RE.findall(text) if cid in self.chunks]
        return Answer(text=text, citations=tuple(dict.fromkeys(cited)))
