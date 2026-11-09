from __future__ import annotations

import httpx2 as httpx
import pytest
from anthropic import APIConnectionError, APIStatusError, RateLimitError

from rag_eval_kit.answer import REFUSAL_TEXT, ClaudeAnswerer, ExtractiveAnswerer, ResearchAgent
from rag_eval_kit.answer.claude import CitedAnswer
from rag_eval_kit.llm import FALLBACK_BODY, FALLBACK_HEADERS, MODEL, LLMError, call_claude
from rag_eval_kit.models import Chunk, ScoredChunk
from rag_eval_kit.retrieval import BM25Retriever
from tests.fakes import FakeClient, message, text_block, tool_use


def scored(chunks: list[Chunk]) -> list[ScoredChunk]:
    return [ScoredChunk(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]


# --- extractive -----------------------------------------------------------------------


def test_extractive_cites_best_sentence(toy_chunks: list[Chunk]) -> None:
    bm25 = BM25Retriever(toy_chunks)
    answerer = ExtractiveAnswerer(idf=bm25.idf)
    contexts = bm25.search("How many requests per minute does the Team plan allow?", 3)
    answer = answerer.answer("How many requests per minute does the Team plan allow?", contexts)
    assert not answer.refused
    assert answer.citations[0] == "limits#0"
    assert "300 requests per minute" in answer.text
    assert "[1]" in answer.text


def test_extractive_adds_second_sentence_for_uncovered_terms(toy_chunks: list[Chunk]) -> None:
    bm25 = BM25Retriever(toy_chunks)
    question = "What does the Team plan cost and how many requests per minute does it allow?"
    answer = ExtractiveAnswerer(idf=bm25.idf, min_coverage=0.3, min_gain=0.2).answer(question, bm25.search(question, 3))
    assert set(answer.citations) == {"pricing#0", "limits#0"}


def test_extractive_refuses_when_terms_are_missing(toy_chunks: list[Chunk]) -> None:
    bm25 = BM25Retriever(toy_chunks)
    question = "Is there a SCIM provisioning endpoint for Kubernetes?"
    answer = ExtractiveAnswerer(idf=bm25.idf).answer(question, bm25.search(question, 3))
    assert answer.refused
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == ()


def test_extractive_refuses_without_contexts_or_terms(toy_chunks: list[Chunk]) -> None:
    answerer = ExtractiveAnswerer()
    assert answerer.answer("team plan", []).refused
    assert answerer.answer("the of and", scored(toy_chunks)).refused


# --- Claude answerer (fake client, no network) -------------------------------------------


def test_claude_answerer_maps_citations_and_sends_required_options(toy_chunks: list[Chunk]) -> None:
    parsed = CitedAnswer(answerable=True, answer="It costs $12 [2].", citations=[2, 9])
    client = FakeClient(message("end_turn", parsed=parsed))
    answer = ClaudeAnswerer(client=client).answer("price?", scored(toy_chunks[:2]))
    assert answer.citations == ("limits#0",)
    assert answer.text == "It costs $12 [2]."
    call = client.messages.calls[0]
    assert call["model"] == MODEL
    assert call["output_format"] is CitedAnswer
    assert call["output_config"] == {"effort": "medium"}
    assert call["extra_headers"] == FALLBACK_HEADERS
    assert call["extra_body"] == FALLBACK_BODY
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(call)


def test_claude_answerer_unanswerable_becomes_refusal(toy_chunks: list[Chunk]) -> None:
    client = FakeClient(message("end_turn", parsed=CitedAnswer(answerable=False, answer="", citations=[])))
    assert ClaudeAnswerer(client=client).answer("?", scored(toy_chunks)).refused


def test_claude_answerer_raises_on_refusal_stop_reason(toy_chunks: list[Chunk]) -> None:
    client = FakeClient(message("refusal"))
    with pytest.raises(LLMError, match="declined"):
        ClaudeAnswerer(client=client).answer("?", scored(toy_chunks))


def test_claude_answerer_skips_call_without_contexts() -> None:
    client = FakeClient()
    assert ClaudeAnswerer(client=client).answer("?", []).refused
    assert client.messages.calls == []


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://api.invalid/v1/messages"))


@pytest.mark.parametrize(
    ("exc", "match"),
    [
        (RateLimitError("slow down", response=_response(429), body=None), "Rate limited"),
        (APIStatusError("boom", response=_response(500), body=None), "API error 500"),
        (APIConnectionError(request=httpx.Request("POST", "https://api.invalid")), "Could not reach"),
    ],
)
def test_call_claude_translates_sdk_errors(exc: Exception, match: str) -> None:
    def fail() -> None:
        raise exc

    with pytest.raises(LLMError, match=match):
        call_claude(fail)


# --- agent --------------------------------------------------------------------------------


def make_agent(toy_chunks: list[Chunk], *responses: object, max_turns: int = 8) -> tuple[ResearchAgent, FakeClient]:
    client = FakeClient(*responses)
    agent = ResearchAgent(
        BM25Retriever(toy_chunks), {c.chunk_id: c for c in toy_chunks}, client=client, k=2, max_turns=max_turns
    )
    return agent, client


def test_agent_multi_hop_loop(toy_chunks: list[Chunk]) -> None:
    agent, client = make_agent(
        toy_chunks,
        message("tool_use", tool_use("t1", "search_docs", {"query": "SAML single sign-on plan"})),
        message("tool_use", tool_use("t2", "search_docs", {"query": "Business plan requests per minute"})),
        message("tool_use", tool_use("t3", "get_chunk", {"chunk_id": "sso#0"})),
        message("end_turn", text_block("SSO needs Business [sso#0]; Team allows 300/min [limits#0] [bogus#9].")),
    )
    result = agent.ask("What rate limit does the SSO plan have?")
    assert [name for name, _ in result.tool_calls] == ["search_docs", "search_docs", "get_chunk"]
    assert result.answer.citations == ("sso#0", "limits#0")
    assert "sso#0" in result.retrieved
    # every tool result was sent back in a user turn, paired with its tool_use id
    last_messages = client.messages.calls[-1]["messages"]
    tool_results = [m["content"][0] for m in last_messages if m["role"] == "user" and isinstance(m["content"], list)]
    assert [r["tool_use_id"] for r in tool_results] == ["t1", "t2", "t3"]
    assert all(not r["is_error"] for r in tool_results)


def test_agent_reports_tool_errors_to_model(toy_chunks: list[Chunk]) -> None:
    agent, client = make_agent(
        toy_chunks,
        message("tool_use", tool_use("t1", "get_chunk", {"chunk_id": "missing#0"})),
        message("end_turn", text_block("NOT_FOUND")),
    )
    result = agent.ask("anything")
    assert result.answer.refused
    error_result = client.messages.calls[1]["messages"][2]["content"][0]
    assert error_result["is_error"] is True


def test_agent_gives_up_after_max_turns(toy_chunks: list[Chunk]) -> None:
    loop = [message("tool_use", tool_use(f"t{i}", "search_docs", {"query": "plan"})) for i in range(2)]
    agent, _ = make_agent(toy_chunks, *loop, max_turns=2)
    with pytest.raises(LLMError, match="did not finish"):
        agent.ask("loop forever")


def test_agent_answerer_adapter_ignores_contexts(toy_chunks: list[Chunk]) -> None:
    agent, _ = make_agent(toy_chunks, message("end_turn", text_block("Costs $12 [pricing#0].")))
    assert agent.answer("price?", []).citations == ("pricing#0",)
