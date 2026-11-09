"""Answer generation: offline extractive, Claude-grounded, and a multi-hop agent."""

from rag_eval_kit.answer.agent import AgentResult, ResearchAgent
from rag_eval_kit.answer.base import REFUSAL_TEXT, Answer, Answerer, format_contexts
from rag_eval_kit.answer.claude import ClaudeAnswerer
from rag_eval_kit.answer.extractive import ExtractiveAnswerer

__all__ = [
    "REFUSAL_TEXT",
    "AgentResult",
    "Answer",
    "Answerer",
    "ClaudeAnswerer",
    "ExtractiveAnswerer",
    "ResearchAgent",
    "format_contexts",
]
