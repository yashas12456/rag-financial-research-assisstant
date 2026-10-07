"""
LLM abstraction layer.

WHAT: A provider-agnostic interface for the final answer-generation step
      of the RAG pipeline. Concrete providers (Anthropic, OpenAI) each
      implement `LLMClient.generate()`; the rest of the app only depends
      on this interface, so switching providers/models is a one-line
      config change (LLM_PROVIDER, LLM_MODEL in .env).
WHY:  RAG's final step is: take the strongest retrieved evidence and ask
      the LLM to answer *using only that evidence*. This module owns the
      anti-hallucination system prompt (section 9 of the spec) so that
      guarantee lives in exactly one place.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.errors import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


SYSTEM_PROMPT = """You are an AI financial research analyst. You answer questions \
strictly using the numbered evidence passages provided below, each extracted from a \
real financial document with a known filename and page number.

Rules you must always follow:
1. Answer ONLY using facts stated in the provided evidence. Never use outside knowledge \
about companies, markets, or financial figures, even if you are confident it is correct.
2. Every factual claim in your answer must be traceable to one or more evidence passages. \
Refer to evidence using its bracketed number, e.g. [1], [2].
3. If the evidence does not contain enough information to answer the question, say so \
explicitly and clearly (e.g. "The provided documents do not contain enough information \
to answer this question."). Do not guess or fill gaps with assumptions.
4. Clearly distinguish stated facts from any inference you make, and avoid inference \
where possible.
5. Do not invent citations, page numbers, or filenames beyond what is given in the \
evidence below.
6. Be concise and precise -- prefer exact figures and direct statements over vague \
paraphrasing."""


@dataclass
class EvidencePassage:
    marker: str        # "[1]", "[2]", ...
    filename: str
    page_number: int
    text: str


@dataclass
class LLMResponse:
    answer: str
    insufficient_evidence: bool


def _build_user_prompt(question: str, evidence: list[EvidencePassage]) -> str:
    evidence_block = "\n\n".join(
        f"{e.marker} (Source: {e.filename}, Page {e.page_number}):\n{e.text}"
        for e in evidence
    )
    return (
        f"Evidence:\n\n{evidence_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the evidence above, citing sources with their "
        f"bracketed number."
    )


class LLMClient(ABC):
    @abstractmethod
    def _call(self, system: str, user: str) -> str:
        ...

    def answer_question(self, question: str, evidence: list[EvidencePassage]) -> LLMResponse:
        if not evidence:
            return LLMResponse(
                answer="The provided documents do not contain enough information to answer "
                       "this question. Try uploading a document that covers this topic.",
                insufficient_evidence=True,
            )
        try:
            raw = self._call(SYSTEM_PROMPT, _build_user_prompt(question, evidence))
        except Exception as exc:
            logger.exception("LLM call failed")
            raise LLMError(f"LLM generation failed: {exc}") from exc

        insufficient = "do not contain enough information" in raw.lower() or \
                       "not contain enough information" in raw.lower()
        return LLMResponse(answer=raw, insufficient_evidence=insufficient)


class AnthropicLLMClient(LLMClient):
    def __init__(self):
        import anthropic
        if not settings.LLM_API_KEY:
            raise LLMError("LLM_API_KEY is not set. Add it to your .env file.")
        self._client = anthropic.Anthropic(api_key=settings.LLM_API_KEY)

    def _call(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAILLMClient(LLMClient):
    def __init__(self):
        from openai import OpenAI
        if not settings.LLM_API_KEY:
            raise LLMError("LLM_API_KEY is not set. Add it to your .env file.")
        self._client = OpenAI(api_key=settings.LLM_API_KEY)

    def _call(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


def get_llm_client() -> LLMClient:
    if settings.LLM_PROVIDER == "anthropic":
        return AnthropicLLMClient()
    elif settings.LLM_PROVIDER == "openai":
        return OpenAILLMClient()
    raise LLMError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}")
