"""
LLM service — provider-agnostic wrapper for chat completion and RAG answers.

Currently implements Google Gemini via the google-genai SDK.
System prompt enforces strict grounding and chunk citation.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from google import genai
from google.genai import types

from app.config import settings

RAG_SYSTEM_PROMPT = """You are DocuMind, an intelligent, grounded document assistant.
Your task is to answer the user's questions truthfully and accurately based ONLY on the provided document excerpts.

CRITICAL RULES:
1. Grounding: Rely strictly on the facts directly mentioned in the provided Context. Do NOT extrapolate, speculate, or bring in outside world knowledge not contained in the context.
2. Honest Refusal: If the provided Context does not contain enough information to answer the question, state clearly: "I cannot find the answer to that in the provided documents." Do not attempt to guess or hallucinate.
3. Citations: Whenever you state a claim derived from a context excerpt, cite the source chunk using brackets formatted as [chunk:<chunk_id>].
4. Clarity: Provide concise, clear, and well-structured answers using markdown formatting.
"""

CITATION_PATTERN = re.compile(r"\[chunk:([a-zA-Z0-9_\-]+)\]")


def build_rag_prompt(query: str, context_chunks: Sequence[dict[str, Any]]) -> str:
    """Build the user prompt combining context chunks and the user question."""
    if not context_chunks:
        context_str = "No document excerpts found."
    else:
        parts = []
        for chunk in context_chunks:
            chunk_id = chunk.get("id", "")
            doc_id = chunk.get("document_id", "")
            text = chunk.get("text", "")
            parts.append(f"---\n[Chunk ID: {chunk_id} | Document ID: {doc_id}]\n{text}\n---")
        context_str = "\n\n".join(parts)

    return (
        f"Context excerpts:\n"
        f"{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def extract_citations(answer: str, context_chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract [chunk:<chunk_id>] citations from the answer string and map them
    to chunk details from context_chunks.
    """
    found_chunk_ids = CITATION_PATTERN.findall(answer)
    chunk_map = {str(c.get("id")): c for c in context_chunks if c.get("id")}

    citations: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk_id in found_chunk_ids:
        if chunk_id not in seen:
            seen.add(chunk_id)
            chunk_data = chunk_map.get(chunk_id, {})
            text = chunk_data.get("text", "")
            snippet = (text[:150] + "...") if len(text) > 150 else (text if text else None)
            citations.append({
                "chunk_id": chunk_id,
                "document_id": str(chunk_data.get("document_id", "")),
                "text_snippet": snippet,
            })

    return citations


class LLMService:
    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider or settings.LLM_PROVIDER
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model = model or settings.LLM_MODEL
        self._gemini_client: genai.Client | None = None

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in the environment or .env."
                )
            self._gemini_client = genai.Client(api_key=self.api_key)
        return self._gemini_client

    def generate_rag_answer(
        self,
        query: str,
        context_chunks: Sequence[dict[str, Any]],
        chat_history: Sequence[dict[str, str]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Generate a cited, grounded answer using the configured LLM provider.

        Returns
        -------
        tuple[str, list[dict]]:
            (answer_text, extracted_citations)
        """
        # If no context chunks exist, short-circuit with honest refusal
        if not context_chunks:
            return "I cannot find the answer to that in the provided documents.", []

        user_content_str = build_rag_prompt(query, context_chunks)

        if self.provider == "gemini":
            client = self._get_gemini_client()

            contents: list[types.Content] = []
            if chat_history:
                for msg in chat_history:
                    role = "model" if msg.get("role") == "assistant" else "user"
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=msg.get("content", ""))],
                        )
                    )

            # Append the latest question with the grounding context
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_content_str)],
                )
            )

            config = types.GenerateContentConfig(
                system_instruction=RAG_SYSTEM_PROMPT,
                temperature=0.2,
            )

            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            answer = response.text or ""
            citations = extract_citations(answer, context_chunks)
            return answer, citations
        else:
            raise NotImplementedError(f"LLM provider '{self.provider}' is not supported.")


# Singleton instance
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
