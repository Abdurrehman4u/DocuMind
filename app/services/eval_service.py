"""
Evaluation Service — assesses factual groundedness of RAG answers.

Design & Architectural Note:
----------------------------
The evaluation harness evaluates an answer against the chunks cited in the
Message.citations metadata (verified against the Chunk table in PostgreSQL).
Limitation:
The Message model (MDD §5) persists Message.citations (the subset of chunks cited
by the LLM), not the entire set of all top-k candidate chunks initially retrieved.
Consequently, evaluation verifies groundedness against the cited evidence rather
than the full candidate pool. If an answer mentions facts from a retrieved chunk that
it failed to cite, those facts are considered ungrounded.

Hybrid Evaluation Strategy:
---------------------------
1. LLM-as-judge: When an LLM provider and API key (GEMINI_API_KEY) are configured,
   Gemini acts as an impartial judge assessing claim entailment on a 0.0 - 1.0 scale.
2. Lexical Heuristic Fallback: When the LLM provider is offline or unconfigured,
   a keyword overlap ratio (excluding stopwords and citation markup) determines
   groundedness. Scores below 0.70 are flagged.
3. Citation Validation: Any citation with a chunk ID not existing in the database
   is rejected as valid evidence.
4. Honest Refusal: Answers explicitly acknowledging that context does not contain
   the answer receive a perfect score (1.0) and are unflagged.
"""

from __future__ import annotations

import json
import re
from typing import Any
from sqlalchemy.orm import Session

from app.models.message import Message
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.conversation import Conversation
from app.models.eval_result import EvalResult
from app.services.llm_service import get_llm_service

EVAL_JUDGE_PROMPT = """You are an impartial evaluator assessing the factual groundedness of an AI response compared to the provided document context.

Context:
{context}

Answer:
{answer}

Task:
Evaluate whether the claims made in the Answer are directly supported by the Context.
- If the answer faithfully reflects the context, assign a high score (0.80 - 1.00).
- If the answer introduces ungrounded facts, contradictions, or extrapolations not in the context, assign a low score (0.00 - 0.69) and set flagged to true.
- A score below 0.70 must have flagged set to true.

Respond strictly with a JSON object adhering to this schema:
{{
  "score": <float between 0.0 and 1.0>,
  "flagged": <boolean>,
  "notes": "<concise explanation of score>"
}}
"""

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is", "are",
    "was", "were", "this", "that", "it", "with", "as", "by", "from", "be", "have", "has",
}


def _heuristic_groundedness(answer: str, context: str) -> tuple[float, bool, str]:
    """
    Lexical overlap heuristic used when LLM judge is unavailable or unconfigured.
    """
    if not context.strip():
        # No context but answer asserts facts
        return 0.0, True, "No retrieved context was associated with this answer."

    # Clean answer by stripping out [chunk:...] citations
    clean_answer = re.sub(r"\[chunk:[^\]]+\]", "", answer)

    # Extract alphanumeric words > 2 chars
    answer_words = [
        w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", clean_answer)
        if w.lower() not in STOPWORDS
    ]

    if not answer_words:
        return 1.0, False, "Answer contains no substantive factual claims."

    context_lower = context.lower()
    matched = sum(1 for w in answer_words if w in context_lower)
    overlap_ratio = matched / len(answer_words)

    score = round(min(1.0, overlap_ratio), 2)
    flagged = score < 0.70
    status_str = "flagged" if flagged else "grounded"
    notes = f"Heuristic evaluation: {score * 100:.0f}% of content keywords supported by context ({status_str})."
    return score, flagged, notes


def evaluate_message_groundedness(
    message: Message,
    db: Session,
    user_id: str | None = None,
) -> EvalResult:
    """
    Evaluate groundedness for an assistant Message and persist the EvalResult row.

    Security & Scoping:
    Only chunks belonging to documents owned by the authenticated user are admitted
    as valid evidence context. Cited chunks belonging to other users or non-existent
    chunks are rejected.
    """
    answer = message.content or ""
    citations: list[dict[str, Any]] = message.citations or []

    # Resolve message owner if user_id was not explicitly provided
    if user_id is None and message.conversation_id:
        conv = db.query(Conversation).filter(Conversation.id == message.conversation_id).first()
        if conv:
            user_id = conv.user_id

    # 1. Check for honest refusal
    refusal_phrases = [
        "cannot find the answer",
        "not found in the provided documents",
        "do not contain the answer",
        "no document excerpts found",
    ]
    if any(phrase in answer.lower() for phrase in refusal_phrases):
        eval_row = EvalResult(
            message_id=message.id,
            groundedness_score=1.0,
            flagged=False,
            notes="Answer truthfully acknowledges lack of context without hallucinating.",
        )
        db.add(eval_row)
        db.commit()
        db.refresh(eval_row)
        return eval_row

    # 2. If citations exist, validate chunk IDs against database and enforce document ownership
    chunk_ids = [c.get("chunk_id") for c in citations if c.get("chunk_id")]
    valid_chunks: list[Chunk] = []

    if chunk_ids:
        chunk_query = (
            db.query(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .filter(Chunk.id.in_(chunk_ids))
        )
        if user_id:
            chunk_query = chunk_query.filter(Document.user_id == user_id)
        valid_chunks = chunk_query.all()

        # If citations were supplied but none exist or belong to the user's documents, reject
        if not valid_chunks:
            eval_row = EvalResult(
                message_id=message.id,
                groundedness_score=0.0,
                flagged=True,
                notes="Citations reference unknown chunk IDs or chunks not owned by the user; rejected as valid evidence.",
            )
            db.add(eval_row)
            db.commit()
            db.refresh(eval_row)
            return eval_row

    # 3. If no citations were provided at all
    if not citations:
        eval_row = EvalResult(
            message_id=message.id,
            groundedness_score=0.0,
            flagged=True,
            notes="No context excerpts were cited to support this answer.",
        )
        db.add(eval_row)
        db.commit()
        db.refresh(eval_row)
        return eval_row

    # Assemble context text from validated chunks
    context_parts = [c.text for c in valid_chunks if c.text]
    context_str = "\n\n".join(context_parts).strip()

    if not context_str:
        eval_row = EvalResult(
            message_id=message.id,
            groundedness_score=0.0,
            flagged=True,
            notes="Validated chunks contained no text content.",
        )
        db.add(eval_row)
        db.commit()
        db.refresh(eval_row)
        return eval_row

    # 4. Attempt LLM-as-judge evaluation
    score = 0.5
    flagged = False
    notes = ""

    try:
        llm = get_llm_service()
        if llm.api_key:
            client = llm._get_gemini_client()
            prompt = EVAL_JUDGE_PROMPT.format(context=context_str, answer=answer)
            from google.genai import types

            resp = client.models.generate_content(
                model=llm.model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            raw_text = resp.text or "{}"
            data = json.loads(raw_text)
            score = float(data.get("score", 0.5))
            flagged = bool(data.get("flagged", score < 0.70))
            notes = str(data.get("notes", "LLM-as-judge evaluated."))
        else:
            score, flagged, notes = _heuristic_groundedness(answer, context_str)
    except Exception:
        score, flagged, notes = _heuristic_groundedness(answer, context_str)

    eval_row = EvalResult(
        message_id=message.id,
        groundedness_score=round(score, 2),
        flagged=flagged,
        notes=notes,
    )
    db.add(eval_row)
    db.commit()
    db.refresh(eval_row)
    return eval_row
