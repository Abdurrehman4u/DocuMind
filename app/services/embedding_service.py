"""
Embedding service.

Wraps the sentence-transformers model so the embedding provider is
easily swappable. The model is loaded once at process startup and
reused across all requests (lazy singleton).

Current provider: sentence-transformers / all-MiniLM-L6-v2
    - Fully local, no API key, no cost.
    - 384-dimensional dense vectors.
    - Fast enough for dev/demo on CPU (< 1 s per batch of 32 chunks).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer

from app.config import settings

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load (and cache) the sentence-transformers model once per process."""
    return SentenceTransformer(settings.EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """
    Generate embedding vectors for a list of texts.

    Parameters
    ----------
    texts:
        Non-empty list of strings to embed. Each string should be at
        most a few hundred tokens long (the chunking step ensures this).

    Returns
    -------
    list[list[float]]
        One embedding vector per input text, in the same order.
        Each vector has 384 dimensions (all-MiniLM-L6-v2).
    """
    model = _get_model()
    embeddings = model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
    # Convert numpy arrays → plain Python lists for JSON-serializability
    return [emb.tolist() for emb in embeddings]


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Convenience wrapper around embed_texts for single-query retrieval.
    """
    return embed_texts([query])[0]
