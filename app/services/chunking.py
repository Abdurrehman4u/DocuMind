"""
Text chunking service.

Strategy: character-based recursive splitting with configurable
chunk size and overlap. Splits on paragraph breaks first (\n\n),
then single newlines, then spaces — matching the spirit of
LangChain's RecursiveCharacterTextSplitter without the dependency.

Default values (confirmed in Phase 0 planning):
    chunk_size   = 1500 characters  (~500 GPT tokens @ 3 chars/token average)
    chunk_overlap = 150 characters  (~50 tokens — 10 % of chunk size)
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    """A single chunk of text with its position within the source document."""
    index: int        # 0-based position in the chunk sequence
    text: str         # The chunk content


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", " ", ""]


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """
    Recursive character splitter.

    Tries each separator in order; if a segment is still too long,
    recurses with the remaining separators.
    """
    if not text:
        return []

    # Pick the first separator that actually appears in the text
    separator = separators[-1]  # fallback: split on every character
    for sep in separators:
        if sep == "" or sep in text:
            separator = sep
            break

    splits = text.split(separator) if separator else list(text)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for part in splits:
        part_len = len(part)
        # If a single part is too large, recurse with the next separators
        if part_len > chunk_size:
            if current_parts:
                chunks.append(separator.join(current_parts).strip())
                current_parts = []
                current_len = 0
            remaining_seps = separators[separators.index(separator) + 1:] if separator in separators else [""]
            chunks.extend(_split_text(part, chunk_size, chunk_overlap, remaining_seps or [""]))
            continue

        if current_len + len(separator) + part_len > chunk_size and current_parts:
            chunk_text = separator.join(current_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Keep overlap: drop earliest parts until we are within (chunk_size - chunk_overlap)
            while current_parts and current_len > chunk_overlap:
                removed = current_parts.pop(0)
                current_len -= len(removed) + len(separator)
            current_len = max(current_len, 0)

        current_parts.append(part)
        current_len += part_len + len(separator)

    # Flush remaining
    if current_parts:
        chunk_text = separator.join(current_parts).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """
    Split *text* into overlapping chunks.

    Parameters
    ----------
    text:
        The full plain-text content to split.
    chunk_size:
        Maximum character length of each chunk.
    chunk_overlap:
        Number of characters to carry over from the previous chunk.

    Returns
    -------
    list[TextChunk]
        Ordered list of chunks with 0-based indexes.
    """
    raw_chunks = _split_text(text.strip(), chunk_size, chunk_overlap, _SEPARATORS)
    # Deduplicate adjacent identical chunks that can arise from edge cases
    seen: set[str] = set()
    unique: list[str] = []
    for c in raw_chunks:
        if c and c not in seen:
            unique.append(c)
            seen.add(c)
    return [TextChunk(index=i, text=c) for i, c in enumerate(unique)]
