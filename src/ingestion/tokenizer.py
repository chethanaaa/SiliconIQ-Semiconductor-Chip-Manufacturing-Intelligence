"""
Tokenizer
---------
OpenAI tiktoken-based tokenization for each chunk.

Adds token_count to every TextChunk and flags chunks that exceed
the embedding model's context window (8191 tokens for text-embedding-3-small).
Oversized chunks are truncated with a warning so they never silently fail
at embedding time.
"""

from typing import List

import tiktoken
from loguru import logger

from src.ingestion.chunker import TextChunk


# text-embedding-3-small / text-embedding-3-large both use cl100k_base
_ENCODING_NAME = "cl100k_base"

# Hard limit for text-embedding-3-* models
_EMBEDDING_MAX_TOKENS = 8191


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


def truncate_to_limit(
    text: str,
    encoder: tiktoken.Encoding,
    max_tokens: int = _EMBEDDING_MAX_TOKENS,
) -> str:
    """Truncate text to max_tokens by decoding back from token ids."""
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def tokenize_chunk(chunk: TextChunk, encoder: tiktoken.Encoding) -> TextChunk:
    """
    Count tokens for a single chunk.
    If the chunk exceeds the embedding context window, truncate it and warn.
    Updates chunk.text, chunk.char_count, and chunk.metadata in-place.
    """
    token_count = count_tokens(chunk.text, encoder)

    if token_count > _EMBEDDING_MAX_TOKENS:
        logger.warning(
            f"Chunk {chunk.chunk_id} has {token_count} tokens "
            f"(limit {_EMBEDDING_MAX_TOKENS}) — truncating."
        )
        chunk.text = truncate_to_limit(chunk.text, encoder)
        chunk.char_count = len(chunk.text)
        token_count = _EMBEDDING_MAX_TOKENS

    chunk.metadata["token_count"] = token_count
    chunk.metadata["encoding"] = _ENCODING_NAME
    chunk.metadata["truncated"] = token_count == _EMBEDDING_MAX_TOKENS

    return chunk


def tokenize_chunks(chunks: List[TextChunk]) -> List[TextChunk]:
    """
    Tokenize all chunks.
    Loads the encoder once and reuses it across all chunks for efficiency.
    """
    encoder = _get_encoder()
    tokenized = [tokenize_chunk(chunk, encoder) for chunk in chunks]

    token_counts = [c.metadata["token_count"] for c in tokenized]
    truncated = sum(1 for c in tokenized if c.metadata.get("truncated"))

    logger.success(
        f"Tokenized {len(tokenized)} chunks | "
        f"min={min(token_counts)} max={max(token_counts)} "
        f"avg={int(sum(token_counts)/len(token_counts))} tokens"
        + (f" | {truncated} truncated" if truncated else "")
    )
    return tokenized
