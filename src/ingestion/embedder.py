"""
Embedder
--------
Generates OpenAI embeddings for each TextChunk using text-embedding-3-small.

Features:
- Batched API calls (max 2048 inputs per request — OpenAI limit)
- Exponential backoff retry on rate-limit / transient errors
- Embedding stored on chunk.metadata["embedding"]
- Skips empty chunks silently
"""

import time
from typing import List

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from loguru import logger
import logging

from src.ingestion.chunker import TextChunk


# OpenAI max inputs per embedding request
_BATCH_SIZE = 512          # conservative — stays well under rate limits
_EMBED_DIMENSIONS = 1536   # text-embedding-3-small default output dim


def _batch(items: list, size: int):
    """Yield successive batches of `size` from items."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
    reraise=True,
)
def _embed_batch(
    client: OpenAI,
    texts: List[str],
    model: str,
) -> List[List[float]]:
    """Call OpenAI embeddings API for a single batch, with retry."""
    response = client.embeddings.create(
        model=model,
        input=texts,
        encoding_format="float",
    )
    # Response is ordered by index
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def embed_chunks(
    chunks: List[TextChunk],
    openai_api_key: str,
    model: str = "text-embedding-3-small",
    batch_size: int = _BATCH_SIZE,
) -> List[TextChunk]:
    """
    Embed all chunks and store the vector on chunk.metadata["embedding"].

    Processes in batches to avoid hitting rate limits.
    Returns the same list of chunks (mutated in-place).
    """
    client = OpenAI(api_key=openai_api_key)

    # Filter out empty chunks
    valid_chunks = [c for c in chunks if c.text.strip()]
    skipped = len(chunks) - len(valid_chunks)
    if skipped:
        logger.warning(f"Skipping {skipped} empty chunks")

    total = len(valid_chunks)
    embedded_count = 0

    logger.info(
        f"Embedding {total} chunks in batches of {batch_size} "
        f"using model '{model}'"
    )

    for batch_idx, batch in enumerate(_batch(valid_chunks, batch_size)):
        texts = [c.text for c in batch]

        t0 = time.perf_counter()
        vectors = _embed_batch(client, texts, model)
        elapsed = time.perf_counter() - t0

        for chunk, vector in zip(batch, vectors):
            chunk.metadata["embedding"] = vector
            chunk.metadata["embedding_model"] = model
            chunk.metadata["embedding_dim"] = len(vector)

        embedded_count += len(batch)
        logger.debug(
            f"Batch {batch_idx + 1}: {len(batch)} chunks embedded "
            f"in {elapsed:.2f}s "
            f"({embedded_count}/{total})"
        )

    logger.success(
        f"Embedding complete — {embedded_count} chunks | "
        f"model={model} | dim={_EMBED_DIMENSIONS}"
    )
    return chunks


def embed_query(
    query: str,
    openai_api_key: str,
    model: str = "text-embedding-3-small",
) -> List[float]:
    """Generate a single query embedding for retrieval."""
    if not query.strip():
        raise ValueError("Query must not be empty")

    client = OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        model=model,
        input=[query],
        encoding_format="float",
    )
    return response.data[0].embedding
