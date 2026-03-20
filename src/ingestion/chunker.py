"""
Semantic Chunker
----------------
Splits extracted page text into semantically coherent chunks using
LlamaIndex's SemanticSplitterNodeParser.

Instead of fixed character windows, it embeds sentences and splits
where cosine similarity between adjacent sentences drops below a threshold —
producing chunks that are topically cohesive rather than arbitrarily cut.
"""

from dataclasses import dataclass, field
from typing import List

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from loguru import logger

from src.ingestion.pdf_extractor import DocumentContent, PageContent


@dataclass
class TextChunk:
    chunk_id: str           # "{source_file}__p{page_num}__c{chunk_idx}"
    text: str
    source_file: str
    page_num: int
    total_pages: int
    chunk_index: int        # index within the document
    char_count: int
    metadata: dict = field(default_factory=dict)


def _build_splitter(
    openai_api_key: str,
    embedding_model: str = "text-embedding-3-small",
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
) -> SemanticSplitterNodeParser:
    """
    buffer_size: number of sentences to group together before computing embeddings.
    breakpoint_percentile_threshold: percentile of cosine distance at which to split.
        Higher = fewer, larger chunks. Lower = more, granular chunks.
    """
    embed_model = OpenAIEmbedding(
        model=embedding_model,
        api_key=openai_api_key,
    )
    return SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
    )


def chunk_document(
    doc: DocumentContent,
    openai_api_key: str,
    embedding_model: str = "text-embedding-3-small",
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
) -> List[TextChunk]:
    """
    Semantically chunk all pages of a document.

    Each page is converted to a LlamaIndex Document, then the splitter
    determines natural semantic boundaries using embeddings.
    """
    splitter = _build_splitter(
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
    )

    # Build one LlamaIndex Document per page (preserves page-level provenance)
    llama_docs = []
    for page in doc.pages:
        if not page.text.strip():
            continue
        llama_docs.append(
            LlamaDocument(
                text=page.text,
                metadata={
                    "source_file": page.source_file,
                    "page_num": page.page_num,
                    "total_pages": doc.total_pages,
                    "extraction_method": page.extraction_method,
                },
            )
        )

    nodes = splitter.get_nodes_from_documents(llama_docs)

    all_chunks: List[TextChunk] = []
    for idx, node in enumerate(nodes):
        meta = node.metadata
        chunk_id = (
            f"{meta.get('source_file', doc.source_file)}"
            f"__p{meta.get('page_num', 0):04d}"
            f"__c{idx:04d}"
        )
        all_chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=node.get_content(),
                source_file=meta.get("source_file", doc.source_file),
                page_num=meta.get("page_num", 0),
                total_pages=meta.get("total_pages", doc.total_pages),
                chunk_index=idx,
                char_count=len(node.get_content()),
                metadata=meta,
            )
        )

    logger.info(
        f"{doc.source_file}: {len(llama_docs)} pages → {len(all_chunks)} semantic chunks"
    )
    return all_chunks


def chunk_documents(
    docs: List[DocumentContent],
    openai_api_key: str,
    embedding_model: str = "text-embedding-3-small",
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
) -> List[TextChunk]:
    """Semantically chunk a list of documents."""
    all_chunks: List[TextChunk] = []
    for doc in docs:
        all_chunks.extend(
            chunk_document(
                doc,
                openai_api_key=openai_api_key,
                embedding_model=embedding_model,
                buffer_size=buffer_size,
                breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            )
        )
    logger.success(f"Total semantic chunks across all documents: {len(all_chunks)}")
    return all_chunks
