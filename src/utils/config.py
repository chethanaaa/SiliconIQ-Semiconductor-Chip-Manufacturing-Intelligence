from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_chat_model: str = Field("gpt-4o", env="OPENAI_CHAT_MODEL")
    rerank_strategy: str = Field("hybrid", env="RERANK_STRATEGY")
    rerank_model_provider: str = Field("openai", env="RERANK_MODEL_PROVIDER")
    openai_rerank_model: str = Field("gpt-4.1-mini", env="OPENAI_RERANK_MODEL")
    anthropic_rerank_model: str = Field(
        "claude-3-5-haiku-latest",
        env="ANTHROPIC_RERANK_MODEL",
    )
    langsmith_monitoring_enabled: bool = Field(
        False,
        env="LANGSMITH_MONITORING_ENABLED",
    )
    langsmith_api_key: str | None = Field(None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        "manufacturing-agentic-rag",
        env="LANGSMITH_PROJECT",
    )
    langsmith_endpoint: str | None = Field(None, env="LANGSMITH_ENDPOINT")

    # News API
    news_api_key: str = Field(..., env="NEWS_API_KEY")
    news_api_base_url: str = Field("https://newsapi.org/v2", env="NEWS_API_BASE_URL")

    # FRED API
    fred_api_key: str = Field(..., env="FRED_API_KEY")
    fred_api_base_url: str = Field("https://api.stlouisfed.org/fred", env="FRED_API_BASE_URL")

    # Vector store
    faiss_index_path: str = Field("data/vector_store/faiss_index", env="FAISS_INDEX_PATH")
    faiss_metadata_path: str = Field("data/vector_store/metadata.json", env="FAISS_METADATA_PATH")

    # Data paths
    pdf_raw_dir: str = Field("data/raw", env="PDF_RAW_DIR")
    pdf_processed_dir: str = Field("data/processed", env="PDF_PROCESSED_DIR")

    # Chunking
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")

    # Retrieval
    top_k_retrieval: int = Field(10, env="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(5, env="TOP_K_RERANK")

    # App
    log_level: str = Field("INFO", env="LOG_LEVEL")
    environment: str = Field("development", env="ENVIRONMENT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
