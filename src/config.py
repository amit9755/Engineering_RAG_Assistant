# ============================================================
# src/config.py - Centralized Configuration Management
#
# Learning Note:
#   Using pydantic-settings lets us load config from environment
#   variables OR a .env file automatically. This is the standard
#   12-factor app pattern for production systems. Every setting
#   has a type, default, and description.
# ============================================================

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """
    All application settings loaded from environment variables.
    Pydantic validates types automatically - if GCP_PORT is set to
    "abc" it will raise an error immediately at startup, not later.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- GCP ---
    # --- Network policy ---
    # false = never contact anything outside the company network: only hosts under
    # INTERNAL_DOMAINS, localhost and private IP addresses (cloud LLMs, Hugging Face,
    # GitHub, telemetry and cloud Bitbucket / Jira are all blocked).
    allow_external_network: bool = Field(default=True, description="Allow calls outside the company network")
    internal_domains: str = Field(default="nxp.com", description="Comma-separated domains treated as internal")

    gcp_project_id: str = Field(default="local-dev", description="GCP Project ID")
    gcp_region: str = Field(default="us-central1", description="GCP Region")
    gcp_bucket_name: str = Field(default="rag-documents", description="GCS Bucket for documents")

    # --- LLM ---
    vertex_ai_model: str = Field(default="gemini-1.5-pro", description="Vertex AI model name")
    openai_api_key: str = Field(default="", description="OpenAI API key for fallback")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")

    # --- LiteLLM Gateway ---
    litellm_master_key: str = Field(default="sk-dev-key", description="LiteLLM gateway master key")
    litellm_budget_limit: float = Field(default=10.0, description="Max spend in USD before blocking")

    # --- Embeddings ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model or Vertex AI model name",
    )
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")

    # --- Vector Store ---
    vector_store_type: Literal["chroma", "vertexai"] = Field(
        default="chroma",
        description="'chroma' for local dev, 'vertexai' for production",
    )
    chroma_persist_dir: str = Field(default="./data/chroma_db", description="ChromaDB persistence directory")
    vertex_ai_index_id: str = Field(default="", description="Vertex AI Vector Search index ID")
    vertex_ai_index_endpoint_id: str = Field(default="", description="Vertex AI index endpoint ID")

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=10, description="Number of chunks to retrieve before re-ranking")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for semantic re-ranking",
    )
    # Learning Note: This threshold separates 'True Data' from 'Noisy Data'.
    # Chunks scoring below this value after re-ranking are considered noisy
    # and filtered OUT before being sent to the LLM.
    reranker_threshold: float = Field(
        default=0.3,
        description="Minimum re-ranker score to keep a chunk (True Data threshold)",
    )
    final_top_k: int = Field(default=5, description="Max chunks passed to LLM after re-ranking")

    # --- Guardrails ---
    enable_input_guardrails: bool = Field(default=True, description="Enable input validation guardrails")
    enable_output_guardrails: bool = Field(default=True, description="Enable output validation guardrails")
    enable_pii_detection: bool = Field(default=True, description="Detect and redact PII in inputs")

    # --- Evaluations ---
    enable_inline_evals: bool = Field(default=True, description="Run RAGAS evals inline per request")
    eval_sample_rate: float = Field(default=1.0, description="Fraction of requests to evaluate (0.0-1.0)")

    # --- Observability ---
    langfuse_public_key: str = Field(default="", description="LangFuse public key")
    langfuse_secret_key: str = Field(default="", description="LangFuse secret key")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", description="LangFuse host URL")
    bigquery_dataset: str = Field(default="rag_metrics", description="BigQuery dataset for metrics")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    log_level: str = Field(default="INFO", description="Logging level")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Deployment environment"
    )


# Singleton instance - import this everywhere instead of creating new instances
settings = Settings()
