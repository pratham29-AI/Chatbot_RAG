from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required keys
    openai_api_key: str
    tavily_api_key: str

    # Model selection
    model_name: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # FAISS persistence
    faiss_index_path: str = "data/faiss_index"

    # Chunking parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval parameters
    retrieval_top_k: int = 4
    retrieval_score_threshold: float = 0.30  # minimum cosine similarity (0–1)

    # Session cap — drop oldest messages beyond this limit to control token usage
    max_session_messages: int = 50


settings = Settings()
