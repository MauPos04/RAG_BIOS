import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    openrouter_http_referer: str
    openrouter_x_title: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_relevance_score: float
    temperature: float
    log_level: str


def load_settings() -> Settings:
    load_dotenv()

    settings = Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip(),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).strip(),
        openrouter_http_referer=os.getenv(
            "OPENROUTER_HTTP_REFERER", "http://localhost:8501"
        ).strip(),
        openrouter_x_title=os.getenv("OPENROUTER_X_TITLE", "RAG_BIOS").strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        top_k=int(os.getenv("TOP_K", "4")),
        min_relevance_score=float(os.getenv("MIN_RELEVANCE_SCORE", "0.20")),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper().strip(),
    )

    logging.getLogger().setLevel(getattr(logging, settings.log_level, logging.INFO))
    return settings
