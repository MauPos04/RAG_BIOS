import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _parse_int(env_name: str, default: int, *, minimum: int | None = None) -> int:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        value = default
    else:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{env_name} debe ser un entero valido.") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{env_name} debe ser mayor o igual a {minimum}.")
    return value


def _parse_float(
    env_name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        value = default
    else:
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{env_name} debe ser un numero valido.") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{env_name} debe ser mayor o igual a {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{env_name} debe ser menor o igual a {maximum}.")
    return value


def _parse_bool(value: str, default: bool) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


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
    retrieval_multiplier: int
    require_citations: bool
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
        chunk_size=_parse_int("CHUNK_SIZE", 1000, minimum=1),
        chunk_overlap=_parse_int("CHUNK_OVERLAP", 200, minimum=0),
        top_k=_parse_int("TOP_K", 6, minimum=1),
        retrieval_multiplier=_parse_int("RETRIEVAL_MULTIPLIER", 3, minimum=1),
        require_citations=_parse_bool(
            os.getenv("REQUIRE_CITATIONS", "true"),
            default=True,
        ),
        temperature=_parse_float("TEMPERATURE", 0.0, minimum=0.0, maximum=2.0),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper().strip(),
    )

    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE.")

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger().setLevel(getattr(logging, settings.log_level, logging.INFO))
    return settings
