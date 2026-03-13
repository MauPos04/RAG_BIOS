import json
import logging
import pickle
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .config import Settings
from .pipeline import load_vector_store


LOGGER = logging.getLogger(__name__)
MANIFEST_FILE_NAME = "manifest.json"
DOCUMENTS_FILE_NAME = "documents.pkl"


@dataclass(slots=True)
class CachedIndexBundle:
    vector_store: Any
    source_documents: list[Document]
    processed_file_names: list[str]
    metadata: dict[str, Any]


def build_cache_key(bundle_hash: str, settings: Settings) -> str:
    key_hasher = sha1()
    key_parts = [
        bundle_hash,
        settings.embedding_model,
        str(settings.chunk_size),
        str(settings.chunk_overlap),
        ",".join(settings.document_languages),
    ]
    for part in key_parts:
        key_hasher.update(part.encode("utf-8"))
    return key_hasher.hexdigest()


def load_cached_bundle(cache_key: str, settings: Settings) -> CachedIndexBundle | None:
    cache_path = _cache_path(cache_key, settings)
    manifest_path = cache_path / MANIFEST_FILE_NAME
    documents_path = cache_path / DOCUMENTS_FILE_NAME

    if not manifest_path.exists() or not documents_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _manifest_matches_settings(manifest, settings):
            LOGGER.info("Cache persistente ignorado por configuracion distinta: %s", cache_path)
            return None

        with documents_path.open("rb") as file_handle:
            source_documents = pickle.load(file_handle)

        vector_store = load_vector_store(cache_path, settings)
        LOGGER.info("Indice cargado desde cache persistente: %s", cache_path)
        return CachedIndexBundle(
            vector_store=vector_store,
            source_documents=source_documents,
            processed_file_names=manifest.get("processed_file_names", []),
            metadata=manifest,
        )
    except Exception:
        LOGGER.exception("No pude cargar el cache persistente desde %s", cache_path)
        return None


def save_cached_bundle(
    cache_key: str,
    settings: Settings,
    *,
    bundle_hash: str,
    vector_store: Any,
    source_documents: list[Document],
    processed_file_names: list[str],
    document_count: int,
    chunk_count: int,
) -> None:
    cache_path = _cache_path(cache_key, settings)
    cache_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "bundle_hash": bundle_hash,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "document_languages": settings.document_languages,
        "processed_file_names": processed_file_names,
        "document_count": document_count,
        "chunk_count": chunk_count,
    }

    vector_store.save_local(str(cache_path))
    with (cache_path / DOCUMENTS_FILE_NAME).open("wb") as file_handle:
        pickle.dump(source_documents, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    (cache_path / MANIFEST_FILE_NAME).write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Indice persistido en cache local: %s", cache_path)


def _cache_path(cache_key: str, settings: Settings) -> Path:
    return Path(settings.index_cache_dir) / cache_key


def _manifest_matches_settings(manifest: dict[str, Any], settings: Settings) -> bool:
    return (
        manifest.get("embedding_model") == settings.embedding_model
        and manifest.get("chunk_size") == settings.chunk_size
        and manifest.get("chunk_overlap") == settings.chunk_overlap
        and manifest.get("document_languages") == settings.document_languages
    )
