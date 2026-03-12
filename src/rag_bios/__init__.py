from .config import Settings

__all__ = [
    "Settings",
    "build_vector_store",
    "answer_question",
    "chunk_documents",
]


def __getattr__(name: str):
    if name in {"build_vector_store", "answer_question", "chunk_documents"}:
        from .pipeline import answer_question, build_vector_store, chunk_documents

        exported_items = {
            "build_vector_store": build_vector_store,
            "answer_question": answer_question,
            "chunk_documents": chunk_documents,
        }
        return exported_items[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
