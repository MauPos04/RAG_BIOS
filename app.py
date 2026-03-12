from hashlib import sha1
import logging
from time import perf_counter

import streamlit as st

from src.rag_bios.config import load_settings
from src.rag_bios.document_loader import SUPPORTED_EXTENSIONS, load_uploaded_documents
from src.rag_bios.pipeline import answer_question, build_vector_store, chunk_documents


LOGGER = logging.getLogger(__name__)
WELCOME_MESSAGE = (
    "Carga uno o varios documentos y luego pregunta solo sobre su contenido. "
    "Si no hay evidencia suficiente, te lo dire explicitamente."
)

st.set_page_config(page_title="RAG BIOS", page_icon="📄", layout="wide")


def main() -> None:
    st.title("RAG BIOS")
    st.caption(
        "Asistente RAG con respuestas basadas unicamente en el contenido de los documentos cargados."
    )

    try:
        settings = load_settings()
    except ValueError as exc:
        st.error(f"Configuracion invalida: {exc}")
        return

    _init_session_state()

    with st.sidebar:
        st.header("Documentos")
        st.caption(f"Modelo configurado: {settings.openrouter_model}")
        st.caption(f"Embeddings: {settings.embedding_model}")

        uploaded_files = st.file_uploader(
            "Carga archivos PDF, DOCX, XLSX o TXT",
            type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
            accept_multiple_files=True,
        )
        process_clicked = st.button("Procesar documentos", use_container_width=True)
        reset_clicked = st.button("Reiniciar sesion", use_container_width=True)

        if reset_clicked:
            _reset_session()
            st.rerun()

        if process_clicked:
            if not uploaded_files:
                st.warning("Carga al menos un archivo antes de procesar.")
            elif not settings.openrouter_api_key:
                st.error("Configura OPENROUTER_API_KEY en tu archivo .env antes de continuar.")
            else:
                _process_documents(uploaded_files, settings)

        if st.session_state.get("processed_file_names"):
            st.success("Documentos listos para preguntas.")
            for name in st.session_state.processed_file_names:
                st.write(f"- {name}")

        if st.session_state.get("processed_file_warnings"):
            st.subheader("Advertencias")
            for warning in st.session_state.processed_file_warnings:
                st.warning(warning)

        if settings.log_level == "DEBUG":
            _render_debug_panel()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for warning in message.get("warnings", []):
                st.warning(warning)
            for evidence in message.get("evidence", []):
                _render_evidence(evidence)

    user_question = st.chat_input("Haz una pregunta sobre los documentos cargados")
    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            answer = "Primero debes cargar y procesar al menos un documento con contenido util."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            return

        if not settings.openrouter_api_key:
            answer = "Configura OPENROUTER_API_KEY antes de hacer preguntas."
            st.error(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            return

        with st.spinner("Buscando evidencia en los documentos..."):
            try:
                result = answer_question(
                    st.session_state.vector_store,
                    user_question,
                    settings,
                    st.session_state.source_documents,
                )
            except RuntimeError as exc:
                LOGGER.exception("Fallo controlado al responder la pregunta.")
                answer = str(exc)
                st.error(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                return
            except Exception:
                LOGGER.exception("Fallo inesperado al responder la pregunta.")
                answer = "Ocurrio un error inesperado al analizar la pregunta."
                st.error(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                return

        st.session_state.last_query_diagnostics = result.diagnostics

        st.markdown(result.answer)
        for warning in result.warnings:
            st.warning(warning)
        for evidence in result.evidence:
            _render_evidence(evidence)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "warnings": result.warnings,
                "evidence": result.evidence,
            }
        )


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "source_documents" not in st.session_state:
        st.session_state.source_documents = []
    if "processed_file_hash" not in st.session_state:
        st.session_state.processed_file_hash = None
    if "processed_file_names" not in st.session_state:
        st.session_state.processed_file_names = []
    if "processed_file_warnings" not in st.session_state:
        st.session_state.processed_file_warnings = []
    if "last_processing_stats" not in st.session_state:
        st.session_state.last_processing_stats = {}
    if "last_query_diagnostics" not in st.session_state:
        st.session_state.last_query_diagnostics = {}


def _process_documents(uploaded_files, settings) -> None:
    current_hash = _hash_uploaded_files(uploaded_files)
    if current_hash == st.session_state.processed_file_hash and st.session_state.vector_store is not None:
        st.info("Los documentos ya estaban procesados en esta sesion.")
        return

    st.session_state.processed_file_warnings = []
    st.session_state.last_processing_stats = {}
    st.session_state.last_query_diagnostics = {}

    try:
        with st.spinner("Extrayendo texto y creando indice vectorial..."):
            extraction_started_at = perf_counter()
            load_result = load_uploaded_documents(uploaded_files)
            extraction_elapsed_ms = round((perf_counter() - extraction_started_at) * 1000, 2)

            if not load_result.documents:
                st.session_state.vector_store = None
                st.session_state.source_documents = []
                st.session_state.processed_file_hash = None
                st.session_state.processed_file_names = []
                st.session_state.processed_file_warnings = load_result.warnings
                st.error(
                    "No pude extraer contenido util de los archivos cargados. "
                    "Revisa las advertencias e intenta con otro documento."
                )
                return

            chunk_started_at = perf_counter()
            chunked_documents = chunk_documents(load_result.documents, settings)
            chunk_elapsed_ms = round((perf_counter() - chunk_started_at) * 1000, 2)
            if not chunked_documents:
                raise ValueError("No se generaron fragmentos utiles para indexar.")

            index_started_at = perf_counter()
            vector_store = build_vector_store(chunked_documents, settings)
            index_elapsed_ms = round((perf_counter() - index_started_at) * 1000, 2)
    except ValueError as exc:
        LOGGER.exception("Error de validacion al procesar documentos.")
        st.error(str(exc))
        return
    except Exception:
        LOGGER.exception("Error inesperado al procesar documentos.")
        st.error("Ocurrio un error inesperado mientras procesaba los documentos.")
        return

    LOGGER.info(
        "Procesamiento completado | archivos=%s | documentos=%s | chunks=%s",
        len(load_result.processed_files),
        len(load_result.documents),
        len(chunked_documents),
    )

    st.session_state.vector_store = vector_store
    st.session_state.source_documents = load_result.documents
    st.session_state.processed_file_hash = current_hash
    st.session_state.processed_file_names = load_result.processed_files
    st.session_state.processed_file_warnings = load_result.warnings
    st.session_state.last_processing_stats = {
        "archivos_procesados": len(load_result.processed_files),
        "documentos_indexados": len(load_result.documents),
        "chunks_indexados": len(chunked_documents),
        "extraccion_ms": extraction_elapsed_ms,
        "chunking_ms": chunk_elapsed_ms,
        "indexacion_ms": index_elapsed_ms,
    }
    st.session_state.messages = st.session_state.messages[:1]


def _hash_uploaded_files(uploaded_files) -> str:
    bundle_hasher = sha1()
    file_signatures = []
    for uploaded_file in uploaded_files:
        content_hasher = sha1()
        content_hasher.update(uploaded_file.getbuffer())
        file_signatures.append((uploaded_file.name, content_hasher.hexdigest()))

    for file_name, content_hash in sorted(file_signatures):
        bundle_hasher.update(file_name.encode("utf-8"))
        bundle_hasher.update(content_hash.encode("utf-8"))

    return bundle_hasher.hexdigest()


def _render_evidence(evidence: dict) -> None:
    metric_value = evidence.get("metric_value")
    metric_label = evidence.get("metric_label", "score")
    title = (
        f"[{evidence['id']}] Fuente: {evidence['source']} | "
        f"{evidence['location']} | {metric_label} {metric_value}"
    )
    with st.expander(title):
        st.write(evidence["content"])


def _render_debug_panel() -> None:
    if st.session_state.last_processing_stats:
        st.subheader("Diagnostico de indexacion")
        for key, value in st.session_state.last_processing_stats.items():
            st.write(f"{key}: {value}")

    if st.session_state.last_query_diagnostics:
        st.subheader("Diagnostico de consulta")
        for key, value in st.session_state.last_query_diagnostics.items():
            st.write(f"{key}: {value}")


def _reset_session() -> None:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.vector_store = None
    st.session_state.source_documents = []
    st.session_state.processed_file_hash = None
    st.session_state.processed_file_names = []
    st.session_state.processed_file_warnings = []
    st.session_state.last_processing_stats = {}
    st.session_state.last_query_diagnostics = {}


if __name__ == "__main__":
    main()
