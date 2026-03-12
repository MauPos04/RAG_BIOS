from hashlib import sha1

import streamlit as st

from src.rag_bios.config import load_settings
from src.rag_bios.document_loader import SUPPORTED_EXTENSIONS, load_uploaded_documents
from src.rag_bios.pipeline import answer_question, build_vector_store, chunk_documents


st.set_page_config(page_title="RAG BIOS", page_icon="📄", layout="wide")


def main() -> None:
    settings = load_settings()

    st.title("RAG BIOS")
    st.caption(
        "Asistente RAG con respuestas basadas unicamente en el contenido de los documentos cargados."
    )

    _init_session_state()

    with st.sidebar:
        st.header("Documentos")
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
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
            answer = "Primero debes cargar y procesar al menos un documento."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            return

        with st.spinner("Buscando evidencia en los documentos..."):
            result = answer_question(st.session_state.vector_store, user_question, settings)

        st.markdown(result.answer)
        for evidence in result.evidence:
            _render_evidence(evidence)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "evidence": result.evidence,
            }
        )


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Carga uno o varios documentos y luego pregunta solo sobre su contenido. "
                    "Si no hay evidencia suficiente, te lo dire explicitamente."
                ),
            }
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_file_hash" not in st.session_state:
        st.session_state.processed_file_hash = None
    if "processed_file_names" not in st.session_state:
        st.session_state.processed_file_names = []


def _process_documents(uploaded_files, settings) -> None:
    current_hash = _hash_uploaded_files(uploaded_files)
    if current_hash == st.session_state.processed_file_hash and st.session_state.vector_store is not None:
        st.info("Los documentos ya estaban procesados en esta sesion.")
        return

    with st.spinner("Extrayendo texto y creando indice vectorial..."):
        documents = load_uploaded_documents(uploaded_files)
        chunked_documents = chunk_documents(documents, settings)
        vector_store = build_vector_store(chunked_documents, settings)

    st.session_state.vector_store = vector_store
    st.session_state.processed_file_hash = current_hash
    st.session_state.processed_file_names = [uploaded_file.name for uploaded_file in uploaded_files]
    st.session_state.messages = st.session_state.messages[:1]


def _hash_uploaded_files(uploaded_files) -> str:
    hasher = sha1()
    for uploaded_file in uploaded_files:
        hasher.update(uploaded_file.name.encode("utf-8"))
        hasher.update(str(uploaded_file.size).encode("utf-8"))
    return hasher.hexdigest()


def _render_evidence(evidence: dict) -> None:
    title = f"Fuente: {evidence['source']} | {evidence['location']} | score {evidence['score']}"
    with st.expander(title):
        st.write(evidence["content"])


def _reset_session() -> None:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Carga uno o varios documentos y luego pregunta solo sobre su contenido. "
                "Si no hay evidencia suficiente, te lo dire explicitamente."
            ),
        }
    ]
    st.session_state.vector_store = None
    st.session_state.processed_file_hash = None
    st.session_state.processed_file_names = []


if __name__ == "__main__":
    main()
