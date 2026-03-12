import os

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .prompts import GROUNDING_PROMPT


@dataclass(slots=True)
class RetrievalResult:
    answer: str
    evidence: list[dict]


def chunk_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_vector_store(documents: list[Document], settings: Settings) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="models_cache",
    )
    return FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )


def answer_question(vector_store: FAISS, question: str, settings: Settings) -> RetrievalResult:
    retrieved_pairs = vector_store.similarity_search_with_relevance_scores(
        question,
        k=settings.top_k,
    )
    filtered_pairs = [
        (document, score)
        for document, score in retrieved_pairs
        if score >= settings.min_relevance_score
    ]

    if not filtered_pairs:
        return RetrievalResult(
            answer="No encontre esa informacion en los documentos cargados.",
            evidence=[],
        )

    context = "\n\n".join(document.page_content for document, _ in filtered_pairs)
    prompt = ChatPromptTemplate.from_template(GROUNDING_PROMPT)
    llm = ChatOpenAI(
        model=settings.openrouter_model,
        temperature=settings.temperature,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": settings.openrouter_http_referer,
            "X-Title": settings.openrouter_x_title,
        },
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    evidence = []
    for document, score in filtered_pairs:
        source = document.metadata.get("source", "Documento")
        location = _format_location(document.metadata)
        evidence.append(
            {
                "source": source,
                "location": location,
                "score": round(float(score), 3),
                "content": document.page_content,
            }
        )

    return RetrievalResult(answer=answer, evidence=evidence)


def _format_location(metadata: dict) -> str:
    if "page_number" in metadata:
        return f"pagina {metadata['page_number']}"
    if "sheet_name" in metadata:
        return f"hoja {metadata['sheet_name']}"
    if "element_index" in metadata:
        return f"bloque {metadata['element_index']}"
    return "fragmento"
