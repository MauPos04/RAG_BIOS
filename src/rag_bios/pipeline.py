import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

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


LOGGER = logging.getLogger(__name__)
ABSTENTION_ANSWER = "No encontre esa informacion en los documentos cargados."
CITATION_PATTERN = re.compile(r"\[(E\d+)\]")
DATE_PATTERN = re.compile(r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b")
ROW_PATTERN = re.compile(r"\bfila\s+(\d+)\b", re.IGNORECASE)
VALUE_LOOKUP_TERMS = ("cual", "cuál", "valor", "dato", "dime", "muestra", "muéstrame")


@dataclass(slots=True)
class RetrievalResult:
    answer: str
    evidence: list[dict]
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


@dataclass(slots=True)
class StructuredSelection:
    matches: list[tuple[Document, float]] = field(default_factory=list)
    matched_columns: list[str] = field(default_factory=list)
    matched_dates: list[str] = field(default_factory=list)
    matched_rows: list[int] = field(default_factory=list)
    exact_answer: str | None = None


@dataclass(slots=True)
class StructuredTextSelection:
    matches: list[tuple[Document, float]] = field(default_factory=list)
    exact_answer: str | None = None


LEXICAL_STOPWORDS = {
    "como",
    "con",
    "cual",
    "cuanto",
    "de",
    "del",
    "donde",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "hay",
    "las",
    "los",
    "nota",
    "notas",
    "para",
    "por",
    "que",
    "segun",
    "solo",
    "sobre",
    "una",
    "uno",
    "unos",
    "unas",
    "usar",
    "uso",
    "y",
}


def chunk_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunked_documents: list[Document] = []
    for document in documents:
        if document.metadata.get("file_type") == "xlsx":
            chunked_documents.append(document)
            continue
        chunked_documents.extend(splitter.split_documents([document]))
    return chunked_documents


@lru_cache(maxsize=4)
def _get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    LOGGER.info("Inicializando modelo de embeddings: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="models_cache",
    )


def build_vector_store(documents: list[Document], settings: Settings) -> FAISS:
    if not documents:
        raise ValueError("No hay contenido util para construir el indice vectorial.")

    embeddings = _get_embeddings(settings.embedding_model)
    return FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )


def answer_question(
    vector_store: FAISS,
    question: str,
    settings: Settings,
    source_documents: list[Document] | None = None,
) -> RetrievalResult:
    diagnostics = {
        "model": settings.openrouter_model,
        "retrieval_mode": "semantic",
        "llm_called": False,
    }

    structured_selection = _select_structured_xlsx_matches(
        question,
        source_documents or [],
        settings.top_k,
    )
    if structured_selection.matches:
        diagnostics.update(
            {
                "retrieval_mode": "structured_xlsx",
                "retrieved_pairs": len(structured_selection.matches),
                "selected_pairs": len(structured_selection.matches),
                "matched_columns": structured_selection.matched_columns,
                "matched_dates": structured_selection.matched_dates,
                "matched_rows": structured_selection.matched_rows,
            }
        )
        LOGGER.info(
            "Lookup estructurado XLSX | columnas=%s | fechas=%s | filas=%s | matches=%s",
            structured_selection.matched_columns,
            structured_selection.matched_dates,
            structured_selection.matched_rows,
            len(structured_selection.matches),
        )
        evidence = _build_evidence(
            structured_selection.matches,
            metric_label="coincidencia",
        )
        if structured_selection.exact_answer:
            diagnostics["citation_mode"] = "structured_exact"
            return RetrievalResult(
                answer=structured_selection.exact_answer,
                evidence=evidence,
                diagnostics=diagnostics,
            )
        return _answer_from_evidence(question, evidence, settings, diagnostics)

    structured_text_selection = _select_structured_txt_matches(
        question,
        source_documents or [],
        settings.top_k,
    )
    if structured_text_selection.matches:
        diagnostics.update(
            {
                "retrieval_mode": "structured_txt",
                "retrieved_pairs": len(structured_text_selection.matches),
                "selected_pairs": len(structured_text_selection.matches),
            }
        )
        LOGGER.info(
            "Lookup estructurado TXT | matches=%s",
            len(structured_text_selection.matches),
        )
        evidence = _build_evidence(
            structured_text_selection.matches,
            metric_label="coincidencia",
        )
        if structured_text_selection.exact_answer:
            diagnostics["citation_mode"] = "structured_exact"
            return RetrievalResult(
                answer=structured_text_selection.exact_answer,
                evidence=evidence,
                diagnostics=diagnostics,
            )
        return _answer_from_evidence(question, evidence, settings, diagnostics)

    retrieval_started_at = perf_counter()
    retrieval_k = max(settings.top_k, settings.top_k * settings.retrieval_multiplier, 50)
    retrieved_pairs = vector_store.similarity_search_with_score(
        question,
        k=retrieval_k,
    )
    retrieval_elapsed_ms = round((perf_counter() - retrieval_started_at) * 1000, 2)

    ranked_pairs = _rank_retrieved_pairs(question, retrieved_pairs)
    selected_pairs = ranked_pairs[: settings.top_k]

    diagnostics.update(
        {
            "retrieval_k": retrieval_k,
            "retrieved_pairs": len(retrieved_pairs),
            "selected_pairs": len(selected_pairs),
            "retrieval_metric": "distance",
            "retrieval_distances": [round(float(score), 3) for _, score in ranked_pairs],
            "lexical_overlaps": [
                _lexical_overlap(question, document.page_content)
                for document, _ in ranked_pairs
            ],
            "retrieval_ms": retrieval_elapsed_ms,
        }
    )

    LOGGER.info(
        "Retrieval completado | modelo=%s | recuperados=%s | seleccionados=%s | distancias=%s",
        settings.openrouter_model,
        len(retrieved_pairs),
        len(selected_pairs),
        diagnostics["retrieval_distances"],
    )

    if not selected_pairs:
        LOGGER.warning("No se recupero evidencia para la pregunta: %s", question)
        diagnostics["citation_mode"] = "no_evidence"
        return RetrievalResult(
            answer=ABSTENTION_ANSWER,
            evidence=[],
            diagnostics=diagnostics,
        )

    evidence = _build_evidence(selected_pairs, metric_label="distancia")
    return _answer_from_evidence(question, evidence, settings, diagnostics)


def _answer_from_evidence(
    question: str,
    evidence: list[dict],
    settings: Settings,
    diagnostics: dict,
) -> RetrievalResult:
    context = _build_context(evidence)
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

    LOGGER.info(
        "Invocando LLM | modelo=%s | evidencia=%s",
        settings.openrouter_model,
        [item["id"] for item in evidence],
    )
    llm_started_at = perf_counter()
    try:
        answer = chain.invoke({"context": context, "question": question})
    except Exception as exc:
        LOGGER.exception("Fallo la llamada al modelo %s", settings.openrouter_model)
        raise RuntimeError(
            "No pude consultar el modelo configurado. Revisa la clave, el proveedor o la red."
        ) from exc

    llm_elapsed_ms = round((perf_counter() - llm_started_at) * 1000, 2)
    diagnostics["llm_called"] = True
    diagnostics["llm_ms"] = llm_elapsed_ms

    validated_answer, cited_ids, warnings, citation_mode = _validate_citations(
        answer,
        evidence,
        settings,
    )
    diagnostics["citation_mode"] = citation_mode
    diagnostics["cited_ids"] = cited_ids

    LOGGER.info(
        "Respuesta validada | citation_mode=%s | cited_ids=%s | warnings=%s",
        citation_mode,
        cited_ids,
        warnings,
    )

    if validated_answer == ABSTENTION_ANSWER:
        return RetrievalResult(
            answer=validated_answer,
            evidence=evidence,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    cited_evidence = [item for item in evidence if item["id"] in cited_ids] or evidence
    return RetrievalResult(
        answer=validated_answer,
        evidence=cited_evidence,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def _select_structured_xlsx_matches(
    question: str,
    source_documents: list[Document],
    top_k: int,
) -> StructuredSelection:
    xlsx_documents = [
        document
        for document in source_documents
        if document.metadata.get("file_type") == "xlsx"
        and isinstance(document.metadata.get("row_data"), dict)
    ]
    if not xlsx_documents:
        return StructuredSelection()

    query_dates = _extract_query_dates(question)
    query_rows = _extract_query_rows(question)
    if not query_dates and not query_rows:
        return StructuredSelection()

    available_columns = _collect_xlsx_columns(xlsx_documents)
    matched_columns = _extract_query_columns(question, available_columns)

    scored_matches: list[tuple[Document, float]] = []
    for document in xlsx_documents:
        row_data = document.metadata.get("row_data", {})
        match_score = 0.0

        if query_dates:
            document_date = _normalize_date_value(document.metadata.get("date_value", ""))
            if document_date not in query_dates:
                continue
            match_score += 10.0

        if query_rows:
            if document.metadata.get("row_number") not in query_rows:
                continue
            match_score += 8.0

        if matched_columns:
            available_matches = [column for column in matched_columns if column in row_data]
            if not available_matches:
                continue
            match_score += float(len(available_matches))

        if match_score <= 0:
            continue
        scored_matches.append((document, match_score))

    if not scored_matches:
        return StructuredSelection()

    scored_matches.sort(
        key=lambda item: (-item[1], item[0].metadata.get("row_number", 0)),
    )
    selected_matches = scored_matches[:top_k]

    exact_answer = None
    if len(scored_matches) == 1:
        matched_document = scored_matches[0][0]
        if matched_columns and _is_exact_value_lookup(question):
            exact_answer = _build_exact_xlsx_answer(matched_document, matched_columns)
        elif not matched_columns and _is_exact_row_lookup(question):
            exact_answer = _build_exact_xlsx_row_answer(matched_document)

    return StructuredSelection(
        matches=selected_matches,
        matched_columns=matched_columns,
        matched_dates=sorted(query_dates),
        matched_rows=sorted(query_rows),
        exact_answer=exact_answer,
    )


def _select_structured_txt_matches(
    question: str,
    source_documents: list[Document],
    top_k: int,
) -> StructuredTextSelection:
    query_terms = _extract_query_terms(question)
    if not query_terms:
        return StructuredTextSelection()

    txt_documents = [
        document
        for document in source_documents
        if document.metadata.get("file_type") == "txt"
        and document.metadata.get("section_title")
    ]
    if not txt_documents:
        return StructuredTextSelection()

    scored_matches: list[tuple[Document, float]] = []
    for document in txt_documents:
        title = str(document.metadata.get("section_title", ""))
        title_overlap = _lexical_overlap_with_terms(query_terms, title)
        if title_overlap <= 0:
            continue
        scored_matches.append((document, float(title_overlap)))

    if not scored_matches:
        return StructuredTextSelection()

    scored_matches.sort(key=lambda item: (-item[1], item[0].metadata.get("element_index", 0)))
    selected_matches = scored_matches[: min(top_k, 3)]

    exact_answer = None
    if _is_exact_txt_lookup(question):
        exact_answer = _build_exact_txt_answer(selected_matches)

    return StructuredTextSelection(
        matches=selected_matches,
        exact_answer=exact_answer,
    )


def _collect_xlsx_columns(documents: list[Document]) -> list[str]:
    for document in documents:
        column_names = document.metadata.get("column_names")
        if isinstance(column_names, list) and column_names:
            return [str(column) for column in column_names]
    return []


def _extract_query_columns(question: str, available_columns: list[str]) -> list[str]:
    question_lower = question.lower()
    question_compact = _normalize_lookup_token(question)
    matched_columns: list[str] = []

    for column in available_columns:
        column_lower = column.lower()
        column_compact = _normalize_lookup_token(column)
        column_spaced = re.sub(r"[_\W]+", " ", column_lower).strip()
        if column_compact and column_compact in question_compact:
            matched_columns.append(column)
            continue
        if column_spaced and column_spaced in question_lower:
            matched_columns.append(column)

    return matched_columns


def _extract_query_dates(question: str) -> set[str]:
    return {
        _normalize_date_value(match.group(1))
        for match in DATE_PATTERN.finditer(question)
        if _normalize_date_value(match.group(1))
    }


def _extract_query_rows(question: str) -> set[int]:
    return {
        int(match.group(1))
        for match in ROW_PATTERN.finditer(question)
    }


def _normalize_date_value(value: str) -> str:
    normalized = value.strip().replace("/", "-")
    return normalized if DATE_PATTERN.fullmatch(normalized) else ""


def _normalize_lookup_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _is_exact_value_lookup(question: str) -> bool:
    question_lower = question.lower()
    extra_terms = ("dame", "muestrame")
    return any(term in question_lower for term in (*VALUE_LOOKUP_TERMS, *extra_terms))


def _is_exact_row_lookup(question: str) -> bool:
    question_lower = question.lower()
    row_terms = ("muestra", "muestrame", "fila", "datos", "informacion", "contenido")
    return any(term in question_lower for term in row_terms)


def _build_exact_xlsx_answer(document: Document, matched_columns: list[str]) -> str:
    row_data = document.metadata.get("row_data", {})
    date_value = document.metadata.get("date_value")
    row_number = document.metadata.get("row_number")
    row_label = (
        f"la fecha {date_value}"
        if date_value
        else f"la fila {row_number}"
    )

    if len(matched_columns) == 1:
        column = matched_columns[0]
        return f"Para {row_label}, {column} es {row_data[column]} [E1]."

    values = "; ".join(f"{column} = {row_data[column]}" for column in matched_columns)
    return f"Para {row_label}: {values} [E1]."


def _build_exact_xlsx_row_answer(document: Document) -> str:
    row_data = document.metadata.get("row_data", {})
    date_value = document.metadata.get("date_value")
    row_number = document.metadata.get("row_number")
    row_label = (
        f"la fecha {date_value}"
        if date_value
        else f"la fila {row_number}"
    )
    values = "; ".join(f"{column} = {value}" for column, value in row_data.items())
    return f"Para {row_label}: {values} [E1]."


def _rank_retrieved_pairs(
    question: str,
    retrieved_pairs: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    query_terms = _extract_query_terms(question)
    if not query_terms:
        return sorted(retrieved_pairs, key=lambda item: item[1])

    ranked_items = [
        (
            document,
            score,
            _lexical_overlap_with_terms(query_terms, document.page_content),
        )
        for document, score in retrieved_pairs
    ]
    ranked_items.sort(key=lambda item: (-item[2], item[1]))
    return [(document, score) for document, score, _ in ranked_items]


def _lexical_overlap(question: str, content: str) -> int:
    return _lexical_overlap_with_terms(_extract_query_terms(question), content)


def _lexical_overlap_with_terms(query_terms: set[str], content: str) -> int:
    if not query_terms:
        return 0

    content_terms = _extract_query_terms(content)
    overlap = 0
    for query_term in query_terms:
        if any(_terms_match(query_term, content_term) for content_term in content_terms):
            overlap += 1
    return overlap


def _extract_query_terms(value: str) -> set[str]:
    normalized = _normalize_free_text(value)
    return {
        term
        for term in re.findall(r"[a-z0-9]+", normalized)
        if len(term) >= 4 and term not in LEXICAL_STOPWORDS
    }


def _normalize_free_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    return normalized.encode("ascii", "ignore").decode("ascii")


def _terms_match(query_term: str, content_term: str) -> bool:
    if query_term == content_term:
        return True
    minimum_prefix = min(len(query_term), len(content_term), 4)
    if minimum_prefix < 4:
        return False
    return query_term[:minimum_prefix] == content_term[:minimum_prefix]


def _is_exact_txt_lookup(question: str) -> bool:
    question_lower = _normalize_free_text(question)
    trigger_terms = (
        "como",
        "comando",
        "pasos",
        "que dicen",
        "que dice",
        "levanto",
        "mira",
        "modelos disponibles",
    )
    return any(term in question_lower for term in trigger_terms)


def _build_exact_txt_answer(selected_matches: list[tuple[Document, float]]) -> str | None:
    answer_parts: list[str] = []

    for index, (document, _) in enumerate(selected_matches, start=1):
        commands = document.metadata.get("command_lines", [])
        if not isinstance(commands, list) or not commands:
            continue

        title = _clean_txt_section_title(str(document.metadata.get("section_title", "")))
        if not title:
            continue

        action = _describe_txt_section(title)
        answer_parts.append(f"Para {action} usa: `{commands[0]}` [E{index}].")

    if not answer_parts:
        return None
    return " ".join(answer_parts)


def _clean_txt_section_title(title: str) -> str:
    cleaned = re.sub(r"^\d+[.)]\s*", "", title).strip()
    return cleaned.rstrip(":").strip()


def _describe_txt_section(title: str) -> str:
    normalized_title = _normalize_free_text(title)
    if normalized_title.startswith("levanta ollama"):
        return "levantar Ollama"
    if normalized_title.startswith("mira modelos disponibles"):
        return "ver los modelos disponibles"
    if normalized_title.startswith("chatea en terminal con uno"):
        return "chatear en terminal con un modelo"
    return title[0].lower() + title[1:] if title else "esa accion"


def _build_evidence(
    selected_pairs: list[tuple[Document, float]],
    *,
    metric_label: str,
) -> list[dict]:
    evidence = []
    for index, (document, metric_value) in enumerate(selected_pairs, start=1):
        source = document.metadata.get("source", "Documento")
        location = _format_location(document.metadata)
        evidence.append(
            {
                "id": f"E{index}",
                "source": source,
                "location": location,
                "metric_label": metric_label,
                "metric_value": round(float(metric_value), 3),
                "content": document.page_content,
            }
        )
    return evidence


def _build_context(evidence: list[dict]) -> str:
    return "\n\n".join(
        (
            f"[{item['id']}] Fuente: {item['source']} | Ubicacion: {item['location']} "
            f"| {item['metric_label'].capitalize()}: {item['metric_value']}\n{item['content']}"
        )
        for item in evidence
    )


def _validate_citations(
    answer: str,
    evidence: list[dict],
    settings: Settings,
) -> tuple[str, list[str], list[str], str]:
    normalized_answer = answer.strip()
    if not normalized_answer:
        LOGGER.warning("El modelo devolvio una respuesta vacia.")
        return ABSTENTION_ANSWER, [], [], "empty"

    if normalized_answer == ABSTENTION_ANSWER:
        return ABSTENTION_ANSWER, [], [], "abstained_by_model"

    valid_ids = {item["id"] for item in evidence}
    cited_ids = CITATION_PATTERN.findall(normalized_answer)
    unique_valid_ids = _unique_valid_ids(cited_ids, valid_ids)

    if not settings.require_citations:
        return normalized_answer, unique_valid_ids, [], "citations_disabled"

    if not cited_ids:
        warning = "Respuesta sin citas explicitas del modelo. Revisa la evidencia recuperada."
        LOGGER.warning(warning)
        return normalized_answer, [], [warning], "missing"

    invalid_citations = [cited_id for cited_id in cited_ids if cited_id not in valid_ids]
    if invalid_citations:
        cleaned_answer = CITATION_PATTERN.sub(
            lambda match: match.group(0) if match.group(1) in valid_ids else "",
            normalized_answer,
        )
        cleaned_answer = _normalize_answer_spacing(cleaned_answer) or normalized_answer
        warning = (
            "El modelo devolvio citas invalidas. Se muestran todos los fragmentos recuperados."
        )
        LOGGER.warning(
            "%s | invalid_citations=%s",
            warning,
            invalid_citations,
        )
        return cleaned_answer, unique_valid_ids, [warning], "invalid_removed"

    return normalized_answer, unique_valid_ids, [], "valid"


def _unique_valid_ids(cited_ids: list[str], valid_ids: set[str]) -> list[str]:
    unique_ids: list[str] = []
    for cited_id in cited_ids:
        if cited_id in valid_ids and cited_id not in unique_ids:
            unique_ids.append(cited_id)
    return unique_ids


def _normalize_answer_spacing(answer: str) -> str:
    compact_answer = re.sub(r"[ \t]{2,}", " ", answer)
    compact_answer = re.sub(r"\s+\n", "\n", compact_answer)
    compact_answer = re.sub(r"\n{3,}", "\n\n", compact_answer)
    return compact_answer.strip()


def _format_location(metadata: dict) -> str:
    if "sheet_name" in metadata:
        location_parts = [f"hoja {metadata['sheet_name']}"]
        if "row_number" in metadata:
            location_parts.append(f"fila {metadata['row_number']}")
        return " | ".join(location_parts)
    if "page_number" in metadata:
        return f"pagina {metadata['page_number']}"
    if "element_index" in metadata:
        return f"bloque {metadata['element_index']}"
    return "fragmento"
