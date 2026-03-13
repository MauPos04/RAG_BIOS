import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
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
from .prompts import CLARIFICATION_PROMPT, GROUNDING_PROMPT


LOGGER = logging.getLogger(__name__)
ABSTENTION_ANSWER = "No encontre esa informacion en los documentos cargados."
CITATION_PATTERN = re.compile(r"\[(E\d+)\]")
DATE_PATTERN = re.compile(
    r"\b((?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:\d{1,2}[-/]\d{1,2}[-/]\d{4}))\b"
)
ROW_PATTERN = re.compile(r"\bfila\s+(\d+)\b", re.IGNORECASE)
FOLLOW_UP_PREFIXES = (
    "y ",
    "e ",
    "tambien",
    "también",
    "entonces",
    "ahora",
    "eso",
    "esa",
    "ese",
    "esto",
    "el otro",
    "la otra",
    "del otro",
    "de la otra",
)
COUNT_FOLLOW_UP_HINTS = (
    "cuantas son",
    "cuantos son",
    "cuantas hay",
    "cuantos hay",
    "cuantas",
    "cuantos",
    "numero de",
)
GENERIC_COLUMN_TOKENS = {"close", "open", "high", "low", "adj", "volume", "price"}
VALUE_LOOKUP_TERMS = ("cual", "cuál", "valor", "dato", "dime", "muestra", "muéstrame")
TXT_COMMAND_HINTS = (
    "comando",
    "levanto",
    "levantar",
    "arranco",
    "arrancar",
    "inicio",
    "iniciar",
    "ejecuto",
    "ejecutar",
    "corro",
    "correr",
    "pasos",
    "mira",
    "modelos disponibles",
)
TXT_PURPOSE_HINTS = (
    "como funciona",
    "que hace",
    "de que sirve",
    "para que sirve",
    "para que funciona",
    "para que",
    "que significa",
)
TXT_SUMMARY_HINTS = (
    "que dicen",
    "que dice",
    "explica",
    "resume",
    "resumen",
    "de que trata",
)
TABULAR_HINTS = (
    "fila",
    "fila ",
    "hoja",
    "columna",
    "column",
    "valor",
    "dato",
    "fecha",
    "date",
    "close",
    "open",
    "high",
    "low",
    "volume",
)
ENUMERATION_HINTS = (
    "opciones",
    "alternativas",
    "situaciones",
    "posibilidades",
    "finalidades",
    "cuales son",
    "tipos",
    "derechos",
    "casos",
    "ejemplos",
)
GENERIC_ENUMERATION_TERMS = {
    "opciones",
    "alternativas",
    "situaciones",
    "posibilidades",
    "cuales",
    "tipos",
    "casos",
    "ejemplos",
    "aparecen",
    "aparece",
    "documento",
    "hay",
    "tengo",
}
ENUMERATION_EXCLUDE_PREFIXES = (
    "adicional a las anteriores",
    "formato ",
    "en cumplimiento",
    "si usted",
    "con el envio",
    "el comite",
    "companias de grupo bios",
    "el codigo de integridad",
    "firma",
    "indique",
    "mecanismos para el conocimiento",
    "numero de documento",
    "nombre",
    "para el ejercicio de mis derechos",
    "correo electronico",
    "ciudad",
    "fecha",
    "manifiesto que la presente autorizacion",
)


@dataclass(slots=True)
class RetrievalResult:
    answer: str
    evidence: list[dict]
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
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
    intent: str = "semantic"


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


def load_vector_store(folder_path: str | os.PathLike[str], settings: Settings) -> FAISS:
    embeddings = _get_embeddings(settings.embedding_model)
    return FAISS.load_local(
        str(folder_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openrouter_model,
        temperature=settings.temperature,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": settings.openrouter_http_referer,
            "X-Title": settings.openrouter_x_title,
        },
    )


def answer_question(
    vector_store: FAISS,
    question: str,
    settings: Settings,
    source_documents: list[Document] | None = None,
    *,
    chat_history: list[dict] | None = None,
) -> RetrievalResult:
    recent_chat_history = chat_history or []
    diagnostics = {
        "model": settings.openrouter_model,
        "retrieval_mode": "semantic",
        "llm_called": False,
        "chat_memory_turns": len(recent_chat_history) // 2,
    }

    count_follow_up_result = _answer_count_follow_up(question, recent_chat_history, diagnostics)
    if count_follow_up_result is not None:
        return count_follow_up_result

    structured_selection = _select_structured_xlsx_matches(
        question,
        source_documents or [],
        settings.top_k,
        recent_chat_history,
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
        return _answer_from_evidence(
            question,
            evidence,
            settings,
            diagnostics,
            recent_chat_history,
        )

    structured_text_selection = _select_structured_txt_matches(
        question,
        source_documents or [],
        settings.top_k,
        recent_chat_history,
    )
    if structured_text_selection.matches:
        diagnostics.update(
            {
                "retrieval_mode": "structured_txt",
                "retrieved_pairs": len(structured_text_selection.matches),
                "selected_pairs": len(structured_text_selection.matches),
                "txt_intent": structured_text_selection.intent,
            }
        )
        LOGGER.info(
            "Lookup estructurado TXT | intent=%s | matches=%s",
            structured_text_selection.intent,
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
        return _answer_from_evidence(
            question,
            evidence,
            settings,
            diagnostics,
            recent_chat_history,
        )

    retrieval_question = _build_retrieval_question(question, recent_chat_history)
    retrieval_started_at = perf_counter()
    retrieval_k = max(settings.top_k, settings.top_k * settings.retrieval_multiplier, 50)
    retrieved_pairs = vector_store.similarity_search_with_score(
        retrieval_question,
        k=retrieval_k,
    )
    retrieval_elapsed_ms = round((perf_counter() - retrieval_started_at) * 1000, 2)

    ranked_pairs = _rank_retrieved_pairs(retrieval_question, retrieved_pairs)
    selected_pairs = ranked_pairs[: settings.top_k]

    diagnostics.update(
        {
            "memory_used_for_retrieval": retrieval_question != question,
            "retrieval_k": retrieval_k,
            "retrieval_query": retrieval_question,
            "retrieved_pairs": len(retrieved_pairs),
            "selected_pairs": len(selected_pairs),
            "retrieval_metric": "distance",
            "retrieval_distances": [round(float(score), 3) for _, score in ranked_pairs],
            "lexical_overlaps": [
                _lexical_overlap(retrieval_question, document.page_content)
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

    expanded_pairs = _expand_semantic_neighbor_pairs(
        question,
        selected_pairs,
        source_documents or [],
        limit=max(settings.top_k * 2, 12),
    )
    evidence = _build_evidence(expanded_pairs, metric_label="distancia")
    enumeration_answer, enumeration_evidence = _try_build_enumeration_answer(question, evidence)
    if enumeration_answer:
        diagnostics["citation_mode"] = "enumeration_exact"
        diagnostics["cited_ids"] = [item["id"] for item in enumeration_evidence]
        return RetrievalResult(
            answer=enumeration_answer,
            evidence=enumeration_evidence,
            diagnostics=diagnostics,
        )
    return _answer_from_evidence(
        question,
        evidence,
        settings,
        diagnostics,
        recent_chat_history,
    )


def _answer_count_follow_up(
    question: str,
    chat_history: list[dict],
    diagnostics: dict,
) -> RetrievalResult | None:
    if not chat_history or not _is_count_follow_up(question):
        return None

    previous_user_question = _last_user_question(chat_history)
    previous_assistant_answer = _last_assistant_answer(chat_history)
    if not previous_user_question or not previous_assistant_answer:
        return None
    if not (
        _is_enumeration_question(previous_user_question)
        or _normalize_free_text(previous_assistant_answer).startswith(
            "en el documento se mencionan estas opciones"
        )
    ):
        return None

    previous_evidence = _last_assistant_evidence(chat_history)
    if len(previous_evidence) < 2:
        return None

    evidence = _renumber_evidence(previous_evidence)
    subject = _infer_count_subject(previous_user_question)
    citation_text = " ".join(f"[{item['id']}]" for item in evidence)
    answer = f"Son {len(evidence)} {subject} en total {citation_text}."
    diagnostics.update(
        {
            "retrieval_mode": "count_follow_up",
            "citation_mode": "count_from_previous_answer",
            "cited_ids": [item["id"] for item in evidence],
        }
    )
    return RetrievalResult(
        answer=answer,
        evidence=evidence,
        diagnostics=diagnostics,
    )


def _answer_from_evidence(
    question: str,
    evidence: list[dict],
    settings: Settings,
    diagnostics: dict,
    chat_history: list[dict],
) -> RetrievalResult:
    context = _build_context(evidence)
    formatted_chat_history = _format_chat_history(chat_history)
    prompt = ChatPromptTemplate.from_template(GROUNDING_PROMPT)
    llm = _build_llm(settings)
    chain = prompt | llm | StrOutputParser()

    LOGGER.info(
        "Invocando LLM | modelo=%s | evidencia=%s",
        settings.openrouter_model,
        [item["id"] for item in evidence],
    )
    llm_started_at = perf_counter()
    try:
        answer = chain.invoke(
            {
                "context": context,
                "question": question,
                "chat_history": formatted_chat_history,
            }
        )
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

    LOGGER.info(
        "Respuesta validada | citation_mode=%s | cited_ids=%s | warnings=%s",
        citation_mode,
        cited_ids,
        warnings,
    )

    if validated_answer == ABSTENTION_ANSWER:
        suggestions = _suggest_clarifying_questions(
            question,
            evidence,
            settings,
            chat_history,
        )
        diagnostics["suggestions_count"] = len(suggestions)
        return RetrievalResult(
            answer=validated_answer,
            evidence=evidence,
            warnings=warnings,
            suggestions=suggestions,
            diagnostics=diagnostics,
        )

    normalized_answer, cited_evidence, normalized_cited_ids = _normalize_cited_output(
        validated_answer,
        evidence,
        cited_ids,
    )
    diagnostics["cited_ids"] = normalized_cited_ids
    return RetrievalResult(
        answer=normalized_answer,
        evidence=cited_evidence,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def _select_structured_xlsx_matches(
    question: str,
    source_documents: list[Document],
    top_k: int,
    chat_history: list[dict],
) -> StructuredSelection:
    xlsx_documents = [
        document
        for document in source_documents
        if document.metadata.get("file_type") == "xlsx"
        and isinstance(document.metadata.get("row_data"), dict)
    ]
    if not xlsx_documents:
        return StructuredSelection()

    available_columns = _collect_xlsx_columns(xlsx_documents)
    lookup_question = _enrich_structured_xlsx_question(question, chat_history, available_columns)
    query_dates = _extract_query_dates(lookup_question)
    query_rows = _extract_query_rows(lookup_question)
    if not query_dates and not query_rows:
        return StructuredSelection()

    matched_columns = _extract_query_columns(lookup_question, available_columns)
    if not _should_use_structured_xlsx(
        question,
        lookup_question,
        query_dates,
        query_rows,
        matched_columns,
        bool(chat_history),
    ):
        return StructuredSelection()

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
        if matched_columns and (
            _is_exact_value_lookup(lookup_question) or _is_follow_up_question(question)
        ):
            exact_answer = _build_exact_xlsx_answer(matched_document, matched_columns)
        elif not matched_columns and (
            _is_exact_row_lookup(lookup_question) or _is_follow_up_question(question)
        ):
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
    chat_history: list[dict],
) -> StructuredTextSelection:
    lookup_question = _build_retrieval_question(question, chat_history)
    query_terms = _extract_query_terms(lookup_question)
    if not query_terms:
        return StructuredTextSelection()
    txt_intent = _classify_txt_intent(question, chat_history)

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
        body_overlap = _lexical_overlap_with_terms(query_terms, document.page_content)
        if title_overlap <= 0:
            continue
        scored_matches.append((document, float(title_overlap * 3 + body_overlap)))

    if not scored_matches:
        return StructuredTextSelection(intent=txt_intent)

    scored_matches.sort(key=lambda item: (-item[1], item[0].metadata.get("element_index", 0)))
    selected_matches = _expand_txt_neighbor_matches(
        scored_matches,
        txt_documents,
        min(top_k, 4),
    )

    exact_answer = None
    if txt_intent == "command" and _is_exact_txt_lookup(lookup_question):
        exact_answer = _build_exact_txt_answer(selected_matches)

    return StructuredTextSelection(
        matches=selected_matches,
        exact_answer=exact_answer,
        intent=txt_intent,
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
    question_terms = _extract_query_terms(question)
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
            continue

        column_terms = [
            token
            for token in re.findall(r"[a-z0-9]+", _normalize_free_text(column_spaced))
            if len(token) >= 4 and token not in GENERIC_COLUMN_TOKENS
        ]
        if any(term in question_terms for term in column_terms):
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
    if not DATE_PATTERN.fullmatch(normalized):
        return ""

    parts = normalized.split("-")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return ""

    if len(parts[0]) == 4:
        year, month, day = parts
    elif len(parts[2]) == 4:
        day, month, year = parts
    else:
        return ""

    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return ""


def _contains_date_like_token(value: str) -> bool:
    return bool(DATE_PATTERN.search(value))


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


def _should_use_structured_xlsx(
    question: str,
    lookup_question: str,
    query_dates: set[str],
    query_rows: set[int],
    matched_columns: list[str],
    has_chat_history: bool,
) -> bool:
    if query_rows or matched_columns:
        return True
    if not query_dates:
        return False
    if _is_exact_row_lookup(lookup_question):
        return True
    if has_chat_history and _is_follow_up_question(question):
        return True
    normalized_lookup = _normalize_free_text(lookup_question)
    return any(term in normalized_lookup for term in TABULAR_HINTS)


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


def _build_retrieval_question(question: str, chat_history: list[dict]) -> str:
    if not chat_history or not _is_follow_up_question(question):
        return question

    last_user_question = _last_user_question(chat_history)
    last_assistant_answer = _last_assistant_answer(chat_history)
    if not last_user_question and not last_assistant_answer:
        return question

    retrieval_parts = [question]
    if last_user_question:
        retrieval_parts.append(f"Pregunta previa: {last_user_question}")
    if last_assistant_answer:
        cleaned_answer = CITATION_PATTERN.sub("", last_assistant_answer).strip()
        if cleaned_answer:
            retrieval_parts.append(f"Respuesta previa: {cleaned_answer}")
    return "\n".join(retrieval_parts)


def _enrich_structured_xlsx_question(
    question: str,
    chat_history: list[dict],
    available_columns: list[str],
) -> str:
    if not chat_history:
        return question

    current_dates = _extract_query_dates(question)
    current_has_date_token = _contains_date_like_token(question)
    current_rows = _extract_query_rows(question)
    current_columns = _extract_query_columns(question, available_columns)

    if current_dates and (current_rows or current_columns):
        return question

    inherited_dates: set[str] = set()
    inherited_rows: set[int] = set()
    inherited_columns: list[str] = []

    for previous_question in _recent_user_questions(chat_history):
        if not current_dates and not current_has_date_token:
            inherited_dates = _extract_query_dates(previous_question)
        if not current_rows:
            inherited_rows = _extract_query_rows(previous_question)
        if not current_columns:
            inherited_columns = _extract_query_columns(previous_question, available_columns)

        if inherited_dates or inherited_rows or inherited_columns:
            break

    if not inherited_dates and not inherited_rows and not inherited_columns:
        return question

    hint_parts = [question]
    if not current_dates and not current_has_date_token:
        hint_parts.extend(f"fecha {value}" for value in sorted(inherited_dates))
    if not current_rows:
        hint_parts.extend(f"fila {value}" for value in sorted(inherited_rows))
    if not current_columns:
        hint_parts.extend(f"columna {value}" for value in inherited_columns)
    return " ".join(hint_parts)


def _is_follow_up_question(question: str) -> bool:
    normalized_question = _normalize_free_text(question).strip()
    if not normalized_question:
        return False

    if any(normalized_question.startswith(prefix) for prefix in FOLLOW_UP_PREFIXES):
        return True

    return len(_extract_query_terms(question)) <= 2


def _is_count_follow_up(question: str) -> bool:
    normalized_question = _normalize_free_text(question)
    return any(hint in normalized_question for hint in COUNT_FOLLOW_UP_HINTS)


def _recent_user_questions(chat_history: list[dict]) -> list[str]:
    return [
        str(message.get("content", "")).strip()
        for message in reversed(chat_history)
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ]


def _last_user_question(chat_history: list[dict]) -> str:
    recent_questions = _recent_user_questions(chat_history)
    return recent_questions[0] if recent_questions else ""


def _last_assistant_answer(chat_history: list[dict]) -> str:
    for message in reversed(chat_history):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", "")).strip()
        if content:
            return content
    return ""


def _last_assistant_evidence(chat_history: list[dict]) -> list[dict]:
    for message in reversed(chat_history):
        if message.get("role") != "assistant":
            continue
        evidence = message.get("evidence")
        if isinstance(evidence, list) and evidence:
            return evidence
    return []


def _infer_count_subject(previous_user_question: str) -> str:
    normalized_question = _normalize_free_text(previous_user_question)
    if "finalidades" in normalized_question:
        return "finalidades"
    if "derechos" in normalized_question:
        return "derechos"
    if "opciones" in normalized_question:
        return "opciones"
    if "situaciones" in normalized_question:
        return "situaciones"
    if "alternativas" in normalized_question:
        return "alternativas"
    if "casos" in normalized_question:
        return "casos"
    return "elementos"


def _format_chat_history(chat_history: list[dict]) -> str:
    if not chat_history:
        return "Sin historial reciente."

    formatted_messages = []
    for message in chat_history:
        role = "Usuario" if message.get("role") == "user" else "Asistente"
        content = _truncate_text(str(message.get("content", "")).strip(), 240)
        if content:
            formatted_messages.append(f"{role}: {content}")
    return "\n".join(formatted_messages) or "Sin historial reciente."


def _truncate_text(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3].rstrip()}..."


def _terms_match(query_term: str, content_term: str) -> bool:
    if query_term == content_term:
        return True
    shorter_term, longer_term = sorted((query_term, content_term), key=len)
    if len(shorter_term) < 5:
        return False
    if not longer_term.startswith(shorter_term):
        return False
    return len(longer_term) - len(shorter_term) <= 2


def _is_exact_txt_lookup(question: str) -> bool:
    question_lower = _normalize_free_text(question)
    trigger_terms = (
        "comando",
        "pasos",
        "que dicen",
        "que dice",
        "levanto",
        "levantar",
        "arranco",
        "arrancar",
        "inicio",
        "iniciar",
        "ejecuto",
        "ejecutar",
        "mira",
        "modelos disponibles",
    )
    return any(term in question_lower for term in trigger_terms)


def _classify_txt_intent(question: str, chat_history: list[dict]) -> str:
    normalized_question = _normalize_free_text(question)
    if any(term in normalized_question for term in TXT_PURPOSE_HINTS):
        return "purpose"
    if any(term in normalized_question for term in TXT_COMMAND_HINTS):
        return "command"
    if any(term in normalized_question for term in TXT_SUMMARY_HINTS):
        return "summary"

    retrieval_question = _build_retrieval_question(question, chat_history)
    normalized_retrieval = _normalize_free_text(retrieval_question)
    if any(term in normalized_retrieval for term in TXT_PURPOSE_HINTS):
        return "purpose"
    if any(term in normalized_retrieval for term in TXT_COMMAND_HINTS):
        return "command"
    if any(term in normalized_retrieval for term in TXT_SUMMARY_HINTS):
        return "summary"
    return "reference"


def _build_exact_txt_answer(selected_matches: list[tuple[Document, float]]) -> str | None:
    for index, (document, _) in enumerate(selected_matches, start=1):
        commands = document.metadata.get("command_lines", [])
        if not isinstance(commands, list) or not commands:
            continue

        title = _clean_txt_section_title(str(document.metadata.get("section_title", "")))
        if not title:
            continue

        action = _describe_txt_section(title)
        return f"Para {action} usa: `{commands[0]}` [E{index}]."

    return None


def _expand_txt_neighbor_matches(
    scored_matches: list[tuple[Document, float]],
    txt_documents: list[Document],
    limit: int,
) -> list[tuple[Document, float]]:
    if not scored_matches:
        return []

    documents_by_key = {
        (document.metadata.get("source"), document.metadata.get("element_index")): document
        for document in txt_documents
    }
    expanded_matches: list[tuple[Document, float]] = []
    seen_keys: set[tuple[str | None, int | None]] = set()

    for document, score in scored_matches:
        current_key = (document.metadata.get("source"), document.metadata.get("element_index"))
        if current_key not in seen_keys:
            expanded_matches.append((document, score))
            seen_keys.add(current_key)

        element_index = document.metadata.get("element_index")
        if not isinstance(element_index, int):
            continue

        for offset, penalty in ((-1, 0.35), (1, 0.25)):
            neighbor_key = (document.metadata.get("source"), element_index + offset)
            if neighbor_key in seen_keys:
                continue
            neighbor = documents_by_key.get(neighbor_key)
            if neighbor is None:
                continue
            expanded_matches.append((neighbor, max(score - penalty, 0.1)))
            seen_keys.add(neighbor_key)

        if len(expanded_matches) >= limit:
            break

    expanded_matches.sort(
        key=lambda item: (
            -item[1],
            item[0].metadata.get("element_index", 0),
        )
    )
    return expanded_matches[:limit]


def _expand_semantic_neighbor_pairs(
    question: str,
    selected_pairs: list[tuple[Document, float]],
    source_documents: list[Document],
    *,
    limit: int,
) -> list[tuple[Document, float]]:
    if not selected_pairs or not source_documents or not _is_enumeration_question(question):
        return selected_pairs

    documents_by_key = {
        (document.metadata.get("source"), document.metadata.get("element_index")): document
        for document in source_documents
        if document.metadata.get("file_type") in {"pdf", "docx"}
        and isinstance(document.metadata.get("element_index"), int)
    }
    if not documents_by_key:
        return selected_pairs

    expanded_pairs: list[tuple[Document, float]] = []
    seen_keys: set[tuple[str | None, int | None]] = set()

    for document, score in selected_pairs:
        current_key = (document.metadata.get("source"), document.metadata.get("element_index"))
        if current_key not in seen_keys:
            expanded_pairs.append((document, score))
            seen_keys.add(current_key)

        if document.metadata.get("file_type") not in {"pdf", "docx"}:
            continue

        element_index = document.metadata.get("element_index")
        source = document.metadata.get("source")
        if not isinstance(element_index, int):
            continue

        content = document.page_content.strip()
        if not content.endswith(":"):
            continue

        for offset in range(1, 13):
            neighbor_key = (source, element_index + offset)
            if neighbor_key in seen_keys:
                continue
            neighbor = documents_by_key.get(neighbor_key)
            if neighbor is None:
                break
            neighbor_content = neighbor.page_content.strip()
            if offset > 1 and _looks_like_new_section(neighbor_content):
                break
            expanded_pairs.append((neighbor, max(score - (offset * 0.05), 0.1)))
            seen_keys.add(neighbor_key)
            if len(expanded_pairs) >= limit:
                return expanded_pairs[:limit]

    return expanded_pairs[:limit]


def _clean_txt_section_title(title: str) -> str:
    cleaned = re.sub(r"^\d+[.)]\s*", "", title).strip()
    return cleaned.rstrip(":").strip()


def _looks_like_new_section(content: str) -> bool:
    normalized_content = _normalize_free_text(content).strip()
    if not normalized_content or len(normalized_content) < 20:
        return False
    if ":" in normalized_content[:60] and not normalized_content.endswith(":"):
        return True
    first_sentence, separator, _ = normalized_content.partition(". ")
    if separator and len(first_sentence.split()) <= 10:
        return True
    return False


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
                "file_type": document.metadata.get("file_type"),
                "element_index": document.metadata.get("element_index"),
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


def _normalize_cited_output(
    answer: str,
    evidence: list[dict],
    cited_ids: list[str],
) -> tuple[str, list[dict], list[str]]:
    if not cited_ids:
        return answer, evidence, []

    evidence_by_id = {item["id"]: item for item in evidence}
    cited_evidence = [evidence_by_id[cited_id] for cited_id in cited_ids if cited_id in evidence_by_id]
    if not cited_evidence:
        return answer, evidence, []

    remap = {old_id: f"E{index}" for index, old_id in enumerate(cited_ids, start=1)}
    remapped_answer = CITATION_PATTERN.sub(
        lambda match: f"[{remap.get(match.group(1), match.group(1))}]",
        answer,
    )

    remapped_evidence: list[dict] = []
    for old_id in cited_ids:
        item = evidence_by_id.get(old_id)
        if item is None:
            continue
        remapped_item = dict(item)
        remapped_item["id"] = remap[old_id]
        remapped_evidence.append(remapped_item)

    return remapped_answer, remapped_evidence, list(remap.values())


def _try_build_enumeration_answer(
    question: str,
    evidence: list[dict],
) -> tuple[str | None, list[dict]]:
    if not _is_enumeration_question(question):
        return None, []

    anchor_terms = _extract_enumeration_anchor_terms(question)
    if not anchor_terms:
        return None, []

    candidates = _select_enumeration_candidates(evidence, anchor_terms)
    if len(candidates) < 2:
        return None, []

    renumbered_candidates = _renumber_evidence(candidates)
    statements = [f"{item['content']} [{item['id']}]" for item in renumbered_candidates]
    if len(statements) == 2:
        statement_text = " y ".join(statements)
    else:
        statement_text = "; ".join(statements[:-1]) + f"; y {statements[-1]}"

    return (
        f"En el documento se mencionan estas opciones: {statement_text}.",
        renumbered_candidates,
    )


def _is_enumeration_question(question: str) -> bool:
    normalized_question = _normalize_free_text(question)
    return any(term in normalized_question for term in ENUMERATION_HINTS)


def _extract_enumeration_anchor_terms(question: str) -> set[str]:
    return {
        term
        for term in _extract_query_terms(question)
        if term not in GENERIC_ENUMERATION_TERMS
    }


def _select_enumeration_candidates(evidence: list[dict], anchor_terms: set[str]) -> list[dict]:
    selected: list[dict] = []
    seen_contents: set[str] = set()
    last_anchor_source: str | None = None
    last_anchor_index: int | None = None
    include_following = 0

    for item in evidence:
        content = str(item.get("content", "")).strip()
        normalized_content = _normalize_free_text(content)
        if not content or normalized_content in seen_contents:
            continue
        item_source = str(item.get("source", ""))
        item_index = item.get("element_index")
        is_anchor = _lexical_overlap_with_terms(anchor_terms, content) > 0
        is_following_item = (
            include_following > 0
            and item_source == last_anchor_source
            and isinstance(item_index, int)
            and isinstance(last_anchor_index, int)
            and item_index == last_anchor_index + 1
        )
        if not is_anchor and not is_following_item:
            include_following = 0
            continue
        if is_anchor and content.endswith(":"):
            last_anchor_source = item_source
            last_anchor_index = item_index if isinstance(item_index, int) else None
            include_following = 10
            if not is_following_item:
                continue
        if len(content) < 12 or len(content) > 400:
            if is_following_item:
                last_anchor_index = item_index
                include_following = max(include_following - 1, 0)
            continue
        if "@" in content or "http" in normalized_content:
            if is_following_item:
                last_anchor_index = item_index
                include_following = max(include_following - 1, 0)
            continue
        if any(normalized_content.startswith(prefix) for prefix in ENUMERATION_EXCLUDE_PREFIXES):
            if is_following_item:
                last_anchor_index = item_index
                include_following = max(include_following - 1, 0)
            continue
        seen_contents.add(normalized_content)
        selected.append(item)
        if is_following_item:
            last_anchor_index = item_index
            include_following = max(include_following - 1, 0)
        else:
            include_following = 0

    return selected


def _renumber_evidence(evidence: list[dict]) -> list[dict]:
    remapped: list[dict] = []
    for index, item in enumerate(evidence, start=1):
        remapped_item = dict(item)
        remapped_item["id"] = f"E{index}"
        remapped.append(remapped_item)
    return remapped


def _suggest_clarifying_questions(
    question: str,
    evidence: list[dict],
    settings: Settings,
    chat_history: list[dict],
) -> list[str]:
    if not evidence:
        return []

    prompt = ChatPromptTemplate.from_template(CLARIFICATION_PROMPT)
    llm = _build_llm(settings)
    chain = prompt | llm | StrOutputParser()

    try:
        raw_suggestions = chain.invoke(
            {
                "context": _build_context(evidence),
                "question": question,
                "chat_history": _format_chat_history(chat_history),
            }
        ).strip()
    except Exception:
        LOGGER.exception("No pude generar sugerencias de reformulacion.")
        return []

    if not raw_suggestions or raw_suggestions == "SIN_SUGERENCIAS":
        return []

    suggestions: list[str] = []
    for line in raw_suggestions.splitlines():
        cleaned = line.strip()
        cleaned = re.sub(r"^[-*]\s*", "", cleaned)
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
        cleaned = cleaned.strip()
        if not cleaned or cleaned in suggestions:
            continue
        suggestions.append(cleaned)
        if len(suggestions) == 3:
            break

    return suggestions


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
