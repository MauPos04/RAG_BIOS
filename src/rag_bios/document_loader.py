import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain_core.documents import Document
from openpyxl import load_workbook
from unstructured.partition.auto import partition


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx"}


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def load_uploaded_documents(uploaded_files) -> list[Document]:
    documents: list[Document] = []
    temp_paths: list[Path] = []

    try:
        for uploaded_file in uploaded_files:
            temp_path = save_uploaded_file(uploaded_file)
            temp_paths.append(temp_path)
            documents.extend(load_file(temp_path, uploaded_file.name))
    finally:
        for temp_path in temp_paths:
            if temp_path.exists():
                os.remove(temp_path)

    return documents


def load_file(file_path: Path, display_name: str | None = None) -> list[Document]:
    suffix = file_path.suffix.lower()
    source_name = display_name or file_path.name

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Formato no soportado: {suffix}")

    if suffix == ".xlsx":
        return _load_xlsx(file_path, source_name)

    elements = partition(filename=str(file_path))
    documents: list[Document] = []

    for index, element in enumerate(elements, start=1):
        text = getattr(element, "text", "")
        if not text or not text.strip():
            continue

        metadata = {
            "source": source_name,
            "element_index": index,
            "file_type": suffix.lstrip("."),
        }
        page_number = getattr(getattr(element, "metadata", None), "page_number", None)
        if page_number is not None:
            metadata["page_number"] = page_number

        documents.append(Document(page_content=text.strip(), metadata=metadata))

    return documents


def _load_xlsx(file_path: Path, source_name: str) -> list[Document]:
    workbook = load_workbook(filename=file_path, data_only=True, read_only=True)
    documents: list[Document] = []

    for sheet in workbook.worksheets:
        rows: list[str] = []
        for row_index, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            values = [str(value).strip() for value in row if value is not None and str(value).strip()]
            if not values:
                continue
            rows.append(f"Fila {row_index}: " + " | ".join(values))

        if not rows:
            continue

        content = f"Hoja: {sheet.title}\n" + "\n".join(rows)
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": source_name,
                    "sheet_name": sheet.title,
                    "file_type": "xlsx",
                },
            )
        )

    workbook.close()
    return documents
