"""Load CV documents (PDF, Word) from a directory."""

from pathlib import Path

import pymupdf  # PyMuPDF for PDF
from docx import Document


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        doc = pymupdf.open(file_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts).strip()
    except Exception as e:
        return f"[Error reading PDF {file_path.name}: {e}]"


def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from a Word document (.docx)."""
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        return f"[Error reading DOCX {file_path.name}: {e}]"


def load_documents_from_dir(directory: str | Path) -> tuple[list[str], str]:
    """
    Load all PDF and Word documents from a directory.

    Returns:
        Tuple of (list of per-file texts, combined full text)
    """
    path = Path(directory).expanduser().resolve()
    if not path.exists():
        return [], f"[Directory not found: {path}]"

    texts: list[str] = []
    supported = {".pdf", ".docx"}

    for file_path in sorted(path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in supported:
            if file_path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(file_path)
            else:
                text = extract_text_from_docx(file_path)
            if text and not text.startswith("[Error"):
                texts.append(f"--- {file_path.name} ---\n{text}")

    combined = "\n\n".join(texts) if texts else "[No documents found in directory]"
    return texts, combined
