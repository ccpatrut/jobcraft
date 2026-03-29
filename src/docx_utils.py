"""Convert markdown CV/cover letter to Word (.docx) format."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def markdown_to_docx(md_content: str, docx_path: str | Path) -> bool:
    """Convert CV/cover letter markdown to a Word document.

    Returns True on success, False on error.
    """
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt
    except ImportError:
        logger.warning("python-docx not installed — skipping .docx generation")
        return False

    docx_path = Path(docx_path)
    docx_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        doc = Document()

        style = doc.styles["Normal"]
        font = style.font
        font.name = "Calibri"
        font.size = Pt(10.5)

        for line in md_content.split("\n"):
            stripped = line.strip()

            if not stripped or stripped in ("---", "***", "___"):
                continue

            if stripped.startswith("### "):
                text = _strip_md(stripped[4:])
                p = doc.add_heading(text, level=3)
                p.style.font.size = Pt(11)
                continue

            if stripped.startswith("## "):
                text = _strip_md(stripped[3:])
                if _is_name_heading(text, md_content):
                    p = doc.add_heading(text, level=1)
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    doc.add_heading(text, level=2)
                continue

            if stripped.startswith("# "):
                text = _strip_md(stripped[2:])
                p = doc.add_heading(text, level=1)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                continue

            if stripped.startswith("**") and stripped.endswith("**"):
                text = _strip_md(stripped)
                p = doc.add_paragraph()
                run = p.add_run(text)
                run.bold = True
                run.font.size = Pt(9)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                continue

            if stripped.startswith(("- ", "* ")):
                text = _strip_md(stripped[2:])
                doc.add_paragraph(text, style="List Bullet")
                continue

            text = _strip_md(stripped)
            doc.add_paragraph(text)

        doc.save(str(docx_path))
        return True
    except Exception as exc:
        logger.warning("DOCX generation failed for %s: %s", docx_path, exc)
        return False


def _strip_md(text: str) -> str:
    """Remove markdown formatting (bold, links) from text."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text.strip()


def _is_name_heading(text: str, full_content: str) -> bool:
    """Heuristic: the first ## heading in a CV is usually the candidate's name."""
    first_h2 = re.search(r"^## (.+)$", full_content, re.MULTILINE)
    if first_h2:
        return _strip_md(first_h2.group(1).strip()) == text
    return False
