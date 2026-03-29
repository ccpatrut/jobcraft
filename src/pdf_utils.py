"""Convert markdown to PDF using fpdf2 (pure Python, no system dependencies)."""

import logging
import re
from pathlib import Path

from fpdf import FPDF

logger = logging.getLogger(__name__)

_UNICODE_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "--",
    "\u2026": "...",
    "\u00a0": " ",
    "\u200b": "",
    "\u2022": "-",
    "\ufeff": "",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u00ad": "",
    "\u2028": "\n",
    "\u2029": "\n",
}


def _sanitize_text(text: str) -> str:
    """Replace problematic Unicode characters with Latin-1 safe equivalents."""
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    cleaned = []
    for ch in text:
        try:
            ch.encode("latin-1")
            cleaned.append(ch)
        except UnicodeEncodeError:
            cleaned.append("?")
    return "".join(cleaned)


def _strip_inline_md(text: str) -> str:
    """Remove markdown links and bold markers from text."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return text


# ---------------------------------------------------------------------------
# CV-specific PDF renderer
# ---------------------------------------------------------------------------


class _CvPdf(FPDF):
    """FPDF subclass with helpers for CV rendering."""

    PAGE_W = 210  # A4 mm
    MARGIN = 15
    CONTENT_W = PAGE_W - 2 * MARGIN
    COLOR_BLACK = (0, 0, 0)
    COLOR_DARK = (40, 40, 40)
    COLOR_ACCENT = (0, 102, 204)
    COLOR_GRAY = (100, 100, 100)

    def _reset(self):
        self.set_x(self.l_margin)
        self.set_text_color(*self.COLOR_BLACK)

    def _draw_rule(self, thickness: float = 0.3):
        y = self.get_y()
        self.set_draw_color(160, 160, 160)
        self.set_line_width(thickness)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(2)

    def render_name(self, name: str):
        self._reset()
        self.set_font("Helvetica", "B", 20)
        self.cell(0, 10, _sanitize_text(name.upper()), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def render_contact(self, contact: str):
        self._reset()
        self.set_font("Helvetica", size=9)
        self.set_text_color(*self.COLOR_GRAY)
        self.cell(0, 5, _sanitize_text(contact), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*self.COLOR_BLACK)
        self.ln(3)
        self._draw_rule(0.5)
        self.ln(2)

    def render_section_header(self, title: str):
        self._reset()
        self.ln(4)
        self.set_font("Helvetica", "B", 11)
        self.cell(
            0,
            7,
            _sanitize_text(title.upper()),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self._draw_rule(0.3)
        self.ln(1)

    def render_experience_title(self, role: str, dates: str = ""):
        self._reset()
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        if dates:
            role_w = self.CONTENT_W * 0.7
            date_w = self.CONTENT_W * 0.3
            x_start = self.l_margin
            self.set_x(x_start)
            self.cell(role_w, 6, _sanitize_text(role))
            self.set_font("Helvetica", size=9)
            self.set_text_color(*self.COLOR_GRAY)
            self.cell(date_w, 6, _sanitize_text(dates), align="R")
            self.set_text_color(*self.COLOR_BLACK)
            self.ln(6)
        else:
            self.multi_cell(0, 6, _sanitize_text(role))
        self._reset()
        self.set_font("Helvetica", size=10)
        self.ln(1)

    def render_bullet(self, text: str, indent: float = 5):
        self._reset()
        self.set_font("Helvetica", size=9.5)
        x = self.l_margin + indent
        self.set_x(x)
        bullet_w = 6
        text_w = self.CONTENT_W - indent - bullet_w
        self.cell(bullet_w, 5, " - ")
        self.multi_cell(text_w, 5, _sanitize_text(text))
        self._reset()

    def render_paragraph(self, text: str):
        self._reset()
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 5.5, _sanitize_text(text))
        self._reset()
        self.ln(1)

    def render_link(self, label: str, url: str):
        self._reset()
        self.ln(6)
        self._draw_rule(0.2)
        self.ln(2)
        self.set_font("Helvetica", size=8)
        self.set_text_color(*self.COLOR_ACCENT)
        self.cell(0, 4, _sanitize_text(label), link=url, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", size=7)
        self.cell(0, 4, _sanitize_text(url), link=url, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*self.COLOR_BLACK)


def _parse_experience_heading(text: str) -> tuple[str, str]:
    """Split '### Role at Company | DATES' into (role_and_company, dates)."""
    if " | " in text:
        parts = text.rsplit(" | ", 1)
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ""


def _parse_education_bullet(text: str) -> tuple[str, str]:
    """Split 'Degree at Institution | (dates)' into (degree_text, dates)."""
    if " | " in text:
        parts = text.rsplit(" | ", 1)
        dates = parts[1].strip().strip("()")
        return parts[0].strip(), dates
    return text.strip(), ""


def markdown_to_pdf(md_content: str, pdf_path: str | Path) -> bool:
    """
    Convert CV/cover letter markdown to a professionally formatted PDF.

    Returns True on success, False on error.
    """
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Separate content from the apply link
        lines = md_content.split("\n")
        content_lines = []
        apply_url = None
        for line in lines:
            if line.strip().startswith("**Apply for this position:**"):
                match = re.search(r"\]\(([^)]+)\)", line)
                if match:
                    apply_url = match.group(1)
                break
            content_lines.append(line)

        pdf = _CvPdf()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_margins(left=_CvPdf.MARGIN, top=12, right=_CvPdf.MARGIN)
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)

        current_section = ""
        i = 0
        while i < len(content_lines):
            line = content_lines[i]
            stripped = line.strip()

            if not stripped or stripped in ("---", "***", "___"):
                i += 1
                continue

            # ## Section header
            if stripped.startswith("## "):
                section_name = stripped[3:].strip()
                current_section = section_name.lower()

                if current_section == "header":
                    # Parse the next lines for Name: and Contact:
                    i += 1
                    name = ""
                    contact = ""
                    while i < len(content_lines):
                        hl = content_lines[i].strip()
                        if hl.startswith("## ") or hl.startswith("### "):
                            break
                        if hl.lower().startswith("name:"):
                            name = hl.split(":", 1)[1].strip()
                        elif hl.lower().startswith("contact:"):
                            contact = hl.split(":", 1)[1].strip()
                        elif not hl:
                            i += 1
                            continue
                        else:
                            if not name:
                                name = hl
                        i += 1
                    pdf.render_name(name or "Candidate")
                    if contact:
                        pdf.render_contact(contact)
                    continue
                else:
                    # Look ahead: skip sections whose only content is N/A or empty
                    next_lines = []
                    j = i + 1
                    while j < len(content_lines):
                        nxt = content_lines[j].strip()
                        if nxt.startswith("## ") or nxt.startswith("---"):
                            break
                        if nxt:
                            bullet = nxt.lstrip("-* ").strip()
                            next_lines.append(bullet.lower())
                        j += 1
                    if all(t in ("n/a", "none", "") for t in next_lines) or not next_lines:
                        i = j
                        continue

                    pdf.render_section_header(section_name)
                    i += 1
                    continue

            # ### Experience / sub-heading
            if stripped.startswith("### "):
                heading_text = _strip_inline_md(stripped[4:].strip())
                role, dates = _parse_experience_heading(heading_text)
                pdf.render_experience_title(role, dates)
                i += 1
                continue

            # Bullet point
            if stripped.startswith("- ") or stripped.startswith("* "):
                text = _strip_inline_md(stripped[2:].strip())
                if current_section == "education":
                    degree, dates = _parse_education_bullet(text)
                    if dates:
                        pdf.render_experience_title(degree, dates)
                    else:
                        pdf.render_bullet(text)
                else:
                    pdf.render_bullet(text)
                i += 1
                continue

            # Regular paragraph
            text = _strip_inline_md(stripped)
            pdf.render_paragraph(text)
            i += 1

        # Apply link at the bottom
        if apply_url:
            pdf.render_link("Apply for this position", apply_url)

        pdf.output(str(pdf_path))
        return True
    except Exception as exc:
        logger.warning("PDF generation failed for %s: %s", pdf_path, exc)
        return False
