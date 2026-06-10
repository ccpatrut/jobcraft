"""Extract structured profile from CV documents using an LLM provider."""

import json
import logging
import re
from typing import Any

from .llm_provider import LLMProvider
from .models import UserProfile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert professional CV/resume parser with years of experience in recruitment and talent acquisition.

Your specialty is identifying the greatest strengths and most distinctive characteristics of each candidate. You excel at:
- Spotting standout achievements, unique skills, and transferable competencies
- Recognizing leadership, initiative, and impact even when understated
- Distilling complex career paths into clear, compelling narratives
- Preserving the candidate's voice and language while ensuring clarity

You extract only factual information from the document. You never invent or exaggerate. You are precise, thorough, and consistent."""

EXTRACTION_PROMPT = """You are a precise CV/resume parser.

Task:
Extract the candidate's information from the CV text and return ONLY one valid JSON object.
Do not add explanations, markdown, code fences, comments, or extra text.

Rules:
- Use exactly these top-level keys:
  "name", "email", "phone", "location", "summary", "skills", "experience", "education", "certifications", "languages", "industries"
- If a value is not found, use:
  - null for single-value fields
  - [] for array fields
  - "" only for "summary" if there is not enough information
- Preserve the original language of extracted content when possible.
- Normalize obvious OCR or formatting issues only when the intended meaning is clear.
- Do not invent information.
- Deduplicate repeated items.
- Prefer the most complete and specific version of a value when duplicates/conflicts appear.
- Extract concise, factual content from the CV only.

Field requirements:
- "name": candidate full name, or null
- "email": best primary email, or null
- "phone": best primary phone number, or null
- "location": city/country or location line if present, or null
- "summary": write a short 2-3 sentence professional summary based only on the CV content; do not exaggerate
- "skills": array of hard skills, tools, technologies, and relevant professional competencies (do NOT include spoken languages here)
- "experience": array of objects, one per role, in reverse chronological order when possible
- "education": array of objects, one per education entry
- "certifications": array of certification names
- "languages": array of objects for spoken/written languages with proficiency levels
- "industries": array of industry/sector keywords that describe the candidate's professional domain

Output schema:
{
  "name": null,
  "email": null,
  "phone": null,
  "location": null,
  "summary": "",
  "skills": [],
  "experience": [
    {
      "job_title": "",
      "company": "",
      "start_date": "",
      "end_date": "",
      "location": "",
      "description": ""
    }
  ],
  "education": [
    {
      "degree": "",
      "institution": "",
      "start_date": "",
      "end_date": "",
      "location": ""
    }
  ],
  "certifications": [],
  "languages": [
    {
      "language": "",
      "level": ""
    }
  ],
  "industries": []
}

Extraction guidance:
- Experience:
  - Extract job title, company, dates, location, and a concise description of key responsibilities and achievements.
  - Keep descriptions factual and include metrics where present.
  - If dates are partial, keep them as written.
  - If company or role is unclear, use an empty string for that field.
- Education:
  - Extract degree, institution, dates, and location where available.
  - Include certifications or training here only if they are clearly education entries rather than certifications.
- Skills:
  - Extract INDIVIDUAL technical skills, tools, technologies, frameworks, platforms, and methods.
  - If the CV groups skills under category headers (e.g. "Platforms & Runtime: Kafka, AMQ, MuleSoft"),
    extract each individual item (Kafka, AMQ, MuleSoft) — NOT the category header itself.
  - Aim for 10-25 specific, concrete skills. Examples: "Kafka", "Kubernetes", "Java", "Python", "MuleSoft", "PostgreSQL", "CI/CD", "API Management".
  - Do NOT include spoken languages in skills — put them in "languages" instead.
  - Do not include vague personality traits unless clearly framed as professional competencies.
- Certifications:
  - Extract named certifications only.
  - Include culinary/hospitality school awards (e.g. Kitchen & Larder, Pastry & Baking) as certifications when listed under education or training.
  - Do not move university degrees into certifications.
- Languages:
  - Extract each spoken/written language with its proficiency level.
  - Use standard levels when possible: Native, Fluent, Proficient, Advanced, Intermediate, Elementary, Beginner.
  - Preserve the original level description if it differs (e.g., "C1", "B2", "Mother tongue").
  - If a language is listed without a level, use "Proficient" as default.
- Industries:
  - Identify the industries/sectors the candidate has worked in based on their employers and job descriptions.
  - Use short, specific keywords: "iGaming", "online casino", "fintech", "investment banking", "SaaS", "healthtech", "automotive", "e-commerce", etc.
  - Include both broad sectors (e.g. "gaming") and specific niches (e.g. "iGaming", "sports betting").
  - Aim for 3-8 keywords. More is fine if the candidate has diverse experience.
  - Do NOT include generic terms like "technology" or "business" — be specific.

Important:
- Return ONLY valid JSON.
- Every key must be present exactly as specified.
- No trailing commas.
- No surrounding text.

CV TEXT:
"""


def _format_experience_item(item: Any) -> str:
    """Convert experience object to readable string."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    parts = []
    title = item.get("job_title", "")
    company = item.get("company", "")
    if title or company:
        parts.append(f"{title} at {company}".strip(" at "))
    start = item.get("start_date", "")
    end = item.get("end_date", "")
    if start or end:
        dates = " - ".join(filter(None, [start, end]))
        parts.append(f"({dates})")
    loc = item.get("location", "")
    if loc:
        parts.append(loc)
    desc = item.get("description", "")
    if desc:
        parts.append(desc)
    return " | ".join(parts) if parts else " | ".join(f"{k}: {v}" for k, v in item.items() if v)


def _format_education_item(item: Any) -> str:
    """Convert education object to readable string."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    parts = []
    degree = item.get("degree", "")
    institution = item.get("institution", "")
    if degree or institution:
        parts.append(f"{degree} at {institution}".strip(" at "))
    start = item.get("start_date", "")
    end = item.get("end_date", "")
    if start or end:
        dates = " - ".join(filter(None, [start, end]))
        parts.append(f"({dates})")
    loc = item.get("location", "")
    if loc:
        parts.append(loc)
    return " | ".join(parts) if parts else " | ".join(f"{k}: {v}" for k, v in item.items() if v)


def _format_language_item(item: Any) -> str:
    """Convert language object to 'Language - Level' string."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    lang = item.get("language", "")
    level = item.get("level", "")
    if lang and level:
        return f"{lang} - {level}"
    return lang or level or str(item)


def _to_strings(items: list, formatter: callable = None) -> list[str]:
    """Normalize list items to strings. Handles strings, dicts, and objects."""
    result = []
    for item in items or []:
        if isinstance(item, str):
            result.append(item)
        elif formatter:
            result.append(formatter(item))
        elif isinstance(item, dict):
            result.append(" | ".join(f"{k}: {v}" for k, v in item.items() if v))
        else:
            result.append(str(item))
    return result


FALLBACK_PROMPT = """Extract from this CV and return ONLY valid JSON with these keys: "name", "skills".
- name: full name or null
- skills: array of 5-15 skills (technologies, tools, languages)
Example: {"name": "John Doe", "skills": ["Python", "Java", "SQL"]}
CV:
"""


def _fallback_extraction(
    combined_text: str,
    provider: LLMProvider,
) -> UserProfile:
    """Simpler extraction when the main prompt fails or returns empty."""
    text = combined_text[:8000] + ("..." if len(combined_text) > 8000 else "")
    try:
        content = provider.chat(
            messages=[{"role": "user", "content": FALLBACK_PROMPT + text}],
            temperature=0.1,
            max_tokens=500,
        )
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            content = json_match.group(0)
        data = json.loads(content)
        return UserProfile(
            name=data.get("name"),
            email=None,
            phone=None,
            location=None,
            summary="",
            skills=_to_strings(data.get("skills", [])),
            experience=[],
            education=[],
            certifications=[],
            raw_text=combined_text,
        )
    except Exception:
        return UserProfile(raw_text=combined_text)


_INDUSTRY_PROMPT = """Based on this CV, list the industries and sectors the candidate has worked in.

Return ONLY a JSON array of short, specific keywords. Examples: "iGaming", "online casino", "fintech", "investment banking", "SaaS", "e-commerce", "healthtech".
Include both broad sectors (e.g. "gaming") and specific niches (e.g. "iGaming", "sports betting").
Do NOT include generic terms like "technology" or "business".
Aim for 3-8 keywords.

CV TEXT:
"""


def _extract_industries(
    text: str,
    provider: LLMProvider,
) -> list[str]:
    """Focused follow-up call to extract industry keywords from the CV."""
    try:
        content = provider.chat(
            messages=[{"role": "user", "content": _INDUSTRY_PROMPT + text[:6000]}],
            temperature=0.1,
            max_tokens=200,
        )
        json_match = re.search(r"\[[\s\S]*?\]", content)
        if not json_match:
            json_match = re.search(r"\[[\s\S]*\]", content)
        if json_match:
            items = json.loads(json_match.group(0))
            if isinstance(items, list):
                return [str(i).strip() for i in items if isinstance(i, str) and len(i.strip()) > 2]
    except Exception:
        pass
    return []


def extract_profile(
    combined_text: str,
    provider: LLMProvider,
    warmup: bool = True,
) -> UserProfile:
    """Use the LLM provider to extract structured profile info from raw CV text."""
    max_chars = 12000
    text = combined_text[:max_chars] + ("..." if len(combined_text) > max_chars else "")
    user_content = EXTRACTION_PROMPT + text

    if warmup:
        provider.warmup()

    content = provider.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        content = json_match.group(0)

    try:
        data = json.loads(content)
        profile = UserProfile(
            name=data.get("name"),
            email=data.get("email"),
            phone=data.get("phone"),
            location=data.get("location"),
            summary=data.get("summary") or "",
            skills=_to_strings(data.get("skills", [])),
            experience=_to_strings(data.get("experience", []), _format_experience_item),
            education=_to_strings(data.get("education", []), _format_education_item),
            certifications=_to_strings(data.get("certifications", [])),
            languages=_to_strings(data.get("languages", []), _format_language_item),
            industries=_to_strings(data.get("industries", [])),
            raw_text=combined_text,
        )
        if not profile.name and not profile.skills and len(combined_text) > 100:
            return _fallback_extraction(combined_text, provider)
        if not profile.industries:
            profile.industries = _extract_industries(text, provider)
            if profile.industries:
                logger.info(
                    "Industry follow-up extracted: %s",
                    ", ".join(profile.industries),
                )
        return profile
    except json.JSONDecodeError:
        if len(combined_text) > 100:
            return _fallback_extraction(combined_text, provider)
        return UserProfile(raw_text=combined_text)
    except Exception:
        if len(combined_text) > 100:
            return _fallback_extraction(combined_text, provider)
        return UserProfile(raw_text=combined_text)
