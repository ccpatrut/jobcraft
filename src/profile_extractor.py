"""Extract structured profile from CV documents using Ollama."""

import json
import re
from typing import Optional

import ollama

from .models import UserProfile


def extract_profile_with_ollama(
    combined_text: str,
    model: str = "llama3.2",
    host: Optional[str] = None,
) -> UserProfile:
    """
    Use Ollama to extract structured profile info from raw CV text.
    """
    # Truncate if too long (models have context limits)
    max_chars = 12000
    text = combined_text[:max_chars] + ("..." if len(combined_text) > max_chars else "")

    prompt = """You are a CV/resume analyzer. Extract structured information from the following CV and document text.
Output ONLY valid JSON with these exact keys (use empty arrays/strings if not found):
{
  "name": "full name or null",
  "email": "email or null",
  "phone": "phone or null",
  "summary": "2-3 sentence professional summary",
  "skills": ["skill1", "skill2", ...],
  "experience": ["role at company - brief", ...],
  "education": ["degree/institution", ...],
  "certifications": ["cert name", ...]
}

CV TEXT:
"""
    full_prompt = prompt + text

    client = ollama.Client(host=host) if host else ollama.Client()

    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        options={"temperature": 0.2},  # Lower for more consistent extraction
    )

    content = response["message"]["content"].strip()
    # Try to parse JSON from response (sometimes wrapped in markdown)
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        content = json_match.group(0)

    try:
        data = json.loads(content)
        return UserProfile(
            name=data.get("name"),
            email=data.get("email"),
            phone=data.get("phone"),
            summary=data.get("summary", ""),
            skills=data.get("skills", []) or [],
            experience=data.get("experience", []) or [],
            education=data.get("education", []) or [],
            certifications=data.get("certifications", []) or [],
            raw_text=combined_text,
        )
    except json.JSONDecodeError:
        return UserProfile(raw_text=combined_text)
