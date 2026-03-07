"""Generate tailored CVs for specific jobs using Ollama."""

from typing import Optional

import ollama

from .models import JobListing, UserPreferences, UserProfile


def generate_tailored_cv(
    profile: UserProfile,
    job: JobListing,
    preferences: UserPreferences,
    model: str = "llama3.2",
    host: Optional[str] = None,
) -> str:
    """
    Generate a CV tailored for a specific job, in markdown format.
    """
    tone_guide = {
        "formal": "Use formal, professional language throughout.",
        "semi-formal": "Use professional but approachable language.",
        "casual": "Use friendly, conversational tone where appropriate.",
        "enthusiastic": "Be energetic and show genuine enthusiasm.",
        "professional": "Balance professionalism with warmth.",
    }
    tone = tone_guide.get(preferences.tone.lower(), tone_guide["professional"])

    style_guide = {
        "concise": "Keep bullets and descriptions brief and scannable.",
        "detailed": "Include more context and achievements.",
        "balanced": "Balance brevity with impact.",
    }
    style = style_guide.get(preferences.style.lower(), style_guide["balanced"])

    focus = ""
    if preferences.focus_areas:
        focus = f"Emphasize these areas: {', '.join(preferences.focus_areas)}. "

    prompt = f"""You are an expert CV writer. Create a tailored CV/resume in MARKDOWN format for this candidate to apply to this specific job.

CANDIDATE PROFILE (from their existing CV):
Name: {profile.name or "Candidate"}
{f"Summary: {profile.summary}" if profile.summary else ""}
Skills: {", ".join(profile.skills) if profile.skills else "N/A"}
Experience: {" | ".join(profile.experience) if profile.experience else "N/A"}
Education: {" | ".join(profile.education) if profile.education else "N/A"}
Certifications: {", ".join(profile.certifications) if profile.certifications else "N/A"}

TARGET JOB:
Title: {job.title}
Company: {job.company}
Description: {job.description[:800]}

INSTRUCTIONS:
- Rewrite and tailor the CV to highlight experience and skills most relevant to THIS job.
- {tone} {style} {focus}
- Output ONLY the CV in clean markdown (use ## for sections, - for bullets).
- Sections: Header (name, contact), Summary, Skills, Experience, Education, Certifications.
- Do not add any preamble or explanation, just the CV."""

    client = ollama.Client(host=host) if host else ollama.Client()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.6},
    )
    return response["message"]["content"].strip()
