"""Generate tailored CVs for specific jobs using Ollama."""

from typing import Optional

import ollama

from .models import JobListing, UserPreferences, UserProfile


def generate_tailored_cv(
    profile: UserProfile,
    job: JobListing,
    preferences: UserPreferences,
    model: str = "qwen3:8b",
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
{f"Location: {profile.location}" if profile.location else ""}
{f"Email: {profile.email}" if profile.email else ""}
{f"Phone: {profile.phone}" if profile.phone else ""}
{f"Summary: {profile.summary}" if profile.summary else ""}
Skills: {", ".join(profile.skills) if profile.skills else "N/A"}
Experience: {" | ".join(profile.experience) if profile.experience else "N/A"}
Education: {" | ".join(profile.education) if profile.education else "N/A"}
Certifications: {", ".join(profile.certifications) if profile.certifications else "N/A"}
Languages: {", ".join(profile.languages) if profile.languages else "N/A"}

TARGET JOB:
Title: {job.title}
Company: {job.company}
Description: {job.description[:800]}

INSTRUCTIONS:
- Rewrite and tailor the CV to highlight experience and skills most relevant to THIS job.
- {tone} {style} {focus}
- Output ONLY the CV. Follow this EXACT structure (do not add extra sections or change the format):

---
## Header
Name: [candidate full name]
Contact: [email] | [phone]

## Summary
[2-3 sentences tailored to this job]

## Skills
- [skill 1]
- [skill 2]
- [skill 3]
...

## Experience
### [Job Title 1]
* [achievement or responsibility]
* [achievement or responsibility]

### [Job Title 2]
* [achievement or responsibility]
...

## Education
- [degree/institution]

## Languages
- [Language - Level]

## Certifications
- [certification 1]
- [certification 2]
---
- Use ## for main sections, ### for each role under Experience, - for Skills/Education/Certifications/Languages, * for Experience bullets.
- For Languages, preserve the exact proficiency levels from the profile (e.g., "English - Fluent", "German - Intermediate"). Do not change or omit the levels.
- If the candidate has NO certifications (listed as "N/A" or empty), OMIT the ## Certifications section entirely. Do not include it at all.
- Do not add preamble, explanation, or anything outside the structure above."""

    client = ollama.Client(host=host) if host else ollama.Client()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.6},
    )
    content = response["message"]["content"].strip()

    # Append job link so the candidate knows where to apply
    if job.url:
        content += f"\n\n---\n**Apply for this position:** [{job.title} at {job.company}]({job.url})"
    return content
