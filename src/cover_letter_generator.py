"""Generate tailored cover letters for specific jobs using Ollama."""

from typing import Optional

import ollama

from .models import JobListing, UserPreferences, UserProfile


def generate_cover_letter(
    profile: UserProfile,
    job: JobListing,
    preferences: UserPreferences,
    model: str = "qwen3:8b",
    host: Optional[str] = None,
) -> str:
    """
    Generate a letter of intention (cover letter) tailored for a specific job.
    """
    tone_guide = {
        "formal": "Use formal, respectful language. Address the hiring manager formally.",
        "semi-formal": "Use professional but warm language.",
        "casual": "Use friendly, conversational tone.",
        "enthusiastic": "Show strong excitement and motivation.",
        "professional": "Balance professionalism with genuine interest.",
    }
    tone = tone_guide.get(preferences.tone.lower(), tone_guide["professional"])

    prompt = f"""You are an expert at writing cover letters. Write a compelling letter of intention (cover letter) for this candidate applying to this job.

CANDIDATE:
Name: {profile.name or "The candidate"}
{f"Location: {profile.location}" if profile.location else ""}
Summary: {profile.summary or "Experienced professional"}
Key skills: {", ".join(profile.skills[:10]) if profile.skills else "Various"}
Relevant experience: {"; ".join(profile.experience[:3]) if profile.experience else "Professional experience"}

JOB:
Title: {job.title}
Company: {job.company}
Key details: {job.description[:600]}

INSTRUCTIONS:
- {tone}
- Length: 3-4 short paragraphs.
- Open with a strong hook about why they're interested in this specific role and company.
- Connect their experience to the job requirements.
- End with a clear call to action (e.g., requesting an interview).
- Do not use placeholder text like [Your Name] - use the candidate's name.
- Output ONLY the letter body, no subject line or meta-commentary.
- Format: Start with "Dear Hiring Manager," or "Dear [Company] Team," and end with "Sincerely," followed by the candidate's name."""

    client = ollama.Client(host=host) if host else ollama.Client()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7},
    )
    content = response["message"]["content"].strip()

    # Append job link so the candidate knows where to apply
    if job.url:
        content += f"\n\n---\n**Apply for this position:** [{job.title} at {job.company}]({job.url})"
    return content
