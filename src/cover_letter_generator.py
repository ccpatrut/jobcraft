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

    raw_cv = (profile.raw_text or "")[:4000]

    prompt = f"""You are a senior career consultant who writes cover letters that get interviews. Your job is to make the candidate sound compelling, confident, and perfectly matched for this role.

CANDIDATE'S BACKGROUND:
\"\"\"
{raw_cv}
\"\"\"

Name: {profile.name or "The candidate"}
{f"Location: {profile.location}" if profile.location else ""}
Languages: {", ".join(profile.languages) if profile.languages else "N/A"}

TARGET ROLE:
Title: {job.title}
Company: {job.company}
Description: {job.description[:800]}

INSTRUCTIONS:
- {tone}
- Write 3-4 compelling paragraphs.
- Opening: a confident, specific hook about why this candidate is drawn to THIS role at THIS company. Not generic — reference something concrete from the job description.
- Body: connect the candidate's strongest achievements to what the role demands. Draw directly from their CV — mention real projects, technologies, metrics, and outcomes. Make the reader think "this person has already done exactly what we need."
- Closing: express enthusiasm and request an interview. Be assertive, not passive.
- The letter must make the candidate sound MORE impressive than a plain reading of their CV. Expand on their impact, frame their experience in the language of the target role.
- Do not use placeholder text like [Your Name] — use the candidate's actual name.
- Output ONLY the letter body. Start with "Dear Hiring Manager," or "Dear {job.company} Team," and end with "Sincerely," followed by the name.
- No subject line, no meta-commentary."""

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
