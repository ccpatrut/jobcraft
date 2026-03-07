"""Match jobs to user profile and rank them using Ollama."""

from typing import Optional

import ollama

from .models import JobListing, UserProfile


def build_profile_summary(profile: UserProfile) -> str:
    """Build a compact summary of the profile for matching."""
    parts = []
    if profile.summary:
        parts.append(profile.summary)
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    if profile.experience:
        parts.append("Experience: " + " | ".join(profile.experience[:5]))
    if profile.certifications:
        parts.append("Certifications: " + ", ".join(profile.certifications))
    return "\n".join(parts)


def rank_jobs_with_ollama(
    profile: UserProfile,
    jobs: list[JobListing],
    model: str = "llama3.2",
    top_n: int = 5,
    host: Optional[str] = None,
    max_jobs_to_rank: int = 25,
) -> list[JobListing]:
    """
    Use Ollama to rank jobs by fit and return top N.
    """
    jobs = jobs[:max_jobs_to_rank]  # Limit to avoid token overflow
    if len(jobs) <= top_n:
        return jobs[:top_n]

    profile_summary = build_profile_summary(profile)

    # Build compact job list for the prompt (shorten descriptions)
    job_lines = []
    for i, j in enumerate(jobs):
        desc = (j.description or "")[:200].replace("\n", " ")
        job_lines.append(f"{i}: {j.title} @ {j.company} - {desc}...")
    jobs_text = "\n".join(job_lines)

    prompt = f"""You are a job matching expert. Given this candidate profile and a list of jobs, select the TOP {top_n} jobs that best match the candidate's experience, skills, and background.

CANDIDATE PROFILE:
{profile_summary}

JOBS (index: title @ company - description):
{jobs_text}

Respond with ONLY the indices of the top {top_n} best-matching jobs, one per line, in order of best match first. Example:
0
3
7
12
2
"""

    client = ollama.Client(host=host) if host else ollama.Client()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3},
    )

    content = response["message"]["content"].strip()
    indices = []
    for line in content.split("\n"):
        line = line.strip().strip(".-)")
        # Extract first number on each line
        for word in line.split():
            if word.isdigit():
                idx = int(word)
                if 0 <= idx < len(jobs) and idx not in indices:
                    indices.append(idx)
                    break
        if len(indices) >= top_n:
            break

    # Build result preserving order
    result = []
    seen = set()
    for idx in indices:
        if idx not in seen and 0 <= idx < len(jobs):
            result.append(jobs[idx])
            seen.add(idx)

    # Pad with remaining if we got fewer
    for j in jobs:
        if len(result) >= top_n:
            break
        if j not in result:
            result.append(j)

    return result[:top_n]
