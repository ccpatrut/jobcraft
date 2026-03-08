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

    raw_cv = (profile.raw_text or "")[:6000]

    prompt = f"""You are a senior career consultant who writes CVs that win interviews. Your job is to take this candidate's background and build the STRONGEST possible CV for the target role.

CANDIDATE'S ORIGINAL CV:
\"\"\"
{raw_cv}
\"\"\"

PROFILE:
Name: {profile.name or "Candidate"}
{f"Location: {profile.location}" if profile.location else ""}
{f"Email: {profile.email}" if profile.email else ""}
{f"Phone: {profile.phone}" if profile.phone else ""}
Skills: {", ".join(profile.skills) if profile.skills else "N/A"}
Certifications: {", ".join(profile.certifications) if profile.certifications else "N/A"}
Languages: {", ".join(profile.languages) if profile.languages else "N/A"}

TARGET ROLE:
Title: {job.title}
Company: {job.company}
Description: {job.description[:1200]}

YOUR MANDATE:
You must make this candidate look HIGHLY competent and perfectly suited for this role. {tone} {style} {focus}

Rules:
1. EXPAND every role: take the candidate's original achievements and amplify them. If the original says "Led modernization of API Management", expand it into a rich, impactful bullet that demonstrates scope, scale, technology depth, and business outcome. Aim for 4-6 strong bullets per role.
2. REFRAME for the target job: connect each bullet to what the target role needs. Use the job description's language and priorities.
3. NEVER diminish: the tailored CV must make the candidate appear MORE qualified than their original CV, not less. Every role should read like the candidate was a high performer.
4. ADD IMPLIED DEPTH: if the candidate worked with Kafka, Kubernetes, and CI/CD, you can reasonably expand on the architectural decisions, scale, and impact those imply. Stay truthful but professional — write like a recruiter who understands the candidate's work deeply.
5. INCLUDE ALL ROLES from the original CV. Do not drop any.
6. Skills section should be tailored: lead with skills most relevant to the target role, group logically.

Output ONLY the CV in this EXACT markdown structure:

---
## Header
Name: [full name]
Contact: [email] | [phone]

## Summary
[3-4 impactful sentences positioning the candidate as an ideal fit for this role]

## Skills
- [most relevant skill group]
- [second skill group]
...

## Experience
### [Job Title at Company | Dates]
* [expanded achievement with scope, scale, and impact]
* [responsibility reframed for target role]
* [technical depth and business outcome]
* [leadership, collaboration, or process improvement]

### [Next Role at Company | Dates]
* [4-6 bullets following the same pattern]
...

## Education
- [degree at institution | dates]

## Languages
- [Language - Level]

## Certifications
- [certification]
---
Formatting rules:
- ## for sections, ### for roles, * for experience bullets, - for skills/education/languages/certifications.
- For Languages, keep the exact proficiency levels (e.g., "English - Proficient", "German - Intermediate").
- If there are NO certifications, OMIT the ## Certifications section entirely.
- No preamble, no commentary, no explanation — only the CV."""

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
