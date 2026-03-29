"""Generate tailored CVs for specific jobs using the LLM provider."""

import logging
import re

from .llm_provider import LLMProvider
from .models import JobListing, UserPreferences, UserProfile

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\[(?:Your|your|EMAIL|PHONE|email|phone|Name|name)[^\]]*\]")
_REQUIRED_SECTIONS = {"summary", "experience", "skills"}


def _validate_cv(text: str) -> list[str]:
    """Return a list of validation issues found in the generated CV."""
    issues: list[str] = []
    if len(text) < 400:
        issues.append(f"Too short ({len(text)} chars, expected 400+)")
    placeholders = _PLACEHOLDER_RE.findall(text)
    if placeholders:
        issues.append(f"Contains placeholders: {', '.join(placeholders[:3])}")
    text_lower = text.lower()
    for section in _REQUIRED_SECTIONS:
        if f"## {section}" not in text_lower and f"**{section}" not in text_lower:
            issues.append(f"Missing required section: {section}")
    return issues


def generate_tailored_cv(
    profile: UserProfile,
    job: JobListing,
    preferences: UserPreferences,
    provider: LLMProvider,
    temperature: float = 0.6,
) -> str:
    """Generate a CV tailored for a specific job, in markdown format.

    Validates the output and retries once with a stricter prompt if issues are found.
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

    pivot_block = ""
    if preferences.pivot_enabled:
        from_ind = (
            ", ".join(preferences.pivot_from)
            if preferences.pivot_from
            else "their previous industry"
        )
        pivot_block = f"""
CAREER PIVOT CONTEXT:
This candidate is transitioning FROM {from_ind} into the target role's industry.
{f"Motivation: {preferences.pivot_motivation}" if preferences.pivot_motivation else ""}

PIVOT-SPECIFIC RULES (these override general rules when they conflict):
- Frame every past achievement in terms of TRANSFERABLE SKILLS: client management, revenue growth, stakeholder engagement, process optimization, data-driven decisions, cross-functional leadership.
- TRANSLATE industry jargon: replace {from_ind}-specific terminology with universal business language the target industry understands. For example, "gaming operators" becomes "enterprise clients", "player engagement" becomes "user engagement / customer retention".
- The Summary must position the candidate as someone whose skills DIRECTLY apply to the new industry — not as an outsider trying to break in. Lead with the transferable value, not the old industry.
- DO NOT hide the previous industry — instead frame it as a strength: fast-paced, data-heavy, regulation-aware, global-scale operations.
- Emphasize metrics and outcomes over domain-specific context. "$2M portfolio, 95% retention, 30% growth" work in any industry.
"""

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
{pivot_block}

Rules:
1. EXPAND every role: take the candidate's original achievements and amplify them. If the original says "Led modernization of API Management", expand it into a rich, impactful bullet that demonstrates scope, scale, technology depth, and business outcome. Aim for 4-6 strong bullets per role.
2. REFRAME for the target job: connect each bullet to what the target role needs. Use the job description's language and priorities.
3. NEVER diminish: the tailored CV must make the candidate appear MORE qualified than their original CV, not less. Every role should read like the candidate was a high performer.
4. ADD IMPLIED DEPTH: if the candidate worked with Kafka, Kubernetes, and CI/CD, you can reasonably expand on the architectural decisions, scale, and impact those imply. Stay truthful but professional — write like a recruiter who understands the candidate's work deeply.
5. INCLUDE ALL ROLES from the original CV. Do not drop any.
6. Skills section should be tailored: lead with skills most relevant to the target role, group logically.

CRITICAL: Use the candidate's REAL contact details in the header — do NOT use placeholders like "[Your Email]".

Output ONLY the CV in this EXACT markdown structure:

---
## {profile.name or "Candidate"}
**Email:** {profile.email or "[email]"} | **Phone:** {profile.phone or "[phone]"}

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

    for attempt in range(2):
        extra = ""
        if attempt > 0:
            extra = (
                "\n\nSTRICT RETRY — the previous attempt had issues. "
                "Ensure: minimum 400 chars, NO placeholder brackets like [Your Email], "
                "and include ## Summary, ## Experience, ## Skills sections."
            )
        content = provider.chat(
            messages=[{"role": "user", "content": prompt + extra}],
            temperature=temperature,
        )
        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()

        issues = _validate_cv(content)
        if not issues:
            return content
        if attempt == 0:
            logger.warning("CV validation issues (retrying): %s", "; ".join(issues))

    logger.warning("CV still has issues after retry: %s", "; ".join(issues))
    return content
