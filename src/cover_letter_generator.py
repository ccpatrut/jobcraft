"""Generate tailored cover letters for specific jobs using the LLM provider."""

import logging
import re

from .llm_provider import LLMProvider
from .models import JobListing, UserPreferences, UserProfile

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\[(?:Your|your|NAME|Name|name|Company|company)[^\]]*\]")


def _validate_cover_letter(text: str) -> list[str]:
    """Return a list of validation issues found in the generated cover letter."""
    issues: list[str] = []
    if len(text) < 200:
        issues.append(f"Too short ({len(text)} chars, expected 200+)")
    placeholders = _PLACEHOLDER_RE.findall(text)
    if placeholders:
        issues.append(f"Contains placeholders: {', '.join(placeholders[:3])}")
    text_lower = text.lower()
    if "sincerely" not in text_lower and "regards" not in text_lower and "best" not in text_lower:
        issues.append("Missing closing (Sincerely/Regards)")
    if "dear" not in text_lower:
        issues.append("Missing opening (Dear)")
    return issues


def generate_cover_letter(
    profile: UserProfile,
    job: JobListing,
    preferences: UserPreferences,
    provider: LLMProvider,
    temperature: float = 0.7,
) -> str:
    """Generate a letter of intention (cover letter) tailored for a specific job.

    Validates the output and retries once with a stricter prompt if issues are found.
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

    pivot_instructions = ""
    if preferences.pivot_enabled:
        from_ind = (
            ", ".join(preferences.pivot_from)
            if preferences.pivot_from
            else "their previous industry"
        )
        motivation = preferences.pivot_motivation
        pivot_instructions = f"""
CAREER PIVOT CONTEXT:
This candidate is transitioning FROM {from_ind} into the target role's industry.
{f"Their motivation: {motivation}" if motivation else ""}

PIVOT-SPECIFIC INSTRUCTIONS:
- Dedicate one paragraph to the transition itself: why the candidate is moving, and why their background is an ASSET (not a liability). Frame the old industry as demanding, fast-paced, and transferable.
- Do NOT apologize for the industry change. Instead, position it as bringing a fresh perspective and proven cross-industry skills.
- Translate achievements into the target industry's language. "Managing iGaming operator accounts" becomes "managing enterprise client relationships in a regulated, high-velocity environment."
- Lead with what's UNIVERSAL: revenue growth, client retention, stakeholder management, data-driven decisions.
"""

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
- No subject line, no meta-commentary.
{pivot_instructions}"""

    for attempt in range(2):
        extra = ""
        if attempt > 0:
            extra = (
                "\n\nSTRICT RETRY — the previous attempt had issues. "
                "Ensure: minimum 200 chars, NO placeholder brackets like [Your Name], "
                "start with 'Dear', end with 'Sincerely' or 'Best regards'."
            )
        content = provider.chat(
            messages=[{"role": "user", "content": prompt + extra}],
            temperature=temperature,
        )
        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()

        issues = _validate_cover_letter(content)
        if not issues:
            return content
        if attempt == 0:
            logger.warning("Cover letter validation issues (retrying): %s", "; ".join(issues))

    logger.warning("Cover letter still has issues after retry: %s", "; ".join(issues))
    return content
