"""Match jobs to user profile and rank them using Ollama."""

import logging
import re
from typing import Optional

import ollama

from .models import JobListing, UserProfile

logger = logging.getLogger(__name__)

# Proficiency tiers (higher number = higher proficiency)
_LEVEL_TIERS = {
    "native": 6, "mother tongue": 6, "muttersprache": 6, "langue maternelle": 6,
    "madrelingua": 6, "c2": 6,
    "fluent": 5, "fließend": 5, "fliessend": 5, "couramment": 5, "courant": 5,
    "fluente": 5, "c1": 5,
    "proficient": 4, "advanced": 4, "fortgeschritten": 4, "avancé": 4, "avanzato": 4,
    "b2": 4,
    "intermediate": 3, "intermediary": 3, "mittel": 3, "intermédiaire": 3,
    "intermedio": 3, "b1": 3,
    "elementary": 2, "basic": 2, "grundkenntnisse": 2, "élémentaire": 2,
    "elementare": 2, "a2": 2,
    "beginner": 1, "anfänger": 1, "débutant": 1, "principiante": 1, "a1": 1,
}

# Language name variants → language code
_LANG_NAMES: dict[str, str] = {
    "english": "en", "englisch": "en", "anglais": "en", "inglese": "en",
    "german": "de", "deutsch": "de", "allemand": "de", "tedesco": "de",
    "french": "fr", "französisch": "fr", "français": "fr", "francese": "fr",
    "italian": "it", "italienisch": "it", "italien": "it", "italiano": "it",
    "spanish": "es", "spanisch": "es", "espagnol": "es", "español": "es",
    "portuguese": "pt", "dutch": "nl", "polish": "pl",
}

# CEFR levels and their tiers
_CEFR_TIER = {"c2": 6, "c1": 5, "b2": 4, "b1": 3, "a2": 2, "a1": 1}

# Detect CEFR levels near a language name (within ~40 chars of each other)
# e.g. "Deutschkenntnisse Niveau C2", "Français niveau C1", "English C2 required"
_CEFR_NEAR_LANG = re.compile(
    r"(?P<lang>deutsch\w*|englisch\w*|english|fran[çc]ais\w*|french|"
    r"italiano?\w*|italian|spanish|spanisch\w*|espagnol\w*)"
    r"[\w\s,.:;-]{0,30}?"
    r"(?P<level>[CcBbAa][12])",
    re.IGNORECASE,
)

# Also detect the reverse: "C2 ... Deutsch"
_CEFR_BEFORE_LANG = re.compile(
    r"(?P<level>[CcBbAa][12])"
    r"[\w\s,.:;-]{0,30}?"
    r"(?P<lang>deutsch\w*|englisch\w*|english|fran[çc]ais\w*|french|"
    r"italiano?\w*|italian|spanish|spanisch\w*|espagnol\w*)",
    re.IGNORECASE,
)

# Keyword patterns that imply fluent/native (tier 5+) without a CEFR code
_FLUENCY_KEYWORDS: list[tuple[re.Pattern, str]] = [
    # German
    (re.compile(r"flie[ßs]end\w*\s+deutsch", re.IGNORECASE), "de"),
    (re.compile(r"muttersprach\w*\s+deutsch", re.IGNORECASE), "de"),
    (re.compile(r"deutsch\w*\s+(?:als\s+)?muttersprache", re.IGNORECASE), "de"),
    (re.compile(r"sehr\s+gute\w*\s+deutsch", re.IGNORECASE), "de"),
    (re.compile(r"stilsicher\w*\s+deutsch", re.IGNORECASE), "de"),
    (re.compile(r"perfekte\w*\s+deutsch", re.IGNORECASE), "de"),
    # French
    (re.compile(r"fran[çc]ais\s+(?:courant|maternel)", re.IGNORECASE), "fr"),
    (re.compile(r"couramment?\s+(?:le\s+)?fran[çc]ais", re.IGNORECASE), "fr"),
    (re.compile(r"langue\s+maternelle\s*:?\s*fran[çc]ais", re.IGNORECASE), "fr"),
    (re.compile(r"ma[îi]trise\s+(?:du\s+)?fran[çc]ais", re.IGNORECASE), "fr"),
    # Italian
    (re.compile(r"italiano\s+(?:fluente|madrelingua)", re.IGNORECASE), "it"),
    (re.compile(r"madrelingua\s+italian[ao]", re.IGNORECASE), "it"),
    # English
    (re.compile(r"fluent\s+(?:in\s+)?english", re.IGNORECASE), "en"),
    (re.compile(r"english\s+(?:native|fluent)", re.IGNORECASE), "en"),
    (re.compile(r"flie[ßs]end\w*\s+englisch", re.IGNORECASE), "en"),
    (re.compile(r"anglais\s+(?:courant|maternel)", re.IGNORECASE), "en"),
]

# Also used in profile parsing
_LANG_NAME_TO_CODE = _LANG_NAMES


def _parse_candidate_languages(profile: UserProfile) -> dict[str, int]:
    """Parse the candidate's language list into {lang_code: tier} map."""
    result = {}
    for entry in profile.languages:
        parts = entry.split(" - ", 1)
        lang_name = parts[0].strip().lower()
        level_str = parts[1].strip().lower() if len(parts) > 1 else "proficient"
        code = _LANG_NAME_TO_CODE.get(lang_name, lang_name[:2])
        tier = _LEVEL_TIERS.get(level_str, 3)
        result[code] = max(result.get(code, 0), tier)
    return result


def _resolve_lang_code(name: str) -> str | None:
    """Map a language name variant to its ISO code."""
    return _LANG_NAMES.get(name.lower().rstrip("ekenntnisse"))


def _job_requires_fluency_beyond(job: JobListing, candidate_langs: dict[str, int]) -> str | None:
    """Check if a job requires fluency the candidate doesn't have.

    Returns a human-readable reason string, or None if OK.
    """
    text = f"{job.title} {job.description}"
    lang_display = {"de": "German", "fr": "French", "it": "Italian", "en": "English"}

    # Pass 1: detect explicit CEFR levels near language names
    for match in _CEFR_NEAR_LANG.finditer(text):
        lang_raw = match.group("lang").lower()
        # Strip common suffixes like "kenntnisse"
        for base, code in _LANG_NAMES.items():
            if lang_raw.startswith(base):
                cefr = match.group("level").lower()
                required_tier = _CEFR_TIER.get(cefr, 0)
                candidate_tier = candidate_langs.get(code, 0)
                if required_tier > candidate_tier:
                    name = lang_display.get(code, code)
                    return f"requires {name} {cefr.upper()} (candidate: tier {candidate_tier}/6)"
                break

    for match in _CEFR_BEFORE_LANG.finditer(text):
        lang_raw = match.group("lang").lower()
        for base, code in _LANG_NAMES.items():
            if lang_raw.startswith(base):
                cefr = match.group("level").lower()
                required_tier = _CEFR_TIER.get(cefr, 0)
                candidate_tier = candidate_langs.get(code, 0)
                if required_tier > candidate_tier:
                    name = lang_display.get(code, code)
                    return f"requires {name} {cefr.upper()} (candidate: tier {candidate_tier}/6)"
                break

    # Pass 2: keyword-based detection (fluent/native without CEFR)
    for pattern, lang_code in _FLUENCY_KEYWORDS:
        if pattern.search(text):
            required_tier = 5
            candidate_tier = candidate_langs.get(lang_code, 0)
            if candidate_tier < required_tier:
                name = lang_display.get(lang_code, lang_code)
                return f"requires fluent {name} (candidate: tier {candidate_tier}/6)"

    return None


def filter_jobs_by_language(
    jobs: list[JobListing], profile: UserProfile
) -> tuple[list[JobListing], int]:
    """Remove jobs whose language requirements exceed the candidate's levels.

    Returns (filtered_jobs, removed_count).
    """
    if not profile.languages:
        return jobs, 0

    candidate_langs = _parse_candidate_languages(profile)
    kept = []
    removed = 0
    for job in jobs:
        reason = _job_requires_fluency_beyond(job, candidate_langs)
        if reason:
            logger.warning(
                "Excluded: %s @ %s (%s)", job.title, job.company, reason
            )
            removed += 1
        else:
            kept.append(job)
    return kept, removed


# ---------------------------------------------------------------------------
# AI-powered language validation loop
# ---------------------------------------------------------------------------

_VALIDATION_PROMPT = """You are a language-requirement analyst. Given a candidate's language proficiency and a job posting, determine if the candidate meets the language requirements.

CANDIDATE LANGUAGES:
{languages}

JOB: {title} @ {company}
DESCRIPTION:
{description}

Does this job require a language proficiency level HIGHER than what the candidate has?
Look for explicit requirements like CEFR levels (A1-C2), words like "fluent", "native", "fließend", "Muttersprache", "courant", "madrelingua", or any phrasing that implies a required level.

Respond with ONLY valid JSON:
- If the candidate MEETS all language requirements: {{"pass": true}}
- If the candidate DOES NOT meet a requirement:
  {{"pass": false, "language": "<language name in English>", "required_phrase": "<exact phrase from description that states the requirement>", "required_level": "<level like C2, fluent, native>"}}
"""


def _ai_validate_job(
    job: JobListing,
    profile: UserProfile,
    client: "ollama.Client",
    model: str,
) -> dict | None:
    """Use LLM to check if a job's language requirements exceed the candidate's.

    Returns None if the job passes, or a dict with failure details.
    """
    import json as _json

    lang_display = ", ".join(profile.languages) if profile.languages else "Not specified"
    prompt = _VALIDATION_PROMPT.format(
        languages=lang_display,
        title=job.title,
        company=job.company,
        description=(job.description or "")[:1500],
    )

    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 200},
        )
        content = resp["message"]["content"].strip()
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            content = json_match.group(0)
        data = _json.loads(content)
        if data.get("pass") is False:
            return data
    except Exception:
        pass
    return None


def _learn_pattern_from_phrase(phrase: str, language: str) -> re.Pattern | None:
    """Build a regex pattern from a phrase the AI flagged as a language requirement."""
    if not phrase or len(phrase) < 4:
        return None
    escaped = re.escape(phrase.strip())
    # Allow minor whitespace variations
    flexible = re.sub(r"\\ ", r"\\s+", escaped)
    try:
        return re.compile(flexible, re.IGNORECASE)
    except re.error:
        return None


def ai_validate_and_filter(
    jobs: list[JobListing],
    profile: UserProfile,
    model: str = "qwen3:8b",
    host: str | None = None,
    max_rounds: int = 3,
) -> tuple[list[JobListing], int, list[str]]:
    """AI validation loop: LLM re-checks jobs, learns missed patterns, re-filters.

    Returns (filtered_jobs, total_removed, learned_phrases).
    """
    import sys
    import time

    if not profile.languages:
        return jobs, 0, []

    client = ollama.Client(host=host) if host else ollama.Client()
    candidate_langs = _parse_candidate_languages(profile)
    learned_patterns: list[tuple[re.Pattern, str]] = []
    learned_phrases: list[str] = []
    total_removed = 0

    for round_num in range(1, max_rounds + 1):
        flagged_indices = []
        print(f"  Round {round_num}/{max_rounds}: checking {len(jobs)} jobs...", flush=True)
        t0 = time.time()

        for idx, job in enumerate(jobs):
            elapsed = time.time() - t0
            sys.stdout.write(
                f"\r  Round {round_num}: [{idx + 1}/{len(jobs)}] "
                f"{elapsed:.0f}s — {job.title[:40]}"
                + " " * 20
            )
            sys.stdout.flush()

            text = f"{job.title} {job.description}"
            blocked = False
            for pattern, lang_code in learned_patterns:
                if pattern.search(text):
                    required_tier = 5
                    candidate_tier = candidate_langs.get(lang_code, 0)
                    if candidate_tier < required_tier:
                        flagged_indices.append(idx)
                        blocked = True
                        break
            if blocked:
                continue

            failure = _ai_validate_job(job, profile, client, model)
            if failure:
                flagged_indices.append(idx)
                phrase = failure.get("required_phrase", "")
                lang_en = failure.get("language", "").lower()
                lang_code = _LANG_NAMES.get(lang_en, lang_en[:2] if lang_en else "")

                if phrase and lang_code:
                    new_pattern = _learn_pattern_from_phrase(phrase, lang_en)
                    if new_pattern:
                        learned_patterns.append((new_pattern, lang_code))
                        learned_phrases.append(phrase)
                        logger.warning(
                            "AI learned new pattern: '%s' -> %s (round %d)",
                            phrase, lang_code, round_num,
                        )

        elapsed = time.time() - t0
        print(
            f"\r  Round {round_num} done — {elapsed:.1f}s, "
            f"{len(flagged_indices)} flagged" + " " * 30,
            flush=True,
        )

        if not flagged_indices:
            print("  No violations found — validation complete.", flush=True)
            break

        flagged_set = set(flagged_indices)
        removed_jobs = [jobs[i] for i in flagged_indices]
        jobs = [j for i, j in enumerate(jobs) if i not in flagged_set]
        total_removed += len(removed_jobs)

        for rj in removed_jobs:
            print(f"    ✗ {rj.title} @ {rj.company}", flush=True)
            logger.warning("AI excluded (round %d): %s @ %s", round_num, rj.title, rj.company)

        if not jobs:
            break

    return jobs, total_removed, learned_phrases


def build_profile_summary(profile: UserProfile) -> str:
    """Build a compact summary of the profile for matching."""
    parts = []
    if profile.summary:
        parts.append(profile.summary)
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    if profile.languages:
        parts.append("Languages: " + ", ".join(profile.languages))
    if profile.experience:
        parts.append("Experience: " + " | ".join(profile.experience[:5]))
    if profile.certifications:
        parts.append("Certifications: " + ", ".join(profile.certifications))
    return "\n".join(parts)


def rank_jobs_with_ollama(
    profile: UserProfile,
    jobs: list[JobListing],
    model: str = "qwen3:8b",
    top_n: int = 5,
    host: Optional[str] = None,
    max_jobs_to_rank: int = 25,
) -> list[JobListing]:
    """
    Use Ollama to rank jobs by fit and return top N.
    """
    import sys
    import time

    jobs = jobs[:max_jobs_to_rank]
    if len(jobs) <= top_n:
        return jobs[:top_n]

    print(f"  Preparing {len(jobs)} jobs for ranking...", flush=True)
    profile_summary = build_profile_summary(profile)

    job_lines = []
    for i, j in enumerate(jobs):
        desc = (j.description or "")[:200].replace("\n", " ")
        job_lines.append(f"{i}: {j.title} @ {j.company} - {desc}...")
    jobs_text = "\n".join(job_lines)

    prompt = f"""You are a job matching expert. Given this candidate profile and a list of jobs, select the TOP {top_n} jobs that best match the candidate's experience, skills, and background.

IMPORTANT RULES:
- Pay close attention to the candidate's LANGUAGE proficiency levels.
- EXCLUDE jobs that require fluent/native proficiency in a language the candidate only has at Intermediate, Elementary, or Beginner level.
  For example: if the candidate has "German - Intermediate" or "German - B1", do NOT select a job that requires "fließende Deutschkenntnisse" (fluent German) or "Muttersprache Deutsch" (native German).
- Prefer jobs where the candidate meets ALL stated language requirements.
- Among qualifying jobs, rank by best skill and experience match.

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

    print(f"  Sending {len(jobs)} jobs to {model} for ranking...", flush=True)
    t0 = time.time()

    # Use streaming to show progress while the model thinks
    content_parts = []
    token_count = 0
    stream = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3},
        stream=True,
    )
    for chunk in stream:
        token = chunk.get("message", {}).get("content", "")
        content_parts.append(token)
        token_count += 1
        elapsed = time.time() - t0
        if token_count % 20 == 0:
            sys.stdout.write(f"\r  Ranking in progress... {elapsed:.0f}s elapsed, {token_count} tokens received")
            sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\r  Ranking complete — {elapsed:.1f}s, {token_count} tokens" + " " * 20, flush=True)

    content = "".join(content_parts).strip()
    indices = []
    for line in content.split("\n"):
        line = line.strip().strip(".-)")
        for word in line.split():
            if word.isdigit():
                idx = int(word)
                if 0 <= idx < len(jobs) and idx not in indices:
                    indices.append(idx)
                    break
        if len(indices) >= top_n:
            break

    result = []
    seen = set()
    for idx in indices:
        if idx not in seen and 0 <= idx < len(jobs):
            result.append(jobs[idx])
            seen.add(idx)

    for j in jobs:
        if len(result) >= top_n:
            break
        if j not in result:
            result.append(j)

    print(f"  Top {len(result[:top_n])} matches selected", flush=True)
    return result[:top_n]
