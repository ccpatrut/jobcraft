"""Language detection for job descriptions using a HuggingFace classifier."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MODEL_NAME = "papluca/xlm-roberta-base-language-detection"
_classifier = None


def _get_classifier():
    """Lazy-load the language detection pipeline (downloads on first use)."""
    global _classifier  # noqa: PLW0603
    if _classifier is None:
        from transformers import pipeline

        logger.info("Loading language detector: %s", _MODEL_NAME)
        _classifier = pipeline(
            "text-classification",
            model=_MODEL_NAME,
            top_k=1,
            truncation=True,
            max_length=512,
        )
    return _classifier


def detect_language(text: str) -> str:
    """Detect the primary language of a text.

    Returns an ISO 639-1 code (e.g., "en", "de", "fr", "it").
    Falls back to "en" on errors.
    """
    if not text or len(text.strip()) < 10:
        return "en"

    try:
        clf = _get_classifier()
        # Use the first ~500 chars for speed
        result = clf(text[:500])
        if result and result[0]:
            label = result[0][0]["label"]
            return label.lower()[:2]
    except Exception as e:
        logger.warning("Language detection failed: %s", e)

    return "en"


def filter_jobs_by_detected_language(
    jobs: list, exclude_languages: list[str]
) -> tuple[list, int]:
    """Filter out jobs whose description is primarily in an excluded language.

    Returns (kept_jobs, removed_count).
    """
    if not exclude_languages:
        return jobs, 0

    exclude_set = {lang.lower() for lang in exclude_languages}
    kept = []
    removed = 0

    for job in jobs:
        text = f"{job.title} {job.description or ''}"
        lang = detect_language(text)
        if lang in exclude_set:
            logger.info("Excluded [%s]: %s @ %s", lang, job.title, job.company)
            removed += 1
        else:
            kept.append(job)

    return kept, removed


# Minimum proficiency tier needed to work in a language (job description written in it)
_WORKING_PROFICIENCY_TIER = 4  # "Proficient" / "Advanced"

_LANG_CODE_TO_TIER_KEY = {
    "de": "german", "fr": "french", "it": "italian",
    "es": "spanish", "nl": "dutch", "pt": "portuguese", "pl": "polish",
}

_PROFICIENCY_TIERS = {
    "native": 6, "fluent": 5, "proficient": 4, "advanced": 4,
    "intermediate": 3, "elementary": 2, "beginner": 1,
    "c2": 6, "c1": 5, "b2": 4, "b1": 3, "a2": 2, "a1": 1,
    "mother tongue": 6, "muttersprache": 6,
}


def _parse_candidate_tiers(languages: list[str]) -> dict[str, int]:
    """Parse candidate language list into {lang_code: tier}."""
    result: dict[str, int] = {}
    lang_name_to_code = {
        "english": "en", "german": "de", "french": "fr", "italian": "it",
        "spanish": "es", "dutch": "nl", "portuguese": "pt", "polish": "pl",
        "romanian": "ro",
    }
    for entry in languages:
        parts = entry.split(" - ", 1)
        name = parts[0].strip().lower()
        level = parts[1].strip().lower() if len(parts) > 1 else "proficient"
        code = lang_name_to_code.get(name, name[:2])
        tier = _PROFICIENCY_TIERS.get(level, 3)
        result[code] = max(result.get(code, 0), tier)
    return result


def filter_jobs_by_description_language(
    jobs: list, candidate_languages: list[str]
) -> tuple[list, int]:
    """Filter out jobs whose description is in a language the candidate isn't proficient in.

    If a job posting is written entirely in German, the candidate needs at least
    Advanced/Proficient German to be a realistic applicant. English postings are
    always kept regardless.

    Returns (kept_jobs, removed_count).
    """
    if not candidate_languages:
        return jobs, 0

    candidate_tiers = _parse_candidate_tiers(candidate_languages)
    kept = []
    removed = 0

    for job in jobs:
        text = f"{job.title} {job.description or ''}"
        lang = detect_language(text)

        # English postings are always fine
        if lang == "en":
            kept.append(job)
            continue

        candidate_tier = candidate_tiers.get(lang, 0)
        if candidate_tier >= _WORKING_PROFICIENCY_TIER:
            kept.append(job)
        else:
            lang_name = {
                "de": "German", "fr": "French", "it": "Italian",
            }.get(lang, lang)
            logger.warning(
                "Excluded [%s posting, candidate tier %d/%d]: %s @ %s",
                lang_name, candidate_tier, _WORKING_PROFICIENCY_TIER,
                job.title, job.company,
            )
            removed += 1

    return kept, removed
