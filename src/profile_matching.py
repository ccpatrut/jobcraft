"""Build profile text used for job search queries and embedding similarity."""

from __future__ import annotations

from .models import UserPreferences, UserProfile


def build_matching_profile_text(
    profile: UserProfile,
    *,
    search_profile_summary: str = "",
    pivot_motivation: str = "",
    pivot_enabled: bool = False,
) -> str:
    """Compact profile blob for embeddings and LLM ranking (not CV generation)."""
    parts: list[str] = []

    if pivot_enabled and pivot_motivation:
        parts.append(f"Career target: {pivot_motivation.strip()}")

    if search_profile_summary:
        parts.append(search_profile_summary.strip())
    elif profile.summary:
        parts.append(profile.summary)

    if profile.location:
        parts.append(f"Location: {profile.location}")

    if profile.languages:
        parts.append("Languages: " + ", ".join(profile.languages))

    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))

    if profile.education:
        for edu in profile.education[:5]:
            parts.append(edu[:300])

    if profile.certifications:
        parts.append("Certifications: " + ", ".join(profile.certifications))

    if profile.experience:
        for exp in profile.experience[:3]:
            parts.append(exp[:300])

    return "\n".join(parts)


def matching_context_from_config(
    profile: UserProfile,
    prefs: UserPreferences | None,
    search_profile_summary: str = "",
) -> tuple[str, bool, str]:
    """Return (summary_override, pivot_enabled, pivot_motivation) for matching helpers."""
    pivot_enabled = bool(prefs and prefs.pivot_enabled)
    pivot_motivation = (prefs.pivot_motivation if prefs else "") or ""
    return search_profile_summary, pivot_enabled, pivot_motivation
