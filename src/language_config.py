"""Apply language proficiency settings from config onto UserProfile."""

from __future__ import annotations

from .models import UserProfile

_PROFICIENCY_TIERS: dict[str, int] = {
    "native": 6,
    "fluent": 5,
    "proficient": 4,
    "advanced": 4,
    "intermediate": 3,
    "intermediary": 3,
    "elementary": 2,
    "beginner": 1,
    "c2": 6,
    "c1": 5,
    "b2": 4,
    "b1": 3,
    "a2": 2,
    "a1": 1,
    "mother tongue": 6,
    "muttersprache": 6,
}

_LANG_NAME_TO_CODE: dict[str, str] = {
    "english": "en",
    "german": "de",
    "french": "fr",
    "italian": "it",
    "spanish": "es",
    "dutch": "nl",
    "portuguese": "pt",
    "polish": "pl",
    "romanian": "ro",
}


def _parse_entry(entry: str) -> tuple[str, int]:
    parts = entry.split(" - ", 1)
    name = parts[0].strip().lower()
    level = parts[1].strip().lower() if len(parts) > 1 else "proficient"
    code = _LANG_NAME_TO_CODE.get(name, name[:2])
    tier = 3
    for key, t in _PROFICIENCY_TIERS.items():
        if key in level:
            tier = max(tier, t)
    return code, tier


def _format_entry(code: str, tier: int, original: str | None = None) -> str:
    if original and _parse_entry(original)[1] == tier:
        return original
    inv = {v: k.title() for k, v in _LANG_NAME_TO_CODE.items()}
    name = inv.get(code, code)
    tier_to_label = {
        6: "Native",
        5: "Fluent",
        4: "Proficient",
        3: "Intermediate",
        2: "Elementary",
        1: "Beginner",
    }
    return f"{name} - {tier_to_label.get(tier, 'Proficient')}"


def merge_language_tiers(existing: list[str], config_langs: list[str]) -> list[str]:
    """Merge two language lists, keeping the higher tier per language code."""
    tiers: dict[str, int] = {}
    originals: dict[str, str] = {}
    for entry in existing + config_langs:
        code, tier = _parse_entry(entry)
        if tier >= tiers.get(code, 0):
            tiers[code] = tier
            originals[code] = entry
    order = ["en", "de", "es", "fr", "it", "nl", "pt", "pl", "ro"]
    result: list[str] = []
    seen: set[str] = set()
    for code in order:
        if code in tiers:
            result.append(_format_entry(code, tiers[code], originals.get(code)))
            seen.add(code)
    for code, tier in tiers.items():
        if code not in seen:
            result.append(_format_entry(code, tier, originals.get(code)))
    return result


def apply_language_config(
    profile: UserProfile,
    config_languages: list[str],
    mode: str = "override",
) -> None:
    """Apply preferences.languages onto profile (in-place)."""
    if not config_languages:
        return
    if mode == "merge":
        profile.languages = merge_language_tiers(profile.languages or [], config_languages)
    else:
        profile.languages = list(config_languages)
