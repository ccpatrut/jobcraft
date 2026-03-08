"""Translate job search queries into national languages for target markets."""

import json
import re
from typing import Optional

import ollama

COUNTRY_LANGUAGES: dict[str, list[dict[str, str]]] = {
    "ch": [
        {"code": "de", "name": "German"},
        {"code": "fr", "name": "French"},
        {"code": "it", "name": "Italian"},
    ],
    "de": [{"code": "de", "name": "German"}],
    "at": [{"code": "de", "name": "German"}],
    "fr": [{"code": "fr", "name": "French"}],
    "it": [{"code": "it", "name": "Italian"}],
    "gb": [],
    "us": [],
    "au": [],
    "ca": [{"code": "fr", "name": "French"}],
    "be": [
        {"code": "fr", "name": "French"},
        {"code": "nl", "name": "Dutch"},
        {"code": "de", "name": "German"},
    ],
    "nl": [{"code": "nl", "name": "Dutch"}],
    "es": [{"code": "es", "name": "Spanish"}],
    "pt": [{"code": "pt", "name": "Portuguese"}],
    "pl": [{"code": "pl", "name": "Polish"}],
}

TRANSLATION_PROMPT = """Translate this job search query into the listed languages.
Return ONLY a JSON object mapping each language code to the translated query.
The translation should be the natural way this job title or role is advertised in that country's job market.
Do not transliterate — use the actual local job market terminology.

Query: "{query}"
Languages: {languages}

Example input:  Query: "Account Manager"  Languages: de (German), fr (French)
Example output: {{"de": "Account Manager", "fr": "Responsable de comptes"}}

Return ONLY the JSON object, nothing else."""


def get_localized_queries(
    query: str,
    country: str,
    model: str = "qwen3:8b",
    host: Optional[str] = None,
    language_strategy: str = "auto",
) -> list[str]:
    """
    Generate localized search queries for a given country.

    language_strategy:
      "auto"          - English first, then local languages
      "english_only"  - English only, no translations
      "local_first"   - local languages first, then English as fallback

    Returns a list of unique queries ordered by the chosen strategy.
    """
    if language_strategy == "english_only":
        return [query]

    languages = COUNTRY_LANGUAGES.get(country, [])
    if not languages:
        return [query]

    lang_desc = ", ".join(f"{lang['code']} ({lang['name']})" for lang in languages)
    prompt = TRANSLATION_PROMPT.format(query=query, languages=lang_desc)

    try:
        client = ollama.Client(host=host) if host else ollama.Client()
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 300},
        )
        content = response["message"]["content"].strip()
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            content = json_match.group(0)
        translations = json.loads(content)
    except Exception:
        return [query]

    local_queries = []
    seen = {query.lower()}
    for lang in languages:
        translated = translations.get(lang["code"], "")
        if isinstance(translated, str) and translated and translated.lower() not in seen:
            seen.add(translated.lower())
            local_queries.append(translated)

    if language_strategy == "local_first":
        return local_queries + [query]

    # "auto" (default): English first, then local
    return [query] + local_queries


# Common marker words per language for lightweight detection.
_LANG_MARKERS: dict[str, set[str]] = {
    "de": {
        "und", "oder", "für", "mit", "bei", "wir", "sie", "ihre", "unser",
        "ist", "sind", "werden", "suchen", "aufgaben", "anforderungen",
        "berufserfahrung", "kenntnisse", "stellenangebot", "arbeitsort",
        "bewerbung", "verantwortung", "deutsch", "gmbh",
    },
    "fr": {
        "nous", "vous", "avec", "pour", "dans", "notre", "votre", "sont",
        "les", "des", "une", "qui", "est", "recherchons", "poste",
        "missions", "profil", "entreprise", "expérience", "candidature",
    },
    "it": {
        "con", "per", "nel", "sono", "della", "delle", "nostro", "nostra",
        "cerchiamo", "offerta", "esperienza", "competenze", "candidatura",
        "responsabilità", "requisiti", "lavoro", "azienda",
    },
    "nl": {
        "wij", "voor", "met", "van", "een", "het", "zijn", "onze",
        "zoeken", "vacature", "ervaring", "functie", "werkzaamheden",
    },
    "es": {
        "para", "con", "que", "los", "las", "una", "del", "nuestro",
        "buscamos", "experiencia", "puesto", "empresa", "requisitos",
    },
    "pt": {
        "para", "com", "que", "uma", "nosso", "nossa", "procuramos",
        "experiência", "empresa", "requisitos", "candidatura",
    },
    "pl": {
        "dla", "jest", "lub", "nasz", "szukamy", "wymagania",
        "doświadczenie", "stanowisko", "firma", "oferta",
    },
}


def detect_language_markers(text: str, lang_code: str) -> bool:
    """Return True if the text likely contains content in the given language."""
    markers = _LANG_MARKERS.get(lang_code)
    if not markers:
        return False
    words = set(text.lower().split())
    hits = words & markers
    return len(hits) >= 3


def get_country_name(country_code: str) -> str:
    """Return a human-readable country name for display."""
    names = {
        "ch": "Switzerland",
        "de": "Germany",
        "at": "Austria",
        "fr": "France",
        "it": "Italy",
        "gb": "United Kingdom",
        "us": "United States",
        "au": "Australia",
        "ca": "Canada",
        "be": "Belgium",
        "nl": "Netherlands",
        "es": "Spain",
        "pt": "Portugal",
        "pl": "Poland",
    }
    return names.get(country_code, country_code.upper())


def get_country_languages_display(country_code: str) -> str:
    """Return a display string of national languages for a country."""
    languages = COUNTRY_LANGUAGES.get(country_code, [])
    if not languages:
        return "English"
    names = [lang["name"] for lang in languages]
    return ", ".join(names)
