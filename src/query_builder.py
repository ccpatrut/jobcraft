"""Search query generation for job discovery."""

from __future__ import annotations

import json as _json
import logging
import re

from .llm_provider import LLMProvider
from .models import UserPreferences, UserProfile
from .spinner import Spinner

logger = logging.getLogger(__name__)

QUERY_GEN_PROMPT = """You are a job market expert. Given a candidate's profile, generate the best job search queries to find roles they are qualified for.

CANDIDATE:
Name: {name}
Summary: {summary}
Skills: {skills}
Experience: {experience}
Industries: {industries}
Country: {country}

Generate 5-8 job search queries that would find the best matching open positions on a job board. Rules:
- Each query should be a realistic job TITLE that employers actually post (e.g., "Solution Architect", "API Engineer", "Technical Account Manager").
- Order from most relevant to broadest.
- Include the candidate's current/most recent role type, plus related roles they could realistically apply for based on their skills.
- CRITICAL: At least 2-3 queries MUST combine the role with the candidate's industry (e.g., if the candidate works in iGaming, include "Account Manager iGaming", "iGaming Client Manager"). This ensures results from the right sector.
- Consider what the {country} job market calls these roles — use common local job title conventions.
- Keep each query 2-5 words. No full sentences, no descriptions, no skills — just job titles (optionally with industry qualifier).
- Do NOT include single generic words like "Engineer" or "Architect" alone.

Return ONLY a JSON array of strings. Example:
["Account Manager iGaming", "iGaming Client Manager", "Account Manager", "Technical Account Manager", "Partnership Manager gaming"]
"""

QUERY_GEN_PIVOT_PROMPT = """You are a job market expert helping a candidate transition into a new industry.

CANDIDATE:
Name: {name}
Location: {location}
Summary: {summary}
Skills (transferable): {skills}
Education and training: {education}
Certifications: {certifications}
Previous experience: {experience}
Previous industries: {from_industries}

TARGET INDUSTRIES: {target_industries}
Motivation: {motivation}

Country: {country}

The candidate is making a CAREER PIVOT from {from_industries} into {target_industries}. Generate 6-10 job search queries that would find ENTRY-LEVEL to MID-LEVEL roles in the TARGET industries where the candidate's transferable skills would be valued.

Rules:
- Focus ENTIRELY on job titles in the TARGET industries ({target_industries}).
- Do NOT generate queries for the candidate's OLD industry ({from_industries}).
- Prioritize culinary training and education when generating kitchen/bakery/service queries.
- Include a mix of: direct role titles (e.g. "Commis Chef", "Küchenhilfe", "Bäckerei"), entry-level roles, and Aushilfe/Stundenlohn-style roles.
- Consider what the {country} job market calls these roles — use German and English local conventions.
- Keep each query 2-5 words. Job titles only, no full sentences.
- Order from most accessible (easiest transition) to aspirational.

Return ONLY a JSON array of strings. Example for hospitality pivot:
["Küchenhilfe", "Commis de Cuisine", "Kitchen Assistant", "Bäckerei", "Konditor", "Aushilfe Gastronomie", "Spüler", "Kellner"]
"""


def extract_role_from_text(text: str) -> str | None:
    """Find a job title/role in free text using common patterns.

    Matches both "domain + role" (e.g. "software engineer") and
    "seniority + role" (e.g. "senior architect") patterns.
    """
    role_suffixes = (
        r"manager|engineer|developer|designer|analyst|consultant|"
        r"specialist|coordinator|director|officer|lead|associate|"
        r"executive|administrator|supervisor|representative|advisor|"
        r"architect|strategist|planner"
    )
    domains = (
        r"account|project|product|marketing|sales|digital|software|data|"
        r"business|operations|hr|finance|it|technical|ux|ui|design|"
        r"client|customer|creative|content|brand|media|communications?|"
        r"community|event|logistics|supply chain|quality|procurement|"
        r"web|frontend|backend|full\s*stack|devops|cloud|security|"
        r"cook|chef|barista|hospitality|restaurant|catering|service|"
        r"retail|warehouse|driver|nurse|care|assistant|"
        r"platform|integration|infrastructure|api|solutions|systems|ai|ml"
    )
    seniority = r"senior|junior|lead|head of|chief|principal|staff"

    domain_role = re.compile(
        rf"\b((?:{seniority})\s+)?({domains})\s+({role_suffixes})",
        re.IGNORECASE,
    )
    seniority_role = re.compile(
        rf"\b({seniority})\s+({role_suffixes})\b",
        re.IGNORECASE,
    )

    matches: list[str] = []
    for m in domain_role.finditer(text):
        role = re.sub(r"\s+", " ", m.group(0)).strip()
        if len(role) > 5 and " " in role:
            matches.append(role)
    for m in seniority_role.finditer(text):
        role = re.sub(r"\s+", " ", m.group(0)).strip()
        if len(role) > 5 and " " in role:
            matches.append(role)

    if matches:
        seen: set[str] = set()
        for r in matches:
            key = r.lower()
            if key not in seen:
                seen.add(key)
                return r
    return None


def ai_generate_queries(
    profile: UserProfile,
    country: str,
    provider: LLMProvider,
    prefs: UserPreferences | None = None,
) -> list[str]:
    """Use the LLM to generate job search queries from the profile."""
    skills_str = ", ".join(profile.skills[:15]) if profile.skills else "N/A"
    exp_str = "; ".join(e[:100] for e in profile.experience[:4]) if profile.experience else "N/A"
    edu_str = "; ".join(e[:120] for e in profile.education[:3]) if profile.education else "N/A"
    cert_str = ", ".join(profile.certifications[:8]) if profile.certifications else "N/A"
    location = profile.location or "N/A"

    if prefs and prefs.pivot_enabled and profile.industries:
        from_ind = ", ".join(prefs.pivot_from) if prefs.pivot_from else "their previous industry"
        prompt = QUERY_GEN_PIVOT_PROMPT.format(
            name=profile.name or "Candidate",
            location=location,
            summary=(profile.summary or "")[:300],
            skills=skills_str,
            education=edu_str,
            certifications=cert_str,
            experience=exp_str,
            from_industries=from_ind,
            target_industries=", ".join(profile.industries),
            motivation=prefs.pivot_motivation or "Career change",
            country=country,
        )
    else:
        industries_str = ", ".join(profile.industries) if profile.industries else "N/A"
        prompt = QUERY_GEN_PROMPT.format(
            name=profile.name or "Candidate",
            summary=(profile.summary or "")[:300],
            skills=skills_str,
            experience=exp_str,
            industries=industries_str,
            country=country,
        )

    try:
        content = provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )

        json_match = re.search(r"\[[\s\S]*\]", content)
        if not json_match:
            logger.warning("AI query generation returned no JSON array.")
            return []

        queries = _json.loads(json_match.group(0))
        if isinstance(queries, list) and queries:
            return [q.strip() for q in queries if isinstance(q, str) and len(q.strip()) > 3]
    except Exception as e:
        logger.warning("AI query generation failed, using fallback: %s", e)
    return []


def fallback_search_queries(
    profile: UserProfile,
    prefs: UserPreferences | None = None,
) -> list[str]:
    """Simple heuristic fallback if the LLM query generation fails.

    When career pivot is active, generates queries for the TARGET industry
    instead of the candidate's existing role titles.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        q = re.sub(r"\s*\([^)]*\)\s*", " ", q).strip()
        q = re.sub(r"\s+", " ", q)
        if q and q.lower() not in seen and len(q) > 3:
            seen.add(q.lower())
            queries.append(q)

    is_pivot = prefs and prefs.pivot_enabled and profile.industries

    if is_pivot:
        for ind in profile.industries:
            _add(ind)
        # Don't add old CV role titles -- they pull results from the wrong industry
    else:
        role_titles: list[str] = []

        for exp in profile.experience[:5]:
            if " at " in exp:
                title = exp.split(" at ")[0].strip()
                if title and len(title) < 60:
                    role_titles.append(title)
                    _add(title)
            role = extract_role_from_text(exp)
            if role:
                role_titles.append(role)
                _add(role)

        if profile.summary:
            role = extract_role_from_text(profile.summary)
            if role:
                role_titles.append(role)
                _add(role)

        industry = profile.industries[0] if profile.industries else None
        if industry and role_titles:
            for title in dict.fromkeys(role_titles):
                qualified = f"{title} {industry}"
                if len(qualified) < 50:
                    _add(qualified)

        if profile.skills:
            for skill in profile.skills[:3]:
                if len(skill) < 40:
                    _add(skill)

    if not queries:
        _add("jobs")

    return queries


def build_search_queries(
    keywords: str,
    profile: UserProfile,
    country: str = "ch",
    provider: LLMProvider | None = None,
    prefs: UserPreferences | None = None,
) -> list[str]:
    """Build search queries using AI, with heuristic fallback.

    If ``keywords`` is provided it is included as an extra query but does NOT
    suppress the AI/fallback queries — they are combined and deduplicated.
    When career pivot is active, queries target the new industry.
    """
    queries: list[str] = []

    if provider is not None:
        with Spinner("Generating search queries with AI"):
            queries = ai_generate_queries(profile, country, provider, prefs)

    if not queries:
        logger.warning("AI returned no queries, using heuristic fallback.")
        queries = fallback_search_queries(profile, prefs)

    if keywords:
        kw_lower = keywords.strip().lower()
        if kw_lower and kw_lower not in {q.lower() for q in queries}:
            queries.insert(0, keywords.strip())

    # Append industry-qualified variants (skip in pivot mode -- queries
    # already target the right industry, and cross-multiplying gives
    # duplicates like "hospitality hospitality").
    is_pivot = prefs and prefs.pivot_enabled
    if profile.industries and not is_pivot:
        existing_lower = {q.lower() for q in queries}
        industry_queries: list[str] = []
        for ind in profile.industries[:3]:
            for q in queries[:3]:
                qualified = f"{q} {ind}"
                if qualified.lower() not in existing_lower and len(qualified) < 55:
                    industry_queries.append(qualified)
                    existing_lower.add(qualified.lower())
                    if len(industry_queries) >= 4:
                        break
            if len(industry_queries) >= 4:
                break
        queries = industry_queries + queries

    return queries
