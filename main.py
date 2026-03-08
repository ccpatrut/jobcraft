#!/usr/bin/env python3
"""
Job Finder - AI-powered job matching and tailored CV/cover letter generation.

1. Drop your CVs (PDF, Word) in the cv_input/ directory
2. Set ADZUNA_APP_ID and ADZUNA_APP_KEY in .env (free at developer.adzuna.com)
3. Run: python main.py

Ensure Ollama is running locally (ollama serve) and you have pulled a model:
  ollama pull qwen3:8b
"""

import logging
import os
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ollama

from src.config import get_preferences, load_config
from src.cover_letter_generator import generate_cover_letter
from src.cv_generator import generate_tailored_cv
from src.database import (
    get_cached_profile,
    get_connection,
    get_job_id,
    get_stats,
    init_db,
    save_generated_output,
    save_jobs,
    save_profile,
    search_cached_jobs,
)
from src.document_loader import load_documents_from_dir
from src.job_fetcher import fetch_adzuna_jobs
from src.job_matcher import ai_validate_and_filter, filter_jobs_by_language, rank_jobs_with_ollama
from src.pdf_utils import markdown_to_pdf
from src.profile_extractor import extract_profile_with_ollama
from src.query_translator import (
    detect_language_markers,
    get_country_languages_display,
    get_country_name,
    get_localized_queries,
)

logger = logging.getLogger(__name__)


def _extract_role_from_text(text: str) -> str | None:
    """Find a job title/role in free text using common patterns."""
    import re

    role_pattern = re.compile(
        r"\b((?:senior|junior|lead|head of|chief|principal|staff)\s+)?"
        r"(account|project|product|marketing|sales|digital|software|data|"
        r"business|operations|hr|finance|it|technical|ux|ui|design|"
        r"client|customer|creative|content|brand|media|communications?|"
        r"community|event|logistics|supply chain|quality|procurement|"
        r"web|frontend|backend|full\s*stack|devops|cloud|security|"
        r"cook|chef|barista|hospitality|restaurant|catering|service|"
        r"retail|warehouse|driver|nurse|care|assistant)"
        r"(\s+(?:manager|engineer|developer|designer|analyst|consultant|"
        r"specialist|coordinator|director|officer|lead|associate|"
        r"executive|administrator|supervisor|representative|advisor|"
        r"architect|strategist|planner))?",
        re.IGNORECASE,
    )
    matches = role_pattern.findall(text)
    if matches:
        roles = [re.sub(r"\s+", " ", " ".join(parts)).strip() for parts in matches if any(parts)]
        roles = [r for r in roles if len(r) > 3]
        if roles:
            seen = set()
            unique = []
            for r in roles:
                key = r.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            return unique[0]
    return None


_QUERY_GEN_PROMPT = """You are a job market expert. Given a candidate's profile, generate the best job search queries to find roles they are qualified for.

CANDIDATE:
Name: {name}
Summary: {summary}
Skills: {skills}
Experience: {experience}
Country: {country}

Generate 5-8 job search queries that would find the best matching open positions on a job board. Rules:
- Each query should be a realistic job TITLE that employers actually post (e.g., "Solution Architect", "API Engineer", "Technical Account Manager").
- Order from most relevant to broadest.
- Include the candidate's current/most recent role type, plus related roles they could realistically apply for based on their skills.
- Consider what the {country} job market calls these roles — use common local job title conventions.
- Keep each query 2-4 words. No full sentences, no descriptions, no skills — just job titles.
- Do NOT include single generic words like "Engineer" or "Architect" alone.

Return ONLY a JSON array of strings. Example:
["Solution Architect", "Integration Engineer", "Platform Engineer", "API Lead", "DevOps Architect"]
"""


def _ai_generate_queries(
    profile, country: str, model: str, host: str | None
) -> list[str]:
    """Use the LLM to generate job search queries from the profile."""
    import json as _json

    skills_str = ", ".join(profile.skills[:15]) if profile.skills else "N/A"
    exp_str = "; ".join(e[:100] for e in profile.experience[:4]) if profile.experience else "N/A"

    prompt = _QUERY_GEN_PROMPT.format(
        name=profile.name or "Candidate",
        summary=(profile.summary or "")[:300],
        skills=skills_str,
        experience=exp_str,
        country=country,
    )

    try:
        client = ollama.Client(host=host) if host else ollama.Client()
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 300},
        )
        content = resp["message"]["content"].strip()
        json_match = re.search(r"\[[\s\S]*\]", content)
        if json_match:
            content = json_match.group(0)
        queries = _json.loads(content)
        if isinstance(queries, list) and queries:
            return [q.strip() for q in queries if isinstance(q, str) and len(q.strip()) > 3]
    except Exception as e:
        logger.warning("AI query generation failed, using fallback: %s", e)
    return []


def _fallback_search_queries(profile) -> list[str]:
    """Simple heuristic fallback if the LLM query generation fails."""
    queries: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        q = re.sub(r"\s*\([^)]*\)\s*", " ", q).strip()
        q = re.sub(r"\s+", " ", q)
        if q and q.lower() not in seen and len(q) > 3:
            seen.add(q.lower())
            queries.append(q)

    for exp in profile.experience[:3]:
        title = exp.split(" at ")[0].strip() if " at " in exp else exp.split(" | ")[0].strip()
        if title and len(title) < 60:
            _add(title)

    if profile.summary:
        role = _extract_role_from_text(profile.summary)
        if role:
            _add(role)

    if not queries:
        _add("jobs")

    return queries


def _build_search_queries(
    keywords: str, profile, country: str = "ch",
    model: str = "qwen3:8b", host: str | None = None,
) -> list[str]:
    """Build search queries using AI, with heuristic fallback."""
    if keywords:
        return [keywords]

    print("  Generating search queries with AI...", flush=True)
    queries = _ai_generate_queries(profile, country, model, host)

    if not queries:
        logger.warning("AI returned no queries, using heuristic fallback.")
        queries = _fallback_search_queries(profile)

    return queries


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Job Finder - AI-Powered Job Matching")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore cached profile and re-extract from CV",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("Job Finder - AI-Powered Job Matching & CV Generation")
    print("=" * 60)

    cfg = load_config()
    prefs = get_preferences(cfg)
    cv_dir = Path(cfg["cv_input_dir"]).expanduser().resolve()
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    ai_cfg = cfg.get("ai", {})
    model = ai_cfg.get("model", "qwen3:8b")
    ollama_host = ai_cfg.get("host") or os.environ.get("OLLAMA_HOST") or None
    job_cfg = cfg.get("job_search", {})
    country = job_cfg.get("country", "gb")
    max_results = job_cfg.get("max_results", 50)
    keywords = job_cfg.get("keywords", "").strip()
    language_strategy = job_cfg.get("language", "auto")
    exclude_languages = job_cfg.get("exclude_languages", []) or []
    locations = job_cfg.get("locations", []) or []
    radius_km = int(job_cfg.get("radius_km", 0))

    # Initialize database
    conn = get_connection()
    init_db(conn)

    # 1. Load documents
    print("\n[1/6] Loading CV documents from:", cv_dir)
    _, combined_text = load_documents_from_dir(cv_dir)
    if (
        combined_text.startswith("[")
        and "not found" in combined_text
        or "No documents" in combined_text
    ):
        print("  ERROR: No documents found. Add PDF or Word files to:", cv_dir)
        return 1
    print("  Loaded successfully. (CV text length:", len(combined_text), "chars)")
    if len(combined_text) < 200:
        print(
            "  WARNING: CV text is very short. If using a scanned PDF, try a text-based PDF or add keywords in config.yaml."
        )

    # 2. Extract profile (use cache if CV unchanged)
    print("\n[2/6] Extracting profile...")
    profile = None
    if args.fresh:
        print("  --fresh flag set: skipping cache, re-extracting from CV.")
    else:
        profile = get_cached_profile(conn, combined_text)
        if profile:
            print("  Using cached profile (CV text unchanged).")
            print("  Tip: run with --fresh to force re-extraction.")
    if not profile:
        print("  Calling Ollama for profile extraction (this may take a minute)...")
        try:
            profile = extract_profile_with_ollama(combined_text, model=model, host=ollama_host)
            print("  Profile extracted from Ollama (fresh).")
        except Exception as e:
            print("  ERROR: Could not connect to Ollama. Is it running? Try: ollama serve")
            print("  Also ensure you have pulled a model: ollama pull", model)
            print("  Details:", e)
            return 1
    # Apply default languages from config if none extracted
    default_langs = cfg.get("preferences", {}).get("default_languages", []) or []
    if not profile.languages and default_langs:
        profile.languages = default_langs
        print("  Using default languages from config.yaml")

    profile_id = save_profile(conn, combined_text, profile)
    print("  Profile:", profile.name or "Unknown", "-", len(profile.skills), "skills")
    if profile.skills:
        print("  Skills:", ", ".join(profile.skills[:10]))
    if profile.languages:
        print("  Languages:", ", ".join(profile.languages))
    if profile.experience:
        print("  Experience:", len(profile.experience), "roles")
        for exp in profile.experience[:3]:
            print("    -", exp[:80] + ("..." if len(exp) > 80 else ""))
    if profile.summary:
        print("  Summary:", profile.summary[:120] + ("..." if len(profile.summary) > 120 else ""))

    # 3. Build search queries (English + national languages) and fetch jobs
    base_queries = _build_search_queries(keywords, profile, country=country, model=model, host=ollama_host)
    lang_info = get_country_languages_display(country)
    country_name = get_country_name(country)
    print(f"\n[3/6] Fetching jobs from Adzuna API (live) for {country_name} ({lang_info})")
    print(f"  Search queries ({len(base_queries)}):")
    for bq in base_queries:
        print(f"    • {bq}")
    if locations:
        loc_display = ", ".join(locations)
        print(f"  Locations: {loc_display} ({radius_km} km radius)")
    print(f"  Language strategy: {language_strategy}")
    if exclude_languages:
        print(f"  Excluding languages: {', '.join(exclude_languages)}")

    # For each base query, expand with local-language translations
    all_queries: list[str] = []
    seen_queries: set[str] = set()
    for bq in base_queries:
        if language_strategy == "english_only":
            expanded = [bq]
        else:
            expanded = get_localized_queries(
                bq, country, model=model, host=ollama_host,
                language_strategy=language_strategy,
            )
        for eq in expanded:
            if eq.lower() not in seen_queries:
                seen_queries.add(eq.lower())
                all_queries.append(eq)

    if len(all_queries) > len(base_queries):
        print(f"  Expanded to {len(all_queries)} queries (with translations)")

    # Search locations in order, then fall back to country-wide
    search_locations = locations + [""] if locations else [""]

    jobs = []
    seen_urls: set[str] = set()

    for loc in search_locations:
        if len(jobs) >= max_results:
            break
        loc_label = loc or "all " + country_name
        for q in all_queries:
            if len(jobs) >= max_results:
                break
            try:
                batch = fetch_adzuna_jobs(
                    search_query=q,
                    location=loc,
                    country=country,
                    results_per_page=max_results,
                    distance_km=radius_km if loc else 0,
                )
                new_in_batch = 0
                for job in batch:
                    if job.url not in seen_urls:
                        seen_urls.add(job.url)
                        jobs.append(job)
                        new_in_batch += 1
                if new_in_batch:
                    print(f"  [{loc_label} | {q[:30]}] -> {new_in_batch} new (total: {len(jobs)})")
            except ValueError as e:
                print("  ERROR:", e)
                return 1
            except Exception as e:
                print(f"  [{loc_label} | {q[:30]}] -> ERROR: {e}")
                continue

    # Filter out jobs in excluded languages
    if exclude_languages:
        before = len(jobs)
        jobs = [
            job for job in jobs
            if not any(
                detect_language_markers(f"{job.title} {job.description}", lang)
                for lang in exclude_languages
            )
        ]
        filtered = before - len(jobs)
        if filtered:
            print(f"  Filtered out {filtered} jobs in excluded languages")

    new_count = save_jobs(conn, jobs, search_query=base_queries[0], country=country)
    print(f"  Total: {len(jobs)} unique jobs from API ({new_count} new saved to database)")

    # Fallback to cached jobs if API returned too few results
    if len(jobs) < 5:
        search_terms = list(base_queries)

        if jobs:
            print(f"  Only {len(jobs)} jobs from API — checking database for more...")
        else:
            print("  No jobs from API — searching database cache...")

        cached = search_cached_jobs(conn, search_terms, country=country, limit=max_results)
        seen_urls = {j.url for j in jobs}
        added = 0
        for cj in cached:
            if cj.url not in seen_urls:
                jobs.append(cj)
                seen_urls.add(cj.url)
                added += 1
        if added:
            print(f"  Found {added} additional jobs from database cache (total: {len(jobs)})")
        else:
            print("  No additional matches found in database cache.")

    if not jobs:
        print("  No jobs found (API or cache). Try adjusting keywords or location/language in config.yaml.")
        return 1

    # 4. Filter by language requirements, then rank
    if profile.languages:
        jobs, lang_removed = filter_jobs_by_language(jobs, profile)
        if lang_removed:
            print(f"  Regex filter removed {lang_removed} jobs requiring higher language proficiency")
            print(f"  {len(jobs)} jobs remaining after regex filter")

    if not jobs:
        print("  No jobs left after language filtering. Try broadening language settings.")
        return 1

    # 4b. Optional AI-powered language validation loop
    ai_validation_enabled = cfg.get("job_search", {}).get("ai_language_validation", False)
    ai_validation_rounds = cfg.get("job_search", {}).get("ai_validation_rounds", 3)

    if ai_validation_enabled and profile.languages:
        print("\n  AI language validation enabled — verifying remaining jobs...")
        jobs, ai_removed, learned = ai_validate_and_filter(
            jobs,
            profile,
            model=model,
            host=ollama_host,
            max_rounds=ai_validation_rounds,
        )
        if ai_removed:
            print(f"  AI validation removed {ai_removed} additional jobs")
            if learned:
                print(f"  Learned {len(learned)} new pattern(s):")
                for phrase in learned:
                    print(f"    - \"{phrase}\"")
            print(f"  {len(jobs)} jobs remaining after AI validation")

    if not jobs:
        print("  No jobs left after AI validation. Try broadening language settings.")
        return 1

    print("\n[4/6] Ranking jobs by fit with Ollama...")
    try:
        top_jobs = rank_jobs_with_ollama(profile, jobs, model=model, top_n=10, host=ollama_host)
    except Exception as e:
        print("  ERROR ranking jobs:", e)
        top_jobs = jobs[:5]
    print("  Selected top", len(top_jobs), "matches")

    # 5. Generate CVs and cover letters
    output_dir.mkdir(parents=True, exist_ok=True)

    def safe_name(s: str) -> str:
        return "".join(c if c.isalnum() or c in " -_" else "_" for c in s)[:60]

    print("\n[5/6] Generating tailored CVs and cover letters...")
    for i, job in enumerate(top_jobs, 1):
        company_safe = safe_name(job.company)
        title_safe = safe_name(job.title)
        prefix = f"{i}_{company_safe}_{title_safe}"
        job_id = get_job_id(conn, job.url)

        print(f"  [{i}/{len(top_jobs)}] {job.title} @ {job.company}")

        try:
            cv_text = generate_tailored_cv(profile, job, prefs, model=model, host=ollama_host)
            cv_md_path = output_dir / f"{prefix}_cv.md"
            cv_md_path.write_text(cv_text, encoding="utf-8")
            cv_pdf_path = output_dir / f"{prefix}_cv.pdf"
            if not markdown_to_pdf(cv_text, cv_pdf_path):
                print("        (PDF generation failed for CV, .md saved)")
            print(f"        CV -> {cv_md_path.name}, {cv_pdf_path.name}")
            if job_id:
                save_generated_output(conn, job_id, profile_id, "cv", str(cv_md_path), cv_text)
        except Exception as e:
            print(f"        CV ERROR: {e}")

        try:
            letter_text = generate_cover_letter(profile, job, prefs, model=model, host=ollama_host)
            letter_md_path = output_dir / f"{prefix}_cover_letter.md"
            letter_md_path.write_text(letter_text, encoding="utf-8")
            letter_pdf_path = output_dir / f"{prefix}_cover_letter.pdf"
            if not markdown_to_pdf(letter_text, letter_pdf_path):
                print("        (PDF generation failed for cover letter, .md saved)")
            print(f"        Cover letter -> {letter_md_path.name}, {letter_pdf_path.name}")
            if job_id:
                save_generated_output(
                    conn, job_id, profile_id, "cover_letter", str(letter_md_path), letter_text
                )
        except Exception as e:
            print(f"        Cover letter ERROR: {e}")

    # 6. Summary
    stats = get_stats(conn)
    conn.close()
    print("\n[6/6] Done!")
    print(f"\nOutput saved to: {output_dir}")
    print(
        f"\nDatabase: {stats['total_jobs']} total jobs cached, "
        f"{stats['new_jobs']} new, {stats['applied']} applied, "
        f"{stats['generated_outputs']} documents generated"
    )
    print("\nNext steps:")
    print("  - Review the generated CVs and cover letters (.md and .pdf)")
    print("  - Adjust config.yaml (tone, style) to match your preferences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
