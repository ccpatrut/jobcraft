#!/usr/bin/env python3
"""
Job Finder - AI-powered job matching and tailored CV/cover letter generation.

1. Drop your CVs (PDF, Word) in the cv_input/ directory
2. Set ADZUNA_APP_ID and ADZUNA_APP_KEY in .env (free at developer.adzuna.com)
3. Run: python main.py

Ensure Ollama is running locally (ollama serve) and you have pulled a model:
  ollama pull llama3.2
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_preferences, load_config
from src.cover_letter_generator import generate_cover_letter
from src.cv_generator import generate_tailored_cv
from src.document_loader import load_documents_from_dir
from src.job_fetcher import fetch_adzuna_jobs
from src.job_matcher import rank_jobs_with_ollama
from src.profile_extractor import extract_profile_with_ollama


def main():
    print("=" * 60)
    print("Job Finder - AI-Powered Job Matching & CV Generation")
    print("=" * 60)

    cfg = load_config()
    prefs = get_preferences(cfg)
    cv_dir = Path(cfg["cv_input_dir"]).expanduser().resolve()
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    ai_cfg = cfg.get("ai", {})
    model = ai_cfg.get("model", "llama3.2")
    ollama_host = ai_cfg.get("host") or os.environ.get("OLLAMA_HOST") or None
    job_cfg = cfg.get("job_search", {})
    country = job_cfg.get("country", "gb")
    max_results = job_cfg.get("max_results", 50)
    keywords = job_cfg.get("keywords", "").strip()

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
    print("  Loaded successfully.")

    # 2. Extract profile
    print("\n[2/6] Extracting profile with Ollama (this may take a minute)...")
    try:
        profile = extract_profile_with_ollama(combined_text, model=model, host=ollama_host)
    except Exception as e:
        print("  ERROR: Could not connect to Ollama. Is it running? Try: ollama serve")
        print("  Also ensure you have pulled a model: ollama pull", model)
        print("  Details:", e)
        return 1
    print("  Profile extracted:", profile.name or "Unknown", "-", len(profile.skills), "skills")

    # 3. Build search query and fetch jobs
    search_query = keywords or " ".join(profile.skills[:5]) if profile.skills else "developer"
    print("\n[3/6] Fetching jobs from Adzuna (query:", search_query[:50] + "...)")
    try:
        jobs = fetch_adzuna_jobs(
            search_query=search_query,
            country=country,
            results_per_page=max_results,
        )
    except ValueError as e:
        print("  ERROR:", e)
        return 1
    except Exception as e:
        print("  ERROR fetching jobs:", e)
        return 1
    print("  Fetched", len(jobs), "jobs")

    if not jobs:
        print("  No jobs found. Try adjusting keywords in config.yaml.")
        return 1

    # 4. Rank jobs
    print("\n[4/6] Ranking jobs by fit with Ollama...")
    try:
        top_jobs = rank_jobs_with_ollama(profile, jobs, model=model, top_n=5, host=ollama_host)
    except Exception as e:
        print("  ERROR ranking jobs:", e)
        top_jobs = jobs[:5]  # Fallback to first 5
    print("  Selected top", len(top_jobs), "matches")

    # 5. Generate CVs and cover letters
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filenames
    def safe_name(s: str) -> str:
        return "".join(c if c.isalnum() or c in " -_" else "_" for c in s)[:60]

    print("\n[5/6] Generating tailored CVs and cover letters...")
    for i, job in enumerate(top_jobs, 1):
        company_safe = safe_name(job.company)
        title_safe = safe_name(job.title)
        prefix = f"{i}_{company_safe}_{title_safe}"

        print(f"  [{i}/5] {job.title} @ {job.company}")

        try:
            cv_text = generate_tailored_cv(profile, job, prefs, model=model, host=ollama_host)
            cv_path = output_dir / f"{prefix}_cv.md"
            cv_path.write_text(cv_text, encoding="utf-8")
            print(f"        CV -> {cv_path.name}")
        except Exception as e:
            print(f"        CV ERROR: {e}")

        try:
            letter_text = generate_cover_letter(profile, job, prefs, model=model, host=ollama_host)
            letter_path = output_dir / f"{prefix}_cover_letter.md"
            letter_path.write_text(letter_text, encoding="utf-8")
            print(f"        Cover letter -> {letter_path.name}")
        except Exception as e:
            print(f"        Cover letter ERROR: {e}")

    print("\n[6/6] Done!")
    print(f"\nOutput saved to: {output_dir}")
    print("\nNext steps:")
    print("  - Review the generated CVs and cover letters")
    print("  - Copy into Word or use a markdown converter")
    print("  - Adjust config.yaml (tone, style) to match your preferences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
