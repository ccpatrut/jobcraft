"""Fetch job listings from APIs (Adzuna primary)."""

import os
from typing import Optional

import requests

from .models import JobListing


def fetch_adzuna_jobs(
    search_query: str,
    location: str = "",
    country: str = "gb",
    results_per_page: int = 20,
    app_id: Optional[str] = None,
    app_key: Optional[str] = None,
) -> list[JobListing]:
    """
    Fetch jobs from Adzuna API.
    Get free API keys at: https://developer.adzuna.com/signup
    """
    app_id = app_id or os.environ.get("ADZUNA_APP_ID")
    app_key = app_key or os.environ.get("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        raise ValueError(
            "Adzuna API credentials required. Set ADZUNA_APP_ID and ADZUNA_APP_KEY "
            "in .env or pass explicitly. Get keys at https://developer.adzuna.com/signup"
        )

    base_url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": search_query or "developer",  # fallback
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }
    if location:
        params["where"] = location

    resp = requests.get(base_url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    jobs = []
    for item in data.get("results", []):
        jobs.append(
            JobListing(
                title=item.get("title", "Unknown"),
                company=item.get("company", {}).get("display_name", "Unknown"),
                description=item.get("description", "") or "",
                location=item.get("location", {}).get("display_name", "") or "",
                url=item.get("redirect_url", ""),
                salary=item.get("salary_min")
                and item.get("salary_max")
                and f"{item['salary_min']}-{item['salary_max']}"
                or None,
                contract_type=item.get("contract_type"),
                source="adzuna",
            )
        )
    return jobs
