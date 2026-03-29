"""Fetch job listings from APIs (Adzuna primary)."""

import logging
import os
import time
from typing import Optional

import requests

from .models import JobListing

log = logging.getLogger(__name__)

_MAX_RETRIES = 4
_BACKOFF_BASE = 1.5  # seconds; doubles each retry: 1.5, 3, 6, 12


def fetch_adzuna_jobs(
    search_query: str,
    location: str = "",
    country: str = "gb",
    results_per_page: int = 20,
    distance_km: int = 0,
    contract_type: str = "",
    app_id: Optional[str] = None,
    app_key: Optional[str] = None,
) -> list[JobListing]:
    """
    Fetch jobs from Adzuna API with automatic retry on 429 rate-limit errors.

    Args:
        distance_km: Radius in km around `location`. Only used when location is set.
                     0 means Adzuna default (exact area match).
        contract_type: "permanent", "contract", or "" (any).
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
        "what": search_query or "jobs",
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }
    if location:
        params["where"] = location
        if distance_km > 0:
            params["distance"] = distance_km
    if contract_type:
        params["contract_type"] = contract_type

    data = _request_with_retry(base_url, params)

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


def enrich_job_descriptions(
    jobs: list[JobListing],
    max_workers: int = 4,
    throttle: float = 0.35,
) -> int:
    """Fetch full descriptions from Adzuna detail pages (in-place update).

    The Adzuna search API truncates descriptions to ~500 chars.  The detail
    pages contain the complete text including requirements sections.

    Returns the number of jobs successfully enriched.
    """
    import html as html_mod
    import re as re_mod
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _lock = threading.Lock()
    _last_t = [0.0]

    def _throttled_get(url: str) -> requests.Response | None:
        with _lock:
            elapsed = time.time() - _last_t[0]
            if elapsed < throttle:
                time.sleep(throttle - elapsed)
            _last_t[0] = time.time()
        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (compatible; JobCraft/1.0)"},
            )
            return resp if resp.status_code == 200 else None
        except requests.RequestException:
            return None

    def _extract(resp_text: str) -> str | None:
        m = re_mod.search(
            r'class="[^"]*adp-body[^"]*"[^>]*>([\s\S]*?)</section',
            resp_text,
            re_mod.DOTALL | re_mod.IGNORECASE,
        )
        if not m:
            return None
        txt = re_mod.sub(r"<[^>]+>", " ", m.group(1))
        txt = html_mod.unescape(txt)
        txt = re_mod.sub(r"\s+", " ", txt).strip()
        return txt if len(txt) > 600 else None

    enriched = 0

    def _enrich_one(job: JobListing) -> bool:
        detail_url = job.url.split("?")[0]
        if "/details/" not in detail_url and "/land/ad/" not in detail_url:
            return False
        resp = _throttled_get(detail_url)
        if resp is None:
            return False
        full = _extract(resp.text)
        if full and len(full) > len(job.description) * 2:
            job.description = full
            return True
        return False

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_enrich_one, j): j for j in jobs}
        for fut in as_completed(futs):
            if fut.result():
                enriched += 1

    return enriched


def _request_with_retry(url: str, params: dict) -> dict:
    """GET with exponential backoff on 429 / 5xx responses."""
    for attempt in range(_MAX_RETRIES + 1):
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == _MAX_RETRIES:
                resp.raise_for_status()
            wait = _BACKOFF_BASE * (2**attempt)
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = max(wait, float(retry_after))
                except ValueError:
                    pass
            log.debug("429/5xx on attempt %d, retrying in %.1fs", attempt + 1, wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    return {}  # unreachable, satisfies type checker
