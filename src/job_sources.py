"""Pluggable job source abstraction.

Supports Adzuna (API key required) and Arbeitnow (free, no key).
Sources are selected via config.yaml ``job_search.sources``.
"""

from __future__ import annotations

import html as html_mod
import logging
import re
from abc import ABC, abstractmethod

import requests

from .models import JobListing

logger = logging.getLogger(__name__)


class JobSource(ABC):
    """Abstract base for job board backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Source identifier (e.g. 'adzuna', 'arbeitnow')."""

    @abstractmethod
    def fetch(
        self,
        query: str,
        location: str = "",
        country: str = "gb",
        max_results: int = 50,
        distance_km: int = 0,
        contract_type: str = "",
        salary_min: int = 0,
    ) -> list[JobListing]:
        """Fetch jobs matching the query. Returns a list of JobListing."""


class AdzunaSource(JobSource):
    """Adzuna job board (requires ADZUNA_APP_ID + ADZUNA_APP_KEY)."""

    @property
    def name(self) -> str:
        return "adzuna"

    def fetch(
        self,
        query: str,
        location: str = "",
        country: str = "gb",
        max_results: int = 50,
        distance_km: int = 0,
        contract_type: str = "",
        salary_min: int = 0,
    ) -> list[JobListing]:
        from .job_fetcher import fetch_adzuna_jobs

        return fetch_adzuna_jobs(
            search_query=query,
            location=location,
            country=country,
            results_per_page=max_results,
            distance_km=distance_km,
            contract_type=contract_type,
            salary_min=salary_min,
        )


class ArbeitnowSource(JobSource):
    """Arbeitnow free job board (Europe + remote focus, no API key)."""

    _BASE_URL = "https://www.arbeitnow.com/api/job-board-api"

    @property
    def name(self) -> str:
        return "arbeitnow"

    def fetch(
        self,
        query: str,
        location: str = "",
        country: str = "gb",
        max_results: int = 50,
        distance_km: int = 0,
        contract_type: str = "",
        salary_min: int = 0,
    ) -> list[JobListing]:
        jobs: list[JobListing] = []
        seen_slugs: set[str] = set()
        query_lower = query.lower().split()
        max_pages = min(max_results // 20 + 1, 5)

        for page in range(1, max_pages + 1):
            try:
                resp = requests.get(
                    self._BASE_URL,
                    params={"page": page},
                    timeout=15,
                    headers={"Accept": "application/json"},
                )
                if resp.status_code != 200:
                    break
                data = resp.json().get("data", [])
                if not data:
                    break
            except Exception as exc:
                logger.warning("Arbeitnow page %d failed: %s", page, exc)
                break

            for item in data:
                slug = item.get("slug", "")
                if slug in seen_slugs:
                    continue
                seen_slugs.add(slug)

                title = item.get("title", "")
                description_html = item.get("description", "")
                description = self._strip_html(description_html)

                text = f"{title} {description}".lower()
                if not any(w in text for w in query_lower):
                    continue

                loc = item.get("location", "") or ""
                if location and location.lower() not in loc.lower():
                    continue

                url = item.get("url", "")
                if not url and slug:
                    url = f"https://www.arbeitnow.com/view/{slug}"

                jobs.append(
                    JobListing(
                        title=title,
                        company=item.get("company_name", "Unknown"),
                        description=description[:3000],
                        location=loc,
                        url=url,
                        salary=None,
                        contract_type=None,
                        source="arbeitnow",
                    )
                )
                if len(jobs) >= max_results:
                    return jobs

        return jobs

    @staticmethod
    def _strip_html(html: str) -> str:
        text = re.sub(r"<[^>]+>", " ", html)
        text = html_mod.unescape(text)
        return re.sub(r"\s+", " ", text).strip()


_SOURCE_REGISTRY: dict[str, type[JobSource]] = {
    "adzuna": AdzunaSource,
    "arbeitnow": ArbeitnowSource,
}


def create_sources(source_names: list[str] | None = None) -> list[JobSource]:
    """Instantiate job sources from config names. Defaults to Adzuna only."""
    names = source_names or ["adzuna"]
    sources: list[JobSource] = []
    for name in names:
        cls = _SOURCE_REGISTRY.get(name.lower())
        if cls:
            sources.append(cls())
        else:
            logger.warning("Unknown job source '%s' — skipping", name)
    return sources or [AdzunaSource()]
