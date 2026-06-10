"""Filter job listings to a configured geographic area."""

from __future__ import annotations

import re

from .models import JobListing

# Major Swiss cities/regions far from Basel (~200 km) — drop when anchoring on Basel area.
_FAR_FROM_BASEL_RE = re.compile(
    r"(?i)"
    r"\b(?:genf|geneva|genève|carouge|vernier|meyrin|"
    r"grand-saconnex|plan-les-ouates|kanton\s+genf|"
    r"zürich|zurich|winterthur|lausanne|lugano|locarno|"
    r"st\.?\s*gallen|sankt\s+gallen|luzern|lucerne|"
    r"neuchâtel|neuenburg|fribourg|freiburg|sion|sitten)\b"
    r"|"
    r"(?:^|[,\s])(?:bern|berne)(?:[,\s]|$)",
)

# Communes commonly within ~25 km of Basel (Basel-Stadt/Land agglomeration).
_BASEL_AREA_RE = re.compile(
    r"(?i)\b(?:"
    r"birsfelden|allschwil|muttenz|liestal|pratteln|reinach|"
    r"münchenstein|munchenstein|riehen|binningen|arlesheim|"
    r"bottmingen|oberwil|therwil|dornach|aesch|münchenstein|"
    r"füllinsdorf|fullinsdorf|lausen|sissach|gelterkinden|"
    r"rheinfelden|grenzach|hegenheim|saint-louis|"
    r"liestal|liestal|augst|ettingen|"
    r"basel-landschaft|basel-stadt"
    r")\b",
)

_VAGUE_LOCATION_RE = re.compile(r"(?i)^(?:schweiz|switzerland|suisse|switzerland\s*\(.*\)?)\s*$")


def _anchors_include_basel(anchors: list[str]) -> bool:
    return any("basel" in a.lower() for a in anchors)


def job_in_search_area(job: JobListing, anchors: list[str]) -> bool:
    """Return True if the job location matches configured anchor city/region."""
    loc = (job.location or "").strip()
    if not loc:
        return False

    if _FAR_FROM_BASEL_RE.search(loc) and _anchors_include_basel(anchors):
        return False

    loc_lower = loc.lower()
    for anchor in anchors:
        if anchor.lower() in loc_lower:
            return True

    if _anchors_include_basel(anchors) and _BASEL_AREA_RE.search(loc):
        return True

    return False


def filter_jobs_by_search_area(
    jobs: list[JobListing],
    anchors: list[str],
) -> tuple[list[JobListing], int]:
    """Keep only jobs in the configured search area. Returns (kept, removed_count)."""
    if not anchors:
        return jobs, 0

    kept: list[JobListing] = []
    removed = 0
    for job in jobs:
        loc = (job.location or "").strip()
        if _VAGUE_LOCATION_RE.match(loc):
            removed += 1
            continue
        if job_in_search_area(job, anchors):
            kept.append(job)
        else:
            removed += 1
    return kept, removed
