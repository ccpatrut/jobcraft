"""Six-step job matching pipeline with typed results.

Each step is a standalone function that returns a typed dataclass, making
the pipeline independently testable and reusable from CLI, API, or tests.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_preferences
from .cover_letter_generator import generate_cover_letter
from .cv_generator import generate_tailored_cv
from .database import (
    get_cached_profile,
    get_job_id,
    get_stats,
    save_generated_output,
    save_jobs,
    save_profile,
    search_cached_jobs,
)
from .document_loader import load_documents_from_dir
from .job_matcher import (
    _TITLE_LANG_SIGNALS,
    ai_validate_and_filter,
    filter_english_only_jobs,
    filter_jobs_by_language,
    rank_jobs_with_embeddings,
    rank_jobs_with_llm,
)
from .lang_detect import (
    detect_language,
    detect_languages_batch,
    filter_jobs_by_description_language,
    filter_jobs_by_detected_language,
)
from .llm_provider import LLMProvider, create_provider
from .models import JobListing, UserPreferences, UserProfile
from .pdf_utils import markdown_to_pdf
from .profile_extractor import extract_profile
from .query_builder import build_search_queries
from .query_translator import (
    get_country_languages_display,
    get_country_name,
    get_localized_queries,
)
from .language_config import apply_language_config
from .location_filter import filter_jobs_by_search_area
from .profile_matching import build_matching_profile_text
from .spinner import Spinner

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────


ADZUNA_EU_COUNTRIES = [
    "gb",
    "de",
    "fr",
    "at",
    "ch",
    "it",
    "nl",
    "pl",
    "be",
    "es",
]


@dataclass
class PipelineConfig:
    """All configuration needed by the pipeline, extracted from config.yaml."""

    cv_dir: Path
    output_dir: Path
    provider: LLMProvider
    countries: list[str]
    max_results: int
    keywords: str
    language_strategy: str
    exclude_languages: list[str]
    locations: list[str]
    radius_km: int
    search_nationwide: bool
    use_embeddings: bool
    rerank_with_llm: bool
    ai_validation_enabled: bool
    ai_validation_rounds: int
    default_languages: list[str]
    config_languages: list[str]
    languages_mode: str
    search_profile_summary: str
    description_language_min_tier: int
    min_similarity: float
    target_industries: list[str]
    prefs: UserPreferences
    temperature: float = 0.7
    contract_type: str = ""
    salary_min: int = 0
    remote_only: bool = False
    fresh: bool = False
    cached: bool = False
    output_formats: list[str] = field(default_factory=lambda: ["md", "pdf"])
    source_names: list[str] = field(default_factory=lambda: ["adzuna"])
    parallel_workers: int = 2

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        fresh: bool = False,
        cached: bool = False,
    ) -> PipelineConfig:
        """Build PipelineConfig from the loaded YAML config dict."""
        ai_cfg = cfg.get("ai", {})
        job_cfg = cfg.get("job_search", {})
        output_cfg = cfg.get("output", {})

        mode = job_cfg.get("mode", "default")

        if mode == "remote_europe":
            countries = job_cfg.get("countries", None) or list(ADZUNA_EU_COUNTRIES)
            remote_only = True
            language_strategy = "english_only"
            locations: list[str] = []
        else:
            countries = [job_cfg.get("country", "gb")]
            remote_only = False
            language_strategy = job_cfg.get("language", "auto")
            locations = job_cfg.get("locations", []) or []

        provider = create_provider(ai_cfg)
        prefs = get_preferences(cfg)
        pref_cfg = cfg.get("preferences", {})
        job_min_sim = job_cfg.get("min_similarity")
        if job_min_sim is None and prefs.pivot_enabled:
            min_similarity = 0.25
        elif job_min_sim is not None:
            min_similarity = float(job_min_sim)
        else:
            min_similarity = 0.35

        return cls(
            cv_dir=Path(cfg["cv_input_dir"]).expanduser().resolve(),
            output_dir=Path(cfg["output_dir"]).expanduser().resolve(),
            provider=provider,
            countries=countries,
            max_results=job_cfg.get("max_results", 50),
            keywords=job_cfg.get("keywords", "").strip(),
            language_strategy=language_strategy,
            exclude_languages=job_cfg.get("exclude_languages", []) or [],
            locations=locations,
            radius_km=int(job_cfg.get("radius_km", 0)),
            search_nationwide=bool(job_cfg.get("search_nationwide", False)),
            use_embeddings=ai_cfg.get("use_embeddings", True),
            rerank_with_llm=ai_cfg.get("rerank_with_llm", True),
            ai_validation_enabled=job_cfg.get("ai_language_validation", False),
            ai_validation_rounds=job_cfg.get("ai_validation_rounds", 3),
            default_languages=pref_cfg.get("default_languages", []) or [],
            config_languages=pref_cfg.get("languages", []) or [],
            languages_mode=str(pref_cfg.get("languages_mode", "override")),
            search_profile_summary=str(pref_cfg.get("search_profile_summary", "") or "").strip(),
            description_language_min_tier=int(
                job_cfg.get("description_language_min_tier", 4),
            ),
            min_similarity=min_similarity,
            target_industries=job_cfg.get("industries", []) or [],
            prefs=prefs,
            temperature=float(ai_cfg.get("temperature", 0.7)),
            contract_type=job_cfg.get("contract_type", "").strip(),
            salary_min=int(job_cfg.get("salary_min", 0)),
            remote_only=remote_only,
            fresh=fresh,
            cached=cached,
            output_formats=output_cfg.get("formats", ["md", "pdf"]) or ["md", "pdf"],
            source_names=job_cfg.get("sources", ["adzuna"]) or ["adzuna"],
        )


# ── Typed result objects ─────────────────────────────────────────────


@dataclass
class LoadResult:
    combined_text: str
    char_count: int
    is_short: bool


@dataclass
class ProfileResult:
    profile: UserProfile
    profile_id: int
    from_cache: bool


@dataclass
class FetchResult:
    jobs: list[JobListing]
    base_queries: list[str]
    all_queries: list[str]
    new_saved: int
    total_fetched: int


@dataclass
class RankResult:
    top_jobs: list[JobListing]
    method: str


@dataclass
class GeneratedDoc:
    job: JobListing
    cv_md_path: Path | None = None
    cv_pdf_path: Path | None = None
    cv_docx_path: Path | None = None
    cv_ok: bool = False
    letter_md_path: Path | None = None
    letter_pdf_path: Path | None = None
    letter_docx_path: Path | None = None
    letter_ok: bool = False
    cv_error: str | None = None
    letter_error: str | None = None


@dataclass
class GenerateResult:
    outputs: list[GeneratedDoc] = field(default_factory=list)


@dataclass
class SummaryResult:
    total_jobs: int
    new_jobs: int
    applied: int
    generated_outputs: int


# ── Step 1: Load documents ──────────────────────────────────────────


def step_load_documents(cfg: PipelineConfig) -> LoadResult:
    """Load CV documents from the input directory.

    Raises FileNotFoundError when no documents are found.
    """
    print("\n[1/6] Loading CV documents from:", cfg.cv_dir)
    _, combined_text = load_documents_from_dir(cfg.cv_dir)

    if (
        combined_text.startswith("[")
        and "not found" in combined_text
        or "No documents" in combined_text
    ):
        raise FileNotFoundError(f"No documents found. Add PDF or Word files to: {cfg.cv_dir}")

    result = LoadResult(
        combined_text=combined_text,
        char_count=len(combined_text),
        is_short=len(combined_text) < 200,
    )
    print("  Loaded successfully. (CV text length:", result.char_count, "chars)")
    if result.is_short:
        print(
            "  WARNING: CV text is very short. If using a scanned PDF, "
            "try a text-based PDF or add keywords in config.yaml."
        )
    return result


# ── Step 2: Extract profile ─────────────────────────────────────────


def step_extract_profile(
    conn,
    combined_text: str,
    cfg: PipelineConfig,
) -> ProfileResult:
    """Extract or load cached candidate profile.

    Raises ConnectionError if the AI provider is unreachable.
    """
    print("\n[2/6] Extracting profile...")
    profile = None
    from_cache = False

    if cfg.fresh:
        print("  --fresh flag set: skipping cache, re-extracting from CV.")
    else:
        profile = get_cached_profile(conn, combined_text)
        if profile:
            print("  Using cached profile (CV text unchanged).")
            print("  Tip: run with --fresh to force re-extraction.")
            from_cache = True

    if not profile:
        try:
            with Spinner(f"Extracting profile with {cfg.provider.name}"):
                profile = extract_profile(
                    combined_text,
                    provider=cfg.provider,
                )
        except Exception as e:
            provider = cfg.provider
            if provider.name == "ollama":
                raise ConnectionError(
                    f"Could not connect to Ollama. Is it running? Try: ollama serve\n"
                    f"Also ensure you have pulled a model: ollama pull {provider.model}\n"
                    f"Details: {e}"
                ) from e
            raise ConnectionError(f"AI provider '{provider.name}' failed. Details: {e}") from e

    if cfg.config_languages:
        apply_language_config(profile, cfg.config_languages, cfg.languages_mode)
        print(f"  Languages ({cfg.languages_mode}): {', '.join(profile.languages)}")
    elif not profile.languages and cfg.default_languages:
        profile.languages = cfg.default_languages
        print("  Using default languages from config.yaml")

    if cfg.target_industries:
        profile.industries = cfg.target_industries
        print(f"  Industries (from config): {', '.join(cfg.target_industries)}")
    elif profile.industries:
        print(f"  Industries (auto-detected): {', '.join(profile.industries)}")

    profile_id = save_profile(conn, combined_text, profile)
    return ProfileResult(profile=profile, profile_id=profile_id, from_cache=from_cache)


# ── Step 3: Fetch and filter jobs ───────────────────────────────────


def _build_candidate_langs(profile: UserProfile) -> dict[str, int]:
    """Build a language-code -> proficiency-tier mapping from profile languages."""
    candidate_langs: dict[str, int] = {}
    if not profile.languages:
        return candidate_langs
    for entry in profile.languages:
        parts = entry.split(" - ", 1)
        name = parts[0].strip().lower()
        level = parts[1].strip().lower() if len(parts) > 1 else "proficient"
        code = {
            "english": "en",
            "german": "de",
            "french": "fr",
            "italian": "it",
            "spanish": "es",
        }.get(name, name[:2])
        tier = {
            "native": 6,
            "fluent": 5,
            "proficient": 4,
            "advanced": 4,
            "c2": 6,
            "c1": 5,
            "b2": 4,
            "b1": 3,
            "a2": 2,
            "a1": 1,
            "intermediate": 3,
            "elementary": 2,
            "beginner": 1,
        }.get(level, 3)
        candidate_langs[code] = max(candidate_langs.get(code, 0), tier)
    return candidate_langs


def _filter_english_only(
    jobs: list[JobListing],
    candidate_langs: dict[str, int],
) -> tuple[list[JobListing], dict[str, int]]:
    """Apply english-only language filtering using batched HF classifier + title signals."""
    print(f"  Classifying {len(jobs)} jobs with HF language model (batched)...")
    texts = [f"{j.title} {j.description or ''}" for j in jobs]
    langs = detect_languages_batch(texts)

    kept: list[JobListing] = []
    removed_count = 0
    for j, lang in zip(jobs, langs):
        if lang == "en":
            kept.append(j)
        else:
            removed_count += 1
            print(f"    \u2717 [{lang}] {j.title} @ {j.company}")
    jobs = kept
    print(f"  HF model kept {len(jobs)} English postings, removed {removed_count} non-English")

    jobs, title_removed = filter_english_only_jobs(jobs, candidate_langs)
    if title_removed:
        print(f"  Title-language filter removed {title_removed} more ({len(jobs)} remaining)")

    return jobs, candidate_langs


def _filter_multilingual(
    jobs: list[JobListing],
    profile: UserProfile,
    cfg: PipelineConfig,
) -> list[JobListing]:
    """Apply multilingual language filtering (exclude list + proficiency check)."""
    if cfg.exclude_languages:
        with Spinner(f"Detecting languages (excluding: {', '.join(cfg.exclude_languages)})"):
            jobs, filtered = filter_jobs_by_detected_language(jobs, cfg.exclude_languages)
        if filtered:
            print(f"  Filtered out {filtered} jobs in excluded languages")

    if profile.languages:
        with Spinner("Checking posting languages against your proficiency"):
            jobs, desc_lang_removed = filter_jobs_by_description_language(
                jobs,
                profile.languages,
                min_tier=cfg.description_language_min_tier,
            )
        if desc_lang_removed:
            print(f"  Removed {desc_lang_removed} jobs in languages beyond your proficiency")
            print(f"  {len(jobs)} remaining")

    return jobs


_REMOTE_POSITIVE = re.compile(
    r"\b(?:fully\s+remote|100\s*%\s*remote|remote\s*(?:work|position|role|job|opportunity)"
    r"|work\s*from\s*(?:home|anywhere)|remote\s*(?:first|friendly)|remote)\b",
    re.IGNORECASE,
)
_REMOTE_NEGATIVE_TITLE = re.compile(
    r"\bhybrid\b|\bon[- ]?site\s+only\b",
    re.IGNORECASE,
)


def _filter_remote_only(jobs: list[JobListing]) -> list[JobListing]:
    """Keep only jobs with an explicit remote indicator; exclude hybrid-only titles."""
    kept: list[JobListing] = []
    removed = 0
    for job in jobs:
        text = f"{job.title} {job.description or ''}"
        has_remote = bool(_REMOTE_POSITIVE.search(text))
        hybrid_title = bool(_REMOTE_NEGATIVE_TITLE.search(job.title or ""))
        if has_remote and not hybrid_title:
            kept.append(job)
        else:
            removed += 1
    print(f"  Remote filter: kept {len(kept)}, removed {removed} non-remote jobs")
    return kept


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _deduplicate_jobs(
    jobs: list[JobListing],
) -> tuple[list[JobListing], int]:
    """Remove near-duplicate listings (same title + company, different URLs)."""
    seen: set[str] = set()
    unique: list[JobListing] = []
    for job in jobs:
        key = (
            _NORMALIZE_RE.sub("", (job.title or "").lower())
            + "|"
            + _NORMALIZE_RE.sub("", (job.company or "").lower())
        )
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique, len(jobs) - len(unique)


def _apply_location_filter(
    jobs: list[JobListing],
    cfg: PipelineConfig,
) -> list[JobListing]:
    """Drop jobs outside configured locations (e.g. Geneva when searching Basel)."""
    if not cfg.locations:
        return jobs
    filtered, removed = filter_jobs_by_search_area(jobs, cfg.locations)
    if removed:
        print(
            f"  Location filter ({', '.join(cfg.locations)}): "
            f"kept {len(filtered)}, removed {removed} outside area"
        )
    return filtered


def _cache_fallback(
    conn,
    jobs: list[JobListing],
    base_queries: list[str],
    cfg: PipelineConfig,
    candidate_langs: dict[str, int],
) -> list[JobListing]:
    """Fall back to cached jobs when too few API results."""
    if jobs:
        print(f"  Only {len(jobs)} jobs \u2014 checking database for more...")
    else:
        print("  No jobs remaining \u2014 searching database cache...")

    seen_urls = {j.url for j in jobs}
    added = 0

    for country in cfg.countries:
        cached = search_cached_jobs(
            conn,
            list(base_queries),
            country=country,
            limit=cfg.max_results,
        )
        for cj in cached:
            if cj.url not in seen_urls:
                if cfg.language_strategy == "english_only":
                    lang = detect_language(f"{cj.title} {cj.description or ''}")
                    if lang != "en":
                        print(f"    \u2717 cached [{lang}] {cj.title} @ {cj.company}")
                        continue
                    flagged = False
                    for pat, lc in _TITLE_LANG_SIGNALS:
                        if pat.search(cj.title or "") and candidate_langs.get(lc, 0) < 4:
                            print(f"    \u2717 cached [title:{lc}] {cj.title} @ {cj.company}")
                            flagged = True
                            break
                    if flagged:
                        continue
                jobs.append(cj)
                seen_urls.add(cj.url)
                added += 1

    if added:
        print(f"  Found {added} additional jobs from database cache (total: {len(jobs)})")
    else:
        print("  No additional matches found in database cache.")
    return _apply_location_filter(jobs, cfg)


def _ai_validation_pass(
    jobs: list[JobListing],
    profile: UserProfile,
    cfg: PipelineConfig,
) -> list[JobListing]:
    """Run AI-powered language validation on remaining jobs."""
    print("\n  AI language validation enabled \u2014 verifying remaining jobs...")
    jobs, ai_removed, learned = ai_validate_and_filter(
        jobs,
        profile,
        provider=cfg.provider,
        max_rounds=cfg.ai_validation_rounds,
    )
    if ai_removed:
        print(f"  AI validation removed {ai_removed} additional jobs")
        if learned:
            print(f"  Learned {len(learned)} new pattern(s):")
            for phrase in learned:
                print(f'    - "{phrase}"')
        print(f"  {len(jobs)} jobs remaining after AI validation")
    return jobs


def _fetch_from_sources(
    cfg: PipelineConfig,
    all_queries: list[str],
) -> tuple[list[JobListing], dict[str, list[JobListing]]]:
    """Fetch jobs from all configured sources concurrently."""
    from .job_sources import create_sources

    sources = create_sources(cfg.source_names)
    if cfg.locations:
        search_locations = list(cfg.locations)
        if cfg.search_nationwide:
            search_locations.append("")
    else:
        search_locations = [""]

    tasks: list[tuple[str, str, str, str]] = []
    for src in sources:
        for country in cfg.countries:
            for loc in search_locations:
                for q in all_queries:
                    tasks.append((src.name, country, loc, q))

    n_workers = min(len(tasks), 4)
    req_interval = 0.35
    _throttle_lock = threading.Lock()
    _last_request_time = [0.0]

    print(
        f"  Dispatching {len(tasks)} API calls across {len(sources)} source(s) "
        f"({n_workers} workers)..."
    )

    jobs: list[JobListing] = []
    seen_urls: set[str] = set()
    per_country_jobs: dict[str, list[JobListing]] = {}

    source_map = {src.name: src for src in sources}

    def _fetch_one(
        source_name: str,
        country: str,
        loc: str,
        query: str,
    ) -> tuple[str, str, str, str, list[JobListing] | Exception]:
        with _throttle_lock:
            elapsed = time.monotonic() - _last_request_time[0]
            if elapsed < req_interval:
                time.sleep(req_interval - elapsed)
            _last_request_time[0] = time.monotonic()

        c_name = get_country_name(country)
        label = f"{source_name}:{loc or c_name}"
        try:
            src = source_map[source_name]
            batch = src.fetch(
                query=query,
                location=loc,
                country=country,
                max_results=cfg.max_results,
                distance_km=cfg.radius_km if loc else 0,
                contract_type=cfg.contract_type,
                salary_min=cfg.salary_min,
            )
            return source_name, country, label, query, batch
        except ValueError:
            raise
        except Exception as exc:
            return source_name, country, label, query, exc

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_fetch_one, sn, c, loc, q): (sn, c, loc, q) for sn, c, loc, q in tasks
        }
        for fut in as_completed(futures):
            source_name, country, label, query, result = fut.result()
            if isinstance(result, Exception):
                print(f"  [{label} | {query[:30]}] -> ERROR: {result}")
                continue
            new_in_batch = 0
            for job in result:
                if job.url not in seen_urls:
                    seen_urls.add(job.url)
                    jobs.append(job)
                    per_country_jobs.setdefault(country, []).append(job)
                    new_in_batch += 1
            if new_in_batch:
                print(f"  [{label} | {query[:30]}] -> {new_in_batch} new (total: {len(jobs)})")

    return jobs, per_country_jobs


def step_fetch_and_filter_jobs(
    conn,
    profile: UserProfile,
    cfg: PipelineConfig,
) -> FetchResult:
    """Fetch jobs from API, apply language filters, save to DB, and cache-fallback.

    When cfg.cached is True, skips the API fetch and works from the DB cache.

    Raises ValueError for API credential issues and RuntimeError when no jobs survive
    filtering.
    """
    primary_country = cfg.countries[0]
    base_queries = build_search_queries(
        cfg.keywords,
        profile,
        country="Europe" if cfg.remote_only else primary_country,
        provider=cfg.provider,
        prefs=cfg.prefs,
    )

    # ── Header ──
    if cfg.cached:
        print("\n[3/6] Loading jobs from database cache (--cached mode)")
    elif cfg.remote_only:
        country_names = [get_country_name(c) for c in cfg.countries]
        print(
            f"\n[3/6] Fetching REMOTE jobs from Adzuna API across {len(cfg.countries)} EU countries"
        )
        print(f"  Countries: {', '.join(country_names)}")
    else:
        lang_info = get_country_languages_display(primary_country)
        country_name = get_country_name(primary_country)
        print(f"\n[3/6] Fetching jobs from Adzuna API (live) for {country_name} ({lang_info})")

    print(f"  Search queries ({len(base_queries)}):")
    for bq in base_queries:
        print(f"    \u2022 {bq}")
    if cfg.locations and not cfg.cached:
        print(f"  Locations: {', '.join(cfg.locations)} ({cfg.radius_km} km radius)")
    print(f"  Language strategy: {cfg.language_strategy}")
    if cfg.contract_type:
        print(f"  Contract type filter: {cfg.contract_type}")
    if cfg.remote_only:
        print("  Remote filter: STRICT (fully remote only, no hybrid)")
    if cfg.exclude_languages:
        print(f"  Excluding languages: {', '.join(cfg.exclude_languages)}")

    # ── Cached mode: skip API, load from DB ──
    if cfg.cached:
        jobs: list[JobListing] = []
        seen_urls: set[str] = set()
        for country in cfg.countries:
            cached = search_cached_jobs(
                conn,
                list(base_queries),
                country=country,
                limit=cfg.max_results,
            )
            for cj in cached:
                if cj.url not in seen_urls:
                    seen_urls.add(cj.url)
                    jobs.append(cj)
        print(f"  Loaded {len(jobs)} jobs from cache")
        if not jobs:
            raise RuntimeError(
                "No cached jobs found. Run without --cached first to populate the cache."
            )
        candidate_langs = _build_candidate_langs(profile)
        if cfg.language_strategy == "english_only":
            jobs, candidate_langs = _filter_english_only(jobs, candidate_langs)
        else:
            jobs = _filter_multilingual(jobs, profile, cfg)
        jobs = _apply_location_filter(jobs, cfg)
        jobs, n_dupes = _deduplicate_jobs(jobs)
        return FetchResult(
            jobs=jobs,
            base_queries=base_queries,
            all_queries=list(base_queries),
            new_saved=0,
            total_fetched=len(jobs),
        )

    # ── Expand queries with local-language translations ──
    all_queries: list[str] = []
    seen_queries: set[str] = set()

    if cfg.language_strategy == "english_only":
        all_queries = list(base_queries)
        seen_queries = {q.lower() for q in all_queries}
    else:
        with Spinner("Translating queries to local languages"):
            for bq in base_queries:
                expanded = get_localized_queries(
                    bq,
                    primary_country,
                    provider=cfg.provider,
                    language_strategy=cfg.language_strategy,
                )
                for eq in expanded:
                    if eq.lower() not in seen_queries:
                        seen_queries.add(eq.lower())
                        all_queries.append(eq)

    if len(all_queries) > len(base_queries):
        print(f"  Expanded to {len(all_queries)} queries (with translations)")

    # ── Fetch from sources (concurrent + rate-limited) ──
    jobs, per_country_jobs = _fetch_from_sources(cfg, all_queries)

    new_saved_total = 0
    for country, cjobs in per_country_jobs.items():
        n = save_jobs(conn, cjobs, search_query=base_queries[0], country=country)
        new_saved_total += n

    total_fetched = len(jobs)

    jobs = _apply_location_filter(jobs, cfg)

    # ── Remote filter ──
    if cfg.remote_only:
        jobs = _filter_remote_only(jobs)

    # ── Language filtering ──
    candidate_langs = _build_candidate_langs(profile)

    if cfg.language_strategy == "english_only":
        jobs, candidate_langs = _filter_english_only(jobs, candidate_langs)
    else:
        jobs = _filter_multilingual(jobs, profile, cfg)

    print(f"  Total: {len(jobs)} jobs after filtering ({new_saved_total} new saved to database)")

    if len(jobs) < 5:
        jobs = _cache_fallback(conn, jobs, base_queries, cfg, candidate_langs)

    if not jobs:
        raise RuntimeError(
            "No jobs found (API or cache). "
            "Try adjusting keywords or location/language in config.yaml."
        )

    if profile.languages:
        jobs, lang_removed = filter_jobs_by_language(jobs, profile)
        if lang_removed:
            print(f"  Regex filter removed {lang_removed} more jobs (explicit requirements)")
            print(f"  {len(jobs)} jobs remaining after regex filter")

    if not jobs:
        raise RuntimeError(
            "No jobs left after language filtering. Try broadening language settings."
        )

    if cfg.ai_validation_enabled and profile.languages:
        jobs = _ai_validation_pass(jobs, profile, cfg)

    if not jobs:
        raise RuntimeError("No jobs left after AI validation. Try broadening language settings.")

    jobs, n_dupes = _deduplicate_jobs(jobs)
    if n_dupes:
        print(f"  Removed {n_dupes} near-duplicate listings ({len(jobs)} unique jobs)")

    return FetchResult(
        jobs=jobs,
        base_queries=base_queries,
        all_queries=all_queries,
        new_saved=new_saved_total,
        total_fetched=total_fetched,
    )


# ── Step 4: Rank jobs ───────────────────────────────────────────────


def step_rank_jobs(
    profile: UserProfile,
    jobs: list[JobListing],
    cfg: PipelineConfig,
) -> RankResult:
    """Rank jobs by relevance to the candidate profile."""
    pivot_mode = cfg.prefs.pivot_enabled
    pivot_target_keywords = list(cfg.target_industries) if pivot_mode else []
    matching_text = build_matching_profile_text(
        profile,
        search_profile_summary=cfg.search_profile_summary,
        pivot_motivation=cfg.prefs.pivot_motivation,
        pivot_enabled=pivot_mode,
    )

    if cfg.use_embeddings:
        print("\n[4/6] Ranking jobs by semantic similarity (embeddings)...")
        try:
            top_jobs = rank_jobs_with_embeddings(
                profile,
                jobs,
                top_n=20,
                rerank_with_llm=cfg.rerank_with_llm,
                provider=cfg.provider,
                pivot_mode=pivot_mode,
                pivot_target_keywords=pivot_target_keywords or None,
                prefs=cfg.prefs,
                min_similarity=cfg.min_similarity,
                matching_text=matching_text,
            )
            return RankResult(top_jobs=top_jobs, method="embeddings")
        except Exception as e:
            print(f"  Embedding ranking failed ({e}), falling back to LLM...")

    print(f"\n[4/6] Ranking jobs by fit with {cfg.provider.name}...")
    try:
        top_jobs = rank_jobs_with_llm(
            profile,
            jobs,
            provider=cfg.provider,
            top_n=20,
            prefs=cfg.prefs,
            pivot_target_keywords=pivot_target_keywords or None,
            matching_text=matching_text,
        )
    except Exception as e:
        print("  ERROR ranking jobs:", e)
        top_jobs = jobs[:5]
    return RankResult(top_jobs=top_jobs, method=cfg.provider.name)


# ── Step 5: Generate outputs ────────────────────────────────────────


def _generate_for_job(
    job: JobListing,
    index: int,
    total: int,
    profile: UserProfile,
    cfg: PipelineConfig,
    profile_id: int,
    db_path: str,
    prefix: str,
) -> GeneratedDoc:
    """Generate CV + cover letter for a single job (used by ThreadPoolExecutor).

    Each call opens its own SQLite connection so it is safe to run from any thread.
    """
    from .database import get_connection
    from .docx_utils import markdown_to_docx

    conn = get_connection(db_path)
    try:
        doc = GeneratedDoc(job=job)
        job_id = get_job_id(conn, job.url)
        generate_docx = "docx" in cfg.output_formats
        generate_pdf = "pdf" in cfg.output_formats

        try:
            cv_text = generate_tailored_cv(
                profile,
                job,
                cfg.prefs,
                provider=cfg.provider,
                temperature=cfg.temperature,
            )
            cv_md_path = cfg.output_dir / f"{prefix}_cv.md"
            cv_md_path.write_text(cv_text, encoding="utf-8")
            doc.cv_md_path = cv_md_path
            doc.cv_ok = True

            if generate_pdf:
                cv_pdf_path = cfg.output_dir / f"{prefix}_cv.pdf"
                if markdown_to_pdf(cv_text, cv_pdf_path):
                    doc.cv_pdf_path = cv_pdf_path

            if generate_docx:
                cv_docx_path = cfg.output_dir / f"{prefix}_cv.docx"
                if markdown_to_docx(cv_text, cv_docx_path):
                    doc.cv_docx_path = cv_docx_path

            if job_id:
                save_generated_output(conn, job_id, profile_id, "cv", str(cv_md_path), cv_text)
        except Exception as e:
            doc.cv_error = str(e)

        try:
            letter_text = generate_cover_letter(
                profile,
                job,
                cfg.prefs,
                provider=cfg.provider,
                temperature=cfg.temperature,
            )
            letter_md_path = cfg.output_dir / f"{prefix}_cover_letter.md"
            letter_md_path.write_text(letter_text, encoding="utf-8")
            doc.letter_md_path = letter_md_path
            doc.letter_ok = True

            if generate_pdf:
                letter_pdf_path = cfg.output_dir / f"{prefix}_cover_letter.pdf"
                if markdown_to_pdf(letter_text, letter_pdf_path):
                    doc.letter_pdf_path = letter_pdf_path

            if generate_docx:
                letter_docx_path = cfg.output_dir / f"{prefix}_cover_letter.docx"
                if markdown_to_docx(letter_text, letter_docx_path):
                    doc.letter_docx_path = letter_docx_path

            if job_id:
                save_generated_output(
                    conn,
                    job_id,
                    profile_id,
                    "cover_letter",
                    str(letter_md_path),
                    letter_text,
                )
        except Exception as e:
            doc.letter_error = str(e)
    finally:
        conn.close()

    return doc


def step_generate_outputs(
    conn,
    profile_id: int,
    profile: UserProfile,
    selected_jobs: list[JobListing],
    cfg: PipelineConfig,
) -> GenerateResult:
    """Generate tailored CVs and cover letters for selected jobs.

    Uses ThreadPoolExecutor for parallel generation when more than 1 job is selected.
    Each worker thread gets its own SQLite connection to avoid cross-thread errors.
    """
    from .database import DEFAULT_DB_PATH

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = DEFAULT_DB_PATH

    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in " -_" else "_" for c in s)[:60]

    result = GenerateResult()
    total = len(selected_jobs)
    fmt_list = ", ".join(cfg.output_formats)
    print(f"\n[5/6] Generating tailored CVs and cover letters for {total} jobs...")
    print(f"  Output formats: {fmt_list}")

    tasks_info: list[tuple[int, JobListing, str]] = []
    for i, job in enumerate(selected_jobs, 1):
        company_safe = _safe(job.company)
        title_safe = _safe(job.title)
        prefix = f"{i}_{company_safe}_{title_safe}"
        tasks_info.append((i, job, prefix))

    workers = min(cfg.parallel_workers, total) if total > 1 else 1

    if workers > 1:
        print(f"  Generating in parallel ({workers} workers)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _generate_for_job,
                    job,
                    idx,
                    total,
                    profile,
                    cfg,
                    profile_id,
                    db_path,
                    prefix,
                ): (idx, job)
                for idx, job, prefix in tasks_info
            }
            for fut in as_completed(futures):
                idx, job = futures[fut]
                try:
                    doc = fut.result()
                except Exception as e:
                    doc = GeneratedDoc(job=job, cv_error=str(e), letter_error=str(e))
                result.outputs.append(doc)
                _print_doc_result(doc, idx, total)
    else:
        for idx, job, prefix in tasks_info:
            print(f"  [{idx}/{total}] {job.title} @ {job.company}")
            doc = _generate_for_job(
                job,
                idx,
                total,
                profile,
                cfg,
                profile_id,
                db_path,
                prefix,
            )
            result.outputs.append(doc)
            _print_doc_result(doc, idx, total)

    return result


def _print_doc_result(doc: GeneratedDoc, idx: int, total: int) -> None:
    """Print generation result for a single job."""
    label = f"  [{idx}/{total}] {doc.job.title} @ {doc.job.company}"
    if doc.cv_ok:
        paths = [p.name for p in [doc.cv_md_path, doc.cv_pdf_path, doc.cv_docx_path] if p]
        print(f"{label} CV -> {', '.join(paths)}")
    elif doc.cv_error:
        print(f"{label} CV ERROR: {doc.cv_error}")
    if doc.letter_ok:
        paths = [
            p.name for p in [doc.letter_md_path, doc.letter_pdf_path, doc.letter_docx_path] if p
        ]
        print(f"        Cover letter -> {', '.join(paths)}")
    elif doc.letter_error:
        print(f"        Cover letter ERROR: {doc.letter_error}")


# ── Step 6: Summary ─────────────────────────────────────────────────


def step_summary(conn) -> SummaryResult:
    """Gather final pipeline statistics."""
    stats = get_stats(conn)
    return SummaryResult(
        total_jobs=stats["total_jobs"],
        new_jobs=stats["new_jobs"],
        applied=stats["applied"],
        generated_outputs=stats["generated_outputs"],
    )
