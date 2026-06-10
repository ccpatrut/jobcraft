"""Semantic job ranking via sentence-transformer embeddings.

Scoring combines three signals:
  1. Role similarity   — embedding cosine between profile summary and job text
  2. Industry keyword   — fraction of the candidate's LLM-extracted industry
                          terms found in the job text (direct string match)
  3. Skills mismatch   — penalty when a job requires hard technical skills
                          absent from the candidate's profile
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from .models import JobListing, UserProfile

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None

INDUSTRY_BOOST = 0.25  # max score bonus for a perfect industry keyword overlap
PIVOT_INDUSTRY_BOOST = 0.40  # stronger boost when career pivot is active
MISMATCH_PENALTY = 0.30  # max score penalty for hard skill mismatch

# Career pivot: penalise jobs that demand many years in the *target* sector the
# candidate is entering (they typically lack that tenure even when the role title fits).
PIVOT_SECTOR_TENURE_MIN_YEARS = 5
PIVOT_SECTOR_TENURE_PENALTY = 0.45  # multiplied by weight 0.85–1.0 from signal strength

_SECTOR_SYNONYMS: dict[str, tuple[str, ...]] = {
    "hospitality": (
        "hospitality",
        "hotel",
        "hotels",
        "hotellerie",
        "hôtellerie",
        "horeca",
        "hostellerie",
    ),
    "hotel": ("hotel", "hotels", "hotellerie", "hôtellerie", "lodging"),
    "restaurant": ("restaurant", "restaurants", "gastronomy", "gastronomie", "dining"),
    "gastronomy": ("gastronomy", "gastronomie", "culinary", "restaurant"),
    "catering": ("catering", "banquet", "banqueting"),
    "bar": ("beverage", "sommelier", "mixology"),
    "food service": ("food service", "foodservice", "f&b", "f and b", "f/b"),
    "front office": ("front office", "front desk", "réception", "reception", "concierge"),
}

_YEAR_MENTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:minimum|min\.?|at least|über|mind\.?|>)\s*(\d{1,2})\s*\+?\s*"
        r"(?:years?|yrs?|jahren?|jährige?|ans?)\b",
        re.I,
    ),
    re.compile(
        r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)\s*(?:of\s+)?"
        r"(?:experience|exp\.?|background|track record)\b",
        re.I,
    ),
    re.compile(
        r"(\d{1,2})\s*(?:ans?|années)\s*d[''']?\s*expérience\b",
        re.I,
    ),
    re.compile(
        r"(\d{1,2})\s*\+?\s*Jahren?\s+"
        r"(?:Erfahrung|Berufserfahrung|Berufserfahrung\s+im)\b",
        re.I,
    ),
    re.compile(
        r"(\d{1,2})\s*\+?\s*(?:years?|yrs?|jahren?)\b[^.]{0,50}\b"
        r"(?:in|within|inside|im|dans|en|au sein)\b",
        re.I,
    ),
)

_TECHNICAL_SKILL_GROUPS: list[list[str]] = [
    [
        "c++",
        "c#",
        "java",
        "python",
        "golang",
        "rust",
        "scala",
        "kotlin",
        "ruby",
        "typescript",
        "javascript",
        "swift",
        "objective-c",
    ],
    [
        "openstack",
        "kubernetes",
        "docker",
        "terraform",
        "ansible",
        "jenkins",
        "ci/cd",
        "devops",
        "linux administration",
        "sysadmin",
        "vmware",
    ],
    [
        "sql server",
        "postgresql",
        "mongodb",
        "oracle database",
        "redis",
        "elasticsearch",
        "database administration",
        "dba",
    ],
    [
        "machine learning",
        "deep learning",
        "pytorch",
        "tensorflow",
        "nlp",
        "computer vision",
        "neural network",
    ],
    ["network engineer", "cisco", "ccna", "ccnp", "firewall", "tcp/ip", "routing", "switching"],
    [
        "electrical engineer",
        "mechanical engineer",
        "civil engineer",
        "chemical engineer",
        "hardware design",
        "pcb design",
        "fpga",
        "vhdl",
    ],
]


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model (downloads on first use)."""
    global _model  # noqa: PLW0603
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _profile_text(profile: UserProfile) -> str:
    """Build a text representation focused on role, skills, and qualifications."""
    from .profile_matching import build_matching_profile_text

    return build_matching_profile_text(profile)


def _job_text(job: JobListing) -> str:
    """Build a single text representation of a job for embedding."""
    desc = (job.description or "")[:500]
    return f"{job.title} at {job.company}. {desc}"


def _skills_mismatch(profile_blob: str, job_text: str) -> float:
    """Return 0-1 penalty: 1 means the job needs hard technical skills the candidate lacks."""
    job_lower = job_text.lower()
    profile_lower = profile_blob.lower()
    penalty_groups = 0
    for group in _TECHNICAL_SKILL_GROUPS:
        job_hits = [s for s in group if re.search(rf"\b{re.escape(s)}\b", job_lower)]
        if len(job_hits) < 2:
            continue
        profile_hits = [s for s in group if re.search(rf"\b{re.escape(s)}\b", profile_lower)]
        if not profile_hits:
            penalty_groups += 1
    if not penalty_groups:
        return 0.0
    return min(penalty_groups / 2.0, 1.0)


def _industry_overlap(industries: list[str], job_text: str) -> float:
    """Fraction of candidate industry keywords found in the job text (0-1)."""
    if not industries:
        return 0.0
    text = job_text.lower()
    hits = sum(1 for kw in industries if kw.lower() in text)
    return hits / len(industries)


def _expand_pivot_sector_terms(keywords: list[str]) -> list[str]:
    """Flatten config target-industry keywords plus light synonyms for matching JDs."""
    seen: set[str] = set()
    terms: list[str] = []
    for raw in keywords:
        k = raw.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        terms.append(k)
        for syn in _SECTOR_SYNONYMS.get(k, ()):
            s = syn.lower()
            if s not in seen:
                seen.add(s)
                terms.append(s)
    return sorted(terms, key=len, reverse=True)


def _sector_term_in_window(window_lower: str, term: str) -> bool:
    t = term.strip().lower()
    if not t:
        return False
    if " " in t:
        return t in window_lower
    if len(t) <= 3:
        return bool(re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", window_lower))
    return t in window_lower


def _collect_year_mentions(text: str) -> list[tuple[int, int, int]]:
    """Return (start, end, years) for year-of-experience phrases (deduped by start)."""
    found: list[tuple[int, int, int]] = []
    seen_starts: set[int] = set()
    for pat in _YEAR_MENTION_PATTERNS:
        for m in pat.finditer(text):
            try:
                y = int(m.group(1))
            except (IndexError, ValueError):
                continue
            if not (2 <= y <= 40):
                continue
            if m.start() in seen_starts:
                continue
            seen_starts.add(m.start())
            found.append((m.start(), m.end(), y))
    return found


def _pivot_sector_tenure_weight(job_text: str, sector_terms: list[str]) -> float:
    """0–1: how strongly the JD ties long tenure to the pivot target sector (bad for pivoters)."""
    if not job_text or not sector_terms:
        return 0.0
    text_lower = job_text.lower()
    mentions = _collect_year_mentions(job_text)
    if not mentions:
        return 0.0
    window_radius = 120
    best = 0.0
    for start, end, years in mentions:
        if years < PIVOT_SECTOR_TENURE_MIN_YEARS:
            continue
        lo = max(0, start - window_radius)
        hi = min(len(text_lower), end + window_radius)
        window = text_lower[lo:hi]
        if not any(_sector_term_in_window(window, term) for term in sector_terms):
            continue
        if years >= 10:
            w = 1.0
        elif years >= 7:
            w = 0.95
        else:
            w = 0.85
        best = max(best, w)
    return best


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of strings into embeddings. Returns (N, dim) array."""
    model = _get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def _cosine_scores(job_embs: np.ndarray, ref_emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(job_embs, axis=1) * np.linalg.norm(ref_emb) + 1e-9
    return np.dot(job_embs, ref_emb) / norms


def rank_jobs_by_similarity(
    profile: UserProfile,
    jobs: list[JobListing],
    top_n: int = 10,
    pivot_mode: bool = False,
    pivot_target_keywords: list[str] | None = None,
    matching_text: str | None = None,
) -> list[tuple[JobListing, float]]:
    """Rank jobs by embedding similarity + industry keyword bonus.

    score = embedding_cosine + INDUSTRY_BOOST * keyword_overlap

    Industry keywords come from profile.industries (extracted by the LLM
    during profile parsing). The overlap is a direct string match: what
    fraction of those terms appear in the job text. This reliably separates
    "Account Manager at iGaming company" from "Account Manager at power
    electronics company."

    When ``pivot_mode`` is True and ``pivot_target_keywords`` is set (from
    config target industries), jobs that demand many years of experience in
    that sector are penalised so pivoting candidates are not surfaced for
    senior insider-only roles.

    Returns list of (job, score) tuples sorted best-first.
    """
    if not jobs:
        return []

    role_str = matching_text if matching_text else _profile_text(profile)
    job_strs = [_job_text(j) for j in jobs]
    # Full descriptions for industry/skills checks — requirements are
    # typically near the bottom, well past the 500-char embedding window.
    job_full = [f"{j.title} at {j.company}. {j.description or ''}" for j in jobs]

    all_texts = [role_str] + job_strs
    embeddings = embed_texts(all_texts)

    role_emb = embeddings[0]
    job_embs = embeddings[1:]
    role_scores = _cosine_scores(job_embs, role_emb)

    if profile.industries:
        boost = PIVOT_INDUSTRY_BOOST if pivot_mode else INDUSTRY_BOOST
        logger.info(
            "Industry boost active (%d terms, boost=%.2f, pivot=%s): %s",
            len(profile.industries),
            boost,
            pivot_mode,
            ", ".join(profile.industries),
        )
        industry_scores = np.array(
            [_industry_overlap(profile.industries, jf) for jf in job_full],
        )
        scores = role_scores + boost * industry_scores
    else:
        scores = role_scores

    profile_blob = role_str + " " + " ".join(profile.industries)
    mismatch_scores = np.array(
        [_skills_mismatch(profile_blob, jf) for jf in job_full],
    )
    penalised = int(np.sum(mismatch_scores > 0))
    if penalised:
        logger.info("Skills mismatch penalty applied to %d jobs", penalised)
        scores = scores - MISMATCH_PENALTY * mismatch_scores

    if pivot_mode and pivot_target_keywords:
        sector_terms = _expand_pivot_sector_terms(pivot_target_keywords)
        if sector_terms:
            tenure_weights = np.array(
                [_pivot_sector_tenure_weight(jf, sector_terms) for jf in job_full],
            )
            n_tenure = int(np.sum(tenure_weights > 0))
            if n_tenure:
                logger.info(
                    "Pivot sector-tenure penalty applied to %d jobs (target sectors: %s)",
                    n_tenure,
                    ", ".join(pivot_target_keywords[:8]),
                )
                scores = scores - PIVOT_SECTOR_TENURE_PENALTY * tenure_weights

    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices[:top_n]:
        results.append((jobs[idx], float(scores[idx])))

    return results
