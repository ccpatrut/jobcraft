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
    parts = []
    if profile.summary:
        parts.append(profile.summary)
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    if profile.experience:
        for exp in profile.experience[:5]:
            parts.append(exp[:300])
    if profile.certifications:
        parts.append("Certifications: " + ", ".join(profile.certifications))
    return "\n".join(parts)


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
) -> list[tuple[JobListing, float]]:
    """Rank jobs by embedding similarity + industry keyword bonus.

    score = embedding_cosine + INDUSTRY_BOOST * keyword_overlap

    Industry keywords come from profile.industries (extracted by the LLM
    during profile parsing). The overlap is a direct string match: what
    fraction of those terms appear in the job text. This reliably separates
    "Account Manager at iGaming company" from "Account Manager at power
    electronics company."

    Returns list of (job, score) tuples sorted best-first.
    """
    if not jobs:
        return []

    role_str = _profile_text(profile)
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

    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices[:top_n]:
        results.append((jobs[idx], float(scores[idx])))

    return results
