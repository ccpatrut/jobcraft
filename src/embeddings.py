"""Semantic job ranking via sentence-transformer embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from .models import JobListing, UserProfile

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model (downloads on first use)."""
    global _model  # noqa: PLW0603
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _profile_text(profile: UserProfile) -> str:
    """Build a single text representation of the candidate for embedding."""
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


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of strings into embeddings. Returns (N, dim) array."""
    model = _get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def rank_jobs_by_similarity(
    profile: UserProfile,
    jobs: list[JobListing],
    top_n: int = 10,
) -> list[tuple[JobListing, float]]:
    """Rank jobs by cosine similarity to the profile.

    Returns list of (job, score) tuples sorted best-first.
    """
    if not jobs:
        return []

    profile_str = _profile_text(profile)
    job_strs = [_job_text(j) for j in jobs]

    all_texts = [profile_str] + job_strs
    embeddings = embed_texts(all_texts)

    profile_emb = embeddings[0]
    job_embs = embeddings[1:]

    # Cosine similarity (embeddings are already normalized by sentence-transformers)
    scores = np.dot(job_embs, profile_emb) / (
        np.linalg.norm(job_embs, axis=1) * np.linalg.norm(profile_emb) + 1e-9
    )

    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices[:top_n]:
        results.append((jobs[idx], float(scores[idx])))

    return results
