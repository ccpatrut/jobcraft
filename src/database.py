"""SQLite database for caching jobs, profiles, and generated outputs."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import JobListing, UserProfile

DEFAULT_DB_PATH = "job_finder.db"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for better concurrent reads."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS profiles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cv_hash     TEXT UNIQUE NOT NULL,
            name        TEXT,
            email       TEXT,
            phone       TEXT,
            location    TEXT,
            summary     TEXT,
            skills      TEXT,
            experience  TEXT,
            education   TEXT,
            certifications TEXT,
            languages   TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            url           TEXT UNIQUE NOT NULL,
            title         TEXT NOT NULL,
            company       TEXT NOT NULL,
            description   TEXT,
            location      TEXT,
            salary        TEXT,
            contract_type TEXT,
            source        TEXT DEFAULT 'adzuna',
            search_query  TEXT,
            country       TEXT,
            fetched_at    TEXT NOT NULL,
            status        TEXT DEFAULT 'new'
        );

        CREATE TABLE IF NOT EXISTS generated_outputs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      INTEGER NOT NULL REFERENCES jobs(id),
            profile_id  INTEGER NOT NULL REFERENCES profiles(id),
            output_type TEXT NOT NULL,
            file_path   TEXT,
            content     TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_url ON jobs(url);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_fetched ON jobs(fetched_at);
        CREATE INDEX IF NOT EXISTS idx_profiles_hash ON profiles(cv_hash);
        CREATE INDEX IF NOT EXISTS idx_outputs_job ON generated_outputs(job_id);
    """)
    conn.commit()


# -- Profile caching --


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def get_cached_profile(conn: sqlite3.Connection, cv_text: str) -> Optional[UserProfile]:
    """Return cached profile if CV text hasn't changed."""
    cv_hash = _hash_text(cv_text)
    row = conn.execute("SELECT * FROM profiles WHERE cv_hash = ?", (cv_hash,)).fetchone()
    if not row:
        return None
    return UserProfile(
        name=row["name"],
        email=row["email"],
        phone=row["phone"],
        location=row["location"],
        summary=row["summary"] or "",
        skills=json.loads(row["skills"] or "[]"),
        experience=json.loads(row["experience"] or "[]"),
        education=json.loads(row["education"] or "[]"),
        certifications=json.loads(row["certifications"] or "[]"),
        languages=json.loads(row["languages"] or "[]"),
        raw_text=cv_text,
    )


def save_profile(conn: sqlite3.Connection, cv_text: str, profile: UserProfile) -> int:
    """Save extracted profile to DB. Returns the profile row id."""
    cv_hash = _hash_text(cv_text)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO profiles (cv_hash, name, email, phone, location, summary,
                              skills, experience, education, certifications, languages, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cv_hash) DO UPDATE SET
            name=excluded.name, email=excluded.email, phone=excluded.phone,
            location=excluded.location, summary=excluded.summary,
            skills=excluded.skills, experience=excluded.experience,
            education=excluded.education, certifications=excluded.certifications,
            languages=excluded.languages,
            created_at=excluded.created_at
        """,
        (
            cv_hash,
            profile.name,
            profile.email,
            profile.phone,
            profile.location,
            profile.summary,
            json.dumps(profile.skills),
            json.dumps(profile.experience),
            json.dumps(profile.education),
            json.dumps(profile.certifications),
            json.dumps(profile.languages),
            now,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM profiles WHERE cv_hash = ?", (cv_hash,)).fetchone()
    return row["id"]


# -- Job caching --


def save_jobs(
    conn: sqlite3.Connection,
    jobs: list[JobListing],
    search_query: str = "",
    country: str = "",
) -> int:
    """Save jobs to DB, skipping duplicates by URL. Returns count of new jobs inserted."""
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for job in jobs:
        try:
            conn.execute(
                """
                INSERT INTO jobs (url, title, company, description, location,
                                  salary, contract_type, source, search_query, country, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title=excluded.title, company=excluded.company,
                    description=excluded.description, location=excluded.location,
                    salary=excluded.salary, contract_type=excluded.contract_type,
                    fetched_at=excluded.fetched_at
                """,
                (
                    job.url,
                    job.title,
                    job.company,
                    job.description,
                    job.location,
                    job.salary,
                    job.contract_type,
                    job.source,
                    search_query,
                    country,
                    now,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return inserted


def get_job_id(conn: sqlite3.Connection, job_url: str) -> Optional[int]:
    """Get the DB row id for a job by URL."""
    row = conn.execute("SELECT id FROM jobs WHERE url = ?", (job_url,)).fetchone()
    return row["id"] if row else None


def get_all_jobs(
    conn: sqlite3.Connection,
    status: Optional[str] = None,
    country: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """Retrieve jobs from DB with optional filtering."""
    query = "SELECT * FROM jobs WHERE 1=1"
    params: list = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if country:
        query += " AND country = ?"
        params.append(country)
    query += " ORDER BY fetched_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def update_job_status(conn: sqlite3.Connection, job_url: str, status: str) -> None:
    """Update the status of a job (new, applied, interview, rejected, saved)."""
    conn.execute("UPDATE jobs SET status = ? WHERE url = ?", (status, job_url))
    conn.commit()


# -- Generated output tracking --


def save_generated_output(
    conn: sqlite3.Connection,
    job_id: int,
    profile_id: int,
    output_type: str,
    file_path: str = "",
    content: str = "",
) -> None:
    """Track a generated CV or cover letter."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO generated_outputs (job_id, profile_id, output_type, file_path, content, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (job_id, profile_id, output_type, file_path, content, now),
    )
    conn.commit()


# -- Stats --


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get summary stats from the database."""
    total_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    new_jobs = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'new'").fetchone()[0]
    applied = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'applied'").fetchone()[0]
    profiles = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    outputs = conn.execute("SELECT COUNT(*) FROM generated_outputs").fetchone()[0]
    return {
        "total_jobs": total_jobs,
        "new_jobs": new_jobs,
        "applied": applied,
        "profiles": profiles,
        "generated_outputs": outputs,
    }
