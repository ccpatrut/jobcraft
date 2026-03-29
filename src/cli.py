"""CLI interface built with Typer: subcommands for pipeline, job browsing, and profile access."""

import logging
import textwrap
from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .database import (
    add_job_note,
    get_all_jobs,
    get_connection,
    get_funnel_stats,
    get_job_by_url,
    get_job_notes,
    get_latest_profile,
    get_stats,
    init_db,
    search_cached_jobs,
    update_job_status,
)
from .models import UserProfile

app = typer.Typer(
    name="jobcraft",
    help="JobCraft — AI-Powered Job Matching & CV Generation",
    add_completion=False,
    no_args_is_help=True,
)

jobs_app = typer.Typer(help="Browse and search cached jobs.", no_args_is_help=True)
profile_app = typer.Typer(help="Manage candidate profiles.", no_args_is_help=True)
app.add_typer(jobs_app, name="jobs")
app.add_typer(profile_app, name="profile")

VALID_STATUSES = ("new", "applied", "interview", "rejected", "saved", "offer")


@app.callback()
def _app_callback() -> None:
    """Configure logging before every subcommand."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ── run ──────────────────────────────────────────────────────────────


@app.command()
def run(
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Re-extract profile from CV, ignoring cache",
    ),
    cached: bool = typer.Option(
        False,
        "--cached",
        help="Skip API fetch, work from cached jobs only (offline mode)",
    ),
) -> None:
    """Run the full job matching and CV generation pipeline."""
    from .pipeline import (
        PipelineConfig,
        step_extract_profile,
        step_fetch_and_filter_jobs,
        step_generate_outputs,
        step_load_documents,
        step_rank_jobs,
        step_summary,
    )

    _print_banner()

    cfg_dict = load_config()
    cfg = PipelineConfig.from_config(cfg_dict, fresh=fresh, cached=cached)

    conn = get_connection()
    init_db(conn)

    try:
        load_result = step_load_documents(cfg)

        profile_result = step_extract_profile(conn, load_result.combined_text, cfg)
        _print_profile_compact(profile_result.profile)

        fetch_result = step_fetch_and_filter_jobs(conn, profile_result.profile, cfg)

        if fetch_result.jobs and not cached:
            from .job_fetcher import enrich_job_descriptions

            n_jobs = len(fetch_result.jobs)
            print(f"\n  Enriching {n_jobs} job descriptions from detail pages...")
            enriched = enrich_job_descriptions(fetch_result.jobs)
            print(f"  Enriched {enriched}/{n_jobs} with full descriptions")

        rank_result = step_rank_jobs(profile_result.profile, fetch_result.jobs, cfg)
        _print_ranked_jobs(rank_result.top_jobs)

        selected = interactive_select(rank_result.top_jobs)
        if not selected:
            print("\n  No jobs selected — exiting.")
            raise typer.Exit()

        step_generate_outputs(
            conn,
            profile_result.profile_id,
            profile_result.profile,
            selected,
            cfg,
        )

        summary = step_summary(conn)
        _print_pipeline_done(summary.__dict__, cfg.output_dir)

    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        raise typer.Exit(1) from e
    except ConnectionError as e:
        print(f"  ERROR: {e}")
        raise typer.Exit(1) from e
    except (ValueError, RuntimeError) as e:
        print(f"  ERROR: {e}")
        raise typer.Exit(1) from e
    finally:
        conn.close()


# ── jobs list ────────────────────────────────────────────────────────


@jobs_app.command("list")
def jobs_list(
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (new, applied, interview, rejected, saved, offer)",
    ),
    country: Optional[str] = typer.Option(
        None,
        "--country",
        "-c",
        help="Filter by country code (e.g. gb, ch, de)",
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Max jobs to display"),
) -> None:
    """List cached jobs from the database."""
    if status and status not in VALID_STATUSES:
        print(f"  Invalid status '{status}'. Choose from: {', '.join(VALID_STATUSES)}")
        raise typer.Exit(1)

    conn = get_connection()
    init_db(conn)
    jobs = get_all_jobs(conn, status=status, country=country, limit=limit)
    conn.close()

    if not jobs:
        print("  No jobs found matching the filters.")
        raise typer.Exit()

    _print_jobs_table(jobs)


# ── jobs search ──────────────────────────────────────────────────────


@jobs_app.command("search")
def jobs_search(
    query: str = typer.Argument(..., help="Search term to match against titles and descriptions"),
    country: Optional[str] = typer.Option(
        None,
        "--country",
        "-c",
        help="Filter by country code",
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
) -> None:
    """Search cached jobs by keyword."""
    conn = get_connection()
    init_db(conn)
    results = search_cached_jobs(conn, [query], country=country, limit=limit)
    conn.close()

    if not results:
        print(f'  No jobs found matching "{query}".')
        raise typer.Exit()

    print(f'\n  Found {len(results)} job(s) matching "{query}":\n')
    for i, job in enumerate(results, 1):
        print(f"  {i:>3}. {job.title}")
        print(f"       {job.company} — {job.location}")
        print(f"       {job.url}")
    print()


# ── jobs show ────────────────────────────────────────────────────────


@jobs_app.command("show")
def jobs_show(
    job_url: str = typer.Argument(..., help="URL of the job to display"),
) -> None:
    """Display full details of a cached job."""
    conn = get_connection()
    init_db(conn)
    result = get_job_by_url(conn, job_url)
    notes = get_job_notes(conn, job_url)
    conn.close()

    if not result:
        print(f"  No job found with URL: {job_url}")
        raise typer.Exit(1)

    _row_id, job = result
    print(f"\n  Title:    {job.title}")
    print(f"  Company:  {job.company}")
    print(f"  Location: {job.location}")
    if job.salary:
        print(f"  Salary:   {job.salary}")
    if job.contract_type:
        print(f"  Contract: {job.contract_type}")
    print(f"  Source:   {job.source}")
    print(f"  URL:      {job.url}")

    if job.description:
        print("\n  Description:")
        for line in textwrap.wrap(job.description[:2000], width=76):
            print(f"    {line}")
        if len(job.description) > 2000:
            print(f"    ... ({len(job.description)} chars total)")

    if notes:
        print("\n  Notes:")
        for note_line in notes.split("\n"):
            print(f"    {note_line}")
    print()


# ── jobs note ────────────────────────────────────────────────────────


@jobs_app.command("note")
def jobs_note(
    job_url: str = typer.Argument(..., help="URL of the job to add a note to"),
    note: str = typer.Argument(..., help="Note text to add"),
) -> None:
    """Add a timestamped note to a job."""
    conn = get_connection()
    init_db(conn)

    result = get_job_by_url(conn, job_url)
    if not result:
        print(f"  No job found with URL: {job_url}")
        conn.close()
        raise typer.Exit(1)

    _row_id, job = result
    add_job_note(conn, job_url, note)
    conn.close()

    print(f"  Note added to: {job.title} @ {job.company}")


# ── profile show ─────────────────────────────────────────────────────


@profile_app.command("show")
def profile_show() -> None:
    """Display the most recently extracted candidate profile."""
    conn = get_connection()
    init_db(conn)
    result = get_latest_profile(conn)
    stats = get_stats(conn)
    conn.close()

    if not result:
        print("  No profile found. Run 'jobcraft run' first to extract a profile from your CV.")
        raise typer.Exit(1)

    _profile_id, profile = result
    _print_full_profile(profile)
    print(
        f"  Database: {stats['total_jobs']} jobs cached, "
        f"{stats['applied']} applied, {stats['generated_outputs']} docs generated\n"
    )


# ── generate ─────────────────────────────────────────────────────────


@app.command()
def generate(
    job_url: str = typer.Argument(
        ..., help="URL of the cached job to generate CV/cover letter for"
    ),
) -> None:
    """Regenerate CV and cover letter for a specific cached job."""
    from .pipeline import PipelineConfig, step_generate_outputs

    cfg_dict = load_config()
    cfg = PipelineConfig.from_config(cfg_dict)

    conn = get_connection()
    init_db(conn)

    job_result = get_job_by_url(conn, job_url)
    if not job_result:
        print(f"  No job found in database with URL: {job_url}")
        print("  Run 'jobcraft jobs list' to see cached jobs.")
        conn.close()
        raise typer.Exit(1)
    _job_id, job = job_result

    profile_result = get_latest_profile(conn)
    if not profile_result:
        print("  No profile found. Run 'jobcraft run' first to extract a profile.")
        conn.close()
        raise typer.Exit(1)
    profile_id, profile = profile_result

    print("\n  Generating CV and cover letter for:")
    print(f"    {job.title} @ {job.company}")

    step_generate_outputs(conn, profile_id, profile, [job], cfg)
    conn.close()
    print("\n  Done!")


# ── status ───────────────────────────────────────────────────────────


@app.command()
def status(
    job_url: str = typer.Argument(..., help="URL of the job to update"),
    new_status: str = typer.Argument(
        ...,
        help="New status: new, applied, interview, rejected, saved, offer",
    ),
) -> None:
    """Update the status of a cached job."""
    if new_status not in VALID_STATUSES:
        print(f"  Invalid status '{new_status}'. Choose from: {', '.join(VALID_STATUSES)}")
        raise typer.Exit(1)

    conn = get_connection()
    init_db(conn)

    job_result = get_job_by_url(conn, job_url)
    if not job_result:
        print(f"  No job found with URL: {job_url}")
        conn.close()
        raise typer.Exit(1)

    _job_id, job = job_result
    update_job_status(conn, job_url, new_status)
    conn.close()

    print(f"  Updated: {job.title} @ {job.company} \u2192 {new_status}")


# ── stats ────────────────────────────────────────────────────────────


@app.command()
def stats() -> None:
    """Show application funnel statistics."""
    conn = get_connection()
    init_db(conn)
    funnel = get_funnel_stats(conn)
    general = get_stats(conn)
    conn.close()

    total = general["total_jobs"]
    print("\n  Application Funnel")
    print("  " + "=" * 40)

    stages = ["new", "saved", "applied", "interview", "offer", "rejected"]
    max_count = max(funnel.values()) if funnel else 1

    for stage in stages:
        count = funnel.get(stage, 0)
        bar_len = int(30 * count / max_count) if max_count else 0
        bar = "\u2588" * bar_len
        pct = (count / total * 100) if total else 0
        print(f"  {stage:<12} {bar:<30} {count:>4} ({pct:.0f}%)")

    other = {k: v for k, v in funnel.items() if k not in stages}
    for stage, count in other.items():
        pct = (count / total * 100) if total else 0
        print(f"  {stage:<12} {'':30} {count:>4} ({pct:.0f}%)")

    print(f"\n  Total: {total} jobs | {general['generated_outputs']} documents generated")
    print()


# ── Display helpers (internal) ───────────────────────────────────────


def _print_banner() -> None:
    print("=" * 60)
    print("Job Finder - AI-Powered Job Matching & CV Generation")
    print("=" * 60)


def interactive_select(jobs: list) -> list:
    """Prompt the user to select which jobs to generate CVs for.

    Shows description snippets and salary to help the user decide.
    Accepts comma-separated numbers, ranges (e.g. 1-5), or 'all'.
    Returns the selected subset of jobs.
    """
    total = len(jobs)
    if total == 0:
        return []

    print(f"\n  Top {total} matches:\n")
    for i, job in enumerate(jobs, 1):
        print(f"  {i:>3}. {job.title} @ {job.company}")
        if job.salary:
            print(f"       Salary: {job.salary}")
        if job.description:
            snippet = job.description[:150].replace("\n", " ").strip()
            if len(job.description) > 150:
                snippet += "..."
            print(f"       {snippet}")
        print(f"       {job.url}")

    print(f"\n  Select jobs to generate CVs for (1-{total}).")
    print("  Enter numbers separated by commas, ranges (e.g. 1-3,5,7), or 'all'.")
    print("  Press Enter for all, or 'q' to quit.\n")

    while True:
        try:
            raw = input("  Your selection: ").strip()
        except (EOFError, KeyboardInterrupt):
            return []

        if not raw or raw.lower() == "all":
            return list(jobs)

        if raw.lower() in ("q", "quit", "exit"):
            return []

        indices: set[int] = set()
        valid = True
        for part in raw.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    lo, hi = part.split("-", 1)
                    lo, hi = int(lo.strip()), int(hi.strip())
                    if lo < 1 or hi > total or lo > hi:
                        print(f"  Invalid range: {part} (must be 1-{total})")
                        valid = False
                        break
                    indices.update(range(lo, hi + 1))
                except ValueError:
                    print(f"  Invalid range: {part}")
                    valid = False
                    break
            else:
                try:
                    n = int(part)
                    if n < 1 or n > total:
                        print(f"  {n} is out of range (1-{total})")
                        valid = False
                        break
                    indices.add(n)
                except ValueError:
                    print(f"  '{part}' is not a number")
                    valid = False
                    break

        if valid and indices:
            selected = [jobs[i - 1] for i in sorted(indices)]
            print(f"  \u2192 {len(selected)} job(s) selected.")
            return selected


def _print_profile_compact(profile: UserProfile) -> None:
    """Short profile summary used during pipeline run."""
    print("  Profile:", profile.name or "Unknown", "-", len(profile.skills), "skills")
    if profile.skills:
        print("  Skills:", ", ".join(profile.skills[:10]))
    if profile.languages:
        print("  Languages:", ", ".join(profile.languages))
    if profile.industries:
        print("  Industries:", ", ".join(profile.industries))
    if profile.experience:
        print("  Experience:", len(profile.experience), "roles")
        for exp in profile.experience[:3]:
            print("    -", exp[:80] + ("..." if len(exp) > 80 else ""))
    if profile.summary:
        print(
            "  Summary:",
            profile.summary[:120] + ("..." if len(profile.summary) > 120 else ""),
        )


def _print_full_profile(profile: UserProfile) -> None:
    """Detailed profile display for the `profile show` command."""
    print(f"\n  Name:     {profile.name or 'Unknown'}")
    if profile.email:
        print(f"  Email:    {profile.email}")
    if profile.phone:
        print(f"  Phone:    {profile.phone}")
    if profile.location:
        print(f"  Location: {profile.location}")

    if profile.summary:
        print("\n  Summary:")
        for line in textwrap.wrap(profile.summary, width=72):
            print(f"    {line}")

    if profile.skills:
        print(f"\n  Skills ({len(profile.skills)}):")
        print(f"    {', '.join(profile.skills)}")

    if profile.experience:
        print(f"\n  Experience ({len(profile.experience)} roles):")
        for exp in profile.experience:
            print(f"    - {exp}")

    if profile.education:
        print(f"\n  Education ({len(profile.education)}):")
        for edu in profile.education:
            print(f"    - {edu}")

    if profile.certifications:
        print(f"\n  Certifications ({len(profile.certifications)}):")
        for cert in profile.certifications:
            print(f"    - {cert}")

    if profile.languages:
        print(f"\n  Languages ({len(profile.languages)}):")
        for lang in profile.languages:
            print(f"    - {lang}")

    print()


def _print_ranked_jobs(jobs: list) -> None:
    """Display the top-ranked job matches."""
    print(f"  Selected top {len(jobs)} matches:")
    for i, job in enumerate(jobs, 1):
        print(f"    {i}. {job.title} @ {job.company}")
        print(f"       {job.url}")


def _print_jobs_table(jobs: list[dict]) -> None:
    """Format cached jobs as a readable listing."""
    print(f"\n  {len(jobs)} job(s) found:\n")

    hdr = f"  {'#':>4}  {'Status':<10}  {'Title':<35}  {'Company':<20}  {'Location'}"
    print(hdr)
    print(f"  {'─' * 4}  {'─' * 10}  {'─' * 35}  {'─' * 20}  {'─' * 15}")

    for i, job in enumerate(jobs, 1):
        title = (job["title"][:33] + "..") if len(job["title"]) > 35 else job["title"]
        company = (job["company"][:18] + "..") if len(job["company"]) > 20 else job["company"]
        location = ""
        if job.get("location"):
            location = (
                (job["location"][:13] + "..") if len(job["location"]) > 15 else job["location"]
            )
        job_status = job.get("status", "new")
        print(f"  {i:>4}  {job_status:<10}  {title:<35}  {company:<20}  {location}")
        print(f"        {job['url']}")

    print()


def _print_pipeline_done(stats: dict, output_dir: Path) -> None:
    """Summary printed at the end of a full pipeline run."""
    print("\n[6/6] Done!")
    print(f"\nOutput saved to: {output_dir}")
    print(
        f"\nDatabase: {stats['total_jobs']} total jobs cached, "
        f"{stats['new_jobs']} new, {stats['applied']} applied, "
        f"{stats['generated_outputs']} documents generated"
    )
    print("\nNext steps:")
    print("  - Review the generated CVs and cover letters (.md and .pdf)")
    print("  - Adjust config.yaml (tone, style) to match your preferences")
    print("  - Use 'jobcraft stats' to see your application funnel")
    print("  - Use 'jobcraft jobs note <url> \"note\"' to track your progress")
