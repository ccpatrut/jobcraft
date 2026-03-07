# Contributing to Job Finder

## Architecture Overview

Job Finder is a pipeline-based application. Each stage transforms data and passes it forward. All AI inference runs locally through Ollama — nothing leaves your machine except Adzuna API calls.

```
 CV files (PDF/Word)
       │
       ▼
 ┌─────────────────┐
 │ Document Loader  │  Extract raw text from PDFs and Word files
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │Profile Extractor │  Ollama parses text into structured profile
 └────────┬────────┘  (cached in SQLite by CV hash)
          ▼
 ┌─────────────────┐
 │ Query Builder    │  Derives multiple search queries from profile
 │ + Translator     │  Expands into local languages (de/fr/it)
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Job Fetcher     │  Calls Adzuna API across locations + queries
 └────────┬────────┘  Falls back to SQLite cache if API returns few
          ▼
 ┌─────────────────┐
 │  Language Filter  │  Regex pass: blocks jobs requiring languages
 │  (regex)         │  beyond candidate's proficiency
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  AI Validation    │  Ollama re-checks survivors, learns new
 │  Loop (optional) │  patterns, re-filters (configurable)
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  AI Ranking       │  Ollama ranks remaining jobs by fit
 └────────┬────────┘  Returns top 10
          ▼
 ┌─────────────────┐
 │  CV + Letter      │  Ollama generates tailored Markdown
 │  Generator       │  for each job match
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  PDF Renderer     │  Converts Markdown to professional PDFs
 └────────┬────────┘
          ▼
    output/ folder
    (10 CVs + 10 cover letters, .md + .pdf)
```

## Module Reference

### `main.py` — Orchestrator

The entry point. Wires all modules together in a 6-step pipeline:

1. Load documents from `cv_input/`
2. Extract or retrieve cached profile
3. Build search queries, fetch jobs, fall back to DB cache
4. Filter by language (regex, then optional AI validation)
5. Rank with Ollama, generate CVs and cover letters
6. Render PDFs, save to `output/`

Supports `--fresh` flag to bypass profile cache and force re-extraction.

### `src/models.py` — Data Models

Pydantic models shared across all modules:

- **`UserProfile`** — name, email, skills, experience, education, certifications, languages, raw CV text
- **`JobListing`** — title, company, description, location, URL, salary, contract type, source
- **`UserPreferences`** — tone, style, focus areas

### `src/config.py` — Configuration

Loads `config.yaml` with environment variable overrides. All settings have sensible defaults. Key config sections:

- `preferences` — tone, style, default languages
- `job_search` — country, locations, radius, language strategy, exclude languages, AI validation toggle
- `ai` — model name, temperature, timeout

### `src/document_loader.py` — Text Extraction

Reads all `.pdf` and `.docx` files from the input directory:

- **PDF** — PyMuPDF (`pymupdf`) for text layer extraction
- **Word** — `python-docx` for paragraph extraction
- Returns per-file texts and a combined string for profile extraction

### `src/profile_extractor.py` — Profile Extraction

Sends the combined CV text to Ollama with a detailed system prompt that instructs the model to return structured JSON with:

- Name, contact info, summary
- Skills, experience entries, education
- Language proficiencies (e.g., "English - Proficient", "German - Intermediate")
- Certifications

Includes a `_to_strings` normalizer to handle LLM output variations (dicts, lists, nested objects) and a fallback extraction prompt if the primary one fails. Results are validated through Pydantic.

### `src/query_translator.py` — Multi-lingual Search

Handles query expansion for multi-lingual markets:

- **`_build_search_queries`** (in `main.py`) generates multiple base queries from job titles, title variants (splitting `&` and `/` compounds), and top skills
- **`get_localized_queries`** translates each base query into national languages via Ollama (e.g., English → German, French, Italian for Switzerland)
- **Language detection** — `detect_language_markers` uses word-frequency heuristics to identify the language of a job description

Configurable strategies: `auto`, `english_only`, `local_first`.

### `src/job_fetcher.py` — Adzuna API Client

Calls the Adzuna REST API (`/v1/api/jobs/{country}/search/1`) with:

- `what` — search query
- `where` — location (optional)
- `distance` — radius in km (optional)
- `results_per_page` — up to 50

Returns a list of `JobListing` objects. Requires `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` from `.env`.

### `src/job_matcher.py` — Filtering + Ranking

The most complex module. Three layers of job filtering:

#### Layer 1: Regex Language Filter (`filter_jobs_by_language`)

Fast, deterministic pass that removes jobs with language requirements exceeding the candidate's proficiency:

- **CEFR detection** — regex patterns find levels like "C2", "B1" near language names, handling variations like "Deutschkenntnisse Niveau C2"
- **Fluency keyword detection** — catches phrases like "fließend Deutsch", "langue maternelle français", "madrelingua italiana"
- **Tier comparison** — maps both the candidate's stated level and the job's requirement to a numeric tier (1-6) and blocks if `required > candidate`

#### Layer 2: AI Validation Loop (`ai_validate_and_filter`)

Optional (toggle via `ai_language_validation` in config). Uses Ollama to re-check each surviving job:

1. Sends the job description + candidate languages to the LLM
2. If the LLM flags a requirement, it extracts the **exact phrase** from the description
3. That phrase is compiled into a new regex pattern and added to a dynamic pattern list
4. Next round, the learned patterns are applied first (instant), then the LLM checks only remaining unknowns
5. Repeats for up to `ai_validation_rounds` (default 3), exits early if no new violations found

This creates a self-improving filter that gets smarter within a single run.

#### Layer 3: AI Ranking (`rank_jobs_with_ollama`)

Sends all surviving jobs (up to 25) to Ollama in a single prompt. The model selects the top 10 by best fit, respecting language constraints, skill overlap, and experience relevance. Uses streaming to show live progress.

### `src/cv_generator.py` — CV Generation

For each matched job, sends the candidate profile + job details + user preferences to Ollama with a structured Markdown template. The prompt enforces:

- Specific section ordering (Header, Profile, Skills, Experience, Education, Languages, Certifications)
- Formatting conventions (## for sections, ### for roles, * for bullets)
- Language preservation (exact proficiency levels from profile)
- Omission of empty sections (e.g., no Certifications header if none exist)

### `src/cover_letter_generator.py` — Cover Letter Generation

Similar to CV generation but with a letter template. Tailors the tone and content to the specific job, incorporating the user's preference settings (formal/casual, concise/detailed).

### `src/pdf_utils.py` — PDF Rendering

Converts generated Markdown into professional PDFs using `fpdf2`:

- **Custom PDF class** (`_JobFinderPDF`) with methods for rendering names, contact info, section headers, experience titles with right-aligned dates, bullet points, and paragraphs
- **Markdown parser** (`_render_markdown`) that walks the Markdown line-by-line, detecting headers, bullets, horizontal rules, and links
- **Smart section skipping** — looks ahead at section content and omits the entire section if it only contains "N/A" or is empty
- **Text sanitization** — handles non-Latin-1 characters for PDF compatibility
- Each generated PDF includes a clickable "Apply for this position" link to the original job posting

### `src/database.py` — SQLite Caching

Manages a local `job_finder.db` with three tables:

| Table | Purpose | Key |
|-------|---------|-----|
| `profiles` | Cached profile extractions | `cv_hash` (SHA-256 of CV text) |
| `jobs` | All fetched job listings | `url` (unique) |
| `generated_outputs` | Tracks what was generated | `job_id` + `profile_id` |

- **Profile caching** — if the CV text hasn't changed (same hash), the Ollama extraction is skipped entirely. Use `--fresh` to force re-extraction.
- **Job storage** — write-only during the fetch phase (saves all API results for historical tracking). Read-back happens only as a fallback when the API returns fewer than 5 jobs.
- **Keyword search** — `search_cached_jobs` does a `LIKE` search across title and description for DB fallback.
- Uses WAL mode for safe concurrent reads.

## Data Flow

```
config.yaml + .env
       │
       ▼
   load_config()  ──→  UserPreferences
       │                     │
       ▼                     ▼
 load_documents()    extract_profile()  ◄──► SQLite (profiles)
       │                     │
       ▼                     ▼
  raw CV text          UserProfile
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
        _build_search_queries()   get_localized_queries()
                  │                   │
                  └─────────┬─────────┘
                            ▼
                    fetch_adzuna_jobs()  ──→  SQLite (jobs)
                            │
                            ▼ fallback if < 5 results
                    search_cached_jobs()  ◄── SQLite (jobs)
                            │
                            ▼
                    filter_jobs_by_language()   [regex]
                            │
                            ▼
                    ai_validate_and_filter()    [LLM, optional]
                            │
                            ▼
                    rank_jobs_with_ollama()     [LLM]
                            │
                            ▼
              ┌─────────────┴─────────────┐
              ▼                           ▼
    generate_tailored_cv()    generate_cover_letter()
              │                           │
              ▼                           ▼
        markdown_to_pdf()           markdown_to_pdf()
              │                           │
              ▼                           ▼
         output/*.pdf                output/*.pdf
```

## Key Design Decisions

### Why Ollama (local LLM) instead of OpenAI/Claude?

Privacy. CVs contain personal data — names, addresses, phone numbers, employment history. Everything stays on the user's machine. No API keys to manage beyond Adzuna.

### Why Adzuna?

Covers DACH, France, Italy, and the UK with a single free API. Supports location-based search with radius. Most other free job APIs are US-only or require paid plans for European markets.

### Why SQLite for caching?

Zero-config, no server process, single-file database. WAL mode enables safe concurrent reads. The cache serves two purposes: avoiding redundant Ollama calls for unchanged CVs, and providing a job fallback when the API returns thin results.

### Why fpdf2 instead of wkhtmltopdf/WeasyPrint?

Pure Python, no system dependencies. WeasyPrint requires Cairo/Pango, wkhtmltopdf requires a binary install — both are fragile across OS/architecture. fpdf2 works everywhere Python runs.

### Why multi-query search?

Job APIs perform keyword matching, not semantic search. A title like "Solution & Integration Architect" returns almost nothing. Splitting into "Solution Architect" + "Integration Architect" + skill-based queries dramatically improves recall.

### Why a self-learning language filter?

Regex patterns can't anticipate every way a job description states language requirements across German, French, Italian, and English. The AI validation loop catches edge cases and learns patterns it can reuse within the same run, reducing LLM calls on subsequent rounds.

## Development

### Running

```bash
uv sync
uv run python main.py          # normal run
uv run python main.py --fresh   # force profile re-extraction
```

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

### Adding dependencies

```bash
uv add <package>
```

### Database inspection

```bash
# CLI
sqlite3 job_finder.db ".tables"
sqlite3 job_finder.db "SELECT title, company FROM jobs LIMIT 10;"

# Or install the SQLite Viewer extension in VS Code / Cursor
```

### Environment variables

All optional — `config.yaml` is the primary config. Env vars override specific settings:

| Variable | Overrides |
|----------|-----------|
| `ADZUNA_APP_ID` | Adzuna API credentials |
| `ADZUNA_APP_KEY` | Adzuna API credentials |
| `OLLAMA_HOST` | `ai.host` in config |
| `OLLAMA_MODEL` | `ai.model` in config |
| `CV_INPUT_DIR` | `cv_input_dir` in config |
| `OUTPUT_DIR` | `output_dir` in config |
| `JOB_COUNTRY` | `job_search.country` in config |
