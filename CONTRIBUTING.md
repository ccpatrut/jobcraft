# Contributing to Job Finder

## Architecture Overview

Job Finder is a pipeline-based application. Each stage transforms data and passes it forward. All AI inference runs locally through Ollama and HuggingFace models — nothing leaves your machine except Adzuna API calls.

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
 │ AI Query Builder │  Ollama generates realistic job titles from profile
 │ + Translator     │  Expands into local languages (de/fr/it) if configured
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Job Fetcher     │  Calls Adzuna API across locations + queries
 └────────┬────────┘  Falls back to SQLite cache if API returns few
          ▼
 ┌─────────────────┐
 │ HF Language      │  xlm-roberta classifier detects posting language
 │ Classifier       │  Removes non-English postings (in english_only mode)
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ Title-Language   │  Catches English postings with "German speaking",
 │ Filter           │  "French required" etc. in title (english_only mode)
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ Regex Language   │  Blocks jobs requiring CEFR levels or fluency
 │ Filter           │  beyond candidate's proficiency
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ AI Validation    │  Ollama re-checks survivors, learns new
 │ Loop (optional)  │  patterns, re-filters (configurable)
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ Semantic Ranking │  HuggingFace sentence-transformers embeddings
 │ (embeddings)     │  rank by cosine similarity, optional Ollama re-rank
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ Interactive      │  User selects which jobs to generate CVs for
 │ Selection        │  (comma-separated, ranges, or 'all')
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ CV + Letter      │  Ollama generates tailored Markdown
 │ Generator        │  for each selected job
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ PDF Renderer     │  Converts Markdown to professional PDFs
 └────────┬────────┘
          ▼
    output/ folder
    (.md + .pdf for each selected job)
```

## Module Reference

### `main.py` — Orchestrator

The entry point. Wires all modules together in a 6-step pipeline:

1. Load documents from `cv_input/`
2. Extract or retrieve cached profile
3. Build AI-generated search queries, fetch jobs, fall back to DB cache
4. Filter by language (HF classifier → title-language filter → regex → optional AI validation)
5. Rank with embeddings (+ optional Ollama re-rank), interactive job selection
6. Generate CVs and cover letters, render PDFs, save to `output/`

Supports `--fresh` flag to bypass profile cache and force re-extraction.

Installs a `SIGINT` handler (`os._exit(130)`) so Ctrl+C works immediately even during HuggingFace/PyTorch C-level inference.

### `src/models.py` — Data Models

Pydantic models shared across all modules:

- **`UserProfile`** — name, email, skills, experience, education, certifications, languages, raw CV text
- **`JobListing`** — title, company, description, location, URL, salary, contract type, source
- **`UserPreferences`** — tone, style, focus areas

### `src/config.py` — Configuration

Loads `config.yaml` with environment variable overrides. All settings have sensible defaults. Key config sections:

- `preferences` — tone, style, default languages
- `job_search` — country, locations, radius, language strategy, exclude languages, AI validation toggle
- `ai` — model name, temperature, timeout, `use_embeddings`, `rerank_with_llm`

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

- **`_ai_generate_queries`** (in `main.py`) uses Ollama to generate 5-8 realistic, market-relevant job titles from the user's profile, replacing the earlier heuristic approach
- **`get_localized_queries`** translates each base query into national languages via Ollama (e.g., English → German, French, Italian for Switzerland)

Configurable strategies: `auto`, `english_only`, `local_first`.

### `src/job_fetcher.py` — Adzuna API Client

Calls the Adzuna REST API (`/v1/api/jobs/{country}/search/1`) with:

- `what` — search query
- `where` — location (optional)
- `distance` — radius in km (optional)
- `results_per_page` — up to 50

Returns a list of `JobListing` objects. Requires `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` from `.env`.

### `src/lang_detect.py` — HuggingFace Language Detection

Uses the `papluca/xlm-roberta-base-language-detection` transformer for accurate language classification (~1.1GB, auto-downloads on first run):

- **`detect_language(text)`** — returns an ISO 639-1 code (e.g., "en", "de", "fr")
- **`filter_jobs_by_detected_language(jobs, exclude_languages)`** — removes jobs in excluded languages
- **`filter_jobs_by_description_language(jobs, candidate_languages)`** — removes jobs whose posting language requires proficiency the candidate doesn't have (e.g., a full-German posting when the candidate has beginner German)

Used heavily in `english_only` mode: every fetched job is classified and non-English postings are removed before any other filtering.

### `src/embeddings.py` — Semantic Similarity Ranking

Uses `sentence-transformers/all-MiniLM-L6-v2` (~80MB, auto-downloads on first run) for fast semantic job matching:

- **`embed_texts(texts)`** — generates dense vector embeddings for a batch of texts
- **`rank_jobs_by_similarity(profile, jobs, top_n)`** — builds a profile text from the candidate's skills/experience/summary, embeds it alongside all job descriptions, computes cosine similarity, and returns the top-N most similar jobs with scores

This replaces LLM-based ranking as the primary method (much faster and more accurate). Ollama re-ranking can optionally refine the top results for nuance.

### `src/job_matcher.py` — Filtering + Ranking

The most complex module. Multiple layers of job filtering:

#### Layer 1: HF Language Classifier

(Handled in `main.py` using `src/lang_detect.py`) — detects posting language and removes non-English postings when `english_only` is set.

#### Layer 2: Title-Language Filter (`filter_english_only_jobs`)

Catches English-language postings that reference non-English languages in the title (e.g., "German speaking Account Manager", "French Ads Manager"). Uses `_TITLE_LANG_SIGNALS` regex patterns on the job title. Only active in `english_only` mode. Compares against the candidate's proficiency — if below Advanced (tier 4), the job is removed.

#### Layer 3: Regex Language Filter (`filter_jobs_by_language`)

Fast, deterministic pass that removes jobs with language requirements exceeding the candidate's proficiency:

- **CEFR detection** — regex patterns find levels like "C2", "B1" near language names, handling variations like "Deutschkenntnisse Niveau C2"
- **Fluency keyword detection** — catches phrases in both local languages ("fließend Deutsch", "langue maternelle français") and English ("fluent German", "native French", "German speaking", "excellent German")
- **Tier comparison** — maps both the candidate's stated level and the job's requirement to a numeric tier (1-6) and blocks if `required > candidate`

#### Layer 4: AI Validation Loop (`ai_validate_and_filter`)

Optional (toggle via `ai_language_validation` in config). Uses Ollama to re-check each surviving job:

1. Sends the job description + candidate languages to the LLM
2. If the LLM flags a requirement, it extracts the **exact phrase** from the description
3. That phrase is compiled into a new regex pattern and added to a dynamic pattern list
4. Next round, the learned patterns are applied first (instant), then the LLM checks only remaining unknowns
5. Repeats for up to `ai_validation_rounds` (default 3), exits early if no new violations found

This creates a self-improving filter that gets smarter within a single run.

#### Layer 5: Semantic Ranking (`rank_jobs_with_embeddings`)

Primary ranking method. Uses HuggingFace sentence embeddings (cosine similarity) to rank all surviving jobs by fit. Optionally followed by Ollama re-ranking (controlled by `rerank_with_llm` in config) for nuanced refinement.

Fallback: `rank_jobs_with_ollama` — sends all surviving jobs to Ollama in a single prompt for LLM-based ranking (used when `use_embeddings: false`).

### `src/cv_generator.py` — CV Generation

For each selected job, sends the candidate profile + raw CV text + job details + user preferences to Ollama with a structured Markdown template. The prompt enforces:

- Specific section ordering (Header, Profile, Skills, Experience, Education, Languages, Certifications)
- Experience expansion: 4-6 detailed bullet points per role, reframed for the target job
- Language preservation (exact proficiency levels from profile)
- Omission of empty sections (e.g., no Certifications header if none exist)
- Drawing from the raw CV text as the primary source to ensure nothing is lost or diminished

### `src/cover_letter_generator.py` — Cover Letter Generation

Similar to CV generation but with a letter template. Uses raw CV text to write compelling, detailed letters. Tailors the tone and content to the specific job, incorporating the user's preference settings (formal/casual, concise/detailed).

### `src/pdf_utils.py` — PDF Rendering

Converts generated Markdown into professional PDFs using `fpdf2`:

- **Custom PDF class** (`_JobFinderPDF`) with methods for rendering names, contact info, section headers, experience titles with right-aligned dates, bullet points, and paragraphs
- **Markdown parser** (`_render_markdown`) that walks the Markdown line-by-line, detecting headers, bullets, horizontal rules, and links
- **Smart section skipping** — looks ahead at section content and omits the entire section if it only contains "N/A" or is empty
- **Text sanitization** — handles non-Latin-1 characters for PDF compatibility

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

### `src/spinner.py` — CLI Loading Spinner

Provides a `Spinner` context manager for animated terminal feedback during long-running operations:

```python
with Spinner("Loading model"):
    slow_operation()
# prints: ✓ Loading model — done (3.2s)
```

Uses a daemon thread for non-blocking animation with elapsed time display.

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
        _ai_generate_queries()   get_localized_queries()
                  │                   │
                  └─────────┬─────────┘
                            ▼
                    fetch_adzuna_jobs()  ──→  SQLite (jobs)
                            │
                            ▼ fallback if < 5 results
                    search_cached_jobs()  ◄── SQLite (jobs)
                            │
                            ▼
                    detect_language()            [HF xlm-roberta]
                            │
                            ▼
                    filter_english_only_jobs()   [title-language patterns]
                            │
                            ▼
                    filter_jobs_by_language()    [regex]
                            │
                            ▼
                    ai_validate_and_filter()     [LLM, optional]
                            │
                            ▼
                    rank_jobs_by_similarity()    [HF sentence-transformers]
                            │
                            ▼ optional
                    rank_jobs_with_ollama()      [LLM re-rank]
                            │
                            ▼
                    _interactive_select()        [user picks jobs]
                            │
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

### Why HuggingFace models for ranking and language detection?

Specialized models outperform general-purpose LLMs on specific tasks. `all-MiniLM-L6-v2` produces semantic embeddings in milliseconds (vs. seconds for Ollama ranking), and `xlm-roberta-base-language-detection` accurately classifies 20 languages at the token level — something regex and Ollama both struggle with.

### Why Adzuna?

Covers DACH, France, Italy, and the UK with a single free API. Supports location-based search with radius. Most other free job APIs are US-only or require paid plans for European markets.

### Why SQLite for caching?

Zero-config, no server process, single-file database. WAL mode enables safe concurrent reads. The cache serves two purposes: avoiding redundant Ollama calls for unchanged CVs, and providing a job fallback when the API returns thin results.

### Why fpdf2 instead of wkhtmltopdf/WeasyPrint?

Pure Python, no system dependencies. WeasyPrint requires Cairo/Pango, wkhtmltopdf requires a binary install — both are fragile across OS/architecture. fpdf2 works everywhere Python runs.

### Why AI-generated search queries?

Job APIs perform keyword matching, not semantic search. Earlier heuristic approaches (extracting job titles from CVs, splitting compounds) produced too many generic terms. The AI query generator uses Ollama to produce 5-8 realistic job titles that a recruiter would recognize, dramatically improving search relevance.

### Why a self-learning language filter?

Regex patterns can't anticipate every way a job description states language requirements across German, French, Italian, and English. The AI validation loop catches edge cases and learns patterns it can reuse within the same run, reducing LLM calls on subsequent rounds.

### Why multi-layer language filtering?

No single method catches everything. HF classifiers detect posting language but miss English postings that require German. Regex catches "fluent German" but misses "Muttersprache Deutsch" if the pattern isn't pre-defined. Title-level keyword matching catches "German speaking" roles that both of the above miss. The AI validation loop catches whatever the other three miss. The layers complement each other.

### Why interactive job selection?

Not every ranked job is relevant — the user knows best. Instead of blindly generating 10 CVs (which takes several minutes of Ollama inference), the user picks exactly which ones they want. This saves time and produces only useful output.

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
| `HF_TOKEN` | HuggingFace Hub token (optional, suppresses download warnings) |
