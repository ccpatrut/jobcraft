# Job Finder

AI-powered job matching and tailored CV/cover letter generation. Uses **Ollama** for local AI, **HuggingFace** models for semantic ranking and language detection, and the **Adzuna** API for job listings. Everything runs locally — no cloud AI services needed.

## What it does

1. **Reads your CVs** – Loads PDF and Word documents from a local directory
2. **Extracts your profile** – Uses Ollama to pull skills, experience, languages, certifications
3. **Fetches jobs** – Pulls listings from Adzuna API with AI-generated multi-query, multi-lingual search
4. **Filters by language** – HF classifier removes non-target-language postings, title-language filter catches "German speaking" style jobs, regex catches explicit requirements, optional AI validation loop learns missed patterns
5. **Ranks by fit** – HuggingFace sentence embeddings for fast semantic similarity, optional Ollama re-ranking for nuance
6. **Interactive selection** – You choose which ranked jobs to generate documents for
7. **Generates output** – Creates tailored CVs and cover letters as Markdown + PDF

See [CONTRIBUTING.md](CONTRIBUTING.md) for a full architecture overview.

## Prerequisites

- **Python 3.11+** (via [uv](https://docs.astral.sh/uv/) or system)
- **Ollama** – [Download](https://ollama.com) and run locally
- **Adzuna API** – Free keys at [developer.adzuna.com](https://developer.adzuna.com/signup)

## Setup

### 1. Install uv (recommended)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or: pip install uv
```

### 2. Create environment and install dependencies

```bash
cd job-finder
uv sync
```

This creates a `.venv` and installs all dependencies (including PyTorch and HuggingFace models). To run the app:

```bash
uv run python main.py
```

**Alternative (Conda):** If you prefer Conda, use `environment.yml`:

```bash
conda env create -f environment.yml
conda activate job-finder
pip install -r requirements.txt
```

### 2. Install and run Ollama

```bash
# Download from https://ollama.com, then:
ollama pull qwen3:8b
ollama serve   # Usually runs automatically in background
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and add:
# ADZUNA_APP_ID=your_id
# ADZUNA_APP_KEY=your_key
```

### 4. Add your CVs

Put your CV, certifications, and experience documents (PDF or Word) in `cv_input/`:

```
cv_input/
├── my_cv.pdf
├── certifications.docx
└── project_summary.docx
```

### 5. Run

```bash
python main.py
```

On first run, HuggingFace models auto-download (~80MB embedder, ~1.1GB language detector). Subsequent runs are instant.

After ranking, you'll see an interactive prompt to select which jobs to generate CVs for. Generated CVs and cover letters appear in `output/`.

Use `--fresh` to force profile re-extraction (ignores cache):

```bash
python main.py --fresh
```

## Configuration

Edit `config.yaml` to adjust:

| Setting | Description |
|---------|-------------|
| `cv_input_dir` | Where your CV files live |
| `output_dir` | Where generated files are saved |
| `preferences.tone` | `formal` \| `semi-formal` \| `casual` \| `enthusiastic` \| `professional` |
| `preferences.style` | `concise` \| `detailed` \| `balanced` |
| `preferences.languages` | Override or merge CV languages (`languages_mode`: `override` \| `merge`) |
| `preferences.search_profile_summary` | Extra text used only for job matching/ranking (not CV generation) |
| `preferences.default_languages` | Fallback language proficiencies if not extracted from CV |
| `job_search.description_language_min_tier` | Min tier to keep non-English postings (`3` = B1, `4` = B2) |
| `job_search.min_similarity` | Embedding score cutoff (default `0.25` when career pivot enabled) |
| `job_search.search_nationwide` | `false` keeps search within `locations` + `radius_km` only |
| `job_search.country` | Adzuna country code (`ch`, `de`, `at`, `fr`, `it`, `gb`, `us`) |
| `job_search.locations` | List of cities/regions to search (e.g. `["Basel", "Zürich"]`) |
| `job_search.radius_km` | Search radius in km around each location |
| `job_search.language` | `auto` \| `english_only` \| `local_first` — controls search and filtering |
| `job_search.exclude_languages` | ISO codes to exclude (e.g. `["de"]` to skip German postings) |
| `job_search.ai_language_validation` | `true`/`false` — LLM re-checks for hidden language requirements |
| `ai.model` | Ollama model (e.g. `qwen3:8b`, `mistral`, `phi4`) |
| `ai.use_embeddings` | `true`/`false` — use HF sentence embeddings for ranking |
| `ai.rerank_with_llm` | `true`/`false` — refine embedding ranking with Ollama |

## Project structure

```
job-finder/
├── cv_input/              # Drop your CVs here (PDF, Word, .txt supplements)
├── output/                # Generated CVs and cover letters
├── config.yaml            # All user-facing configuration
├── .env                   # API keys (create from .env.example)
├── main.py                # Orchestrator / entry point
├── pyproject.toml         # Dependencies (uv/pip)
├── job_finder.db          # SQLite cache (auto-created)
└── src/
    ├── models.py               # Pydantic data models
    ├── config.py               # YAML + env config loader
    ├── document_loader.py      # PDF/Word text extraction
    ├── profile_extractor.py    # AI profile extraction (Ollama)
    ├── query_translator.py     # Multi-lingual query translation
    ├── job_fetcher.py          # Adzuna API client
    ├── job_matcher.py          # Language filters + AI ranking + AI validation
    ├── embeddings.py           # HuggingFace semantic similarity ranking
    ├── lang_detect.py          # HuggingFace language detection (xlm-roberta)
    ├── cv_generator.py         # Tailored CV generation (Ollama)
    ├── cover_letter_generator.py  # Cover letter generation (Ollama)
    ├── pdf_utils.py            # Markdown → PDF renderer (fpdf2)
    ├── database.py             # SQLite caching layer
    └── spinner.py              # Animated loading spinner for CLI
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR to `main`:

- **Lint** – Ruff check and format
- **Verify** – Import validation, config loading, document loader

The CI workflow does not require API secrets.

## Troubleshooting

- **"Could not connect to Ollama"** – Run `ollama serve` and ensure a model is pulled (`ollama pull qwen3:8b`).
- **"Adzuna API credentials required"** – Add `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` to `.env`.
- **No documents found** – Ensure PDF or `.docx` files are in `cv_input/` (not `.doc`).
- **Slow responses** – Larger models are slower. Try `ollama pull qwen3:8b` for a lighter model.
- **Ctrl+C not stopping** – Fixed: the app installs a SIGINT handler that terminates immediately, even during HuggingFace/PyTorch inference.
- **HF Hub warnings** – Set `HF_TOKEN` in `.env` to suppress "unauthenticated requests" warnings (optional, models download fine without it).
