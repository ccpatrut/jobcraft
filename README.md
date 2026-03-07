# Job Finder

AI-powered job matching and tailored CV/cover letter generation. Uses **Ollama** for local AI (no cloud required) and the **Adzuna** API for job listings.

## What it does

1. **Reads your CVs** – Loads PDF and Word documents from a local directory
2. **Extracts your profile** – Uses Ollama to pull skills, experience, languages, certifications
3. **Fetches jobs** – Pulls listings from Adzuna API with multi-query, multi-lingual search
4. **Filters & ranks** – Regex + AI language validation, then AI-powered ranking by fit
5. **Generates output** – Creates 10 tailored CVs and 10 cover letters as Markdown + PDF

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

This creates a `.venv` and installs all dependencies. To run the app:

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

Generated CVs and cover letters will appear in `output/`.

## Configuration

Edit `config.yaml` to adjust:

| Setting | Description |
|---------|-------------|
| `cv_input_dir` | Where your CV files live |
| `output_dir` | Where generated files are saved |
| `preferences.tone` | `formal` \| `semi-formal` \| `casual` \| `enthusiastic` \| `professional` |
| `preferences.style` | `concise` \| `detailed` \| `balanced` |
| `job_search.country` | Adzuna country code (`gb`, `us`, `de`, etc.) |
| `ai.model` | Ollama model (e.g. `qwen3:8b`, `mistral`, `phi4`) |

## Project structure

```
job-finder/
├── cv_input/              # Drop your CVs here (PDF, Word)
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
    ├── job_matcher.py          # Language filter + AI ranking + AI validation
    ├── cv_generator.py         # Tailored CV generation (Ollama)
    ├── cover_letter_generator.py  # Cover letter generation (Ollama)
    ├── pdf_utils.py            # Markdown → PDF renderer (fpdf2)
    └── database.py             # SQLite caching layer
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
