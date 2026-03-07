# Job Finder

AI-powered job matching and tailored CV/cover letter generation. Uses **Ollama** for local AI (no cloud required) and the **Adzuna** API for job listings.

## What it does

1. **Reads your CVs** – Drops PDF and Word documents from a directory
2. **Extracts your profile** – Uses Ollama to pull skills, experience, certifications
3. **Fetches jobs** – Pulls listings from Adzuna (free API)
4. **Ranks matches** – AI selects the top 5 jobs that fit you best
5. **Generates output** – Creates 5 tailored CVs and 5 cover letters (one per job)

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
ollama pull llama3.2
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
| `ai.model` | Ollama model (e.g. `llama3.2`, `mistral`, `llama2`) |

## Project structure

```
job-finder/
├── cv_input/          # ← Drop your CVs here
├── output/            # ← Generated CVs and letters
├── config.yaml        # User preferences
├── .env               # API keys (create from .env.example)
├── main.py            # Entry point
├── pyproject.toml     # Project config (uv/pip)
├── uv.lock            # Locked deps (uv)
└── src/
    ├── document_loader.py      # PDF/Word extraction
    ├── profile_extractor.py    # Ollama CV parsing
    ├── job_fetcher.py         # Adzuna API
    ├── job_matcher.py         # AI ranking
    ├── cv_generator.py        # Tailored CVs
    └── cover_letter_generator.py
```

## GitHub & CI

The project includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every push and pull request to `main` or `master`:

- **Lint** – Ruff check and format
- **Verify** – Imports, config loading, document loader

To push to GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/job-finder.git
git push -u origin main
```

Add API keys as **GitHub Secrets** (Settings → Secrets) if you run workflows that need them. The current CI does not require secrets (it only verifies imports and config).

## Troubleshooting

- **"Could not connect to Ollama"** – Run `ollama serve` and ensure a model is pulled (`ollama pull llama3.2`).
- **"Adzuna API credentials required"** – Add `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` to `.env`.
- **No documents found** – Ensure PDF or `.docx` files are in `cv_input/` (not `.doc`).
- **Slow responses** – Larger models are slower. Try `ollama pull llama3.2:1b` for a lighter model.
