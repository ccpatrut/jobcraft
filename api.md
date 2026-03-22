# Job Finder – API Reference & Rationale

This document describes each API used by the Job Finder application, with documentation links and the rationale for including it.

---

## Job Search APIs

### 1. Adzuna

**Rationale:** Adzuna is a major job aggregator with broad European coverage. It supports Germany (`de`), Austria (`at`), Switzerland (`ch`), France (`fr`), and Italy (`it`), and includes corporate and hospitality roles (cook, barista, etc.). The free tier (250 requests/day) is sufficient for personal use.

**Documentation:**
- [Developer portal & overview](https://developer.adzuna.com/overview)
- [Interactive API docs](https://developer.adzuna.com/activedocs)
- [Search endpoint](https://developer.adzuna.com/docs/search)
- [Regional/country setup](https://developer.adzuna.com/docs/regional)
- [Job categories](https://developer.adzuna.com/docs/categories)
- [Sign up for API keys](https://developer.adzuna.com/signup)

**Cost:** Free (25/min, 250/day, 1000/week, 2500/month)

---

### 2. Bundesagentur für Arbeit (Arbeitsagentur)

**Rationale:** Germany’s official job database and one of the largest in Europe. Includes all sectors (office, hospitality, crafts, services). Best option for German jobs. No signup required; uses a fixed client ID.

**Documentation:**
- [OpenAPI documentation](https://jobsuche.api.bund.dev/)
- [GitHub (bundesAPI)](https://github.com/bundesAPI/jobsuche-api)

**Base URL:** `https://rest.arbeitsagentur.de/jobboerse/jobsuche-service`

**Cost:** Free (no published limits)

---

### 3. Arbeitnow

**Rationale:** Free public API without authentication. Covers remote and European jobs from major ATS (Greenhouse, SmartRecruiters, etc.). Good complement for Europe and remote roles.

**Documentation:**
- [API docs (Postman)](https://documenter.getpostman.com/view/18545278/UVJbJdKh)
- [Overview & usage](https://www.arbeitnow.com/blog/job-board-api)

**Base endpoint:** `https://arbeitnow.com/api/job-board-api`

**Cost:** Free (no API key required)

---

## Local AI (Ollama)

**Rationale:** Runs LLMs locally so no cloud API costs and no data leaves your machine. Used for profile extraction from CVs, search query generation, language translation, AI language validation, optional re-ranking, and CV/cover letter generation.

**Documentation:**
- [Ollama docs](https://docs.ollama.com/)
- [API reference](https://ollama.readthedocs.io/)
- [Python client](https://github.com/ollama/ollama-python)

**Model used:** `qwen3:8b` (configurable via `ai.model` in `config.yaml`)

**Cost:** Free (local compute only)

---

## HuggingFace Models

### 4. sentence-transformers/all-MiniLM-L6-v2

**Rationale:** Fast, lightweight sentence embedding model for semantic similarity ranking. Produces dense vectors that allow cosine-similarity comparison between a candidate profile and job descriptions. Much faster and more accurate than LLM-based ranking for this task.

**Documentation:**
- [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [sentence-transformers library](https://www.sbert.net/)

**Size:** ~80MB (auto-downloads on first run)

**Cost:** Free (local inference)

---

### 5. papluca/xlm-roberta-base-language-detection

**Rationale:** Accurate multi-language text classifier fine-tuned on 20 languages. Used to detect the primary language of job postings so non-target-language jobs can be filtered out before ranking. Essential for the `english_only` mode in multi-lingual markets like Switzerland.

**Documentation:**
- [Model card](https://huggingface.co/papluca/xlm-roberta-base-language-detection)

**Size:** ~1.1GB (auto-downloads on first run)

**Cost:** Free (local inference)

---

## Summary

| API / Model    | Primary use                    | Auth required | Cost   |
|----------------|---------------------------------|---------------|--------|
| Adzuna         | Multi-country job search (DE, AT, CH, FR, IT) | Yes (app_id, app_key) | Free   |
| Arbeitsagentur | Germany jobs                   | Fixed client ID | Free   |
| Arbeitnow      | Europe & remote jobs            | No            | Free   |
| Ollama         | CV analysis, query generation, validation, CV/letter generation | No (local)  | Free   |
| all-MiniLM-L6-v2 | Semantic job ranking (embeddings) | No (local) | Free |
| xlm-roberta-base-language-detection | Job posting language detection | No (local) | Free |
