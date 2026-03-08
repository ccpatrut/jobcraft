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

**Rationale:** Runs LLMs locally so no cloud API costs and no data leaves your machine. Used for profile extraction from CVs, job ranking, and CV/cover letter generation.

**Documentation:**
- [Ollama docs](https://docs.ollama.com/)
- [API reference](https://ollama.readthedocs.io/)
- [Python client](https://github.com/ollama/ollama-python)

**Cost:** Free (local compute only)

---

## Summary

| API            | Primary use                    | Auth required | Cost   |
|----------------|---------------------------------|---------------|--------|
| Adzuna         | Multi-country job search (DE, AT, CH, FR, IT) | Yes (app_id, app_key) | Free   |
| Arbeitsagentur | Germany jobs                   | Fixed client ID | Free   |
| Arbeitnow     | Europe & remote jobs            | No            | Free   |
| Ollama        | CV analysis, matching, generation | No (local)  | Free   |
