"""Load configuration from config.yaml and environment."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .models import UserPreferences

load_dotenv()


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load YAML config, with env overrides."""
    path = Path(config_path)
    if not path.exists():
        return _default_config()

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    ai = cfg.setdefault("ai", {})

    # Env overrides
    if os.environ.get("CV_INPUT_DIR"):
        cfg["cv_input_dir"] = os.environ["CV_INPUT_DIR"]
    if os.environ.get("OUTPUT_DIR"):
        cfg["output_dir"] = os.environ["OUTPUT_DIR"]
    if os.environ.get("JOB_COUNTRY"):
        cfg.setdefault("job_search", {})["country"] = os.environ["JOB_COUNTRY"]
    if os.environ.get("OLLAMA_HOST"):
        ai["host"] = os.environ["OLLAMA_HOST"]
    if os.environ.get("OLLAMA_MODEL"):
        ai["model"] = os.environ["OLLAMA_MODEL"]

    return cfg


def _default_config() -> dict:
    return {
        "cv_input_dir": "./cv_input",
        "output_dir": "./output",
        "preferences": {
            "tone": "professional",
            "style": "balanced",
            "focus_areas": [],
        },
        "job_search": {"country": "gb", "max_results": 50, "keywords": ""},
        "ai": {
            "provider": "auto",
            "model": "qwen3:8b",
            "gemini_model": "gemini-2.5-flash",
            "temperature": 0.7,
            "timeout": 120,
        },
    }


def get_preferences(cfg: dict) -> UserPreferences:
    """Extract user preferences from config."""
    prefs = cfg.get("preferences", {})
    pivot = prefs.get("career_pivot", {})
    return UserPreferences(
        tone=prefs.get("tone", "professional"),
        style=prefs.get("style", "balanced"),
        focus_areas=prefs.get("focus_areas", []) or [],
        pivot_enabled=pivot.get("enabled", False),
        pivot_from=pivot.get("from_industries", []) or [],
        pivot_motivation=pivot.get("motivation", "") or "",
    )
