#!/usr/bin/env python3
"""
JobCraft - AI-powered job matching and tailored CV/cover letter generation.

Usage:
  python main.py run [--fresh]              # full pipeline
  python main.py jobs list [--status new]   # browse cached jobs
  python main.py jobs search "python"       # search the cache
  python main.py profile show               # display extracted profile
  python main.py generate <job_url>         # regenerate for one job
  python main.py status <job_url> applied   # update job status

Ensure Ollama is running locally (ollama serve) and you have pulled a model:
  ollama pull qwen3:8b
"""

import os
import signal
import sys
from pathlib import Path


def _force_exit(signum, frame):
    """Handle Ctrl+C immediately even during native C code (HF/PyTorch inference)."""
    print("\n  Interrupted — exiting.", flush=True)
    os._exit(130)


signal.signal(signal.SIGINT, _force_exit)

# Add project root to path so `from src.*` works when invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import app

if __name__ == "__main__":
    app()
