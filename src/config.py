"""
Configuration and Path Management for Catalist Meeting Memory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Base Project Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Load environment variables from .env in project root if present
load_dotenv(PROJECT_ROOT / ".env")

# Key Directories
DATA_DIR = PROJECT_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
STORAGE_DIR = PROJECT_ROOT / "storage"
CACHE_DIR = STORAGE_DIR / "llm_cache"
EVAL_DIR = PROJECT_ROOT / "eval"

# Key Files
QUESTIONS_FILE = DATA_DIR / "questions.json"
DB_PATH = STORAGE_DIR / "meeting_memory.db"
METRICS_FILE = STORAGE_DIR / "llm_metrics.json"
ANSWERS_OUTPUT_FILE = PROJECT_ROOT / "answers.json"
BRIEF_OUTPUT_FILE = PROJECT_ROOT / "brief_ts3012.md"
FAILURES_OUTPUT_FILE = PROJECT_ROOT / "failures.md"
WRITEUP_OUTPUT_FILE = PROJECT_ROOT / "WRITEUP.md"

# Evaluation Files
MY_QUESTIONS_FILE = EVAL_DIR / "my_questions.json"

# API & LLM Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

# Verification & Task Constraints
MIN_CITATION_WORDS = int(os.getenv("MIN_CITATION_WORDS", "6"))
MAX_CITATIONS_PER_ANSWER = int(os.getenv("MAX_CITATIONS_PER_ANSWER", "3"))

# Benchmark Targets
BENCHMARK_BRIEF_TEAM = "TS3012"
BENCHMARK_FINAL_SESSION = "d"  # Final meeting to exclude from pre-final brief


def ensure_dirs() -> None:
    """Ensure runtime output and storage directories exist."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def check_environment() -> Dict[str, Any]:
    """Check readiness of data and API keys."""
    ensure_dirs()
    
    transcripts_exist = TRANSCRIPTS_DIR.exists()
    transcript_count = len(list(TRANSCRIPTS_DIR.glob("*.txt"))) if transcripts_exist else 0
    questions_exist = QUESTIONS_FILE.exists()
    api_key_set = bool(GEMINI_API_KEY)
    
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir_exists": DATA_DIR.exists(),
        "transcripts_exist": transcripts_exist,
        "transcript_count": transcript_count,
        "questions_file_exists": questions_exist,
        "storage_dir_ready": STORAGE_DIR.exists(),
        "gemini_api_key_set": api_key_set,
        "model": GEMINI_MODEL,
        "min_citation_words": MIN_CITATION_WORDS,
    }


if __name__ == "__main__":
    from rich import print as rprint
    rprint("[bold green]Catalist Meeting Memory Configuration:[/bold green]")
    for k, v in check_environment().items():
        rprint(f"  [cyan]{k}[/cyan]: {v}")
