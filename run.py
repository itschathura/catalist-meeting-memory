"""
Master Benchmark Runner for Catalist Meeting Memory.

Executes all 25 official benchmark questions from data/questions.json,
synthesizes answers with verbatim citations, embeds the Team TS3012 brief,
and produces the official submission file: answers.json.
"""

from __future__ import annotations

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.config import (
    QUESTIONS_FILE,
    ANSWERS_OUTPUT_FILE,
    BRIEF_OUTPUT_FILE,
    ensure_dirs,
)
from src.agent import answer_question
from src.brief import generate_ts3012_brief
from src.verify import validate_answers_file, verify_citation

console = Console()


def run_benchmark(
    use_cache: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run full benchmark on all 25 official questions and generate answers.json."""
    ensure_dirs()

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_FILE}")

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    if limit:
        questions = questions[:limit]

    console.print(f"\n[bold blue]Starting Benchmark Execution on {len(questions)} Questions...[/bold blue]\n")

    answers_list: List[Dict[str, Any]] = []
    latencies: List[float] = []

    table = Table(title="Catalist Benchmark Execution: Official 25 Questions")
    table.add_column("QID", style="cyan", width=6)
    table.add_column("Team", style="magenta", width=8)
    table.add_column("Citations", justify="center", width=10)
    table.add_column("Latency (s)", justify="right", width=12)
    table.add_column("Answer Preview", style="dim")

    start_all = time.time()

    for idx, q in enumerate(questions):
        q_id = q["question_id"]
        team = q["team"]
        q_text = q["question"]

        t0 = time.time()
        res = answer_question(q_id, team, q_text, use_cache=use_cache)
        latency = time.time() - t0
        latencies.append(latency)

        citations = res.get("citations", [])
        
        # Verify count
        valid_count = 0
        for c in citations:
            v_res = verify_citation(c["meeting_id"], c["quote"], snap=False)
            if v_res.valid:
                valid_count += 1

        cit_str = f"{valid_count}/{len(citations)}"
        preview = res.get("answer", "")[:60] + "..."
        table.add_row(q_id, team, cit_str, f"{latency:.2f}", preview)

        answers_list.append({
            "question_id": q_id,
            "answer": res.get("answer", ""),
            "citations": citations,
        })

    console.print(table)

    # 2. Ensure TS3012 brief is generated
    console.print("\n[bold blue]Loading Pre-Final Brief for Team TS3012...[/bold blue]")
    if not BRIEF_OUTPUT_FILE.exists():
        brief_text = generate_ts3012_brief(use_cache=use_cache)
    else:
        with open(BRIEF_OUTPUT_FILE, "r", encoding="utf-8") as f:
            brief_text = f.read()

    # 3. Assemble submission payload conforming to submission_format
    submission_payload = {
        "answers": answers_list,
        "brief": brief_text,
    }

    with open(ANSWERS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(submission_payload, f, indent=2)

    total_time = time.time() - start_all

    console.print(f"\n[bold green]Submission saved to:[/bold green] [cyan]{ANSWERS_OUTPUT_FILE}[/cyan]")

    # 4. Perform final compliance audit
    console.print("\n[bold yellow]Performing Automated Compliance Audit on answers.json...[/bold yellow]")
    audit_report = validate_answers_file(ANSWERS_OUTPUT_FILE)

    console.print(f"  Total Answers:            [cyan]{audit_report['total_answers']}[/cyan]")
    console.print(f"  Total Citations:          [cyan]{audit_report['total_citations']}[/cyan]")
    console.print(f"  Valid Citations:          [cyan]{audit_report['valid_citations']}[/cyan]")
    console.print(f"  Citation Validity Rate:   [bold green]{audit_report['validity_rate_pct']}%[/bold green]")
    console.print(f"  Total Execution Time:     [cyan]{total_time:.2f}s[/cyan]")
    console.print(f"  Average Question Latency: [cyan]{sum(latencies)/len(latencies):.2f}s[/cyan]\n")

    return audit_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Catalist Master Benchmark Runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--no-cache", action="store_true", help="Bypass LLM cache")
    args = parser.parse_args()

    run_benchmark(use_cache=not args.no_cache, limit=args.limit)
