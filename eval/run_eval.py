"""
Automated Evaluation Suite for Catalist Meeting Memory.

Evaluates:
1. Citation Validity Rate (% verbatim quotes in single speaker turns)
2. Latency per question (seconds)
3. Token usage and estimated cost economics
4. Completeness and format compliance
"""

from __future__ import annotations

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any

from rich.console import Console
from rich.table import Table

from src.config import QUESTIONS_FILE, MY_QUESTIONS_FILE, EVAL_DIR, ensure_dirs
from src.agent import answer_question
from src.verify import verify_citation
from src.llm import get_llm_client

console = Console()


def evaluate_dataset(
    dataset_path: Path,
    max_questions: Optional[int] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation on a questions dataset."""
    ensure_dirs()
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Handle both dict-based (official questions.json) and list-based (my_questions.json)
    if isinstance(raw_data, dict) and "questions" in raw_data:
        questions = raw_data["questions"]
    elif isinstance(raw_data, list):
        questions = raw_data
    else:
        raise ValueError(f"Unrecognized dataset format in {dataset_path}")

    if max_questions:
        questions = questions[:max_questions]

    console.print(f"\n[bold blue]Running Evaluation on {len(questions)} questions from {dataset_path.name}...[/bold blue]\n")

    results: List[Dict[str, Any]] = []
    total_citations = 0
    valid_citations = 0
    latencies: List[float] = []

    start_eval_time = time.time()

    table = Table(title=f"Evaluation Results: {dataset_path.name}")
    table.add_column("QID", style="cyan", width=6)
    table.add_column("Team", style="magenta", width=8)
    table.add_column("Status", style="bold", width=8)
    table.add_column("Citations", justify="center", width=10)
    table.add_column("Latency (s)", justify="right", width=12)
    table.add_column("Answer Preview", style="dim")

    for q in questions:
        q_id = q["question_id"]
        team = q["team"]
        q_text = q["question"]

        t0 = time.time()
        ans_data = answer_question(q_id, team, q_text, use_cache=use_cache)
        latency = time.time() - t0
        latencies.append(latency)

        citations = ans_data.get("citations", [])
        q_valid_cits = 0
        for c in citations:
            total_citations += 1
            res = verify_citation(c["meeting_id"], c["quote"], snap=False)
            if res.valid:
                valid_citations += 1
                q_valid_cits += 1

        status = "[green]PASS[/green]" if (len(citations) > 0 and q_valid_cits == len(citations)) else "[red]FAIL[/red]"
        cit_display = f"{q_valid_cits}/{len(citations)}"

        preview = ans_data.get("answer", "")[:60] + "..."
        table.add_row(q_id, team, status, cit_display, f"{latency:.2f}", preview)

        results.append({
            "question_id": q_id,
            "team": team,
            "question": q_text,
            "answer": ans_data.get("answer"),
            "citations": citations,
            "latency_sec": round(latency, 2),
            "citations_valid": (q_valid_cits == len(citations)),
        })

    total_eval_elapsed = time.time() - start_eval_time
    citation_validity_pct = (valid_citations / total_citations * 100) if total_citations > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    console.print(table)

    # Fetch token and cost economics from LLM client
    client = get_llm_client()
    metrics = client.get_metrics_summary()

    summary = {
        "dataset": dataset_path.name,
        "total_questions": len(questions),
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "citation_validity_rate_pct": round(citation_validity_pct, 2),
        "average_latency_sec": round(avg_latency, 2),
        "total_elapsed_sec": round(total_eval_elapsed, 2),
        "cumulative_tokens": metrics["total_tokens"],
        "estimated_cost_usd": metrics["estimated_cost_usd"],
        "results": results,
    }

    # Save evaluation report
    output_report = EVAL_DIR / f"eval_report_{dataset_path.stem}.json"
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    console.print("\n[bold green]Evaluation Summary:[/bold green]")
    console.print(f"  Total Questions:          [cyan]{summary['total_questions']}[/cyan]")
    console.print(f"  Citation Validity Rate:   [bold green]{summary['citation_validity_rate_pct']}%[/bold green]")
    console.print(f"  Average Latency:          [cyan]{summary['average_latency_sec']}s[/cyan]")
    console.print(f"  Cumulative Cost:          [cyan]${summary['estimated_cost_usd']}[/cyan]")
    console.print(f"  Report Saved To:          [dim]{output_report}[/dim]\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Catalist Evaluation Suite")
    parser.add_argument("--my-questions", action="store_true", help="Evaluate self-generated 10 benchmark questions")
    parser.add_argument("--official", action="store_true", help="Evaluate official 25 benchmark questions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--no-cache", action="store_true", help="Bypass LLM cache")
    args = parser.parse_args()

    target_file = MY_QUESTIONS_FILE if args.my_questions else QUESTIONS_FILE
    evaluate_dataset(target_file, max_questions=args.limit, use_cache=not args.no_cache)
