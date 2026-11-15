"""
Interactive Live Demo CLI for Catalist Interview.

Allows live querying of meeting memory for any team and arbitrary question.
Displays chronological turns, synthesized answer, and verbatim verified citations.
"""

from __future__ import annotations

import time
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent import answer_question
from src.verify import verify_citation
from src.memory import get_team_roles

console = Console()


def run_live_query(team: str, question: str) -> None:
    """Execute a live question with visual UI output."""
    team_clean = team.upper().strip()
    
    console.print(Panel(
        f"[bold white]Team:[/bold white] [bold cyan]{team_clean}[/bold cyan]\n[bold white]Question:[/bold white] {question}",
        title="[bold yellow]Catalist Meeting Memory - Live Query[/bold yellow]",
        border_style="blue"
    ))

    # 1. Show detected roles for team
    roles = get_team_roles(team_clean)
    role_str = " | ".join([f"[cyan]{spk}[/cyan]: [dim]{r or 'Participant'}[/dim]" for spk, r in roles.items()])
    console.print(f"[bold]Detected Team Roles:[/bold] {role_str}\n")

    # 2. Run Agent
    console.print("[dim]Retrieving chronological evidence and synthesizing answer...[/dim]")
    t0 = time.time()
    result = answer_question("LIVE_DEMO", team_clean, question, use_cache=False)
    elapsed = time.time() - t0

    # 3. Display Answer
    console.print(Panel(
        f"[bold green]{result['answer']}[/bold green]",
        title=f"[bold]Answer (Generated in {elapsed:.2f}s)[/bold]",
        border_style="green"
    ))

    # 4. Display Citations Table
    citations = result.get("citations", [])
    table = Table(title=f"Verbatim Citations ({len(citations)} provided)")
    table.add_column("Meeting", style="cyan", width=12)
    table.add_column("Turn / Speaker", style="magenta", width=18)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Verbatim Quote", style="white")

    for c in citations:
        m_id = c["meeting_id"]
        q = c["quote"]
        v_res = verify_citation(m_id, q, snap=False)
        status = "[bold green]VERIFIED[/bold green]" if v_res.valid else "[bold red]INVALID[/bold red]"
        speaker_info = f"Turn {v_res.turn_id or '?'} ({v_res.speaker or '?'})"
        table.add_row(m_id, speaker_info, status, f"\"{q}\"")

    console.print(table)
    console.print("\n")


def interactive_mode():
    """Run in interactive loop."""
    console.print("[bold green]=== Welcome to Catalist Meeting Memory Live Demo ===[/bold green]")
    console.print("[dim]Type 'exit' or press Ctrl+C to quit.[/dim]\n")

    while True:
        try:
            team = console.input("[bold cyan]Enter Team ID (e.g. TS3007, TS3006, ES2002): [/bold cyan]").strip()
            if team.lower() in ["exit", "quit", "q"]:
                break
            if not team:
                continue

            question = console.input("[bold cyan]Enter your question: [/bold cyan]").strip()
            if question.lower() in ["exit", "quit", "q"]:
                break
            if not question:
                continue

            run_live_query(team, question)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting Live Demo. Good luck with the interview![/dim]")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Catalist Live Demo CLI")
    parser.add_argument("--team", type=str, default=None, help="Team ID (e.g. TS3007)")
    parser.add_argument("--q", type=str, default=None, help="Question text")
    args = parser.parse_args()

    if args.team and args.q:
        run_live_query(args.team, args.q)
    else:
        interactive_mode()
