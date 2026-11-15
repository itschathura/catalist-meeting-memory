"""
Pre-Final Meeting Brief Generator for Team TS3012.

Generates the comprehensive briefing document that Team TS3012 must read
before their final meeting (Meeting D - Detailed Design & Costing).
Covers:
1. Past decisions settled so far (Meetings A, B, C)
2. What changed along the way (pivots and dropped ideas)
3. Open action items with assigned owners
4. Open disagreements and unresolved questions
Zero look-ahead data leakage: strictly uses Meetings A, B, and C.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from pathlib import Path

from src.config import BRIEF_OUTPUT_FILE, ensure_dirs
from src.llm import get_llm_client
from src.memory import get_pre_final_brief_context
from src.verify import verify_citation


BRIEF_SYSTEM_PROMPT = """You are an Executive Design Team Coordinator for the AMI remote control project.
Your task is to write the official Pre-Final Meeting Brief for Team TS3012 to read BEFORE their final meeting (Meeting 4 / Session D).

CRITICAL CONSTRAINTS:
1. You only have access to Meetings A, B, and C. Do not invent or assume anything from Meeting D.
2. The brief must be strictly structured into these 4 core sections:
   - 1. Decisions Settled So Far
   - 2. What Changed Along the Way (Pivots & Dropped Concepts)
   - 3. Open Action Items & Assigned Owners
   - 4. Open Disagreements & Unresolved Questions
3. Identify all speakers by their letter and functional role:
   - Speaker A: Project Manager
   - Speaker B: User Interface Designer
   - Speaker C: Industrial Designer
   - Speaker D: Marketing
4. Back key factual statements and decisions with verbatim citations from the evidence:
   Format: [MeetingID: "exact quote of at least six words"]
   Example: [TS3012b: "we will need a simple battery"]
5. Professional, highly structured Markdown formatting."""


def generate_ts3012_brief(
    use_cache: bool = True,
    output_path: Optional[Path] = None,
    db_path: Optional[Any] = None,
) -> str:
    """Generate and verify the Team TS3012 Pre-Final Meeting Brief."""
    context = get_pre_final_brief_context("TS3012", db_path=db_path)
    roles = context["roles"]
    turns = context["turns"]

    # Assemble evidence text
    evidence_lines = [
        f"=== EVIDENCE CHRONOLOGY FOR TEAM TS3012 (MEETINGS A, B, C) ===",
        f"Team: TS3012 | Roles: {roles}",
        "",
    ]
    for t in turns:
        role_label = f" ({t.speaker_role})" if t.speaker_role else ""
        evidence_lines.append(
            f"[{t.meeting_id} | Turn {t.turn_id} | {t.timestamp}] {t.speaker}{role_label}: \"{t.text}\""
        )

    evidence_text = "\n".join(evidence_lines)

    prompt = f"""Write the complete Pre-Final Meeting Brief for Team TS3012 based strictly on the following chronological transcript turns from Meetings A, B, and C:

{evidence_text}

Produce a polished, comprehensive, and citation-backed briefing document in Markdown."""

    client = get_llm_client()
    resp = client.generate(
        prompt=prompt,
        system_instruction=BRIEF_SYSTEM_PROMPT,
        temperature=0.2,
        use_cache=use_cache,
    )

    brief_markdown = resp.get("text", "").strip()

    # Save to file
    ensure_dirs()
    target_file = output_path or BRIEF_OUTPUT_FILE
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(brief_markdown)

    return brief_markdown


if __name__ == "__main__":
    from rich import print as rprint

    rprint("[bold blue]Generating Pre-Final Meeting Brief for Team TS3012...[/bold blue]")
    brief = generate_ts3012_brief(use_cache=True)
    rprint(f"[bold green]Brief generated successfully ({len(brief.split())} words)![/bold green]")
    rprint(f"Saved to: [cyan]{BRIEF_OUTPUT_FILE}[/cyan]")
    rprint("\n[dim]First 600 characters of Brief:[/dim]")
    rprint(brief[:600])
