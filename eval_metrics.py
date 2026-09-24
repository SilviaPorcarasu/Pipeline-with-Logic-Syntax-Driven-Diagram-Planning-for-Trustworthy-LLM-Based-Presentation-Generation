#!/usr/bin/env python3
"""Compute lexical + syntactic diversity + chunk-grounding metrics.

Usage:
    python scripts/eval_metrics.py outputs/run1/scenario_plan.json [run2/...] [...]

Reads bullets from each `scenario_plan.json` and prints:
  - per-run Distinct-1/2/3
  - per-run grounding via NLI entailment + hallucination checks, evaluated
    slide-by-slide against the retrieved chunks found in sibling
    `retrieval.json`
  - cross-run self-BLEU (sacreBLEU) — only when >=2 runs are provided
  - cross-run mean dependency TED  — only when >=2 runs are provided

Output goes to stdout as a small JSON object so it can be piped into the
dissertation's results table.

Skip grounding (faster, no NLI model load) with:
    --no-grounding
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from pathlib import Path

# Make `metrics.py` at the repo root importable when this script is run
# from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics import (  # noqa: E402 — sys.path tweak above
    chunk_overlap_bleu,
    deck_distinct_ns,
    deck_grounding,
    diagram_set_llm_judge,
    deck_llm_judge,
    diagram_grounding,
    diagram_relation_sentences,
    pairwise_mean_ted,
    pairwise_self_bleu,
    pairwise_semantic_cosine,
    within_deck_diversity,
)


def _extract_bullets(scenario_plan_path: Path) -> list[str]:
    """Pull every bullet text from one scenario_plan.json into a flat list.

    The pipeline writes the plan in two shapes depending on stage:
      - list of slide entries at root (most common)
      - dict with `scenario` / `entries` key wrapping the list
    Handle both.
    """
    data = json.loads(scenario_plan_path.read_text())
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("scenario") or data.get("entries") or data.get("slides") or []
        if isinstance(entries, dict):
            entries = list(entries.values())
    else:
        entries = []
    if not isinstance(entries, list):
        return []
    bullets: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        scenario = entry.get("scenario_plan") if isinstance(entry.get("scenario_plan"), dict) else {}
        bp = scenario.get("bullet_plan") or entry.get("bullet_plan") or []
        if isinstance(bp, list):
            for it in bp:
                if isinstance(it, dict):
                    txt = it.get("bullet") or it.get("text") or it.get("line") or ""
                else:
                    txt = it
                txt = str(txt).strip()
                if txt:
                    bullets.append(txt)
    return bullets


def _extract_bullets_per_slide(scenario_plan_path: Path) -> list[list[str]]:
    """Return one list of bullets per slide (preserves slide grouping
    so we can pair each bullet with the chunks retrieved for ITS slide)."""
    data = json.loads(scenario_plan_path.read_text())
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("scenario") or data.get("entries") or data.get("slides") or []
        if isinstance(entries, dict):
            entries = list(entries.values())
    else:
        entries = []
    if not isinstance(entries, list):
        return []
    out: list[list[str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        scenario = entry.get("scenario_plan") if isinstance(entry.get("scenario_plan"), dict) else {}
        bp = scenario.get("bullet_plan") or entry.get("bullet_plan") or []
        slide_bullets: list[str] = []
        if isinstance(bp, list):
            for it in bp:
                if isinstance(it, dict):
                    txt = it.get("bullet") or it.get("text") or it.get("line") or ""
                else:
                    txt = it
                txt = str(txt).strip()
                if txt:
                    slide_bullets.append(txt)
        out.append(slide_bullets)
    return out


def _extract_text_levels(scenario_plan_path: Path) -> dict[str, dict]:
    """Extract three text streams per the advisor's diversity plan:

    1. `overall_scenario` — outline-wide narrative: teaching_scenario +
       structure_plan concatenated across all slides. One document per
       run that captures the deck's story arc.
    2. `slide_bullets` — every slide's bullets, kept slide-aligned so
       TED can pair bullets across runs.
    3. `speech_scripts` — every slide's `speech_plan` (the narrator
       script that accompanies the figure during a recorded
       presentation).

    Returns a dict like:
      {
        "overall_scenario": {"flat": [str], "per_slide": [str]},
        "slide_bullets":    {"flat": [str], "per_slide": [list[str]]},
        "speech_scripts":   {"flat": [str], "per_slide": [str]},
      }
    where `flat` is the list ready for Distinct-N and per_slide preserves
    the slide grouping for TED pairing.
    """
    data = json.loads(scenario_plan_path.read_text())
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("scenario") or data.get("entries") or data.get("slides") or []
        if isinstance(entries, dict):
            entries = list(entries.values())
    else:
        entries = []
    if not isinstance(entries, list):
        entries = []

    overall_per_slide: list[str] = []
    bullets_per_slide: list[list[str]] = []
    speech_per_slide: list[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        scenario = entry.get("scenario_plan") if isinstance(entry.get("scenario_plan"), dict) else {}
        # 1. Overall narrative — combine teaching_scenario + structure_plan
        teaching = str(scenario.get("teaching_scenario") or entry.get("teaching_scenario") or "").strip()
        structure = str(scenario.get("structure_plan") or entry.get("structure_plan") or "").strip()
        narrative = " ".join(s for s in (teaching, structure) if s)
        overall_per_slide.append(narrative)

        # 2. Bullets — same extraction as _extract_bullets_per_slide
        bp = scenario.get("bullet_plan") or entry.get("bullet_plan") or []
        slide_bullets: list[str] = []
        if isinstance(bp, list):
            for it in bp:
                if isinstance(it, dict):
                    txt = it.get("bullet") or it.get("text") or it.get("line") or ""
                else:
                    txt = it
                txt = str(txt).strip()
                if txt:
                    slide_bullets.append(txt)
        bullets_per_slide.append(slide_bullets)

        # 3. Speech / figure script
        speech = str(scenario.get("speech_plan") or entry.get("speech_plan")
                      or scenario.get("voiceover") or entry.get("voiceover") or "").strip()
        speech_per_slide.append(speech)

    return {
        "overall_scenario": {
            "flat": [s for s in overall_per_slide if s],
            "per_slide": overall_per_slide,
        },
        "slide_bullets": {
            "flat": [b for slide in bullets_per_slide for b in slide if b],
            "per_slide": bullets_per_slide,
        },
        "speech_scripts": {
            "flat": [s for s in speech_per_slide if s],
            "per_slide": speech_per_slide,
        },
    }


def _extract_diagram_descriptions_per_slide(scenario_plan_path: Path) -> list[str]:
    """Return one diagram description string per slide.

    Pulls `diagram_description` (Prompt 2.2 output, the natural-language
    captioning of the diagram) preferentially, falls back to
    `diagram_code` (D2 source, has node/edge labels as text) so the
    metric still works on older scenario plans.
    """
    data = json.loads(scenario_plan_path.read_text())
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("scenario") or data.get("entries") or data.get("slides") or []
        if isinstance(entries, dict):
            entries = list(entries.values())
    else:
        entries = []
    if not isinstance(entries, list):
        return []
    out: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            out.append("")
            continue
        scenario = entry.get("scenario_plan") if isinstance(entry.get("scenario_plan"), dict) else {}
        desc = (
            scenario.get("diagram_description")
            or entry.get("diagram_description")
            or scenario.get("diagram_code")
            or entry.get("diagram_code")
            or scenario.get("diagram_theme")
            or entry.get("diagram_theme")
            or ""
        )
        out.append(str(desc).strip())
    return out


def _extract_diagram_relations_per_slide(scenario_plan_path: Path) -> list[list[str]]:
    """Return one list of canonical relation sentences per slide."""
    descs = _extract_diagram_descriptions_per_slide(scenario_plan_path)
    return [diagram_relation_sentences(desc) for desc in descs]


def _extract_slide_descs(scenario_plan_path: Path) -> list[str]:
    """Return s_i per slide: `teaching_scenario` (falls back to `goal`)."""
    data = json.loads(scenario_plan_path.read_text())
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("scenario") or data.get("entries") or data.get("slides") or []
        if isinstance(entries, dict):
            entries = list(entries.values())
    else:
        entries = []
    if not isinstance(entries, list):
        return []
    out: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            out.append("")
            continue
        scenario = entry.get("scenario_plan") if isinstance(entry.get("scenario_plan"), dict) else {}
        desc = (
            scenario.get("teaching_scenario")
            or entry.get("teaching_scenario")
            or entry.get("goal")
            or ""
        )
        out.append(str(desc).strip())
    return out


def _extract_prologue_descs(scenario_plan_path: Path) -> list[str]:
    """Return d^Prologue_i per slide: `image_plan` (falls back to `diagram_description`)."""
    data = json.loads(scenario_plan_path.read_text())
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("scenario") or data.get("entries") or data.get("slides") or []
        if isinstance(entries, dict):
            entries = list(entries.values())
    else:
        entries = []
    if not isinstance(entries, list):
        return []
    out: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            out.append("")
            continue
        scenario = entry.get("scenario_plan") if isinstance(entry.get("scenario_plan"), dict) else {}
        desc = (
            scenario.get("image_plan")
            or entry.get("image_plan")
            or scenario.get("diagram_description")
            or entry.get("diagram_description")
            or ""
        )
        out.append(str(desc).strip())
    return out


def _extract_slides_for_judge(scenario_plan_path: Path) -> list[dict]:
    """Return per-slide dicts for llm_judge_score: title, goal, bullets, etc."""
    data = json.loads(scenario_plan_path.read_text())
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("scenario") or data.get("entries") or data.get("slides") or []
        if isinstance(entries, dict):
            entries = list(entries.values())
    else:
        entries = []
    if not isinstance(entries, list):
        return []
    out: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        scenario = entry.get("scenario_plan") if isinstance(entry.get("scenario_plan"), dict) else {}
        # Collect bullets from bullet_plan list
        bp = scenario.get("bullet_plan") or entry.get("bullet_plan") or []
        bullets = [str(b.get("bullet", b) if isinstance(b, dict) else b) for b in bp if b]
        out.append({
            "slide_number": entry.get("slide_number"),
            "title": str(entry.get("title") or scenario.get("slide_title") or ""),
            "goal": str(entry.get("goal") or scenario.get("teaching_scenario") or ""),
            "bullets": bullets,
            "diagram_description": str(
                scenario.get("diagram_description") or entry.get("diagram_description") or ""
            ),
            "rag_context": str(entry.get("rag_entries") or entry.get("evidence_excerpt") or ""),
        })
    return out


def _parse_rag_entries_text(rag_entries: str) -> list[str]:
    """Pull individual chunk excerpts from app.py's `rag_entries` format.

    The pipeline writes one line per chunk like:
      "- chunk_id=...; source=...; score=0.5234; excerpt=<chunk text>"
    Split by `excerpt=` and take everything until the next newline / line
    boundary. Returns one string per chunk, in retrieval order.
    """
    import re as _re

    chunks: list[str] = []
    for line in rag_entries.split("\n"):
        m = _re.search(r"excerpt=(.*)$", line)
        if m:
            txt = m.group(1).strip().rstrip("…").rstrip(".").strip()
            if txt:
                chunks.append(txt)
    return chunks


def _extract_chunks_per_slide(retrieval_path: Path) -> list[list[str]]:
    """Pull RAG chunk texts per slide from `retrieval.json`.

    Pipeline writes retrieval.json as a list of dicts shaped like:
      {
        "slide_number": 1,
        "rag_entries": "- chunk_id=...; ...; excerpt=<text>\\n- ...",
        "evidence_excerpt": "<combined chunk text>",
        "hits": [{"chunk_id": ..., "source": ..., "score": ...}, ...],
      }
    Earlier shapes (dict-keyed or list-of-{chunks: [...]}) are also
    tolerated as fallbacks.
    """
    data = json.loads(retrieval_path.read_text())
    if isinstance(data, dict):
        items = []
        for k in sorted(data.keys(), key=lambda x: (len(str(x)), str(x))):
            items.append(data[k])
        entries = items
    elif isinstance(data, list):
        entries = data
    else:
        return []

    out: list[list[str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            out.append([])
            continue
        # Preferred path — parse the formatted `rag_entries` string.
        rag_entries_text = entry.get("rag_entries")
        slide_chunks: list[str] = []
        if isinstance(rag_entries_text, str) and rag_entries_text.strip():
            slide_chunks = _parse_rag_entries_text(rag_entries_text)

        # Fallback: maybe the writer used a structured `chunks` field.
        if not slide_chunks:
            chunks_field = entry.get("chunks") or entry.get("retrieved") or entry.get("docs") or []
            if isinstance(chunks_field, list):
                for c in chunks_field:
                    if isinstance(c, dict):
                        txt = c.get("text") or c.get("excerpt") or c.get("chunk") \
                              or c.get("content") or c.get("body") or ""
                    else:
                        txt = c
                    txt = str(txt).strip()
                    if txt:
                        slide_chunks.append(txt)

        # Last resort: `evidence_excerpt` is a single combined string of
        # all chunks. Treat it as one chunk so grounding still works.
        if not slide_chunks:
            ev = entry.get("evidence_excerpt") or entry.get("evidence") or ""
            ev = str(ev).strip()
            if ev:
                slide_chunks = [ev]

        out.append(slide_chunks)
    return out


def _fmt(v: Any, ndigits: int = 4) -> str:
    """Render a metric value for the markdown table; falls back to '—' when missing."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def _build_markdown_report(summary: dict) -> str:
    """Pretty-print the metrics summary as Markdown tables for human review."""
    per_run = summary.get("per_run", [])
    cross = summary.get("cross_run", {})
    lines: list[str] = []
    lines.append("# Evaluation metrics report")
    lines.append("")
    lines.append(f"**Runs analysed:** {len(per_run)}")
    lines.append("")

    # --- Per-run diversity (Distinct-N) at three text levels ---
    lines.append("## Per-run diversity (Distinct-N)")
    lines.append("")
    lines.append("Higher = more lexical variety inside one deck. d1/d2/d3 = unique unigram/bigram/trigram ratio.")
    lines.append("")
    levels = ("overall_scenario", "slide_bullets", "speech_scripts")
    header = "| Run | Level | Segments | Distinct-1 | Distinct-2 | Distinct-3 |"
    sep = "|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for idx, run in enumerate(per_run, start=1):
        run_label = Path(run.get("scenario_plan", "")).parent.name or f"run{idx}"
        diversity = run.get("diversity", {}) or {}
        for lvl in levels:
            d = diversity.get(lvl) or {}
            lines.append(
                f"| {run_label} | {lvl} | {_fmt(d.get('n_segments'), 0)} | "
                f"{_fmt(d.get('distinct_1'))} | {_fmt(d.get('distinct_2'))} | {_fmt(d.get('distinct_3'))} |"
            )
    lines.append("")

    # --- Cross-run diversity (self-BLEU, dep-TED) ---
    if cross:
        lines.append("## Cross-run diversity")
        lines.append("")
        lines.append("Self-BLEU: lower = decks more different (good). Mean dep-TED: higher = more syntactic variation.")
        lines.append("")
        lines.append("| Level | Self-BLEU | Mean dep-TED |")
        lines.append("|---|---|---|")
        for lvl in levels:
            entry = cross.get(lvl) or {}
            lines.append(f"| {lvl} | {_fmt(entry.get('self_bleu'))} | {_fmt(entry.get('mean_dep_ted'), 2)} |")
        lines.append("")

    # --- LLM-as-judge correctness ---
    if any("llm_judge" in r for r in per_run):
        lines.append("## LLM-as-judge correctness")
        lines.append("")
        lines.append(
            "Scores from 0.0 (poor) to 1.0 (excellent). "
            "concept_coverage = key ideas present; factual_accuracy = no hallucinations; "
            "absence_of_repetition = diverse bullets; diagram_relevance = diagram matches topic."
        )
        lines.append("")
        lines.append("| Run | concept_coverage | factual_accuracy | absence_of_repetition | diagram_relevance | overall | slides |")
        lines.append("|---|---|---|---|---|---|---|")
        for idx, run in enumerate(per_run, start=1):
            run_label = Path(run.get("scenario_plan", "")).parent.name or f"run{idx}"
            j = run.get("llm_judge") or {}
            if "error" in j:
                lines.append(f"| {run_label} | error | error | error | error | error | — |")
            else:
                lines.append(
                    f"| {run_label} "
                    f"| {_fmt(j.get('concept_coverage'))} "
                    f"| {_fmt(j.get('factual_accuracy'))} "
                    f"| {_fmt(j.get('absence_of_repetition'))} "
                    f"| {_fmt(j.get('diagram_relevance'))} "
                    f"| {_fmt(j.get('overall'))} "
                    f"| {_fmt(j.get('n_slides_scored'), 0)} |"
                )
        lines.append("")

    # --- LLM-as-judge on concatenated diagram descriptions ---
    if any("llm_diagram_judge" in r for r in per_run):
        lines.append("## LLM-as-judge diagram set")
        lines.append("")
        lines.append(
            "Scores from 0.0 (poor) to 1.0 (excellent). "
            "diagram_conclusiveness = diagrams clearly communicate concepts and relations; "
            "diagram_coherence = diagrams form a coherent set; "
            "diagram_specificity = diagrams are specific rather than generic; "
            "diagram_groundedness = diagrams stay aligned with the shared slide goals."
        )
        lines.append("")
        lines.append("| Run | diagram_conclusiveness | diagram_coherence | diagram_specificity | diagram_groundedness | overall | diagrams |")
        lines.append("|---|---|---|---|---|---|---|")
        for idx, run in enumerate(per_run, start=1):
            run_label = Path(run.get("scenario_plan", "")).parent.name or f"run{idx}"
            j = run.get("llm_diagram_judge") or {}
            if "error" in j:
                lines.append(f"| {run_label} | error | error | error | error | error | — |")
            else:
                lines.append(
                    f"| {run_label} "
                    f"| {_fmt(j.get('diagram_conclusiveness'))} "
                    f"| {_fmt(j.get('diagram_coherence'))} "
                    f"| {_fmt(j.get('diagram_specificity'))} "
                    f"| {_fmt(j.get('diagram_groundedness'))} "
                    f"| {_fmt(j.get('overall'))} "
                    f"| {_fmt(j.get('n_diagrams_scored'), 0)} |"
                )
        lines.append("")

    # --- Grounding vs retrieved chunks (thesis formulas) ---
    if any("grounding" in r and "G_s_b" in (r.get("grounding") or {}) for r in per_run):
        lines.append("## Grounding vs retrieved chunks")
        lines.append("")
        lines.append(
            "NLI-based grounding computed slide-by-slide against the retrieved chunks of each slide. "
            "G_s_b: per-bullet best entailment; G_s_b_all: concatenated slide bullets; "
            "G_s_d: relation sentences extracted from diagram arrows. "
            "H_* denotes the corresponding hallucination rate (lower is better)."
        )
        lines.append("")
        lines.append("| Run | G_s_b | H_s_b | G_s_b_all | H_s_b_all | G_s_d | H_s_d |")
        lines.append("|---|---|---|---|---|---|---|")
        for idx, run in enumerate(per_run, start=1):
            run_label = Path(run.get("scenario_plan", "")).parent.name or f"run{idx}"
            g = run.get("grounding") or {}
            lines.append(
                f"| {run_label} "
                f"| {_fmt(g.get('G_s_b'))} "
                f"| {_fmt(g.get('H_s_b'))} "
                f"| {_fmt(g.get('G_s_b_all'))} "
                f"| {_fmt(g.get('H_s_b_all'))} "
                f"| {_fmt(g.get('G_s_d'))} "
                f"| {_fmt(g.get('H_s_d'))} |"
            )
        lines.append("")

    # --- Within-deck diversity ---
    if any("within_deck_diversity" in r for r in per_run):
        lines.append("## Within-deck diversity")
        lines.append("")
        lines.append(
            "Pairwise metrics over all items within a single presentation. "
            "Cosine dist: higher = more semantically diverse. "
            "Self-BLEU: lower = more lexically diverse. "
            "Mean dep-TED: higher = more syntactically diverse."
        )
        lines.append("")
        for stream in ("bullets", "diagrams"):
            lines.append(f"### {stream.capitalize()}")
            lines.append("")
            lines.append("| Run | Mean cos dist | Max cos dist | Std cos dist | Mean self-BLEU | Mean dep-TED | Max dep-TED | Pairs |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for idx, run in enumerate(per_run, start=1):
                run_label = Path(run.get("scenario_plan", "")).parent.name or f"run{idx}"
                d = (run.get("within_deck_diversity") or {}).get(stream) or {}
                lines.append(
                    f"| {run_label} "
                    f"| {_fmt(d.get('mean_cosine_dist'))} "
                    f"| {_fmt(d.get('max_cosine_dist'))} "
                    f"| {_fmt(d.get('std_cosine_dist'))} "
                    f"| {_fmt(d.get('mean_self_bleu'))} "
                    f"| {_fmt(d.get('mean_dep_ted'), 2)} "
                    f"| {_fmt(d.get('max_dep_ted'), 0)} "
                    f"| {_fmt(d.get('n_pairs'), 0)} |"
                )
            lines.append("")

    # --- Cross-run diagram diversity ---
    cross_diagrams = (summary.get("cross_run") or {}).get("diagrams")
    if cross_diagrams:
        lines.append("## Cross-run diagram diversity (V1 vs V2 vs V3)")
        lines.append("")
        lines.append("Self-BLEU / TED / cosine computed on diagram relation sentences across runs.")
        lines.append("")
        lines.append("| Self-BLEU | Mean dep-TED | Mean cosine dist |")
        lines.append("|---|---|---|")
        lines.append(
            f"| {_fmt(cross_diagrams.get('self_bleu'))} "
            f"| {_fmt(cross_diagrams.get('mean_dep_ted'), 2)} "
            f"| {_fmt(cross_diagrams.get('mean_cosine_dist'))} |"
        )
        lines.append("")

    # --- Per-run chunk-overlap BLEU (paraphrasing vs verbatim) ---
    if any("chunk_overlap" in r for r in per_run):
        lines.append("## Chunk-overlap BLEU (paraphrasing vs verbatim)")
        lines.append("")
        lines.append("Low BLEU = paraphrased (didactic). High BLEU = verbatim from source chunks. "
                     "`mean` averages per-slide BLEU; `max` is the worst-case slide (most copy-paste).")
        lines.append("")
        streams = ("slide_bullets", "diagram_relations", "speech_scripts", "overall_scenario")
        for stream in streams:
            lines.append(f"### {stream}")
            lines.append("")
            lines.append("| Run | Mean BLEU | Max BLEU | Slides |")
            lines.append("|---|---|---|---|")
            for idx, run in enumerate(per_run, start=1):
                run_label = Path(run.get("scenario_plan", "")).parent.name or f"run{idx}"
                co = (run.get("chunk_overlap") or {}).get(stream) or {}
                lines.append(
                    f"| {run_label} | {_fmt(co.get('mean_bleu_vs_chunks'))} | "
                    f"{_fmt(co.get('max_bleu_vs_chunks'))} | {_fmt(co.get('n_slides_scored'), 0)} |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def main(paths: list[str], with_grounding: bool = True, write_per_run: bool = True,
         markdown_path: str | None = None, json_path: str | None = None,
         llm_judge: bool = False, llm_provider: str = "anthropic",
         llm_judge_mode: str = "diagram_set",
         llm_model_id: str = "") -> int:
    decks: list[list[str]] = []
    decks_overall: list[list[str]] = []
    decks_speech: list[list[str]] = []
    decks_diagrams: list[list[str]] = []
    per_run: list[dict] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 1
        bullets = _extract_bullets(p)
        decks.append(bullets)

        levels = _extract_text_levels(p)
        decks_overall.append(levels["overall_scenario"]["flat"])
        decks_speech.append(levels["speech_scripts"]["flat"])

        run_entry: dict = {
            "scenario_plan": str(p),
            "bullet_count": len(bullets),
            "diversity": {
                "overall_scenario": {
                    "n_segments": len(levels["overall_scenario"]["flat"]),
                    **deck_distinct_ns(levels["overall_scenario"]["flat"]),
                },
                "slide_bullets": {
                    "n_segments": len(levels["slide_bullets"]["flat"]),
                    **deck_distinct_ns(levels["slide_bullets"]["flat"]),
                },
                "speech_scripts": {
                    "n_segments": len(levels["speech_scripts"]["flat"]),
                    **deck_distinct_ns(levels["speech_scripts"]["flat"]),
                },
            },
            **deck_distinct_ns(bullets),
        }

        if with_grounding:
            try:
                bullets_per_slide = _extract_bullets_per_slide(p)
                diagram_descs = _extract_diagram_descriptions_per_slide(p)
                diagram_relations_per_slide = _extract_diagram_relations_per_slide(p)
                diagram_texts_for_metrics = [
                    " ".join(rels) if rels else str(desc or "").strip()
                    for rels, desc in zip(diagram_relations_per_slide, diagram_descs)
                ]
                flat_diagram_texts = [d for d in diagram_texts_for_metrics if (d or "").strip()]
                decks_diagrams.append(flat_diagram_texts)

                grounding: dict = {}

                retrieval_path = p.parent / "retrieval.json"
                if retrieval_path.exists():
                    chunks_per_slide = _extract_chunks_per_slide(retrieval_path)
                    grounding["bullet_vs_chunks"] = deck_grounding(
                        bullets_per_slide, chunks_per_slide)
                    bullet_concat_per_slide = [[" ".join(bs).strip()] if any((b or "").strip() for b in bs) else [] for bs in bullets_per_slide]
                    grounding["bullet_concat_vs_chunks"] = deck_grounding(
                        bullet_concat_per_slide, chunks_per_slide)
                    grounding["diagram_vs_chunks"] = diagram_grounding(
                        diagram_descs, chunks_per_slide)
                    grounding["G_s_b"] = grounding["bullet_vs_chunks"].get("mean_entailment", 0.0)
                    grounding["G_s_b_all"] = grounding["bullet_concat_vs_chunks"].get("mean_entailment", 0.0)
                    grounding["G_s_d"] = grounding["diagram_vs_chunks"].get("mean_entailment", 0.0)
                    grounding["H_s_b"] = grounding["bullet_vs_chunks"].get("hallucination_rate", 0.0)
                    grounding["H_s_b_all"] = grounding["bullet_concat_vs_chunks"].get("hallucination_rate", 0.0)
                    grounding["H_s_d"] = grounding["diagram_vs_chunks"].get("hallucination_rate", 0.0)
                    slide_by_slide = []
                    bullet_slides = grounding["bullet_vs_chunks"].get("per_slide", []) or []
                    concat_slides = grounding["bullet_concat_vs_chunks"].get("per_slide", []) or []
                    diagram_slides = grounding["diagram_vs_chunks"].get("per_slide", []) or []
                    total_slides = max(
                        len(bullet_slides),
                        len(concat_slides),
                        len(diagram_slides),
                    )
                    for idx in range(total_slides):
                        bullet_slide = bullet_slides[idx] if idx < len(bullet_slides) else {}
                        concat_slide = concat_slides[idx] if idx < len(concat_slides) else {}
                        diagram_slide = diagram_slides[idx] if idx < len(diagram_slides) else {}
                        slide_by_slide.append({
                            "slide_number": idx + 1,
                            "G_s_b": bullet_slide.get("mean_entailment", 0.0),
                            "H_s_b": bullet_slide.get("hallucination_rate", 0.0),
                            "G_s_b_all": concat_slide.get("mean_entailment", 0.0),
                            "H_s_b_all": concat_slide.get("hallucination_rate", 0.0),
                            "G_s_d": diagram_slide.get("mean_entailment", 0.0),
                            "H_s_d": diagram_slide.get("hallucination_rate", 0.0),
                            "n_bullets": bullet_slide.get("n_texts_scored", 0.0),
                            "n_relations": diagram_slide.get("n_relations_scored", 0.0),
                        })
                    grounding["slide_by_slide"] = slide_by_slide
                    bullets_joined = [" / ".join(bs) for bs in bullets_per_slide]
                    diagram_relations_joined = [
                        " ".join(rels) if rels else str(desc or "").strip()
                        for rels, desc in zip(diagram_relations_per_slide, diagram_descs)
                    ]
                    speech_per_slide = levels["speech_scripts"]["per_slide"]
                    overall_per_slide = levels["overall_scenario"]["per_slide"]
                    run_entry["chunk_overlap"] = {
                        "slide_bullets": chunk_overlap_bleu(bullets_joined, chunks_per_slide),
                        "diagram_relations": chunk_overlap_bleu(diagram_relations_joined, chunks_per_slide),
                        "speech_scripts": chunk_overlap_bleu(speech_per_slide, chunks_per_slide),
                        "overall_scenario": chunk_overlap_bleu(overall_per_slide, chunks_per_slide),
                    }

                run_entry["grounding"] = grounding

                # Within-deck diversity: bullets vs bullets, diagrams vs diagrams
                flat_bullets = [b for slide in bullets_per_slide for b in slide if (b or "").strip()]
                flat_diagrams = flat_diagram_texts
                run_entry["within_deck_diversity"] = {
                    "bullets": within_deck_diversity(flat_bullets),
                    "diagrams": within_deck_diversity(flat_diagrams),
                }

            except Exception as exc:  # noqa: BLE001
                run_entry["grounding"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            decks_diagrams.append([])

        # LLM-as-judge correctness (opt-in via --llm-judge)
        if llm_judge:
            try:
                import os as _os
                from llm_service import LLMService  # noqa: E402
                _provider = llm_provider or _os.environ.get("LLM_PROVIDER", "anthropic")
                if llm_model_id:
                    _model_id = llm_model_id
                elif _provider == "anthropic":
                    _model_id = _os.environ.get("ANTHROPIC_JUDGE_MODEL_ID") or "claude-haiku-4-5-20251001"
                elif _provider == "openai":
                    _model_id = _os.environ.get("OPENAI_JUDGE_MODEL_ID") or "gpt-4.1-mini"
                else:
                    _model_id = _os.environ.get("GEMINI_JUDGE_MODEL_ID") or _os.environ.get("GEMINI_MODEL_ID") or "gemini-3-flash-preview"
                _llm = LLMService(provider=_provider, model_id=_model_id, max_new_tokens=400)
                slides_for_judge = _extract_slides_for_judge(p)
                if llm_judge_mode in {"slide", "both"}:
                    run_entry["llm_judge"] = deck_llm_judge(slides_for_judge, _llm)
                if llm_judge_mode in {"diagram_set", "both"}:
                    run_entry["llm_diagram_judge"] = diagram_set_llm_judge(slides_for_judge, _llm)
            except Exception as exc:  # noqa: BLE001
                if llm_judge_mode in {"slide", "both"}:
                    run_entry["llm_judge"] = {"error": f"{type(exc).__name__}: {exc}"}
                if llm_judge_mode in {"diagram_set", "both"}:
                    run_entry["llm_diagram_judge"] = {"error": f"{type(exc).__name__}: {exc}"}

        per_run.append(run_entry)

        if write_per_run:
            try:
                (p.parent / "metrics.json").write_text(
                    json.dumps(run_entry, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  warn: could not write metrics.json next to {p}: {exc}", file=sys.stderr)

    summary: dict = {"per_run": per_run}
    if len(decks) >= 2:
        summary["cross_run"] = {
            "slide_bullets": {
                "self_bleu": pairwise_self_bleu(decks),
                "mean_dep_ted": pairwise_mean_ted(decks),
                **pairwise_semantic_cosine(decks),
            },
            "overall_scenario": {
                "self_bleu": pairwise_self_bleu(decks_overall),
                "mean_dep_ted": pairwise_mean_ted(decks_overall),
                **pairwise_semantic_cosine(decks_overall),
            },
            "speech_scripts": {
                "self_bleu": pairwise_self_bleu(decks_speech),
                "mean_dep_ted": pairwise_mean_ted(decks_speech),
                **pairwise_semantic_cosine(decks_speech),
            },
            "diagrams": {
                "self_bleu": pairwise_self_bleu(decks_diagrams),
                "mean_dep_ted": pairwise_mean_ted(decks_diagrams),
                **pairwise_semantic_cosine(decks_diagrams),
            },
            # Top-level shortcuts for backwards compat
            "self_bleu": pairwise_self_bleu(decks),
            "mean_dep_ted": pairwise_mean_ted(decks),
            **pairwise_semantic_cosine(decks),
        }
    summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
    print(summary_json)

    # Optional: drop a JSON copy at a stable path so multiple invocations can
    # accumulate without copy/paste from stdout.
    if json_path:
        Path(json_path).expanduser().write_text(summary_json, encoding="utf-8")
        print(f"  wrote JSON summary: {json_path}", file=sys.stderr)

    # Markdown report — readable tables for the dissertation document.
    if markdown_path:
        md = _build_markdown_report(summary)
        Path(markdown_path).expanduser().write_text(md, encoding="utf-8")
        print(f"  wrote Markdown report: {markdown_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="+", help="scenario_plan.json paths")
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="skip the NLI-based slide-by-slide grounding metric",
    )
    parser.add_argument(
        "--no-per-run-file",
        action="store_true",
        help="do not drop a metrics.json next to each scenario_plan.json",
    )
    parser.add_argument(
        "--markdown",
        default="",
        help="path for a human-readable Markdown report (tables); skipped when empty",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="path for the aggregate JSON summary; default prints to stdout only",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="run LLM-as-judge scoring (requires API key)",
    )
    parser.add_argument(
        "--llm-provider",
        default="anthropic",
        help="provider for LLM judge: anthropic (default), gemini, openai",
    )
    parser.add_argument(
        "--llm-judge-mode",
        default="diagram_set",
        choices=("slide", "diagram_set", "both"),
        help="scope for LLM judge: diagram_set (default), slide, or both",
    )
    parser.add_argument(
        "--llm-model-id",
        default="",
        help="model ID for LLM judge (default: claude-haiku-4-5-20251001 for anthropic)",
    )
    args = parser.parse_args()
    sys.exit(main(
        args.scenarios,
        with_grounding=not args.no_grounding,
        write_per_run=not args.no_per_run_file,
        markdown_path=args.markdown or None,
        json_path=args.json_out or None,
        llm_judge=args.llm_judge,
        llm_provider=args.llm_provider,
        llm_judge_mode=args.llm_judge_mode,
        llm_model_id=args.llm_model_id,
    ))
