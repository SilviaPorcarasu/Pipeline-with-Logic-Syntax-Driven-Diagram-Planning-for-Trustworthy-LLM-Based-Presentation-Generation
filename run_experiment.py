#!/usr/bin/env python3
"""
run_experiment.py

Runs all 3 diagram-concept variants (V1 / V2-spaCy / V3-LLM) from the
same shared global plan and slide detail for each evaluation topic.

Output structure (compatible with scripts/eval_metrics.py):
  experiments/<topic_slug>/
    shared/
      outline.json
      retrieval.json        ← grounding source for eval_metrics.py
      global_plan.json
    v1/
      scenario_plan.json
      retrieval.json        ← copy of shared/
    v2/
      scenario_plan.json
      retrieval.json
      prolog_kb.json        ← spaCy Prolog facts/rules per slide
    v3/
      scenario_plan.json
      retrieval.json
      prolog_kb.json        ← LLM Prolog facts/rules per slide

Usage:
  python run_experiment.py
  python run_experiment.py --topics "Decision Trees" "Backpropagation in Neural Networks"
  python run_experiment.py --slides 8 --out-dir experiments
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from config import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_CONTENT_MAX_BULLETS,
    DEFAULT_CONTENT_MIN_BULLETS,
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_NONCONTENT_MAX_BULLETS,
    DEFAULT_NONCONTENT_MIN_BULLETS,
    DEFAULT_RAG_CONTEXT_MAX_CHARS,
    DEFAULT_RAG_MODE,
    DEFAULT_RAG_TOP_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from llm_service import (
    LLMService,
    SemanticRetriever,
    build_evidence_text,
    load_chunks,
    make_slide_query,
    retrieve_chunks,
)
from models import PresentationRequest
from prompts import (
    diagram_concept_prompt,
    diagram_concept_prompt_v2,
    global_plan_prompt,
    outline_prompt,
    slide_detail_prompt,
)

TOPICS = [
    "Decision Trees",
    "Backpropagation in Neural Networks",
    "Bayesian Classification",
    "K-Nearest Neighbor Classification",
]

SLIDES_PER_TOPIC = 8
OUT_BASE = Path("experiments")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def _rag_entries_text(hits, *, max_entries: int = 5, excerpt_chars: int = 220) -> str:
    rows = []
    for h in hits[: max(1, max_entries)]:
        source = _norm(getattr(h, "source", "") or "")
        excerpt = _norm(getattr(h, "text", "") or "")
        if len(excerpt) > excerpt_chars:
            excerpt = excerpt[:excerpt_chars].rsplit(" ", 1)[0].rstrip() + "..."
        rows.append(
            _norm(
                f"- chunk_id={getattr(h, 'chunk_id', '')}; source={source}; "
                f"score={float(getattr(h, 'score', 0.0)):.4f}; excerpt={excerpt}"
            )
        )
    return "\n".join(rows)


def _bullet_constraints(item_type: str) -> tuple[int, int]:
    if item_type == "content":
        lo = max(1, DEFAULT_CONTENT_MIN_BULLETS)
        hi = min(3, max(lo, DEFAULT_CONTENT_MAX_BULLETS))
    else:
        lo = max(1, DEFAULT_NONCONTENT_MIN_BULLETS)
        hi = min(3, max(lo, DEFAULT_NONCONTENT_MAX_BULLETS))
    return lo, hi


def run_topic(
    topic: str,
    llm: LLMService,
    chunks: list,
    semantic_retriever,
    out_dir: Path,
    slides_count: int = SLIDES_PER_TOPIC,
) -> None:
    print(f"\n{'='*60}")
    print(f"TOPIC: {topic}")
    print(f"{'='*60}")

    req = PresentationRequest(
        topic=topic,
        slides_count=slides_count,
        style="academic",
        include_images=False,
        include_voiceover=False,
    )

    # Step 1: Outline (once)
    print("  [1/4] Outline...")
    outline = llm.generate_outline(req, outline_prompt(req))
    if len(outline) != req.slides_count:
        print(f"    adjusted: {req.slides_count} → {len(outline)} slides")
        req.slides_count = len(outline)

    # Step 2: RAG retrieval per slide (once)
    print("  [2/4] RAG retrieval...")
    hits_by_slide: dict[int, list] = {}
    evidence_by_slide: dict[int, str] = {}
    rag_entries_by_slide: dict[int, str] = {}
    retrieval_manifest: list[dict] = []

    for item in outline:
        query = make_slide_query(req, item)
        hits = retrieve_chunks(
            chunks=chunks, query=query, top_k=DEFAULT_RAG_TOP_K,
            rag_mode=DEFAULT_RAG_MODE, semantic_retriever=semantic_retriever,
        )
        if not hits:
            hits = retrieve_chunks(
                chunks=chunks, query=topic, top_k=DEFAULT_RAG_TOP_K,
                rag_mode=DEFAULT_RAG_MODE, semantic_retriever=semantic_retriever,
            )
        hits_by_slide[item.slide_number] = hits
        evidence_by_slide[item.slide_number] = build_evidence_text(
            hits, max_chars=DEFAULT_RAG_CONTEXT_MAX_CHARS
        )
        rag_entries_by_slide[item.slide_number] = _rag_entries_text(hits)
        retrieval_manifest.append({
            "slide_number": item.slide_number,
            "title": item.title,
            "query": query,
            "rag_entries": rag_entries_by_slide[item.slide_number],
            "evidence_excerpt": evidence_by_slide[item.slide_number],
            "hits": [
                {"chunk_id": h.chunk_id, "source": h.source, "score": round(float(h.score), 6)}
                for h in hits
            ],
        })

    # Step 3: Global plan (once)
    print("  [3/4] Global plan...")
    rag_summary = "\n".join(
        f"Slide {item.slide_number}: {_norm(rag_entries_by_slide.get(item.slide_number, '')[:200])}"
        for item in outline
    )
    gplan = llm.generate_global_plan(global_plan_prompt(req, outline, rag_summary))
    gplan_text = "\n".join(
        f"  Slide {p.get('slide_number', '?')}: "
        f"idea={p.get('main_idea', '')} | "
        f"diagram={p.get('diagram_theme', '')} | "
        f"progression={p.get('progression', '')}"
        for p in gplan
    )

    # Save shared artefacts
    shared_dir = out_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "outline.json").write_text(
        json.dumps([asdict(item) for item in outline], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    shared_retrieval = shared_dir / "retrieval.json"
    shared_retrieval.write_text(
        json.dumps(retrieval_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (shared_dir / "global_plan.json").write_text(
        json.dumps(gplan, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Step 4: Per-slide detail (once) + V1/V2/V3 concepts (3x)
    print("  [4/4] Slide detail + V1/V2/V3 diagram concepts...")

    v1_manifest: list[dict] = []
    v2_manifest: list[dict] = []
    v3_manifest: list[dict] = []
    v2_prolog_kb: dict[int, dict] = {}
    v3_prolog_kb: dict[int, dict] = {}

    for item in outline:
        sn = item.slide_number
        rag_entries = rag_entries_by_slide.get(sn, "")
        evidence_text = evidence_by_slide.get(sn, "")
        hits = hits_by_slide.get(sn, [])
        hit_rows = [
            {"chunk_id": h.chunk_id, "source": h.source, "score": round(float(h.score), 6)}
            for h in hits
        ]

        min_b, max_b = _bullet_constraints(item.type)
        plan_entry = next(
            (p for p in gplan if int(p.get("slide_number", 0) or 0) == sn), {}
        )

        # Shared slide detail — Prompt 2.1
        print(f"    Slide {sn}: detail...", end=" ", flush=True)
        detail_prompt = slide_detail_prompt(
            req, item,
            global_plan_text=gplan_text,
            rag_entries_text=rag_entries,
            min_bullets=min_b,
            max_bullets=max_b,
        )
        slide_detail = llm.generate_slide_detail(
            detail_prompt, min_bullets=min_b, max_bullets=max_b
        )

        diagram_theme = slide_detail.get("diagram_theme", "") or f"Diagram for {item.title}"
        bullets_text = "\n".join(
            f"- {b.get('bullet', '')}" for b in slide_detail.get("bullet_plan", [])
        )
        diagram_signature = str(plan_entry.get("diagram_signature", "") or "").strip().lower()

        def _entry(mode: str, desc: str, code: str, facts: list, rules: list) -> dict:
            plan = dict(slide_detail)
            plan["diagram_description"] = desc
            plan["diagram_code"] = code
            plan["image_plan"] = desc
            return {
                "slide_number": sn,
                "slide_type": item.type,
                "title": item.title,
                "goal": item.goal,
                "query": make_slide_query(req, item),
                "bullet_constraints": {"min": min_b, "max": max_b},
                "rag_entries": rag_entries,
                "retrieval_hits": hit_rows,
                "evidence_excerpt": evidence_text,
                "diagram_concept_mode": mode,
                "scenario_plan": plan,
                "prolog_facts": facts,
                "prolog_rules": rules,
            }

        # V1 — direct LLM diagram concept
        print("V1", end=" ", flush=True)
        cp_v1 = diagram_concept_prompt(
            sn, item.title, diagram_theme, bullets_text, rag_entries,
            diagram_signature=diagram_signature,
        )
        desc_v1 = llm.generate_diagram_concept(cp_v1)
        code_v1 = llm._format_diagram_code(
            slide_number=sn,
            teaching_scenario=slide_detail.get("teaching_scenario", ""),
            structure_plan=slide_detail.get("structure_plan", ""),
            bullet_plan=slide_detail.get("bullet_plan", []),
            diagram_description=desc_v1, diagram_code="", image_plan="",
            slide_title=item.title, diagram_theme=diagram_theme,
            rag_entries_text=rag_entries,
        )
        v1_manifest.append(_entry("v1", desc_v1, code_v1, [], []))

        # V2 — spaCy → Prolog → LLM diagram concept
        print("V2", end=" ", flush=True)
        from prolog_extractor import extract_prolog
        facts_v2, rules_v2 = extract_prolog(
            item.title + ". " + bullets_text.replace("- ", "")
        )
        cp_v2 = diagram_concept_prompt_v2(
            sn, item.title, diagram_theme, bullets_text, rag_entries,
            prolog_facts=facts_v2, prolog_rules=rules_v2,
            diagram_signature=diagram_signature,
        )
        desc_v2 = llm.generate_diagram_concept(cp_v2)
        code_v2 = llm._format_diagram_code(
            slide_number=sn,
            teaching_scenario=slide_detail.get("teaching_scenario", ""),
            structure_plan=slide_detail.get("structure_plan", ""),
            bullet_plan=slide_detail.get("bullet_plan", []),
            diagram_description=desc_v2, diagram_code="", image_plan="",
            slide_title=item.title, diagram_theme=diagram_theme,
            rag_entries_text=rag_entries,
        )
        v2_manifest.append(_entry("v2_spacy", desc_v2, code_v2, facts_v2, rules_v2))
        v2_prolog_kb[sn] = {"facts": facts_v2, "rules": rules_v2, "diagram_desc": desc_v2}

        # V3 — LLM → Prolog → LLM diagram concept
        print("V3", flush=True)
        facts_v3, rules_v3 = llm.generate_prolog_from_text(
            item.title + ". " + bullets_text.replace("- ", "")
        )
        cp_v3 = diagram_concept_prompt_v2(
            sn, item.title, diagram_theme, bullets_text, rag_entries,
            prolog_facts=facts_v3, prolog_rules=rules_v3,
            diagram_signature=diagram_signature,
        )
        desc_v3 = llm.generate_diagram_concept(cp_v3)
        code_v3 = llm._format_diagram_code(
            slide_number=sn,
            teaching_scenario=slide_detail.get("teaching_scenario", ""),
            structure_plan=slide_detail.get("structure_plan", ""),
            bullet_plan=slide_detail.get("bullet_plan", []),
            diagram_description=desc_v3, diagram_code="", image_plan="",
            slide_title=item.title, diagram_theme=diagram_theme,
            rag_entries_text=rag_entries,
        )
        v3_manifest.append(_entry("v3_llm", desc_v3, code_v3, facts_v3, rules_v3))
        v3_prolog_kb[sn] = {"facts": facts_v3, "rules": rules_v3, "diagram_desc": desc_v3}

    # Save per-variant outputs
    for label, manifest, prolog_kb in [
        ("v1", v1_manifest, {}),
        ("v2", v2_manifest, v2_prolog_kb),
        ("v3", v3_manifest, v3_prolog_kb),
    ]:
        vdir = out_dir / label
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "scenario_plan.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        shutil.copy2(shared_retrieval, vdir / "retrieval.json")
        if prolog_kb:
            (vdir / "prolog_kb.json").write_text(
                json.dumps(
                    {str(k): v for k, v in prolog_kb.items()},
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    print(f"  Saved: {out_dir}/{{v1,v2,v3}}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topics", nargs="+", default=TOPICS,
        help="Topics to run (default: all 4)",
    )
    parser.add_argument(
        "--out-dir", default=str(OUT_BASE),
        help=f"Base output directory (default: {OUT_BASE})",
    )
    parser.add_argument(
        "--slides", type=int, default=SLIDES_PER_TOPIC,
        help=f"Slides per topic (default: {SLIDES_PER_TOPIC})",
    )
    args = parser.parse_args()

    out_base = Path(args.out_dir)

    print(f"Loading chunks from {DEFAULT_CHUNKS_PATH}...")
    chunks = load_chunks(Path(DEFAULT_CHUNKS_PATH))

    llm = LLMService(
        provider=DEFAULT_LLM_PROVIDER,
        model_id=DEFAULT_MODEL_ID,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
    )

    semantic_retriever = None
    if DEFAULT_RAG_MODE == "semantic":
        print("Building semantic retriever...")
        semantic_retriever = SemanticRetriever(
            chunks=chunks, model_id=DEFAULT_EMBEDDING_MODEL_ID
        )

    for topic in args.topics:
        run_topic(
            topic, llm, chunks, semantic_retriever,
            out_dir=out_base / _slug(topic),
            slides_count=args.slides,
        )

    print(f"\nAll done. Results in: {out_base}/")
    print("\nNext — run eval_metrics.py on all variants:")
    for topic in args.topics:
        slug = _slug(topic)
        paths = " ".join(
            f"{out_base}/{slug}/{v}/scenario_plan.json" for v in ("v1", "v2", "v3")
        )
        print(f"  python scripts/eval_metrics.py {paths} --llm-judge")


if __name__ == "__main__":
    main()
