#!/usr/bin/env python3
"""Main orchestration pipeline for LLM + semantic RAG + images + PPT + voiceover export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_IMAGE_FALLBACK,
    DEFAULT_IMAGE_PROVIDER,
    DEFAULT_IMAGES_PER_SLIDE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_SLIDE_SECONDS,
    DEFAULT_MODEL_ID,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_CONTENT_MIN_BULLETS,
    DEFAULT_CONTENT_MAX_BULLETS,
    DEFAULT_NONCONTENT_MIN_BULLETS,
    DEFAULT_NONCONTENT_MAX_BULLETS,
    DEFAULT_OUT_DIR,
    DEFAULT_RAG_MODE,
    DEFAULT_RAG_CONTEXT_MAX_CHARS,
    DEFAULT_RAG_TOP_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_RECORDING_MODE,
    DEFAULT_TARGET_TOTAL_SECONDS,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    DEFAULT_VOICE,
    DEFAULT_VOICEOVER_PROFILE,
    DEFAULT_VOICEOVER_ENGINE,
    DEFAULT_WORDS_PER_SECOND,
)
from image_service import build_image_query_candidates, create_placeholder_image, fetch_image
from layout_engine import choose_layout, layout_dict
from llm_service import (
    ChunkItem,
    LLMService,
    SemanticRetriever,
    build_evidence_text,
    load_chunks,
    make_slide_query,
    retrieve_chunks,
)
from models import PresentationRequest, SlideContent
from prompts import (
    outline_prompt,
    slide_content_prompt,
    global_plan_prompt,
    slide_detail_prompt,
    diagram_concept_prompt,
    diagram_concept_prompt_v2,
    slide_combined_prompt,
    batch_deck_prompt,
)


def _lazy_import_render_ppt():
    from ppt_generator import render_ppt
    return render_ppt


def _lazy_import_recording():
    from recording_service import build_recording
    return build_recording


def _lazy_import_voice():
    from voice_service import write_voiceover_outputs
    return write_voiceover_outputs


def _write_text_resilient(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    attempts: int = 4,
    base_delay_seconds: float = 1.0,
) -> None:
    """Write a file with atomic replace and retries for flaky pod volumes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    tmp_path = path.with_name(f"{path.name}.tmp")
    for attempt in range(1, attempts + 1):
        try:
            tmp_path.write_text(text, encoding=encoding)
            tmp_path.replace(path)
            return
        except OSError as exc:
            last_error = exc
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            if attempt == attempts:
                raise
            delay = base_delay_seconds * attempt
            print(
                f"Warning: write failed for {path} (attempt {attempt}/{attempts}, errno={exc.errno}). "
                f"Retrying in {delay:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(delay)
    if last_error is not None:
        raise last_error


def _str_to_bool(value: str | bool | None, default: bool) -> bool:
    """Convert common truthy/falsy values to a boolean with a default fallback."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_space(text: str) -> str:
    """Collapse repeated whitespace and trim leading/trailing spaces."""
    return " ".join(str(text).split())


VOICE_GROUNDING_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "this",
    "that",
    "these",
    "those",
}


def _tokens(text: str) -> set[str]:
    """Tokenize text for quick lexical grounding checks."""
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(tok) >= 3 and tok not in VOICE_GROUNDING_STOPWORDS and not tok.isdigit()
    }


def _voice_grounding_scores(*, voiceover: str, bullets: list[str], evidence_text: str) -> tuple[float, float]:
    """Return (voice-vs-bullets overlap, voice-vs-evidence overlap)."""
    voice_tokens = _tokens(voiceover)
    bullet_tokens = _tokens(" ".join(bullets))
    evidence_tokens = _tokens(evidence_text)
    bullet_overlap = len(voice_tokens.intersection(bullet_tokens)) / max(1, len(bullet_tokens))
    evidence_overlap = len(voice_tokens.intersection(evidence_tokens)) / max(1, len(evidence_tokens))
    return float(bullet_overlap), float(evidence_overlap)


def _fallback_subtitle(goal: str, *, max_words: int = 10, max_chars: int = 70) -> str:
    """Build a short subtitle candidate from slide goal text."""
    clean = _normalize_space(goal)
    if not clean:
        return ""
    words = clean.split()
    if len(words) > max_words:
        clean = " ".join(words[: max(1, int(max_words))]).rstrip(",;:")
    if len(clean) > max_chars:
        clean = clean[: max_chars].rsplit(" ", 1)[0].rstrip(",;:")
    return _normalize_space(clean)


def _rag_entries_text(hits: list[Any], *, max_entries: int = 5, excerpt_chars: int = 220) -> str:
    """Format top retrieval hits as explicit RAG entries passed to LLM prompts."""
    rows: list[str] = []
    for h in hits[: max(1, int(max_entries))]:
        source = _normalize_space(getattr(h, "source", "") or "") or "unknown"
        excerpt = _normalize_space(getattr(h, "text", "") or "")
        if len(excerpt) > excerpt_chars:
            excerpt = excerpt[:excerpt_chars].rsplit(" ", 1)[0].rstrip() + "..."
        rows.append(
            _normalize_space(
                f"- chunk_id={getattr(h, 'chunk_id', '')}; source={source}; "
                f"score={float(getattr(h, 'score', 0.0)):.4f}; excerpt={excerpt}"
            )
        )
    return "\n".join(rows)


def _scenario_plan_text(plan: dict[str, Any]) -> str:
    """Convert structured scenario plan JSON into compact text for the final slide-writer prompt."""
    lines: list[str] = []
    scenario = _normalize_space(str(plan.get("teaching_scenario", "") or ""))
    if scenario:
        lines.append(f"Scenario: {scenario}")
    structure = _normalize_space(str(plan.get("structure_plan", "") or ""))
    if structure:
        lines.append(f"Structure: {structure}")
    bullet_plan = plan.get("bullet_plan") or []
    if isinstance(bullet_plan, list):
        for idx, item in enumerate(bullet_plan, start=1):
            if not isinstance(item, dict):
                continue
            bullet = _normalize_space(str(item.get("bullet", "") or ""))
            expl = _normalize_space(str(item.get("explanation", "") or ""))
            if bullet and expl:
                lines.append(f"Line {idx}: {bullet} | explain: {expl}")
            elif bullet:
                lines.append(f"Line {idx}: {bullet}")
    diagram = _normalize_space(str(plan.get("diagram_description", "") or ""))
    if diagram:
        lines.append(f"Diagram description: {diagram}")
    diagram_code = str(plan.get("diagram_code", "") or "").strip()
    if diagram_code:
        lines.append("Diagram code:")
        lines.extend([ln.rstrip() for ln in diagram_code.splitlines() if _normalize_space(ln)])
    image_plan = _normalize_space(str(plan.get("image_plan", "") or ""))
    if image_plan:
        lines.append(f"Image plan: {image_plan}")
    speech_plan = _normalize_space(str(plan.get("speech_plan", "") or ""))
    if speech_plan:
        lines.append(f"Speech plan: {speech_plan}")
    return "\n".join(lines)


def _build_scenario_teacher_report(*, req: PresentationRequest, scenario_manifest: list[dict[str, Any]]) -> str:
    """Build a professor-friendly markdown report with full slide planning details."""
    lines: list[str] = []
    lines.append("# Scenario Plan (Professor Guide)")
    lines.append("")
    lines.append(f"- Topic: {req.topic}")
    lines.append(f"- Style: {req.style}")
    lines.append(f"- Slides: {req.slides_count}")
    lines.append("")
    lines.append("## How To Read")
    lines.append("- `RAG evidence` shows what was retrieved from the course/book.")
    lines.append("- `Scenario prompt` is the exact planning prompt sent to the model.")
    lines.append("- `Scenario output` is the returned plan used for generation.")
    lines.append("- `Content prompt/output` appear when pipeline stage is `full`.")
    lines.append("")

    for entry in sorted(scenario_manifest, key=lambda x: int(x.get("slide_number") or 0)):
        slide_number = int(entry.get("slide_number") or 0)
        title = _normalize_space(str(entry.get("title", "") or ""))
        slide_type = _normalize_space(str(entry.get("slide_type", "") or ""))
        goal = _normalize_space(str(entry.get("goal", "") or ""))
        query = _normalize_space(str(entry.get("query", "") or ""))
        constraints = entry.get("bullet_constraints") or {}
        min_b = int(constraints.get("min") or 0)
        max_b = int(constraints.get("max") or 0)
        scenario = entry.get("scenario_plan") or {}
        bullet_plan = scenario.get("bullet_plan") or []
        rag_hits = entry.get("retrieval_hits") or []
        evidence_excerpt = _normalize_space(str(entry.get("evidence_excerpt", "") or ""))

        lines.append(f"## Slide {slide_number}: {title}")
        if slide_type:
            lines.append(f"- Type: `{slide_type}`")
        if goal:
            lines.append(f"- Goal: {goal}")
        if query:
            lines.append(f"- Retrieval query: `{query}`")
        if min_b > 0 and max_b > 0:
            lines.append(f"- Bullet constraints: {min_b}-{max_b}")
        lines.append("")

        lines.append("### Scenario Output")
        teach = _normalize_space(str(scenario.get("teaching_scenario", "") or ""))
        struct = _normalize_space(str(scenario.get("structure_plan", "") or ""))
        diag = _normalize_space(str(scenario.get("diagram_description", "") or ""))
        diag_code = str(scenario.get("diagram_code", "") or "").strip()
        img_plan = _normalize_space(str(scenario.get("image_plan", "") or ""))
        speech = _normalize_space(str(scenario.get("speech_plan", "") or ""))
        if teach:
            lines.append(f"- Teaching scenario: {teach}")
        if struct:
            lines.append(f"- Structure plan: {struct}")
        if isinstance(bullet_plan, list) and bullet_plan:
            lines.append("- Bullet plan:")
            for idx, item in enumerate(bullet_plan, start=1):
                if not isinstance(item, dict):
                    continue
                b = _normalize_space(str(item.get("bullet", "") or ""))
                e = _normalize_space(str(item.get("explanation", "") or ""))
                if b and e:
                    lines.append(f"  - {idx}. {b} | Explain: {e}")
                elif b:
                    lines.append(f"  - {idx}. {b}")
        if diag:
            lines.append(f"- Diagram description: {diag}")
        if diag_code:
            lines.append("- Diagram code:")
            lines.append("```text")
            lines.append(diag_code)
            lines.append("```")
        if img_plan:
            lines.append(f"- Image plan: {img_plan}")
        if speech:
            lines.append(f"- Speech plan: {speech}")
        lines.append("")

        lines.append("### RAG Evidence")
        if isinstance(rag_hits, list) and rag_hits:
            for hit in rag_hits[:5]:
                if not isinstance(hit, dict):
                    continue
                cid = _normalize_space(str(hit.get("chunk_id", "") or ""))
                src = _normalize_space(str(hit.get("source", "") or ""))
                score = hit.get("score")
                try:
                    s = f"{float(score):.4f}"
                except Exception:
                    s = str(score or "")
                lines.append(f"- chunk_id={cid} | source={src} | score={s}")
        else:
            lines.append("- No retrieval hits recorded.")
        if evidence_excerpt:
            lines.append(f"- Evidence excerpt: {evidence_excerpt}")
        lines.append("")

        scenario_prompt = str(entry.get("scenario_prompt", "") or "")
        if scenario_prompt:
            lines.append("### Scenario Prompt Sent To LLM")
            lines.append("```text")
            lines.append(scenario_prompt)
            lines.append("```")
            lines.append("")
        content_prompt = str(entry.get("content_prompt", "") or "")
        if content_prompt:
            lines.append("### Content Prompt Sent To LLM")
            lines.append("```text")
            lines.append(content_prompt)
            lines.append("```")
            lines.append("")
        content_output = entry.get("content_output")
        if isinstance(content_output, dict) and content_output:
            lines.append("### Content Output (Used For PPT)")
            lines.append("```json")
            lines.append(json.dumps(content_output, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _file_sha1(path: Path) -> str | None:
    """Compute SHA-1 checksum for a file, returning None if hashing fails."""
    try:
        h = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _parse_request_json(raw: str) -> dict[str, Any]:
    """Parse request JSON from inline text or from a JSON file path."""
    text = raw.strip()
    maybe_path = Path(text).expanduser()
    if maybe_path.exists() and maybe_path.is_file():
        text = maybe_path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("Request JSON must be an object.")
    return parsed


def _parse_request(args: argparse.Namespace) -> PresentationRequest:
    """Build and validate a PresentationRequest from CLI arguments."""
    if args.request_json:
        payload = _parse_request_json(args.request_json)
        topic = str(payload.get("topic", "")).strip()
        slides_count = int(payload.get("slides_count", 0))
        style = str(payload.get("style", "academic")).strip() or "academic"
        include_images = _str_to_bool(payload.get("include_images"), True)
        include_voiceover = _str_to_bool(payload.get("include_voiceover"), True)
    else:
        topic = str(args.topic or "").strip()
        slides_count = int(args.slides_count)
        style = str(args.style or "academic").strip() or "academic"
        include_images = _str_to_bool(args.include_images, True)
        include_voiceover = _str_to_bool(args.include_voiceover, True)

    if not topic:
        raise RuntimeError("Topic is required.")
    # slides_count == 0 means "auto" — the outline prompt + generate_outline
    # handle it by letting the model pick a number based on topic density
    # (range [5, 15] in AUTO mode).
    if slides_count > 0:
        slides_count = max(3, min(20, slides_count))
    else:
        slides_count = 0

    return PresentationRequest(
        topic=topic,
        slides_count=slides_count,
        style=style,
        include_images=include_images,
        include_voiceover=include_voiceover,
    )


def _resolve_chunks_paths(raw_chunks_path: str) -> list[Path]:
    """Parse one or multiple chunks paths passed via --chunks-path."""
    raw = str(raw_chunks_path or "").strip()
    parts = [p.strip() for p in re.split(r"[,\n;]+", raw) if p.strip()]
    if not parts:
        parts = [str(DEFAULT_CHUNKS_PATH)]
    paths: list[Path] = []
    seen: set[str] = set()
    for part in parts:
        path = Path(part).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _load_merged_chunks(chunks_paths: list[Path]) -> list[ChunkItem]:
    """Load chunks from multiple files and merge them for one shared RAG index."""
    merged: list[ChunkItem] = []
    seen_rows: set[str] = set()
    used_chunk_ids: set[str] = set()
    for chunks_path in chunks_paths:
        rows = load_chunks(chunks_path)
        for row in rows:
            row_sig = hashlib.sha1(f"{row.source}\n{row.text}".encode("utf-8")).hexdigest()
            if row_sig in seen_rows:
                continue
            seen_rows.add(row_sig)
            chunk_id = _normalize_space(str(row.chunk_id)) or "chunk"
            if chunk_id in used_chunk_ids:
                suffix = 2
                candidate = f"{chunk_id}__merged_{suffix}"
                while candidate in used_chunk_ids:
                    suffix += 1
                    candidate = f"{chunk_id}__merged_{suffix}"
                chunk_id = candidate
            used_chunk_ids.add(chunk_id)
            merged.append(
                ChunkItem(
                    chunk_id=chunk_id,
                    source=row.source,
                    text=row.text,
                    tokens=set(row.tokens),
                )
            )
    if not merged:
        raise RuntimeError("No usable chunk rows found across all configured chunks paths.")
    return merged


def _args() -> argparse.Namespace:
    """Parse and return CLI arguments for this module."""
    p = argparse.ArgumentParser(description="English-only LLM PPT generation pipeline.")
    p.add_argument("--request-json", default="", help="Inline JSON string or path to JSON file.")
    p.add_argument("--topic", default="")
    p.add_argument("--slides-count", type=int, default=8)
    p.add_argument("--style", default="academic")
    p.add_argument("--include-images", default="true")
    p.add_argument("--include-voiceover", default="true")

    p.add_argument("--chunks-path", default=str(DEFAULT_CHUNKS_PATH))
    p.add_argument("--rag-mode", choices=("semantic", "lexical"), default=DEFAULT_RAG_MODE)
    p.add_argument("--embedding-model-id", default=DEFAULT_EMBEDDING_MODEL_ID)
    p.add_argument("--rag-top-k", type=int, default=DEFAULT_RAG_TOP_K)
    p.add_argument("--rag-context-max-chars", type=int, default=DEFAULT_RAG_CONTEXT_MAX_CHARS)

    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--llm-provider", choices=("huggingface", "gemini", "anthropic", "openai"), default=DEFAULT_LLM_PROVIDER)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)

    p.add_argument(
        "--image-provider",
        default=DEFAULT_IMAGE_PROVIDER,
        choices=("none", "wikimedia", "openverse", "openai", "t2i", "diagram", "auto"),
    )
    p.add_argument("--image-fallback", default=DEFAULT_IMAGE_FALLBACK, choices=("text_only", "placeholder", "require_web"))
    p.add_argument("--images-per-slide", type=int, default=DEFAULT_IMAGES_PER_SLIDE)
    p.add_argument("--recording-mode", default=DEFAULT_RECORDING_MODE, choices=("none", "video"))
    p.add_argument("--voiceover-engine", default=DEFAULT_VOICEOVER_ENGINE, choices=("none", "edge-tts", "espeak", "elevenlabs"))
    p.add_argument("--voiceover-profile", default=DEFAULT_VOICEOVER_PROFILE, choices=("short", "standard", "long"))
    p.add_argument("--target-total-seconds", type=int, default=DEFAULT_TARGET_TOTAL_SECONDS)
    p.add_argument("--pipeline-stage", default=os.environ.get("PIPELINE_STAGE", "full"), choices=("full", "scenario"))
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_WIDTH)
    p.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    p.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS)
    p.add_argument("--min-slide-seconds", type=float, default=DEFAULT_MIN_SLIDE_SECONDS)
    p.add_argument("--words-per-second", type=float, default=DEFAULT_WORDS_PER_SECOND)
    p.add_argument("--content-min-bullets", type=int, default=DEFAULT_CONTENT_MIN_BULLETS)
    p.add_argument("--content-max-bullets", type=int, default=DEFAULT_CONTENT_MAX_BULLETS)
    p.add_argument("--noncontent-min-bullets", type=int, default=DEFAULT_NONCONTENT_MIN_BULLETS)
    p.add_argument("--noncontent-max-bullets", type=int, default=DEFAULT_NONCONTENT_MAX_BULLETS)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return p.parse_args()


def _parse_and_validate_args() -> tuple[argparse.Namespace, PresentationRequest, str, str, Path, list[Path], list[ChunkItem], LLMService, "SemanticRetriever | None"]:
    """Parse CLI args, validate keys, load chunks, and build the LLM service.

    Returns (args, req, pipeline_stage, model_id, out_dir, chunks_paths, chunks, llm, semantic_retriever).
    """
    args = _args()
    req = _parse_request(args)
    pipeline_stage = _normalize_space(str(args.pipeline_stage or "full")).lower() or "full"
    if pipeline_stage not in {"full", "scenario"}:
        pipeline_stage = "full"
    model_id = str(args.model_id or "").strip()
    # For Gemini we normalize model id from env when caller left a HF-like value.
    if args.llm_provider == "gemini" and (not model_id or "/" in model_id):
        model_id = os.environ.get("GEMINI_MODEL_ID", "gemini-3-flash-preview").strip() or "gemini-3-flash-preview"
    # Hard-fail early on missing paid-provider keys, before expensive generation starts.
    if req.include_images and args.image_provider == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("IMAGE_PROVIDER=openai requires OPENAI_API_KEY.")
    if req.include_images and args.image_provider == "t2i" and not os.environ.get("T2I_ENDPOINT", "").strip():
        raise RuntimeError("IMAGE_PROVIDER=t2i requires T2I_ENDPOINT.")
    if (
        args.recording_mode == "video"
        and req.include_voiceover
        and args.voiceover_engine == "elevenlabs"
        and not os.environ.get("ELEVENLABS_API_KEY", "").strip()
    ):
        raise RuntimeError("VOICEOVER_ENGINE=elevenlabs requires ELEVENLABS_API_KEY.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_paths = _resolve_chunks_paths(args.chunks_path)
    missing_chunks = [str(p) for p in chunks_paths if not p.exists()]
    if missing_chunks:
        raise RuntimeError(f"Chunks file(s) not found: {missing_chunks}")

    lora_slide_only = (
        args.llm_provider == "huggingface"
        and bool(os.environ.get("HF_ADAPTER_PATH") or os.environ.get("LORA_ADAPTER_PATH"))
        and _str_to_bool(os.environ.get("LORA_SLIDE_ONLY", "1"), True)
    )
    llm = LLMService(
        provider=args.llm_provider,
        model_id=model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        adapter_path="" if lora_slide_only else None,
    )
    chunks = _load_merged_chunks(chunks_paths)
    semantic_retriever: SemanticRetriever | None = None
    if args.rag_mode == "semantic":
        # Build embedding index once, then reuse for all slide retrieval queries.
        semantic_retriever = SemanticRetriever(
            chunks=chunks,
            model_id=args.embedding_model_id,
        )

    return args, req, pipeline_stage, model_id, out_dir, chunks_paths, chunks, llm, semantic_retriever


def _run_rag_retrieval(
    args: argparse.Namespace,
    req: PresentationRequest,
    pipeline_stage: str,
    chunks: list[ChunkItem],
    llm: LLMService,
    semantic_retriever: "SemanticRetriever | None",
) -> tuple[
    list[Any],                              # outline
    list[SlideContent],                     # slide_contents
    list[dict[str, Any]],                   # retrieval_manifest
    list[dict[str, Any]],                   # scenario_manifest
    dict[int, str],                         # evidence_by_slide
    dict[int, str],                         # diagram_code_by_slide
    dict[int, str],                         # diagram_desc_by_slide
    dict[int, dict[str, list[str]]],        # prolog_kb_by_slide
]:
    """Multi-step pipeline: global plan → per-slide detail → diagram concept → D2 → content."""
    lora_slide_only = (
        args.llm_provider == "huggingface"
        and bool(os.environ.get("HF_ADAPTER_PATH") or os.environ.get("LORA_ADAPTER_PATH"))
        and _str_to_bool(os.environ.get("LORA_SLIDE_ONLY", "1"), True)
    )
    planning_llm = llm
    slide_llm = llm
    if lora_slide_only:
        slide_llm = LLMService(
            provider=args.llm_provider,
            model_id=llm.model_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            adapter_path=(
                _normalize_space(os.environ.get("HF_ADAPTER_PATH", ""))
                or _normalize_space(os.environ.get("LORA_ADAPTER_PATH", ""))
                or None
            ),
        )

    outline = planning_llm.generate_outline(req, outline_prompt(req))
    # The outline may be shorter/longer than the target (density-driven). Sync
    # `req.slides_count` to the real length so downstream voiceover pacing and
    # timeline math use the actual number, not the requested target.
    if len(outline) != req.slides_count:
        print(f"Outline adjusted: target {req.slides_count} → actual {len(outline)} slides (density-based).")
        req.slides_count = len(outline)

    # RAG retrieval per slide (same as before)
    hits_by_slide: dict[int, list] = {}
    evidence_by_slide: dict[int, str] = {}
    rag_entries_by_slide: dict[int, str] = {}
    retrieval_manifest: list[dict[str, Any]] = []

    for item in outline:
        query = make_slide_query(req, item)
        hits = retrieve_chunks(
            chunks=chunks, query=query, top_k=args.rag_top_k,
            rag_mode=args.rag_mode, semantic_retriever=semantic_retriever,
        )
        if not hits:
            hits = retrieve_chunks(
                chunks=chunks, query=req.topic, top_k=args.rag_top_k,
                rag_mode=args.rag_mode, semantic_retriever=semantic_retriever,
            )
        if not hits:
            raise RuntimeError(f"No RAG evidence for slide {item.slide_number}: {item.title}")

        hits_by_slide[item.slide_number] = hits
        evidence_by_slide[item.slide_number] = build_evidence_text(hits, max_chars=args.rag_context_max_chars)
        rag_entries_by_slide[item.slide_number] = _rag_entries_text(hits)
        hit_rows = [{"chunk_id": h.chunk_id, "source": h.source, "score": round(float(h.score), 6)} for h in hits]
        retrieval_manifest.append({
            "slide_number": item.slide_number, "title": item.title,
            "query": query, "rag_entries": rag_entries_by_slide[item.slide_number],
            "hits": hit_rows, "evidence_excerpt": evidence_by_slide[item.slide_number],
        })

    prompt_mode = os.environ.get("PROMPT_MODE", "full").strip().lower()
    scenario_skip_diagrams = (
        _str_to_bool(os.environ.get("SKIP_DIAGRAM_GENERATION", "0"), False)
        or (pipeline_stage == "scenario" and _str_to_bool(os.environ.get("SCENARIO_SKIP_DIAGRAMS", "1"), True))
    )
    batch_slides_by_number: dict[int, dict[str, Any]] = {}

    if prompt_mode == "batch":
        # BATCH MODE: generate the ENTIRE deck in one Gemini call
        print("Running BATCH mode: generating entire deck in one call...")
        batch_prompt = batch_deck_prompt(req, outline, rag_entries_by_slide)
        batch_slides = slide_llm.generate_batch_deck(batch_prompt, expected_slides=len(outline))
        print(f"Batch deck: {len(batch_slides)} slides generated in 1 call.")
        for s in batch_slides:
            sn = int(s.get("slide_number", 0))
            if sn:
                batch_slides_by_number[sn] = s
        global_plan = [
            {"slide_number": s.get("slide_number"), "main_idea": s.get("main_idea", ""),
             "diagram_theme": s.get("diagram_description", "")[:80], "progression": ""}
            for s in batch_slides
        ]
        global_plan_text = "\n".join(
            f"  Slide {p.get('slide_number', '?')}: {p.get('main_idea', '')}"
            for p in global_plan
        )
    else:
        # PROMPT 1: Global plan (runs ONCE for all slides)
        rag_summary = "\n".join(
            f"Slide {item.slide_number}: {_normalize_space(rag_entries_by_slide.get(item.slide_number, '')[:200])}"
            for item in outline
        )
        plan_prompt = global_plan_prompt(req, outline, rag_summary)
        print("Running Prompt 1: Global plan...")
        global_plan = planning_llm.generate_global_plan(plan_prompt)
        global_plan_text = "\n".join(
            f"  Slide {p.get('slide_number', '?')}: idea={p.get('main_idea', '')} | diagram={p.get('diagram_theme', '')} | progression={p.get('progression', '')}"
            for p in global_plan
        )
        print(f"Global plan: {len(global_plan)} slides planned.")

    # Per-slide: Prompt 2.1 → 2.2 → 2.3 → content
    slide_contents: list[SlideContent] = []
    scenario_manifest: list[dict[str, Any]] = []
    diagram_code_by_slide: dict[int, str] = {}
    diagram_desc_by_slide: dict[int, str] = {}
    prolog_kb_by_slide: dict[int, dict[str, list[str]]] = {}  # {slide_num: {"facts": [...], "rules": [...]}}

    # DIAGRAM_CONCEPT_MODE controls how Prompt 2.2 generates the diagram concept:
    #   v1          — direct LLM (current baseline, default)
    #   v2_spacy    — spaCy SVO → Prolog KB (no LLM) → LLM diagram concept
    #   v2_llm      — LLM-extracted Prolog KB → LLM diagram concept
    diagram_concept_mode = os.environ.get("DIAGRAM_CONCEPT_MODE", "v1").strip().lower()
    if diagram_concept_mode not in ("v1", "v2_spacy", "v2_llm"):
        print(f"[warn] Unknown DIAGRAM_CONCEPT_MODE={diagram_concept_mode!r}, falling back to v1")
        diagram_concept_mode = "v1"
    if diagram_concept_mode != "v1":
        print(f"  Diagram concept mode: {diagram_concept_mode}")

    for item in outline:
        sn = item.slide_number
        rag_entries = rag_entries_by_slide.get(sn, "")
        evidence_text = evidence_by_slide.get(sn, "")

        hard_bullet_cap = 3
        if item.type == "content":
            requested_min = max(1, int(args.content_min_bullets))
            requested_max = max(requested_min, int(args.content_max_bullets))
        else:
            requested_min = max(1, int(args.noncontent_min_bullets))
            requested_max = max(requested_min, int(args.noncontent_max_bullets))
        max_bullets = min(hard_bullet_cap, requested_max)
        min_bullets = min(requested_min, max_bullets)

        # Pipeline mode: "batch" | "combined" | "full"


        if prompt_mode == "batch" and sn in batch_slides_by_number:
            # Batch: already generated in the single upfront call
            print(f"  Slide {sn}: using batch-generated data.")
            slide_detail = batch_slides_by_number[sn]
            diagram_desc = slide_detail.get("diagram_description", "")
            diagram_code = slide_detail.get("diagram_code", "")
        elif prompt_mode == "batch" or prompt_mode == "combined":
            # Combined: 1 call per slide (fast, free-tier friendly)
            combined_prompt_text = slide_combined_prompt(
                req, item,
                global_plan_text=global_plan_text,
                rag_entries_text=rag_entries,
                min_bullets=min_bullets,
                max_bullets=max_bullets,
            )
            print(f"  Slide {sn}: Prompt 2 (combined)...")
            # Retry on empty Gemini payload — rate limits and transient parse
            # errors otherwise leave the slide with blank bullets + a generic
            # fallback diagram identical to every other failed slide.
            slide_detail = None
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    slide_detail = slide_llm.generate_slide_combined(
                        combined_prompt_text,
                        min_bullets=min_bullets,
                        max_bullets=max_bullets,
                    )
                    break
                except RuntimeError as exc:
                    last_err = exc
                    print(f"  Slide {sn}: combined attempt {attempt + 1}/3 failed: {exc}")
                    import time; time.sleep(2 ** attempt)
            if slide_detail is None:
                raise RuntimeError(f"Slide {sn} failed after 3 combined-mode retries: {last_err}")
            diagram_desc = slide_detail.get("diagram_description", "")
            diagram_code = slide_detail.get("diagram_code", "")
        else:
            # Full: 3 separate prompts per slide
            # Prompt 2.1: slide detail + diagram theme
            detail_prompt_text = slide_detail_prompt(
                req, item,
                global_plan_text=global_plan_text,
                rag_entries_text=rag_entries,
                min_bullets=min_bullets,
                max_bullets=max_bullets,
            )
            print(f"  Slide {sn}: Prompt 2.1 (slide detail)...")
            slide_detail = slide_llm.generate_slide_detail(
                detail_prompt_text,
                min_bullets=min_bullets,
                max_bullets=max_bullets,
            )

            if scenario_skip_diagrams:
                slide_detail.setdefault("diagram_description", "")
                slide_detail.setdefault("diagram_code", "")
                slide_detail.setdefault("image_plan", "")
                diagram_desc = ""
                diagram_code = ""
            else:
                # Prompt 2.2: diagram concept (nodes, arrows — no code)
                diagram_theme = slide_detail.get("diagram_theme", "") or f"Diagram for {item.title}"
                bullets_text = "\n".join(f"- {b.get('bullet', '')}" for b in slide_detail.get("bullet_plan", []))
                # Pull the per-slide diagram_signature from the global plan so
                # Prompt 2.2 doesn't default everything to a left-to-right `flow`.
                plan_entry = next(
                    (p for p in (global_plan or []) if int(p.get("slide_number", 0) or 0) == sn),
                    {},
                )
                diagram_signature = str(plan_entry.get("diagram_signature", "") or "").strip().lower()

                # V2: extract Prolog KB before building the diagram concept prompt
                prolog_facts: list[str] = []
                prolog_rules: list[str] = []
                if diagram_concept_mode == "v2_spacy":
                    print(f"  Slide {sn}: extracting Prolog KB via spaCy...")
                    from prolog_extractor import extract_prolog  # lazy import
                    prolog_facts, prolog_rules = extract_prolog(
                        item.title + ". " + bullets_text.replace("- ", "")
                    )
                    print(f"    → {len(prolog_facts)} facts, {len(prolog_rules)} rules")
                elif diagram_concept_mode == "v2_llm":
                    print(f"  Slide {sn}: extracting Prolog KB via LLM...")
                    prolog_facts, prolog_rules = slide_llm.generate_prolog_from_text(
                        item.title + ". " + bullets_text.replace("- ", "")
                    )
                    print(f"    → {len(prolog_facts)} facts, {len(prolog_rules)} rules")

                if prolog_facts or prolog_rules:
                    prolog_kb_by_slide[sn] = {"facts": prolog_facts, "rules": prolog_rules}

                # Build the concept prompt (V1 or V2) and call the LLM
                if diagram_concept_mode in ("v2_spacy", "v2_llm"):
                    concept_prompt_text = diagram_concept_prompt_v2(
                        sn, item.title, diagram_theme, bullets_text, rag_entries,
                        prolog_facts=prolog_facts,
                        prolog_rules=prolog_rules,
                        diagram_signature=diagram_signature,
                    )
                else:
                    concept_prompt_text = diagram_concept_prompt(
                        sn, item.title, diagram_theme, bullets_text, rag_entries,
                        diagram_signature=diagram_signature,
                    )
                print(f"  Slide {sn}: Prompt 2.2 (diagram concept, mode={diagram_concept_mode})...")
                diagram_desc = slide_llm.generate_diagram_concept(concept_prompt_text)

                # Prompt 2.3: D2 code (conversion of concept to compilable D2).
                print(f"  Slide {sn}: Prompt 2.3 (D2 code)...")
                diagram_code = slide_llm._format_diagram_code(
                    slide_number=sn,
                    teaching_scenario=slide_detail.get("teaching_scenario", ""),
                    structure_plan=slide_detail.get("structure_plan", ""),
                    bullet_plan=slide_detail.get("bullet_plan", []),
                    diagram_description=diagram_desc,
                    diagram_code="",
                    image_plan="",
                    slide_title=item.title,
                    diagram_theme=diagram_theme,
                    rag_entries_text=rag_entries,
                )

                slide_detail["diagram_description"] = diagram_desc
                slide_detail["diagram_code"] = diagram_code
                slide_detail["image_plan"] = diagram_desc

        if diagram_desc:
            diagram_desc_by_slide[sn] = diagram_desc
        if diagram_code:
            diagram_code_by_slide[sn] = diagram_code

        # Build scenario plan dict (compatible with existing pipeline)
        scenario_plan: dict[str, Any] = dict(slide_detail)
        scenario_text = _scenario_plan_text(scenario_plan)

        hit_rows = [{"chunk_id": h.chunk_id, "source": h.source, "score": round(float(h.score), 6)}
                     for h in hits_by_slide.get(sn, [])]
        prolog_kb = prolog_kb_by_slide.get(sn, {})
        scenario_entry: dict[str, Any] = {
            "slide_number": sn, "slide_type": item.type,
            "title": item.title, "goal": item.goal,
            "query": make_slide_query(req, item),
            "bullet_constraints": {"min": min_bullets, "max": max_bullets},
            "rag_entries": rag_entries, "retrieval_hits": hit_rows,
            "evidence_excerpt": evidence_text,
            "global_plan_entry": next((p for p in global_plan if p.get("slide_number") == sn), {}),
            "scenario_plan": scenario_plan,
            "scenario_plan_text": scenario_text,
            "diagram_concept_mode": diagram_concept_mode,
            "prolog_facts": prolog_kb.get("facts", []),
            "prolog_rules": prolog_kb.get("rules", []),
        }

        if pipeline_stage == "scenario":
            scenario_manifest.append(scenario_entry)
            continue

        # Generate content (same as before)
        content_prompt_text = slide_content_prompt(
            req, item, evidence_text,
            scenario_plan_text=scenario_text,
            min_bullets=min_bullets, max_bullets=max_bullets,
            target_total_seconds=args.target_total_seconds,
            voiceover_profile=args.voiceover_profile,
        )
        content = slide_llm.generate_slide_content(
            sn, content_prompt_text,
            min_bullets=min_bullets, max_bullets=max_bullets,
            fallback_subtitle=_fallback_subtitle(item.goal),
        )
        # RAG grounding retry
        if item.type == "content":
            _, ev_overlap = _voice_grounding_scores(
                voiceover=content.voiceover, bullets=content.bullets, evidence_text=evidence_text,
            )
            if ev_overlap < 0.03:
                try:
                    retry_content = slide_llm.generate_slide_content(
                        sn, f"{content_prompt_text}\n\nSTRICT: reuse RAG facts in voiceover.",
                        min_bullets=min_bullets, max_bullets=max_bullets,
                        fallback_subtitle=_fallback_subtitle(item.goal),
                    )
                    _, retry_ev = _voice_grounding_scores(
                        voiceover=retry_content.voiceover, bullets=retry_content.bullets, evidence_text=evidence_text,
                    )
                    if retry_ev > ev_overlap:
                        content = retry_content
                except Exception:
                    pass

        if diagram_desc:
            content.image_suggestion = diagram_desc
        content.slide_number = sn
        scenario_entry["content_prompt"] = content_prompt_text
        scenario_entry["content_output"] = asdict(content)
        scenario_manifest.append(scenario_entry)
        slide_contents.append(content)

    return outline, slide_contents, retrieval_manifest, scenario_manifest, evidence_by_slide, diagram_code_by_slide, diagram_desc_by_slide, prolog_kb_by_slide


def _run_scenario_stage(
    args: argparse.Namespace,
    req: PresentationRequest,
    pipeline_stage: str,
    model_id: str,
    out_dir: Path,
    chunks_paths: list[Path],
    chunks: list[ChunkItem],
    outline: list[Any],
    retrieval_manifest: list[dict[str, Any]],
    scenario_manifest: list[dict[str, Any]],
) -> None:
    """Write scenario-only outputs and print the manifest. Only called when pipeline_stage == 'scenario'."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outline_path = out_dir / "outline.json"
    retrieval_path = out_dir / "retrieval.json"
    scenario_path = out_dir / "scenario_plan.json"
    teacher_report_path = out_dir / "scenario_teacher_report.md"
    _write_text_resilient(outline_path, json.dumps([asdict(item) for item in outline], indent=2))
    _write_text_resilient(retrieval_path, json.dumps(retrieval_manifest, indent=2))
    _write_text_resilient(scenario_path, json.dumps(scenario_manifest, indent=2))
    _write_text_resilient(
        teacher_report_path,
        _build_scenario_teacher_report(req=req, scenario_manifest=scenario_manifest),
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": asdict(req),
        "pipeline_stage": pipeline_stage,
        "model_id": model_id,
        "llm_provider": args.llm_provider,
        "out_dir": str(out_dir),
        "outline_json": str(outline_path),
        "retrieval_json": str(retrieval_path),
        "scenario_plan_json": str(scenario_path),
        "scenario_complete_json": str(scenario_path),
        "scenario_teacher_report_md": str(teacher_report_path),
        "chunks_path": str(chunks_paths[0]),
        "chunks_paths": [str(p) for p in chunks_paths],
        "chunks_merged_count": len(chunks),
        "rag_mode": args.rag_mode,
        "embedding_model_id": args.embedding_model_id if args.rag_mode == "semantic" else None,
        "content_min_bullets": args.content_min_bullets,
        "content_max_bullets": args.content_max_bullets,
        "noncontent_min_bullets": args.noncontent_min_bullets,
        "noncontent_max_bullets": args.noncontent_max_bullets,
        "style": req.style,
    }
    manifest_path = out_dir / "manifest.json"
    _write_text_resilient(manifest_path, json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _run_full_stage(
    args: argparse.Namespace,
    req: PresentationRequest,
    pipeline_stage: str,
    model_id: str,
    out_dir: Path,
    chunks_paths: list[Path],
    chunks: list[ChunkItem],
    outline: list[Any],
    slide_contents: list[SlideContent],
    retrieval_manifest: list[dict[str, Any]],
    scenario_manifest: list[dict[str, Any]],
    evidence_by_slide: dict[int, str],
    diagram_code_by_slide: dict[int, str],
    diagram_desc_by_slide: dict[int, str],
) -> None:
    """Handle images, PPT rendering, voiceover, recording, and full manifest output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    layout_by_slide: dict[int, str] = {}
    content_by_slide = {item.slide_number: item for item in slide_contents}
    for item in outline:
        content = content_by_slide[item.slide_number]
        layout_by_slide[item.slide_number] = choose_layout(item, content, include_images=req.include_images)

    image_by_slide: dict[int, Path] = {}
    image_manifest: list[dict[str, Any]] = []
    images_dir = out_dir / "images"
    web_images_found = 0
    missing_web_slides: list[int] = []
    used_web_image_urls: set[str] = set()
    used_image_hashes: set[str] = set()
    if req.include_images:
        for content in slide_contents:
            layout_name = layout_by_slide.get(content.slide_number, "text_only")
            if layout_name != "text_left_image_right":
                continue

            evidence = evidence_by_slide.get(content.slide_number, "")
            # Generate multiple cleaned search queries from topic + title + bullets + RAG evidence.
            deduped_queries = build_image_query_candidates(
                topic=req.topic,
                title=content.title,
                image_suggestion=content.image_suggestion,
                bullets=content.bullets,
                evidence_text=evidence,
            )
            if not deduped_queries:
                deduped_queries = [_normalize_space(f"{req.topic} {content.title} diagram")]

            path: Path | None = None
            source_url: str | None = None
            selected_query: str | None = None
            for query in deduped_queries:
                # Try providers in ranked order and skip URLs already used on previous slides.
                path, source_url = fetch_image(
                    query=query,
                    slide_number=content.slide_number,
                    out_dir=images_dir,
                    provider=args.image_provider,
                    max_candidates=args.images_per_slide,
                    avoid_urls=used_web_image_urls,
                    title=content.title,
                    bullets=content.bullets,
                    evidence_text=evidence,
                    diagram_description=diagram_desc_by_slide.get(content.slide_number, ""),
                    diagram_code=diagram_code_by_slide.get(content.slide_number, ""),
                )
                if path is not None:
                    src = (source_url or "").strip()
                    # URL-level dedupe: prevents repeating the same web asset across slides.
                    if src.lower().startswith("http") and src in used_web_image_urls:
                        try:
                            path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        path = None
                        source_url = None
                        continue
                    image_hash = _file_sha1(path)
                    # Content-level dedupe: catches duplicates even if URL differs.
                    if image_hash and image_hash in used_image_hashes:
                        try:
                            path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        path = None
                        source_url = None
                        continue
                    if image_hash:
                        used_image_hashes.add(image_hash)
                    if src.lower().startswith("http"):
                        used_web_image_urls.add(src)
                    selected_query = query
                    break

            if path is None:
                if args.image_fallback == "require_web":
                    # Keep slide text-only when web image is mandatory but unavailable.
                    layout_by_slide[content.slide_number] = "text_only"
                    missing_web_slides.append(content.slide_number)
                    image_manifest.append(
                        {
                            "slide_number": content.slide_number,
                            "query": deduped_queries[0] if deduped_queries else "",
                            "query_candidates": deduped_queries,
                            "selected_query": None,
                            "image_suggestion": content.image_suggestion,
                            "image_path": None,
                            "origin": "missing_web",
                            "source_url": None,
                        }
                    )
                    continue
                if args.image_fallback == "placeholder":
                    placeholder_query = deduped_queries[0] if deduped_queries else _normalize_space(f"{req.topic} {content.title}")
                    path = images_dir / f"slide_{content.slide_number:02d}_placeholder.png"
                    create_placeholder_image(path, content.title, placeholder_query)
                    image_by_slide[content.slide_number] = path
                    image_manifest.append(
                        {
                            "slide_number": content.slide_number,
                            "query": placeholder_query,
                            "query_candidates": deduped_queries,
                            "selected_query": placeholder_query,
                            "image_suggestion": content.image_suggestion,
                            "image_path": str(path),
                            "origin": "generated",
                            "source_url": None,
                        }
                    )
                else:
                    layout_by_slide[content.slide_number] = "text_only"
                    image_manifest.append(
                        {
                            "slide_number": content.slide_number,
                            "query": deduped_queries[0] if deduped_queries else "",
                            "query_candidates": deduped_queries,
                            "selected_query": None,
                            "image_suggestion": content.image_suggestion,
                            "image_path": None,
                            "origin": "none",
                            "source_url": None,
                        }
                    )
                continue

            image_by_slide[content.slide_number] = path
            web_images_found += 1
            image_manifest.append(
                {
                    "slide_number": content.slide_number,
                    "query": selected_query or deduped_queries[0],
                    "query_candidates": deduped_queries,
                    "selected_query": selected_query,
                    "image_suggestion": content.image_suggestion,
                    "image_path": str(path),
                    "origin": "web",
                    "source_url": source_url,
                }
            )

    if req.include_images and args.image_fallback == "require_web" and web_images_found == 0:
        raise RuntimeError(
            f"No images were found for this deck with provider={args.image_provider}. "
            "Tried multiple queries per slide. "
            f"Missing slides: {missing_web_slides}. "
            "Try checking API keys (for openai), increasing IMAGES_PER_SLIDE, or using IMAGE_PROVIDER=wikimedia."
        )

    render_ppt = _lazy_import_render_ppt()
    deck_path = render_ppt(
        out_path=out_dir / "presentation.pptx",
        slides=slide_contents,
        layout_by_slide=layout_by_slide,
        image_by_slide=image_by_slide,
        deck_style=req.style,
    )

    voice_json: Path | None = None
    voice_txt: Path | None = None
    if req.include_voiceover:
        write_voiceover_outputs = _lazy_import_voice()
        voice_json, voice_txt = write_voiceover_outputs(out_dir / "voiceover", slide_contents)

    recording_meta: dict[str, Any] = {
        "timeline_json": None,
        "timeline_objects_json": None,
        "recording_video": None,
        "recording_warning": None,
        "audio_dir": None,
        "audio_tracks_count": 0,
        "voiceover_engine_used": None,
        "video_has_audio_stream": None,
        "voiceover_audio_file": None,
        "voiceover_audio_duration_seconds": None,
    }
    if args.recording_mode == "video":
        if req.include_voiceover and args.voiceover_engine == "none":
            raise RuntimeError(
                "Voiceover was requested, but VOICEOVER_ENGINE=none creates a silent video. "
                "Set VOICEOVER_ENGINE=edge-tts (recommended) or VOICEOVER_ENGINE=espeak."
            )
        effective_engine = args.voiceover_engine if req.include_voiceover else "none"
        build_recording = _lazy_import_recording()
        recording_meta = build_recording(
            slides=slide_contents,
            image_by_slide=image_by_slide,
            recording_dir=out_dir / "recording",
            engine=effective_engine,
            voice=args.voice,
            style=req.style,
            width=args.video_width,
            height=args.video_height,
            fps=args.video_fps,
            min_slide_seconds=args.min_slide_seconds,
            words_per_second=args.words_per_second,
        )
        if req.include_voiceover and int(recording_meta.get("audio_tracks_count") or 0) <= 0:
            raise RuntimeError(
                "Voiceover was requested, but no audio tracks were generated. "
                "Try VOICEOVER_ENGINE=edge-tts or install/configure espeak."
            )
        if req.include_voiceover and recording_meta.get("video_has_audio_stream") is False:
            raise RuntimeError(
                "Voiceover tracks were generated, but final MP4 has no audio stream. "
                "Check ffmpeg/audio codec support in the runtime and re-run."
            )

    outline_path = out_dir / "outline.json"
    slides_path = out_dir / "slide_contents.json"
    retrieval_path = out_dir / "retrieval.json"
    scenario_path = out_dir / "scenario_plan.json"
    teacher_report_path = out_dir / "scenario_teacher_report.md"
    images_path = out_dir / "images.json"

    _write_text_resilient(outline_path, json.dumps([asdict(item) for item in outline], indent=2))
    _write_text_resilient(slides_path, json.dumps([asdict(item) for item in slide_contents], indent=2))
    _write_text_resilient(retrieval_path, json.dumps(retrieval_manifest, indent=2))
    _write_text_resilient(scenario_path, json.dumps(scenario_manifest, indent=2))
    _write_text_resilient(
        teacher_report_path,
        _build_scenario_teacher_report(req=req, scenario_manifest=scenario_manifest),
    )
    _write_text_resilient(images_path, json.dumps(image_manifest, indent=2))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": asdict(req),
        "pipeline_stage": pipeline_stage,
        "model_id": model_id,
        "llm_provider": args.llm_provider,
        "out_dir": str(out_dir),
        "deck": str(deck_path),
        "outline_json": str(outline_path),
        "slide_contents_json": str(slides_path),
        "retrieval_json": str(retrieval_path),
        "scenario_plan_json": str(scenario_path),
        "scenario_complete_json": str(scenario_path),
        "scenario_teacher_report_md": str(teacher_report_path),
        "images_json": str(images_path),
        "voiceover_json": str(voice_json) if voice_json else None,
        "voiceover_txt": str(voice_txt) if voice_txt else None,
        "recording_mode": args.recording_mode,
        "voiceover_engine": args.voiceover_engine,
        "voiceover_profile": args.voiceover_profile,
        "target_total_seconds": args.target_total_seconds,
        "voice": args.voice,
        "recording_timeline_json": recording_meta.get("timeline_json"),
        "recording_timeline_objects_json": recording_meta.get("timeline_objects_json"),
        "recording_video": recording_meta.get("recording_video"),
        "recording_warning": recording_meta.get("recording_warning"),
        "recording_audio_dir": recording_meta.get("audio_dir"),
        "recording_audio_tracks_count": recording_meta.get("audio_tracks_count"),
        "recording_voiceover_engine_used": recording_meta.get("voiceover_engine_used"),
        "recording_video_has_audio_stream": recording_meta.get("video_has_audio_stream"),
        "recording_voiceover_audio_file": recording_meta.get("voiceover_audio_file"),
        "recording_voiceover_audio_duration_seconds": recording_meta.get("voiceover_audio_duration_seconds"),
        "chunks_path": str(chunks_paths[0]),
        "chunks_paths": [str(p) for p in chunks_paths],
        "chunks_merged_count": len(chunks),
        "rag_mode": args.rag_mode,
        "embedding_model_id": args.embedding_model_id if args.rag_mode == "semantic" else None,
        "content_min_bullets": args.content_min_bullets,
        "content_max_bullets": args.content_max_bullets,
        "noncontent_min_bullets": args.noncontent_min_bullets,
        "noncontent_max_bullets": args.noncontent_max_bullets,
        "image_provider": args.image_provider,
        "image_fallback": args.image_fallback,
        "layouts": {str(k): layout_dict(v) for k, v in layout_by_slide.items()},
    }
    manifest_path = out_dir / "manifest.json"
    _write_text_resilient(manifest_path, json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def main() -> None:
    """Run the module entrypoint end-to-end."""
    # Phase 1: parse CLI args, validate keys, load chunks, build LLM service.
    args, req, pipeline_stage, model_id, out_dir, chunks_paths, chunks, llm, semantic_retriever = (
        _parse_and_validate_args()
    )

    # Phase 2: generate outline, run per-slide RAG retrieval, scenario plans, and content generation.
    outline, slide_contents, retrieval_manifest, scenario_manifest, evidence_by_slide, diagram_code_by_slide, diagram_desc_by_slide, prolog_kb_by_slide = (
        _run_rag_retrieval(args, req, pipeline_stage, chunks, llm, semantic_retriever)
    )

    # Phase 3: scenario-only pipeline — write outputs and return early.
    if pipeline_stage == "scenario":
        _run_scenario_stage(
            args, req, pipeline_stage, model_id, out_dir,
            chunks_paths, chunks, outline, retrieval_manifest, scenario_manifest,
        )
        return

    # Phase 4: full pipeline — images, PPT, voiceover, recording, manifest.
    _run_full_stage(
        args, req, pipeline_stage, model_id, out_dir,
        chunks_paths, chunks, outline, slide_contents,
        retrieval_manifest, scenario_manifest,
        evidence_by_slide, diagram_code_by_slide, diagram_desc_by_slide,
    )


if __name__ == "__main__":
    main()
