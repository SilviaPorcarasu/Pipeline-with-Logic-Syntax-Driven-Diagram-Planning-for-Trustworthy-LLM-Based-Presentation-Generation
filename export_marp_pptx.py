#!/usr/bin/env python3
"""Convert scenario_plan.json to a Marp markdown deck and export to PPTX.

Layout: compact title + small subtitle at top, then 2 columns —
short bullets left (~35%), large diagram right (~65%).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any

# Marp 16:9 default canvas in pixels — passed to the layout decider so
# the LLM has the real slide dimensions when picking position + sizes.
SLIDE_WIDTH_PX = 1280
SLIDE_HEIGHT_PX = 720


def _compute_adaptive_layout(image_path: Path, bullet_count: int, avg_bullet_words: float) -> dict:
    """Decide full layout from PNG aspect ratio + bullet density.

    Returns a dict with:
      `mode`         — "stacked" (image top, bullets bottom) or "side" (2 columns)
      `image_first`  — bool; in stacked mode, image goes above bullets
      `text_flex`    — int, flex weight of text column in side mode
      `img_flex`     — int, flex weight of image column in side mode
      `img_max_w`    — int %, max width the image is allowed to occupy
      `img_max_h`    — int %, max height the image is allowed to occupy
      `bullet_em`    — float, bullet font-size in em (auto-shrinks when dense)
      `line_height`  — float, bullet line-height
      `ratio`        — float, image aspect ratio (width / height); used by
                       caller to override LLM layout picks for tall portraits
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            w, h = img.size
    except Exception:
        w, h = 1200, 800
    ratio = w / max(1, h)

    # Stack diagrams with aspect ratio >= 1.8; use side layouts otherwise.
    if ratio >= 1.8:
        mode = "stacked"
        text_flex, img_flex = 100, 100
        img_max_w, img_max_h = 100, 100
    elif ratio >= 1.0:
        mode = "side"
        text_flex, img_flex = 35, 65
        img_max_w, img_max_h = 100, 100
    elif ratio >= 0.75:
        mode = "side"
        text_flex, img_flex = 38, 62
        img_max_w, img_max_h = 100, 100
    elif ratio >= 0.55:
        mode = "side"
        text_flex, img_flex = 42, 58
        img_max_w, img_max_h = 100, 100
    else:
        # Very tall portrait diagrams: image is height-constrained so
        # give it equal horizontal space — text still readable at 50%.
        mode = "side"
        text_flex, img_flex = 48, 52
        img_max_w, img_max_h = 100, 100

    # Bullet sizing: shrink when we have many or long bullets so they never
    # overflow the column and bleed over the title. Tighter line-heights than
    # the earlier version so bullets look compact instead of stretched.
    density = bullet_count * max(1.0, avg_bullet_words)
    if density <= 20:
        bullet_em, line_height = 1.10, 1.38
    elif density <= 30:
        bullet_em, line_height = 1.00, 1.32
    elif density <= 42:
        bullet_em, line_height = 0.92, 1.28
    else:
        bullet_em, line_height = 0.85, 1.24

    # Stacked mode leaves less vertical room for bullets — shrink one more step.
    if mode == "stacked":
        bullet_em = max(0.85, bullet_em - 0.08)
        line_height = max(1.25, line_height - 0.08)

    return {
        "mode": mode,
        "image_first": False if mode == "stacked" else True,
        "text_flex": text_flex,
        "img_flex": img_flex,
        "img_max_w": img_max_w,
        "img_max_h": img_max_h,
        "bullet_em": bullet_em,
        "line_height": line_height,
        "ratio": ratio,
    }


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _compact(text: str, max_words: int) -> str:
    words = _normalize(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    # Cap at max_words, but back up to the nearest sentence end so we never
    # leave the reader hanging on "and", "the", or a trailing clause.
    clipped = words[:max_words]
    for i in range(len(clipped) - 1, -1, -1):
        if clipped[i].endswith((".", "!", "?")):
            return " ".join(clipped[: i + 1])
    # No sentence boundary found — fall back to the nearest comma/colon, then
    # to the raw cap. Never append "…": the trailing ellipsis was leaking into
    # rendered slides whenever a math-heavy bullet exceeded the cap.
    for i in range(len(clipped) - 1, -1, -1):
        if clipped[i].endswith((",", ";", ":")):
            return " ".join(clipped[: i + 1]).rstrip(",;:") + "."
    return " ".join(clipped).rstrip(",;:")


def _build_marp_header() -> str:
    return """---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Times New Roman', 'Liberation Serif', serif;
    background: #ffffff;
    color: #1a2a44;
    padding: 30px 40px;
    overflow: hidden;
  }
  h1 {
    font-size: 1.6em;
    color: #192d54;
    border-bottom: 2px solid #005ab4;
    padding-bottom: 0.15em;
    margin-bottom: 0.25em;
    margin-top: 0;
  }
  h2 {
    font-size: 0.9em;
    color: #506482;
    font-style: italic;
    font-weight: normal;
    margin-top: 0;
    margin-bottom: 0.4em;
  }
  ul {
    font-size: 1.0em;
    line-height: 1.35;
    margin-top: 0.2em;
    padding-left: 1.2em;
  }
  ul {
    list-style: none;
    padding-left: 0;
    margin-left: 0;
  }
  li {
    position: relative;
    padding-left: 1.0em;
    margin-bottom: 0.35em;
    list-style: none;
  }
  /* CSS-drawn dot bullet (no font / glyph dependency).
     `list-style-type: disc` rendered as a half-glyph or arrow on some
     slides because Marp's PPTX export remapped the marker character on
     pastel backgrounds. A flat circle drawn via ::before always renders
     consistently regardless of font fallback. */
  li::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0.5em;
    width: 0.4em;
    height: 0.4em;
    border-radius: 50%;
    background: #1f3a5f;
  }
  img {
    display: block;
    margin: 0 auto;
    object-fit: contain;
  }
  .layout {
    display: flex;
    gap: 1.0em;
    height: 85%;
    align-items: stretch;
    justify-content: center;
    overflow: hidden;
  }
  .col-text {
    flex: 40;
    min-width: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;
    padding-right: 14px;
    position: relative;
    z-index: 1;
  }
  .col-text ul {
    margin: auto 0;
    align-self: center;
    width: 100%;
  }
  .col-img {
    flex: 60;
    min-width: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    overflow: hidden;
    padding: 0;
    background: transparent;
    position: relative;
    z-index: 0;
  }
  .col-img img {
    width: 100%;
    height: 100%;
    display: block;
    margin: 0 auto;
    object-fit: contain;
    background: transparent;
  }
  .stacked {
    display: flex;
    flex-direction: column;
    height: 88%;
    gap: 0.4em;
    align-items: stretch;
  }
  .stacked .row-img {
    flex: 60;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
    z-index: 0;
  }
  .stacked .row-img img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  .stacked .row-text {
    flex: 40;
    min-height: 0;
    position: relative;
    z-index: 1;
  }
  .stacked .row-text ul {
    margin: 0;
  }
  .full-img {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 88%;
    width: 100%;
    overflow: hidden;
  }
  .full-img img {
    width: 100%;
    height: 100%;
    display: block;
    margin: auto;
    object-fit: contain;
  }
---

"""


def _read_image_dims(image_path: Path) -> tuple[int, int]:
    """Return (width, height) in pixels for a PNG, or (0, 0) on failure."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return int(img.size[0]), int(img.size[1])
    except Exception:
        return 0, 0


def _slide_to_markdown(
    entry: dict[str, Any],
    images_dir: Path | None,
    layout_decider: Any | None = None,
) -> str:
    """Convert one scenario entry to a Marp slide."""
    # Title: keep the model's full wording (h1 wraps to 2 lines if needed).
    # Cap only at a hard 16-word safety limit and never append an ellipsis.
    raw_title = _normalize(str(entry.get("title", "") or "Untitled"))
    _title_words = raw_title.split()
    title = " ".join(_title_words[:16]) if len(_title_words) > 16 else raw_title
    scenario = entry.get("scenario_plan") if isinstance(entry.get("scenario_plan"), dict) else {}


    # Bullets: max 3, each a concise complete sentence (<= 13 words).
    # _compact backs up to the nearest sentence end so bullets never trail off.
    bullet_plan = scenario.get("bullet_plan", [])
    if not isinstance(bullet_plan, list):
        bullet_plan = []
    bullets = []
    for item in bullet_plan:
        if isinstance(item, dict):
            b = _compact(str(item.get("bullet", "") or ""), 18)
        else:
            b = _compact(str(item), 18)
        if b:
            # Append a period to bullets without terminal punctuation.
            if not b.endswith((".", "!", "?", "…", ":")):
                b = b.rstrip(",;") + "."
            bullets.append(b)
        if len(bullets) >= 3:
            break

    # Find diagram image
    slide_num = entry.get("slide_number", 0)
    if not isinstance(slide_num, int):
        try:
            slide_num = int(slide_num)
        except (ValueError, TypeError):
            slide_num = 0
    image_path = None
    if images_dir and images_dir.exists():
        for pattern in [
            f"slide_{slide_num:02d}_diagram_text2diagram.png",
            f"slide_{slide_num:02d}_diagram_render.png",
            f"slide_{slide_num:02d}_*.png",
        ]:
            matches = list(images_dir.glob(pattern))
            if matches:
                image_path = matches[0]
                break

    # Model-driven layout with aspect-ratio heuristics as SAFETY FALLBACK.
    # The LLM picks sizes and column split per slide based on its understanding
    # of the diagram content and bullet density; we only clamp to safe ranges
    # so an off-spec value (e.g. text_ratio=99) can't break the slide.
    layout = dict(scenario.get("layout") or {}) if isinstance(scenario.get("layout"), dict) else {}
    kind = str(layout.get("kind", "")).lower()

    # Read image dimensions so we can both call Prompt 3 with real numbers
    # AND apply a deterministic post-LLM guard against bad layout decisions.
    img_w, img_h = (0, 0)
    if image_path and image_path.exists():
        img_w, img_h = _read_image_dims(image_path)
    img_ratio_real = (img_w / img_h) if img_h > 0 else 0.0

    # Prompt 3: ask the LLM to pick the FINAL layout based on the real
    # rendered PNG dimensions and the actual bullet text. We only override
    # the fields that the existing renderer below already understands —
    # rendering structure / CSS classes are unchanged.
    if layout_decider and img_w > 0 and bullets and kind != "diagram-only":
        try:
            decision = layout_decider(
                slide_w=SLIDE_WIDTH_PX,
                slide_h=SLIDE_HEIGHT_PX,
                image_w=img_w,
                image_h=img_h,
                title=title,
                bullets=bullets,
                draft_layout=layout,
                slide_number=slide_num,
            )
        except Exception as exc:
            print(
                f"  Warning: layout-decider LLM failed for slide {slide_num}: {exc}",
                file=sys.stderr,
            )
            decision = None
        if decision:
            for key in (
                "image_position",
                "text_ratio",
                "bullet_size_em",
                "bullet_line_height",
                "title_size_em",
            ):
                if decision.get(key) is not None:
                    layout[key] = decision[key]

    # For diagrams narrower than 1.7:1, use a side layout.
    # Preserve explicit left/right choices; otherwise alternate by slide number.
    if img_ratio_real > 0 and img_ratio_real < 1.7:
        pos = str(layout.get("image_position", "")).lower()
        if pos in {"top", "bottom"} or pos not in {"left", "right"}:
            layout["image_position"] = "left" if slide_num % 2 == 0 else "right"

    def _clamp_float(val, lo, hi):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        return max(lo, min(hi, v))

    def _clamp_int(val, lo, hi):
        try:
            v = int(float(val))
        except (TypeError, ValueError):
            return None
        return max(lo, min(hi, v))

    # Deterministic pastel palette so every slide has a soft tint and
    # successive slides cycle through the colors. The LLM may still
    # override via layout.background / layout.accent_color.
    _PASTEL_BG = [
        ("#f3f8ff", "#1f3a5f"),  # alice blue / deep navy
        ("#fff8f0", "#7a3e2c"),  # warm cream / dark sienna
        ("#f4f9f3", "#2d5a3d"),  # mint / forest
        ("#fdf6f7", "#7a2c4d"),  # blush / wine
        ("#fbf5e7", "#6b4a1c"),  # parchment / olive
        ("#f0f7fa", "#1f5566"),  # ice / teal
    ]
    bg_default, accent_default = _PASTEL_BG[(slide_num - 1) % len(_PASTEL_BG)]

    slide_style_parts: list[str] = []
    if layout.get("title_size_em") is not None:
        ts = _clamp_float(layout["title_size_em"], 1.0, 1.8)
        if ts is not None:
            slide_style_parts.append(f"h1 {{ font-size: {ts:.2f}em; }}")
    accent = layout.get("accent_color") or accent_default
    slide_style_parts.append(f"h1 {{ border-bottom-color: {accent}; }}")
    bg = layout.get("background")
    if not bg or bg == "white":
        bg = bg_default
    slide_style_parts.append(f"section {{ background: {bg}; }}")

    # `diagram-only` forces the image to fill the slide, ignoring bullets.
    effective_bullets = [] if kind == "diagram-only" else bullets

    # Resolve the adaptive layout fields. Model values win when provided and
    # in-range; everything missing or out-of-range gets filled from the
    # aspect-ratio heuristic (the same `_compute_adaptive_layout` we had).
    if image_path and effective_bullets:
        avg_words = (
            sum(len(b.split()) for b in effective_bullets) / len(effective_bullets)
        )
        fallback = _compute_adaptive_layout(image_path, len(effective_bullets), avg_words)
        model_text_ratio = _clamp_int(layout.get("text_ratio"), 30, 60)
        model_pos = str(layout.get("image_position", "")).strip().lower()
        model_img_size = _clamp_int(layout.get("image_size_pct"), 50, 100)
        model_img_w = _clamp_int(layout.get("diagram_max_width_pct"), 50, 100)
        model_img_h = _clamp_int(layout.get("diagram_max_height_pct"), 50, 100)
        model_bullet_em = _clamp_float(layout.get("bullet_size_em"), 0.85, 1.15)
        model_lh = _clamp_float(layout.get("bullet_line_height"), 1.2, 1.5)

        # Aspect ratio veto: portrait diagrams (ratio < 0.85) look tiny when
        # the LLM forces them into stacked mode below text — the image is
        # already height-constrained so stacking just wastes horizontal
        # room. Force side mode for portraits regardless of the LLM pick.
        portrait_image = float(fallback.get("ratio", 1.0)) < 0.85
        if model_pos in {"top", "bottom"} and not portrait_image:
            mode = "stacked"
            image_first = model_pos == "top"
        elif model_pos in {"left", "right"}:
            # Respect explicit side layouts before applying the aspect-ratio fallback.
            mode = "side"
            image_first = model_pos == "left"
        else:
            mode = fallback["mode"]
            image_first = fallback.get("image_first", False)
        if mode == "stacked":
            text_flex = fallback["text_flex"]
            img_flex = fallback["img_flex"]
        else:
            text_flex = model_text_ratio if model_text_ratio is not None else fallback["text_flex"]
            img_flex = 100 - text_flex if model_text_ratio is not None else fallback["img_flex"]

        if model_img_size is not None:
            img_max_w = img_max_h = model_img_size
        else:
            img_max_w = model_img_w if model_img_w is not None else fallback["img_max_w"]
            img_max_h = model_img_h if model_img_h is not None else fallback["img_max_h"]
        bullet_em = model_bullet_em if model_bullet_em is not None else fallback["bullet_em"]
        line_height = model_lh if model_lh is not None else fallback["line_height"]

        auto = {
            "mode": mode,
            "image_first": image_first,
            "text_flex": text_flex,
            "img_flex": img_flex,
            "img_max_w": img_max_w,
            "img_max_h": img_max_h,
            "bullet_em": bullet_em,
            "line_height": line_height,
        }
        slide_style_parts.append(
            f".col-text ul, .stacked .row-text ul {{ "
            f"font-size: {auto['bullet_em']:.2f}em; line-height: {auto['line_height']:.2f}; }}"
        )
    else:
        auto = None

    slide_style = (
        "<style scoped>\n" + "\n".join(slide_style_parts) + "\n</style>"
        if slide_style_parts
        else ""
    )

    # Build slide markdown (no subtitle — removed entirely)
    lines = [f"# {title}"]
    lines.append("")
    if slide_style:
        lines.append(slide_style)
        lines.append("")

    if image_path and effective_bullets and auto is not None:
        if auto["mode"] == "stacked":
            # Wide diagram → bullets and image stacked vertically. `image_first`
            # decides whether the image leads (top) or follows (bottom).
            img_block = [
                '<div class="row-img">',
                "",
                f'<img src="{image_path}" alt="diagram" '
                f'style="max-width: {auto["img_max_w"]}%; max-height: {auto["img_max_h"]}%;">',
                "",
                "</div>",
            ]
            text_block = ['<div class="row-text">', ""]
            for b in effective_bullets:
                text_block.append(f"- {b}")
            text_block.extend(["", "</div>"])
            lines.append('<div class="stacked">')
            if auto.get("image_first", True):
                lines.extend(img_block)
                lines.extend(text_block)
            else:
                lines.extend(text_block)
                lines.extend(img_block)
            lines.append("</div>")
        else:
            # Side-by-side. image_position=="left" puts the diagram on the
            # LEFT (image col first, text col second); anything else uses the
            # standard text-left / image-right classroom layout.
            image_left = str(layout.get("image_position", "")).strip().lower() == "left"
            text_block = [
                f'<div class="col-text" style="flex: {auto["text_flex"]}">',
                "",
                *[f"- {b}" for b in effective_bullets],
                "",
                "</div>",
            ]
            img_block = [
                f'<div class="col-img" style="flex: {auto["img_flex"]}">',
                "",
                (
                    f'<img src="{image_path}" alt="diagram" '
                    f'style="max-width: {auto["img_max_w"]}%; max-height: {auto["img_max_h"]}%;">'
                ),
                "",
                "</div>",
            ]
            lines.append('<div class="layout">')
            for blk in ((img_block, text_block) if image_left else (text_block, img_block)):
                lines.extend(blk)
            lines.append("</div>")
    elif image_path:
        # diagram-only or no bullets at all -> full-width diagram
        lines.append('<div class="full-img">')
        lines.append("")
        lines.append(
            f'<img src="{image_path}" alt="diagram" '
            f'style="max-width: 98%; max-height: 100%;">'
        )
        lines.append("")
        lines.append("</div>")
    elif bullets:
        for b in bullets:
            lines.append(f"- {b}")

    # Speaker notes (hidden)
    speech = _compact(str(scenario.get("speech_plan", "") or ""), 60)
    if speech:
        lines.append("")
        lines.append(f"<!-- {speech} -->")

    return "\n".join(lines)


def _make_layout_decider() -> Any | None:
    """Return a callable that picks final per-slide layout via Anthropic.

    None when LLM_LAYOUT_DECIDER=0 OR ANTHROPIC_API_KEY is missing OR
    LLMService can't be imported. Caller falls back to the deterministic
    aspect-ratio rule baked into _compute_adaptive_layout in that case.
    """
    if _normalize(os.environ.get("LLM_LAYOUT_DECIDER", "1")).lower() in {"0", "false", "no"}:
        print("LLM layout decider: disabled (LLM_LAYOUT_DECIDER=0).")
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("LLM layout decider: skipped (ANTHROPIC_API_KEY not set).")
        return None
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from llm_service import LLMService
    except Exception as exc:
        print(f"LLM layout decider: failed to import LLMService ({exc}).", file=sys.stderr)
        return None
    model_id = os.environ.get("LAYOUT_MODEL_ID") or os.environ.get(
        "ANTHROPIC_MODEL_ID", "claude-haiku-4-5-20251001"
    )
    try:
        svc = LLMService(provider="anthropic", model_id=model_id, temperature=0.2)
    except Exception as exc:
        print(f"LLM layout decider: init failed ({exc}).", file=sys.stderr)
        return None
    print(f"LLM layout decider: enabled ({model_id}).")
    return svc.decide_final_layout


def build_marp_markdown(
    scenario_entries: list[dict[str, Any]],
    images_dir: Path | None = None,
    layout_decider: Any | None = None,
) -> str:
    """Build complete Marp markdown from scenario entries."""
    parts = [_build_marp_header()]
    for i, entry in enumerate(scenario_entries):
        if i > 0:
            parts.append("\n---\n")
        parts.append(_slide_to_markdown(entry, images_dir, layout_decider=layout_decider))
    return "\n".join(parts)


def export_pptx(md_path: Path, out_path: Path) -> Path:
    """Export Marp markdown to PPTX via marp-cli, reading the deck from stdin.

    Marp v3+ has a bad habit of globbing *.md siblings of the input file
    and refusing the `-o` flag when it finds more than one. To dodge this
    completely we feed the markdown via stdin (`-` as the positional
    argument); Marp then has no path to glob from.
    """
    marp = shutil.which("marp")
    if not marp:
        raise RuntimeError("marp-cli not found. Install: npm install -g @marp-team/marp-cli")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_text = md_path.read_text(encoding="utf-8")
    out_abs = str(out_path.resolve())

    cmd = [
        marp,
        "--pptx",
        "--allow-local-files",
        "-o", out_abs,
        "-",  # read markdown from stdin
    ]
    # CWD = the original markdown's parent so any relative image paths
    # in the deck (rare — we usually emit absolute paths — but possible)
    # still resolve against the run's diagrams folder.
    proc = subprocess.run(
        cmd,
        input=md_text,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        cwd=str(md_path.resolve().parent),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Marp export failed: {proc.stderr or proc.stdout}")
    if not out_path.exists():
        raise RuntimeError(f"Marp did not create output: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Export scenario to PPTX via Marp")
    p.add_argument("--scenario-json", required=True, help="Path to scenario_plan.json")
    p.add_argument("--images-dir", default="", help="Directory with diagram PNGs")
    p.add_argument("--out-dir", default="", help="Output directory (default: same as scenario)")
    args = p.parse_args()

    scenario_path = Path(args.scenario_json).expanduser().resolve()
    if not scenario_path.exists():
        raise SystemExit(f"Missing: {scenario_path}")

    entries = json.loads(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit("scenario_plan.json must be a JSON array")

    images_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else None
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else scenario_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "presentation.md"
    pptx_path = out_dir / "presentation_marp.pptx"

    layout_decider = _make_layout_decider()
    md_content = build_marp_markdown(entries, images_dir=images_dir, layout_decider=layout_decider)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Marp markdown: {md_path}")

    export_pptx(md_path, pptx_path)
    print(f"PPTX exported: {pptx_path}")


if __name__ == "__main__":
    main()
