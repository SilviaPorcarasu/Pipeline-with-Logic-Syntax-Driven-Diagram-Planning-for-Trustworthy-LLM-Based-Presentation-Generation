"""Voiceover export helpers for JSON and plain-text outputs."""

from __future__ import annotations

import json
from pathlib import Path

from models import SlideContent


def write_voiceover_outputs(out_dir: Path, slides: list[SlideContent]) -> tuple[Path, Path]:
    """Export voiceover text per slide as JSON and human-readable TXT."""
    out_dir.mkdir(parents=True, exist_ok=True)

    by_slide: list[dict[str, str | int]] = []
    lines: list[str] = []
    for slide in slides:
        by_slide.append(
            {
                "slide_number": slide.slide_number,
                "title": slide.title,
                "voiceover": slide.voiceover,
            }
        )
        lines.append(f"[Slide {slide.slide_number:02d}] {slide.title}")
        lines.append(slide.voiceover.strip())
        lines.append("")

    voice_json = out_dir / "voiceover.json"
    voice_txt = out_dir / "voiceover.txt"
    voice_json.write_text(json.dumps(by_slide, indent=2), encoding="utf-8")
    voice_txt.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return voice_json, voice_txt
