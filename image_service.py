"""Image query construction, provider search, filtering, and download utilities."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.error import URLError

from PIL import Image, ImageDraw, ImageFont


STOPWORDS = {
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
}

NOISY_QUERY_TERMS = {
    "chunk",
    "source",
    "score",
    "jsonl",
    "dataset",
    "data",
    "book",
    "chapter",
    "section",
    "figure",
    "table",
    "file",
    "image",
    "images",
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
# Generic technical-query tokens — kept topic-agnostic so the same
# pipeline works on ML, biology, finance, or any other domain.
TECHNICAL_QUERY_TOKENS = {
    "algorithm",
    "architecture",
    "diagram",
    "equation",
    "flow",
    "graph",
    "math",
    "matrix",
    "model",
    "process",
    "structure",
    "system",
}
DIAGRAM_HINT_TERMS = {
    "diagram",
    "architecture",
    "flow",
    "flowchart",
    "graph",
    "process",
    "schema",
    "structure",
}
REJECT_TITLE_TERMS = {
    "logo",
    "wordmark",
    "flag",
    "icon",
    "coat of arms",
    "disambiguation",
    "locator map",
    "seal",
}

PAGE3_DIAGRAM_TERMS = {
    "input",
    "output",
    "process",
    "step",
    "stage",
    "node",
    "edge",
    "flow",
    "data",
    "result",
}

GENERIC_LABEL_TERMS = {
    "students",
    "undergraduate",
    "objective",
    "slide",
    "slides",
    "example",
    "overview",
    "introduction",
    "conclusion",
    "summary",
}


def _normalize_space(text: str) -> str:
    """Collapse repeated whitespace and trim leading/trailing spaces."""
    return re.sub(r"\s+", " ", text).strip()



def _bool_env(name: str, default: bool) -> bool:
    raw = _normalize_space(os.environ.get(name, "1" if default else "0")).lower()
    return raw not in {"0", "false", "no", "off"}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into normalized terms used by retrieval/scoring logic."""
    return [
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) >= 4 and tok not in STOPWORDS and not tok.isdigit() and tok not in NOISY_QUERY_TERMS
    ]


def _truncate_words(text: str, max_words: int = 14) -> str:
    words = _normalize_space(text).split()
    if not words:
        return ""
    return " ".join(words[: max(1, int(max_words))])


def _safe_name(text: str) -> str:
    """Sanitize arbitrary text into a filesystem-safe filename fragment."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    return cleaned[:96] or "image"


def _top_terms(text: str, limit: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    cleaned = re.sub(r"\[[^\]]+\]", " ", text)
    for tok in _tokenize(cleaned):
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [tok for tok, _count in ranked[: max(1, int(limit))]]


def _clean_image_suggestion(text: str) -> str:
    value = _normalize_space(text).lower()
    value = re.sub(r"\.(png|jpg|jpeg|webp|bmp|svg)\b", "", value, flags=re.IGNORECASE)
    value = value.replace("_", " ").replace("-", " ")
    tokens = [tok for tok in _tokenize(value) if tok not in NOISY_QUERY_TERMS]
    if not tokens:
        return ""
    return _normalize_space(" ".join(tokens[:10]))


def build_image_query(topic: str, image_suggestion: str, evidence_text: str) -> str:
    """Build a single image search query from topic, suggestion, and evidence."""
    terms = _top_terms(evidence_text, limit=4)
    clean_suggestion = _clean_image_suggestion(image_suggestion)
    pieces = [topic, clean_suggestion, " ".join(terms)]
    return _normalize_space(" ".join([part for part in pieces if _normalize_space(part)]))


def _compact_terms(text: str, max_terms: int = 8) -> str:
    tokens = _tokenize(text)
    if not tokens:
        return ""
    return _normalize_space(" ".join(tokens[: max(1, int(max_terms))]))


def build_image_query_candidates(
    *,
    topic: str,
    title: str,
    image_suggestion: str,
    bullets: list[str],
    evidence_text: str,
) -> list[str]:
    """Generate prioritized image search query candidates for one slide."""
    raw_topic = _truncate_words(_normalize_space(re.sub(r"[^A-Za-z0-9 ]+", " ", topic)), max_words=12)
    raw_title = _truncate_words(_normalize_space(re.sub(r"[^A-Za-z0-9 ]+", " ", title)), max_words=12)
    topic_terms = _compact_terms(topic, max_terms=7)
    title_terms = _compact_terms(title, max_terms=7)
    suggestion_terms = _clean_image_suggestion(image_suggestion)
    bullet_terms = _compact_terms(" ".join(bullets[:2]), max_terms=8)
    evidence_terms = " ".join(_top_terms(evidence_text, limit=4))

    # Start from high-signal combinations and progressively broaden to fallback variants.
    base_parts = [topic_terms, suggestion_terms, evidence_terms]
    base = _normalize_space(" ".join([p for p in base_parts if p]))
    candidates: list[str] = []
    if raw_topic and raw_title:
        candidates.append(_normalize_space(f"{raw_topic} {raw_title} diagram"))
        candidates.append(_normalize_space(f"{raw_topic} {raw_title} english diagram"))
    if base:
        candidates.append(base)
        candidates.append(_normalize_space(f"{base} diagram"))
        candidates.append(_normalize_space(f"{base} english diagram"))
    if topic_terms and title_terms:
        candidates.append(_normalize_space(f"{topic_terms} {title_terms} diagram"))
        candidates.append(_normalize_space(f"{topic_terms} {title_terms} english diagram"))
    if suggestion_terms:
        candidates.append(_normalize_space(f"{suggestion_terms} diagram"))
        candidates.append(_normalize_space(f"{suggestion_terms} infographic"))
    if topic_terms:
        candidates.append(_normalize_space(f"{topic_terms} diagram"))
        candidates.append(_normalize_space(f"{topic_terms} infographic"))
    if bullet_terms:
        candidates.append(_normalize_space(f"{topic_terms} {bullet_terms}"))
    # Extra bullet-focused candidates help avoid repeating one generic image for all slides.
    for bullet in bullets[:3]:
        bullet_query = _compact_terms(bullet, max_terms=7)
        if not bullet_query:
            continue
        if raw_topic:
            candidates.append(_normalize_space(f"{raw_topic} {bullet_query} diagram"))
        candidates.append(_normalize_space(f"{bullet_query} diagram"))
        candidates.append(_normalize_space(f"{bullet_query} educational illustration"))

    # No topic-specific hint injection — the pipeline must stay
    # domain-agnostic. Image search candidates are derived purely
    # from the slide title, bullets, and suggestion text supplied
    # upstream by the LLM.

    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        q = _compact_terms(_normalize_space(candidate), max_terms=10)
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    return deduped[:14]


def _label_from_text(text: str, *, max_words: int = 3, max_chars: int = 26) -> str:
    """Create short readable diagram labels from free text (title/bullets)."""
    words = re.findall(r"[A-Za-z0-9]+", _normalize_space(text))
    if not words:
        return ""
    compact: list[str] = []
    for w in words:
        low = w.lower()
        if low in STOPWORDS or low in NOISY_QUERY_TERMS or low in GENERIC_LABEL_TERMS:
            continue
        compact.append(w.capitalize())
        if len(compact) >= max(1, int(max_words)):
            break
    label = " ".join(compact) if compact else words[0].capitalize()
    if len(label) > max_chars:
        label = label[: max_chars].rsplit(" ", 1)[0]
    return _normalize_space(label) or "Step"


def _unique_labels(candidates: list[str], *, needed: int) -> list[str]:
    """Keep non-empty unique labels and pad when there are not enough."""
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        label = _normalize_space(c)
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
        if len(out) >= needed:
            return out
    while len(out) < needed:
        out.append(f"Step {len(out) + 1}")
    return out


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load Times New Roman when available, with safe fallback chain."""
    candidates = [
        # Times New Roman — primary choice
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/Times_New_Roman.ttf",
        "/usr/share/fonts/TTF/times.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "C:\\Windows\\Fonts\\times.ttf",
        # Fallbacks
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=max(10, int(size)))
        except Exception:
            continue
    # Try by name (works if font is on the system path)
    for name in ("Times New Roman", "TimesNewRoman", "times", "LiberationSerif", "DejaVuSerif"):
        try:
            return ImageFont.truetype(name, size=max(10, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_line(draw: ImageDraw.ImageDraw, *, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Wrap text to a max width using the active font."""
    words = _normalize_space(text).split()
    if not words:
        return []
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word]).strip()
        if int(draw.textlength(trial, font=font)) <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _build_caption(*, slide_number: int, title: str, bullets: list[str]) -> str:
    """Build the figure caption preserving the slide title's natural wording.

    Earlier this passed the title through `_label_from_text` (max_words=8 +
    stopword strip + capitalize), which truncated long titles like
    "Backpropagation Algorithm That Trained Modern Neural Networks" to
    "Modern Neural." Now keep the natural title and only cap the length.
    """
    t = _normalize_space(title or "").strip()
    if not t:
        t = "Process overview"
    if len(t) > 110:
        t = t[:110].rsplit(" ", 1)[0]
    return f"Figure {slide_number}. {t}."


def _draw_diagram_caption(
    draw: ImageDraw.ImageDraw,
    *,
    caption: str,
    width: int,
    y: int,
    font: ImageFont.ImageFont,
) -> None:
    """Render diagram caption in up to three wrapped lines (was two)."""
    lines = _wrap_line(draw, text=caption, font=font, max_width=max(240, width - 180))
    if not lines:
        return
    used = lines[:3]
    if len(lines) > 3:
        last = used[-1]
        ell = "..."
        while last and int(draw.textlength(f"{last}{ell}", font=font)) > max(200, width - 220):
            last = last[:-1].rstrip()
        used[-1] = f"{last}{ell}" if last else ell
    cursor = y
    for line in used:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = int(bbox[2] - bbox[0])
        x = max(40, (width - line_w) // 2)
        draw.text((x, cursor), line, fill=(56, 74, 107), font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        cursor += int(bbox[3] - bbox[1]) + 6


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from model outputs."""
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:d2|mermaid|dot|graphviz|plantuml)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _diagram_renderer_strict() -> bool:
    """When enabled, diagram renderer failures should not silently fallback."""
    return _bool_env("DIAGRAM_RENDERER_STRICT", False)


def _cleanup_file(path: Path) -> None:
    """Silently remove a file if it exists."""
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _normalize_shape_value(raw_shape: str) -> str:
    """Normalize a D2 shape value to a known allowed shape name."""
    shape = _normalize_space(str(raw_shape)).strip("\"' ").lower()
    aliases = {
        "note": "rectangle",
        "sticky_note": "rectangle",
        "sticky-note": "rectangle",
        "note_card": "rectangle",
        "note-card": "rectangle",
        "card": "rectangle",
        "box": "rectangle",
        "process": "rectangle",
        "terminator": "oval",
        "start": "oval",
        "end": "oval",
    }
    allowed = {
        "rectangle",
        "square",
        "oval",
        "circle",
        "diamond",
        "hexagon",
        "cloud",
        "parallelogram",
        "document",
        "cylinder",
        "queue",
        "package",
        "step",
        "callout",
        "stored_data",
        "person",
        "page",
        "text",
        "code",
        "class",
        "sql_table",
        "image",
    }
    shape = aliases.get(shape, shape)
    if shape not in allowed:
        return "rectangle"
    return shape


def _normalize_shape_line(line: str) -> str:
    """Normalize a D2 shape declaration line to use a valid shape value."""
    m = re.match(r'^([A-Za-z0-9_.-]+\.)?shape\s*:\s*"?([A-Za-z0-9_-]+)"?\s*$', line, re.IGNORECASE)
    if not m:
        return line
    prefix = m.group(1) or ""
    shape = _normalize_shape_value(m.group(2))
    return f"{prefix}shape: {shape}"


def _extract_d2_block(text: str) -> str:
    """Keep only D2 code from mixed text/code payloads."""
    if not text:
        return ""

    code = _strip_code_fences(text).replace("\\n", "\n")
    raw_lines = [ln.rstrip() for ln in code.splitlines() if _normalize_space(ln)]
    lines: list[str] = []
    for raw in raw_lines:
        ln = raw.strip()
        low = ln.lower()
        # Preserve global direction directive as-is.
        if low.startswith("direction:"):
            direction = "down" if "down" in low else "right"
            lines.append(f"direction: {direction}")
            continue
        # Normalize bare font-size to style.font-size.
        if re.match(r"^font-size\s*:", low):
            ln = re.sub(r"(?i)^font-size\s*:", "style.font-size:", ln, count=1).strip()
            low = ln.lower()
        ln = _normalize_shape_line(ln)
        low = ln.lower()
        # --- Edge hardening ---
        # Normalize undirected edges to directed.
        ln = ln.replace(' -- ', ' -> ')
        # Remove port suffixes (.south, .north, .east, .west).
        ln = re.sub(r'\.(south|north|east|west)\b', '', ln)
        # Validate edges: drop lines where source or destination is empty.
        if '->' in ln:
            parts = ln.split('->')
            if any(p.strip() == '' for p in parts):
                continue
        if _normalize_space(ln):
            lines.append(ln)

    if not lines:
        return ""
    if lines and lines[0].strip().lower() in {"d2", "d2:"}:
        lines = lines[1:]
    if not lines:
        return ""
    if not any(ln.strip().lower().startswith("direction:") for ln in lines):
        lines.insert(0, "direction: right")
    if len(lines) > 36:
        lines = lines[:36]
    out = "\n".join(lines).strip()
    if len(out) > 1800:
        out = out[:1800].rsplit("\n", 1)[0].strip()
    low = out.lower()
    # D2 has flexible syntax; we check for common structural markers.
    if "->" in out and any(
        marker in low for marker in ("direction:", "shape:", "style:", "classes:", "class:", "near:")
    ):
        return out
    if "->" in out and "\n" in out:
        return out
    return ""


def _autocrop_png(path: Path, margin: int = 8) -> None:
    """Trim transparent borders so the diagram fills its bounding box.

    Called after rsvg-convert. Best-effort: any failure is swallowed so
    the original PNG stays in place. Adds a tiny margin so antialiased
    strokes on the edge aren't clipped.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return
    try:
        with Image.open(path) as im:
            if im.mode != "RGBA":
                im = im.convert("RGBA")
            bbox = im.getbbox()
            if not bbox:
                return
            l, t, r, b = bbox
            l = max(0, l - margin)
            t = max(0, t - margin)
            r = min(im.width, r + margin)
            b = min(im.height, b + margin)
            if (r - l) < im.width or (b - t) < im.height:
                im.crop((l, t, r, b)).save(path)
    except Exception:
        return


def _quote_labels_with_parens(d2_code: str) -> str:
    """Wrap node labels in double quotes when they contain `(`, `)`, `=`, or `·`.

    D2 truncates unquoted labels at the first `(`, so `f(x)` renders as `f(x`
    and `y=h(g(f(x)))` becomes `y=h(g(f(x`. The LLM is instructed to quote
    these but sometimes forgets — this is the safety net.
    """
    if not d2_code:
        return d2_code
    out: list[str] = []
    # Aggressive quote trigger: anything outside [letters, digits, _, space,
    # hyphen, period, basic punctuation] forces double-quote wrapping. This
    # catches every paren/bracket/operator/math symbol the LLM might leave
    # bare, so D2 never truncates a label at an unquoted `(`/`[`/`=`.
    needs_quote_re = re.compile(r"[^\w\s\-.,!?'](?<!　)", re.UNICODE)
    edge_label_re = re.compile(
        r"^(\s*[\w.\-]+\s*(?:->|<-)\s*[\w.\-]+\s*):\s*(.+?)\s*$"
    )
    for raw in d2_code.splitlines():
        # Edge lines: handle inline edge labels like `a -> b: f(g(x))`.
        # Without this, D2 truncates the edge label at the first `(`.
        if "->" in raw or "<-" in raw:
            em = edge_label_re.match(raw)
            if em:
                edge_part, edge_label = em.groups()
                edge_label = edge_label.strip()
                if edge_label and not edge_label.startswith('"') and needs_quote_re.search(edge_label):
                    safe = edge_label.replace('"', "'")
                    out.append(f'{edge_part}: "{safe}"')
                    continue
            out.append(raw)
            continue
        s = raw.lstrip()
        if not s or s.startswith(("#", "style.", "direction:", "vars:", "classes:", "}")):
            out.append(raw)
            continue
        m = re.match(r"^(\s*)([\w.\-]+)\s*:\s*(.+?)(\s*\{.*)?$", raw)
        if not m:
            out.append(raw)
            continue
        indent, node_id, label, suffix = m.groups()
        label = (label or "").rstrip()
        suffix = suffix or ""
        if not label or label.startswith('"'):
            out.append(raw)
            continue
        if needs_quote_re.search(label):
            safe = label.replace('"', "'")
            out.append(f"{indent}{node_id}: \"{safe}\"{suffix}")
        else:
            out.append(raw)
    return "\n".join(out)


_PLACEHOLDER_LABEL_RE = re.compile(r"^[\s\?\.\-_]+$|^tbd$|^<.*>$", re.IGNORECASE)
_D2_NODE_LINE_RE = re.compile(r"^(\s*)([\w.\-]+)\s*:\s*(.+?)(\s*\{.*)?$")


def _is_placeholder_label(label: str) -> bool:
    """Return True if a node label is empty/placeholder (`?`, `...`, `TBD`, ...)."""
    text = (label or "").strip().strip('"').strip()
    if not text:
        return True
    return bool(_PLACEHOLDER_LABEL_RE.match(text))


def _find_placeholder_node_ids(d2_code: str) -> list[str]:
    """Return the list of fully-qualified node IDs whose label is a placeholder."""
    if not d2_code:
        return []
    container_stack: list[str] = []
    found: list[str] = []
    for raw in d2_code.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s == "}" or s.startswith("}"):
            if container_stack:
                container_stack.pop()
            continue
        if "->" in s or "<-" in s:
            continue
        if s.startswith(("style.", "direction:", "vars:", "classes:")):
            continue
        m = _D2_NODE_LINE_RE.match(raw)
        if not m:
            if s.endswith("{"):
                cid = s.split(":", 1)[0].strip()
                if cid:
                    container_stack.append(cid)
            continue
        _, node_id, label, suffix = m.groups()
        is_container = bool(suffix and "{" in suffix and "}" not in suffix)
        full_id = ".".join(container_stack + [node_id]) if container_stack else node_id
        if not is_container and _is_placeholder_label(label):
            found.append(full_id)
        if is_container:
            container_stack.append(node_id)
    return found


def _drop_d2_nodes(d2_code: str, drop_ids: set[str]) -> str:
    """Remove placeholder D2 node declarations and any edges that reference them."""
    if not d2_code or not drop_ids:
        return d2_code
    short_drop = {fid.split(".")[-1] for fid in drop_ids}
    out: list[str] = []
    container_stack: list[str] = []
    for raw in d2_code.splitlines():
        s = raw.strip()
        if not s:
            out.append(raw)
            continue
        if s == "}" or s.startswith("}"):
            if container_stack:
                container_stack.pop()
            out.append(raw)
            continue
        if "->" in s or "<-" in s:
            edge_body = s.split(":", 1)[0]
            tokens = re.split(r"\s*->\s*|\s*<-\s*", edge_body)
            if any(tok.strip() in drop_ids or tok.strip() in short_drop for tok in tokens):
                continue
            out.append(raw)
            continue
        m = _D2_NODE_LINE_RE.match(raw)
        if m:
            _, node_id, _, suffix = m.groups()
            is_container = bool(suffix and "{" in suffix and "}" not in suffix)
            full_id = ".".join(container_stack + [node_id]) if container_stack else node_id
            if not is_container and (full_id in drop_ids or node_id in short_drop):
                continue
            out.append(raw)
            if is_container:
                container_stack.append(node_id)
            continue
        out.append(raw)
    return "\n".join(out)


def _suggest_labels_via_openai(
    *,
    placeholder_ids: list[str],
    title: str,
    bullets: list[str],
    diagram_description: str,
) -> dict[str, str]:
    """Ask OpenAI for contextual labels for placeholder D2 nodes.

    Returns a mapping `node_id -> human-readable label`. Returns `{}` when
    `OPENAI_API_KEY` is not set or the call fails. Caller decides what to do
    with missing entries (typically: drop the node).
    """
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or not placeholder_ids:
        return {}
    model_id = (
        (os.environ.get("OPENAI_DIAGRAM_LABEL_MODEL_ID") or "").strip()
        or (os.environ.get("OPENAI_MODEL_ID") or "").strip()
        or "gpt-4o-mini"
    )
    bullet_lines = "\n".join(f"- {b}" for b in bullets[:6] if b)
    desc = _normalize_space(diagram_description)[:400]
    user_prompt = (
        f"Slide title: {title or '(untitled)'}\n"
        f"Slide bullets:\n{bullet_lines or '(none)'}\n"
        f"Diagram description: {desc or '(none)'}\n\n"
        f"Some nodes in the D2 diagram have placeholder labels (`?`, `...`, empty).\n"
        f"For each node id below, suggest a SHORT (2-4 words) human-readable label\n"
        f"that fits the slide topic. Respond ONLY with JSON of the form\n"
        f'{{"labels": {{"<node_id>": "<label>", ...}}}}.\n'
        f"Node ids needing labels: {', '.join(placeholder_ids)}"
    )
    payload = {
        "model": model_id,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "max_tokens": 300,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate short, descriptive English labels for nodes in an "
                    "educational diagram. Never return placeholders like `?`, `...`, "
                    "`TBD`, or single letters. Keep each label 2-4 words."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        import json as _json

        import requests  # local import keeps PIL-only callers light

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            return {}
        content = (
            resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        try:
            parsed = _json.loads(content)
        except Exception:
            return {}
        labels = parsed.get("labels") if isinstance(parsed, dict) else None
        if not isinstance(labels, dict):
            return {}
        out: dict[str, str] = {}
        for nid, lbl in labels.items():
            text = _normalize_space(str(lbl or "")).strip('"').strip()
            if not text or _is_placeholder_label(text):
                continue
            out[str(nid)] = text
        return out
    except Exception:
        return {}


def _apply_label_overrides(d2_code: str, overrides: dict[str, str]) -> str:
    """Replace D2 node labels for the given fully-qualified ids."""
    if not d2_code or not overrides:
        return d2_code
    short_overrides = {fid.split(".")[-1]: text for fid, text in overrides.items()}
    out: list[str] = []
    container_stack: list[str] = []
    for raw in d2_code.splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "->" in s or "<-" in s:
            out.append(raw)
            continue
        if s == "}" or s.startswith("}"):
            if container_stack:
                container_stack.pop()
            out.append(raw)
            continue
        if s.startswith(("style.", "direction:", "vars:", "classes:")):
            out.append(raw)
            continue
        m = _D2_NODE_LINE_RE.match(raw)
        if not m:
            out.append(raw)
            continue
        indent, node_id, label, suffix = m.groups()
        is_container = bool(suffix and "{" in suffix and "}" not in suffix)
        full_id = ".".join(container_stack + [node_id]) if container_stack else node_id
        new_label = overrides.get(full_id) or short_overrides.get(node_id)
        if not is_container and new_label and _is_placeholder_label(label):
            safe = new_label.replace('"', "'")
            out.append(f'{indent}{node_id}: "{safe}"{suffix or ""}')
        else:
            out.append(raw)
        if is_container:
            container_stack.append(node_id)
    return "\n".join(out)


def _resolve_placeholder_labels(
    d2_code: str,
    *,
    title: str = "",
    bullets: list[str] | None = None,
    diagram_description: str = "",
) -> str:
    """Replace placeholder labels (`?`, `...`, empty) with contextual names.

    Strategy:
      1. Find all placeholder nodes.
      2. Ask OpenAI for human-readable labels (only if `OPENAI_API_KEY` is set).
      3. Apply whatever labels came back; drop any nodes still left as
         placeholders so D2 doesn't render `?` glyphs. Edges referencing the
         dropped nodes are removed in the same pass.
    """
    if not d2_code:
        return d2_code
    placeholder_ids = _find_placeholder_node_ids(d2_code)
    if not placeholder_ids:
        return d2_code
    overrides = _suggest_labels_via_openai(
        placeholder_ids=placeholder_ids,
        title=title,
        bullets=list(bullets or []),
        diagram_description=diagram_description,
    )
    code = _apply_label_overrides(d2_code, overrides) if overrides else d2_code
    remaining = [pid for pid in placeholder_ids if pid not in overrides]
    if remaining:
        code = _drop_d2_nodes(code, set(remaining))
    return code


# Positional keywords D2 reserves for things like `near: top-center`. As
# node IDs (even nested: `tree.left -> tree.right`) D2 rejects them with
# `"left" must be the last part of the key`. ALWAYS rename when used as
# an ID, including after a dot.
_D2_POSITIONAL_KEYWORDS = {
    "left", "right", "top", "bottom",
    "north", "south", "east", "west",
    "near",
}

# D2 attribute names. As bare node IDs (`label: "x"` at top level) they'd
# collide, but as property accessors (`Input.shape: oval`, `Node.label: x`)
# they're legitimate D2 syntax — must NOT be rewritten there.
_D2_ATTRIBUTE_KEYWORDS = {
    "shape", "label", "style", "direction", "icon", "link",
    "classes", "class", "source", "target", "layers",
    "scenarios", "steps", "vars", "level", "width", "height",
}

_D2_RESERVED_IDS = _D2_POSITIONAL_KEYWORDS | _D2_ATTRIBUTE_KEYWORDS


def _rename_reserved_ids(d2_code: str) -> str:
    """Rename node IDs that collide with D2 reserved keywords.

    D2 v0.6.5 throws "reserved keywords are prohibited in edges" and
    `"left" must be the last part of the key` when a node ID equals
    a reserved word like `left`, `right`, `top`, `north`. Sonnet
    happily emits `root -> left` because it's natural English. The
    safety net renames any standalone token that matches a reserved
    keyword to `n_<keyword>` (e.g. `left` -> `n_left`) wherever it
    sits as a node id (line start before `:`, on either side of `->`,
    or as the final piece of a dotted path).
    """
    if not d2_code:
        return d2_code
    # Rename reserved node IDs case-insensitively, including nested IDs.
    # Preserve attribute access such as Input.shape.
    pos_pat = re.compile(
        rf"(?<![A-Za-z0-9_])({'|'.join(re.escape(k) for k in sorted(_D2_POSITIONAL_KEYWORDS, key=len, reverse=True))})(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    attr_pat = re.compile(
        rf"(?<![A-Za-z0-9_])(?<!\.)({'|'.join(re.escape(k) for k in sorted(_D2_ATTRIBUTE_KEYWORDS, key=len, reverse=True))})(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    # Rename reserved keywords in intermediate key segments.
    # Preserve final attributes and the style namespace.
    _mid_kw_set = _D2_ATTRIBUTE_KEYWORDS - {"style", "vars", "classes", "layers", "scenarios", "steps"}
    mid_attr_pat = re.compile(
        rf"\.({'|'.join(re.escape(k) for k in sorted(_mid_kw_set, key=len, reverse=True))})(?=\.)",
        re.IGNORECASE,
    )
    # Prefix reserved keywords followed by digits for D2 v0.6.5 compatibility.
    _all_kw = _D2_POSITIONAL_KEYWORDS | _mid_kw_set
    fuzzy_pat = re.compile(
        rf"(?<![A-Za-z0-9_])({'|'.join(re.escape(k) for k in sorted(_all_kw, key=len, reverse=True))})(\d+)(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    out_lines: list[str] = []
    for raw in d2_code.splitlines():
        # Skip lines where the keyword belongs naturally — directives like
        # `direction: right`, `shape: rectangle`, attribute lines like
        # `style.fill`, and any line inside a quoted label.
        stripped = raw.lstrip()
        # Preserve attribute lines, but rename container IDs to match their references.
        stripped_low = stripped.lower()
        is_attr_line = stripped_low.startswith((
            "direction:", "shape:", "style.", "near:",
            "vars:", "classes:", "class:", "icon:",
            "label:", "link:", "source:", "target:",
            "layers:", "scenarios:", "steps:",
            "level:", "width:", "height:",
        ))
        if is_attr_line and "{" not in stripped:
            out_lines.append(raw)
            continue
        # Don't rewrite tokens inside double-quoted strings (labels).
        parts = re.split(r'("[^"]*")', raw)
        rebuilt = []
        for piece in parts:
            if piece.startswith('"') and piece.endswith('"'):
                rebuilt.append(piece)
            else:
                piece = pos_pat.sub(r"n_\1", piece)
                piece = attr_pat.sub(r"n_\1", piece)
                piece = mid_attr_pat.sub(r".n_\1", piece)
                piece = fuzzy_pat.sub(r"n_\1\2", piece)
                rebuilt.append(piece)
        out_lines.append("".join(rebuilt))
    return "\n".join(out_lines)


_SAFE_PASTEL_FILLS = {
    "aliceblue", "honeydew", "lavenderblush", "mistyrose",
    "papayawhip", "lemonchiffon", "mintcream", "lavender",
    "peachpuff", "white", "lightblue", "lightyellow",
    "lightgray", "lightgrey", "beige", "ghostwhite",
    "whitesmoke", "seashell", "ivory", "snow", "linen",
    "antiquewhite", "floralwhite", "oldlace", "cornsilk",
    "blanchedalmond", "bisque", "moccasin", "wheat",
    "navajowhite", "thistle", "plum", "lightpink",
    "pink", "lightskyblue", "powderblue", "lightcyan",
    "paleturquoise", "azure",
}


def _sanitize_d2_colors(d2_code: str) -> str:
    """Replace any `style.fill: "<color>"` value not in our pastel safelist
    with `aliceblue` so D2 v0.6.5 doesn't reject the file outright.

    D2 v0.6.5 only accepts CSS named colors plus a few extras; v0.7+ added
    more. Sonnet sometimes emits names like `lightgreen`, `salmon`, or
    `coral` that v0.6.5 rejects. Keeping our own narrow pastel safelist
    plus a hex-pass-through means the diagram still compiles.
    """
    if not d2_code:
        return d2_code
    pattern = re.compile(
        r'(style\.fill\s*:\s*)"([^"]+)"',
        re.IGNORECASE,
    )

    def _replace(match: re.Match) -> str:
        prefix, color = match.group(1), match.group(2)
        if color.startswith("#") and re.fullmatch(r"#[0-9A-Fa-f]{3,8}", color):
            return f'{prefix}"{color}"'
        if color.lower() in _SAFE_PASTEL_FILLS:
            return f'{prefix}"{color.lower()}"'
        return f'{prefix}"aliceblue"'

    return pattern.sub(_replace, d2_code)


def _wrap_long_labels(d2_code: str, *, max_len: int = 20) -> str:
    """Break long quoted labels onto multiple lines so text stops overflowing
    its shape boundary.

    D2 lays each shape around its label text. Long math labels like
    `"∂L/∂W=∂L/∂a·∂a/∂z·∂z/∂W"` push the bounding box wider than the
    container holding the node, so the label visibly leaks past the
    container border. Inserting line breaks at natural separators
    (=, ·, ⊙, /, comma, space) keeps the label compact.
    """
    if not d2_code:
        return d2_code

    def _wrap(text: str) -> str:
        if len(text) <= max_len:
            return text
        breakpoints = [" = ", "=", " · ", "·", " ⊙ ", "⊙", ", ", ",", " "]

        def _split_once(s: str) -> tuple[str, str] | None:
            for sep in breakpoints:
                idx = s.rfind(sep, 1, max_len + len(sep))
                if idx > 0:
                    cut = idx + len(sep)
                    return s[:cut].rstrip(), s[cut:].lstrip()
            if len(s) > max_len:
                return s[:max_len].rstrip(), s[max_len:].lstrip()
            return None

        out_parts: list[str] = []
        remaining = text
        for _ in range(8):
            if len(remaining) <= max_len:
                break
            split = _split_once(remaining)
            if not split:
                break
            chunk, remaining = split
            if chunk:
                out_parts.append(chunk)
        if remaining:
            out_parts.append(remaining)
        return "\\n".join(out_parts)

    out_lines: list[str] = []
    for raw in d2_code.splitlines():
        m = re.match(r'^(\s*[\w.\-]+\s*:\s*)"([^"]+)"(\s*\{?.*)$', raw)
        if not m:
            out_lines.append(raw)
            continue
        prefix, label, suffix = m.groups()
        new_label = _wrap(label)
        if new_label != label:
            out_lines.append(f'{prefix}"{new_label}"{suffix}')
        else:
            out_lines.append(raw)
    return "\n".join(out_lines)


def _repair_d2_code_for_compile(d2_code: str) -> str:
    """Apply a minimal repair pass for common D2 compile failures from LLM output."""
    if not d2_code:
        return ""

    out_lines: list[str] = []
    for raw in d2_code.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        # D2 v0.6.x rejects "Times New Roman" / similar custom fonts —
        # the LLM keeps emitting them. Drop the directive entirely so
        # the renderer falls back to its bundled font.
        if re.match(r"(?i)^style\.font\s*:", ln):
            continue
        # D2 v0.6.x complains "class must be the last part of the key"
        # when classes are referenced like `node.class: stage`. Drop
        # those lines too — visual styling lost but compile succeeds.
        if re.match(r"(?i)^[\w.\-]+\.class\s*:", ln):
            continue
        low = ln.lower()
        # Normalize bare font-size to style.font-size.
        if re.match(r"^font-size\s*:", low):
            ln = re.sub(r"(?i)^font-size\s*:", "style.font-size:", ln, count=1).strip()
            low = ln.lower()
        ln = _normalize_shape_line(ln)
        low = ln.lower()
        # --- Edge hardening ---
        # Normalize undirected edges to directed.
        ln = ln.replace(' -- ', ' -> ')
        # Remove port suffixes (.south, .north, .east, .west).
        ln = re.sub(r'\.(south|north|east|west)\b', '', ln)
        # Validate edges: drop lines where source or destination is empty.
        if '->' in ln:
            parts = ln.split('->')
            if any(p.strip() == '' for p in parts):
                continue
        if _normalize_space(ln):
            out_lines.append(ln)
    if not out_lines:
        return ""
    if not any(ln.lower().startswith("direction:") for ln in out_lines):
        out_lines.insert(0, "direction: right")
    if len(out_lines) > 36:
        out_lines = out_lines[:36]
    repaired = "\n".join(out_lines).strip()
    # Fix unnamed containers: `group1 {` or `group1: {` → `group1: Group 1 {`
    # D2 uses the raw ID as the visible header when no label follows the colon.
    _unnamed_ids = r'(group|grp|container|stage|phase|section|block)\d*'
    def _label_container(m: re.Match) -> str:
        indent, cid = m.group(1), m.group(2)
        # Insert space before digit runs: group1 → group 1, stage2 → stage 2
        human = re.sub(r'(\D)(\d)', r'\1 \2', cid).replace("_", " ").replace("-", " ").title()
        return f"{indent}{cid}: {human} {{"
    # Case 1: `group1 {`  (no colon at all)
    repaired = re.sub(
        rf'^(\s*)({_unnamed_ids})\s*\{{',
        _label_container,
        repaired,
        flags=re.MULTILINE,
    )
    # Case 2: `group1: {`  (colon but no label before brace)
    repaired = re.sub(
        rf'^(\s*)({_unnamed_ids})\s*:\s*\{{',
        _label_container,
        repaired,
        flags=re.MULTILINE,
    )
    if len(repaired) > 1800:
        repaired = repaired[:1800].rsplit("\n", 1)[0].strip()
    # Close truncated D2 blocks, ignoring braces inside quoted labels.
    no_str = re.sub(r'"[^"\n]*"', "", repaired)
    open_braces = no_str.count("{")
    close_braces = no_str.count("}")
    if open_braces > close_braces:
        repaired = repaired + "\n" + "}\n" * (open_braces - close_braces)
    return repaired


def _render_d2_png(*, d2_code: str, out_path: Path) -> tuple[bool, str]:
    """Render D2 code to PNG via SVG path (preferred) with direct PNG fallback."""
    d2 = shutil.which("d2")
    if not d2:
        return False, "d2_not_installed"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_path = out_path.with_suffix(".d2")
    svg_path = out_path.with_suffix(".svg")
    layout = _normalize_space(os.environ.get("D2_LAYOUT", "elk")) or "elk"

    # Padding prevents clipped strokes. Omit --center for older D2 versions.
    # The rendered PNG is transparent and trimmed to the diagram bounds.
    pad = _normalize_space(os.environ.get("D2_PAD", "10")) or "10"
    base_flags = ["--layout", layout, "--pad", pad]

    def _run_once(code: str) -> tuple[bool, str]:
        # Keep labels readable after scaling the diagram into a slide column.
        font_size = _normalize_space(os.environ.get("D2_FONT_SIZE", "56")) or "56"
        code_with_font = code.rstrip() + f"\nstyle.font-size: {font_size}\n"
        src_path.write_text(code_with_font, encoding="utf-8")
        cmd_svg = [d2, *base_flags, str(src_path), str(svg_path)]
        proc_svg = subprocess.run(cmd_svg, capture_output=True, text=True, timeout=60, check=False)
        if proc_svg.returncode != 0:
            return False, _normalize_space(proc_svg.stderr or proc_svg.stdout or "d2_svg_failed")
        if not svg_path.exists() or svg_path.stat().st_size <= 0:
            return False, "d2_empty_svg_output"
        rsvg = shutil.which("rsvg-convert")
        if rsvg:
            cmd_png = [rsvg, "-b", "none", "-w", "3840", "-h", "2160", str(svg_path), "-o", str(out_path)]
            proc_png = subprocess.run(cmd_png, capture_output=True, text=True, timeout=60, check=False)
            if proc_png.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                _autocrop_png(out_path)
                return True, "ok"
            # rsvg found but failed — fall through to cairosvg / direct
        # cairosvg fallback (pip install cairosvg) — no system dependency needed
        try:
            import cairosvg  # noqa: WPS433
            cairosvg.svg2png(
                url=str(svg_path),
                write_to=str(out_path),
                output_width=3840,
                background_color=None,
            )
            if out_path.exists() and out_path.stat().st_size > 0:
                _autocrop_png(out_path)
                return True, "ok_cairosvg"
        except (ImportError, Exception):
            pass
        cmd_direct = [d2, *base_flags, str(src_path), str(out_path)]
        proc_direct = subprocess.run(cmd_direct, capture_output=True, text=True, timeout=90, check=False)
        if proc_direct.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return True, "ok_direct_png"
        return False, _normalize_space(proc_direct.stderr or proc_direct.stdout or "d2_png_failed")

    succeeded = False
    try:
        # Always pre-quote labels with parens so D2 doesn't truncate them.
        # Placeholder-label rewriting (`?`, `...`) needs slide context, so it
        # runs upstream in `_generate_text_to_diagram` before reaching here.
        prepped = _quote_labels_with_parens(d2_code)
        prepped = _wrap_long_labels(prepped)
        prepped = _sanitize_d2_colors(prepped)
        prepped = _rename_reserved_ids(prepped)
        # Strip incompatible directives the LLM keeps adding (Times New
        # Roman font, dotted class refs) so the FIRST compile pass has a
        # chance instead of always falling through to the repair retry.
        prepped = "\n".join(
            ln for ln in prepped.splitlines()
            if not re.match(r"(?i)^\s*style\.font\s*:", ln)
            and not re.match(r"(?i)^\s*[\w.\-]+\.class\s*:", ln)
        )
        ok, reason = _run_once(prepped)
        if ok:
            succeeded = True
            return True, reason
        # Second pass: repair common malformed tokens and retry.
        repaired = _repair_d2_code_for_compile(prepped)
        if repaired and repaired != prepped:
            ok2, reason2 = _run_once(repaired)
            if ok2:
                succeeded = True
                return True, "ok_repaired"
            return False, reason2
        return False, reason
    except Exception as exc:
        return False, f"d2_exception:{exc}"
    finally:
        # Keep the .d2 source on failure so we can grep the exact LLM
        # output that broke the compile. Strict mode raises immediately,
        # so if we delete here the post-mortem evidence is gone.
        if succeeded:
            _cleanup_file(src_path)
            _cleanup_file(svg_path)


def _compose_diagram_canvas(
    *,
    rendered_path: Path,
    out_path: Path,
    slide_number: int,
    title: str,
    bullets: list[str],
) -> None:
    """Normalize rendered diagram to project style and append figure caption.

    Caption now occupies up to 3 lines (was 2) so long titles like
    "Backpropagation Algorithm That Trained Modern Neural Networks" don't
    get truncated. We reserve ~150 px at the bottom for the caption block
    instead of the previous 56 px (which only fit a single line cleanly).
    """
    width, height = 2560, 1440
    # Preserve transparency so the slide background shows around the diagram.
    canvas = Image.new("RGBA", (width, height), color=(255, 255, 255, 0))
    with Image.open(rendered_path) as src:
        src_rgba = src.convert("RGBA")
        target_w = max(400, width - 60)
        # Leave 170 px at the bottom for the 3-line caption band so the
        # diagram doesn't overlap with the caption text.
        target_h = height - 200
        src_rgba.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
        x = (width - src_rgba.width) // 2
        y = max(40, (target_h - src_rgba.height) // 2)
        canvas.alpha_composite(src_rgba, (x, y))

    caption_layer = Image.new("RGBA", (width, height), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(caption_layer)
    caption_font = _load_font(36)
    caption = _build_caption(slide_number=slide_number, title=title, bullets=bullets)
    _draw_diagram_caption(draw, caption=caption, width=width, y=height - 150, font=caption_font)
    # Save as RGBA so the transparency survives — `.convert("RGB")` would
    # flatten everything onto a white background and defeat the change.
    final = Image.alpha_composite(canvas, caption_layer)
    final.save(out_path)


_ALLOWED_D2_SHAPES = {"rectangle", "oval", "diamond", "hexagon", "cylinder", "cloud"}


def _parse_diagram_description(diagram_description: str) -> str:
    """Convert a `nodes: X (shape, Label), ... | arrows: A -> B, ...` string
    into D2 code. This is the prompt-2.2 format and lets every slide have its
    own structure even when the LLM D2 formatter silently returns empty.

    Returns empty string if parsing fails.
    """
    text = _normalize_space(diagram_description or "")
    if not text or "nodes:" not in text.lower():
        return ""

    import re

    # Split on `|` into sections (nodes / arrows / direction / notes).
    sections: dict[str, str] = {}
    for part in text.split("|"):
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        sections[k.strip().lower()] = v.strip()

    nodes_raw = sections.get("nodes", "")
    arrows_raw = sections.get("arrows", "")
    direction = sections.get("direction", "right").lower()
    if direction not in {"right", "down", "up", "left"}:
        direction = "right"

    # Split at node boundaries rather than closing parentheses,
    # since labels may contain nested parentheses.
    node_chunks = re.split(
        r"\)\s*,\s*(?=[A-Za-z_]\w*\s*\()",
        nodes_raw,
    )
    chunk_pattern = re.compile(
        r"^\s*([A-Za-z_]\w*)\s*\(\s*([a-zA-Z]+)\s*,\s*(.+?)\)?\s*$"
    )
    nodes: list[tuple[str, str, str]] = []
    for chunk in node_chunks:
        chunk = chunk.strip().rstrip(",").strip()
        if not chunk:
            continue
        m = chunk_pattern.match(chunk)
        if not m:
            continue
        nid = m.group(1).strip()
        shape = m.group(2).strip().lower()
        label = m.group(3).strip().strip('"').strip("'")
        # Restore any closing paren the split swallowed when balanced.
        if label.count("(") > label.count(")"):
            label += ")" * (label.count("(") - label.count(")"))
        if shape not in _ALLOWED_D2_SHAPES:
            shape = "rectangle"
        if nid and label:
            nodes.append((nid, shape, label))

    if len(nodes) < 2:
        return ""

    valid_ids = {n[0] for n in nodes}
    arrow_pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)")
    arrows: list[tuple[str, str]] = []
    for m in arrow_pattern.finditer(arrows_raw):
        src, dst = m.group(1).strip(), m.group(2).strip()
        if src in valid_ids and dst in valid_ids and src != dst:
            arrows.append((src, dst))

    if not arrows:
        # Chain them linearly as a graceful degradation.
        for i in range(len(nodes) - 1):
            arrows.append((nodes[i][0], nodes[i + 1][0]))

    # Parse grouping hints from notes and wrap matching nodes in D2 containers.
    notes_raw = sections.get("notes", "")
    container_assignments: dict[str, list[str]] = {}
    name_by_lower_id = {n[0].lower(): n[0] for n in nodes}
    name_aliases: dict[str, str] = {}
    for nid, _, label in nodes:
        for tok in re.findall(r"[A-Za-z_]\w*", label):
            name_aliases[tok.lower()] = nid
    if notes_raw:
        # Pattern A — "group X, Y as <name> container/stage/etc"
        group_re_a = re.compile(
            r"(?:^|[\.\;\,]\s*)(?:Group|group|Place|place|Put|put)\s+"
            r"([A-Za-z0-9_, /+&]+?)\s+"
            r"(?:visually\s+)?(?:as|in|into)\s+(?:the\s+|a\s+|an\s+|lower\s+|upper\s+|side\s+)?"
            r"['\"]?([\w][\w \-]*?)['\"]?\s*(?:container|stage|cluster|column|row|pass|layer|group|side|panel)",
            re.IGNORECASE,
        )
        # Pattern B — "group X, Y in <container/cluster/...> labeled '<name>'"
        # or "<container/cluster/...> called '<name>'" — mirror form.
        group_re_b = re.compile(
            r"(?:^|[\.\;\,]\s*)(?:Group|group|Place|place|Put|put)\s+"
            r"([A-Za-z0-9_, /+&]+?)\s+"
            r"(?:visually\s+)?(?:as|in|into)\s+(?:the\s+|a\s+|an\s+|lower\s+|upper\s+|side\s+)?"
            r"(?:container|stage|cluster|column|row|pass|layer|group|side|panel)\s+"
            r"(?:called|labeled|named|titled)\s+"
            r"['\"]?([\w][\w \-]*?)['\"]?(?=[\s\.\;\,]|$)",
            re.IGNORECASE,
        )
        seen_matches: set[tuple[str, str]] = set()
        all_matches: list[tuple[str, str]] = []
        for m in group_re_a.finditer(notes_raw):
            ids_raw = m.group(1)
            label_raw = (m.group(2) or "Group").strip().title()
            key = (ids_raw, label_raw)
            if key not in seen_matches:
                seen_matches.add(key)
                all_matches.append((ids_raw, label_raw))
        for m in group_re_b.finditer(notes_raw):
            ids_raw = m.group(1)
            label_raw = (m.group(2) or "Group").strip().title()
            key = (ids_raw, label_raw)
            if key not in seen_matches:
                seen_matches.add(key)
                all_matches.append((ids_raw, label_raw))
        for id_blob, container_label in all_matches:
            ids: list[str] = []
            for tok in re.split(r"[,/+&]|\sand\s", id_blob):
                tok = tok.strip()
                if not tok:
                    continue
                low = tok.lower()
                if low in name_by_lower_id:
                    ids.append(name_by_lower_id[low])
                elif low in name_aliases:
                    ids.append(name_aliases[low])
            if len(ids) >= 2:
                # Sanitize container id
                cid = re.sub(r"[^A-Za-z0-9_]", "_", container_label).strip("_") or "Group"
                if not cid[0].isalpha() and cid[0] != "_":
                    cid = "g_" + cid
                container_assignments.setdefault(cid, [])
                for nid in ids:
                    if nid not in container_assignments[cid]:
                        container_assignments[cid].append(nid)

    # Build the assigned set so we know which nodes go in containers vs flat.
    assigned: dict[str, str] = {}
    for cid, members in container_assignments.items():
        for nid in members:
            assigned.setdefault(nid, cid)

    # Auto-grouping: when no explicit container hints, split into 2-3 chunks
    # using positional names (Input / Processing / Output) so containers
    # have meaningful titles without colliding with node labels.
    # Skip if the LLM explicitly said "no container" in the notes.
    _no_containers_requested = bool(re.search(r"no[- ]container", notes_raw, re.IGNORECASE))
    if not container_assignments and len(nodes) >= 5 and not _no_containers_requested:
        # Order nodes by their first appearance as an arrow source/target,
        # falling back to declared order. This puts upstream nodes first.
        order_map = {n[0]: idx for idx, n in enumerate(nodes)}
        for src, dst in arrows:
            order_map.setdefault(src, len(order_map))
            order_map.setdefault(dst, len(order_map))
        ordered = sorted([n[0] for n in nodes], key=lambda x: order_map.get(x, 999))
        if len(ordered) <= 6:
            chunks_split = [ordered[: len(ordered) // 2], ordered[len(ordered) // 2 :]]
        else:
            third = len(ordered) // 3
            chunks_split = [ordered[:third], ordered[third : 2 * third], ordered[2 * third :]]

        # Positional names based on number of chunks and flow position.
        _two_names = ["Input", "Output"]
        _three_names = ["Input", "Processing", "Output"]
        chunk_names = _two_names if len(chunks_split) <= 2 else _three_names
        for chunk_idx, members in enumerate(chunks_split):
            if not members:
                continue
            cname = chunk_names[min(chunk_idx, len(chunk_names) - 1)]
            cid = cname.lower().replace(" ", "_")
            # Avoid duplicate cids when two chunks map to the same name.
            if cid in container_assignments:
                cid = f"{cid}_{chunk_idx + 1}"
            container_assignments[cid] = list(members)
            for nid in members:
                assigned.setdefault(nid, cid)

    pastel_palette = [
        "aliceblue", "honeydew", "lavenderblush", "mistyrose",
        "papayawhip", "lemonchiffon", "mintcream", "lavender",
    ]
    lines = [f"direction: {direction}"]
    # Emit containers first (each holds its assigned nodes with shape+label).
    for idx, (cid, members) in enumerate(container_assignments.items()):
        fill = pastel_palette[idx % len(pastel_palette)]
        # Container header with its own pretty label.
        pretty_label = re.sub(r"_+", " ", cid).strip().title()
        lines.append(f'{cid}: "{pretty_label}" {{')
        lines.append(f'  style.fill: "{fill}"')
        lines.append('  style.stroke-dash: 3')
        for nid in members:
            shape = next(s for n, s, _ in nodes if n == nid)
            label = next(l for n, _, l in nodes if n == nid)
            safe_label = label.replace('"', '')
            lines.append(f'  {nid}: "{safe_label}"')
            lines.append(f'  {nid}.shape: {shape}')
        lines.append('}')
    # Emit unassigned nodes at top level.
    for nid, shape, label in nodes:
        if nid in assigned:
            continue
        safe_label = label.replace('"', '')
        lines.append(f'{nid}: "{safe_label}"')
        lines.append(f"{nid}.shape: {shape}")
    # Edges — rewrite endpoints with their container prefix when assigned.
    for src, dst in arrows:
        src_path = f"{assigned[src]}.{src}" if src in assigned else src
        dst_path = f"{assigned[dst]}.{dst}" if dst in assigned else dst
        lines.append(f"{src_path} -> {dst_path}")
    # Tighter typography keeps the long math labels readable inside
    # narrow shapes (hexagons, diamonds) without truncation.
    lines.append("style.font-size: 28")
    return "\n".join(lines)


def _build_d2_from_text(
    *,
    query: str,
    title: str,
    bullets: list[str],
    evidence_text: str,
    diagram_description: str,
) -> str:
    """Build a heuristic D2 diagram from text when LLM code fails.

    Preference order:
      1. Parse the LLM-generated `diagram_description` (nodes/arrows spec).
         This produces a unique diagram per slide even when the D2 formatter
         step returned empty — critical because otherwise every network-ish
         slide falls back to the SAME hardcoded Input->H1->H2->Output->Loss.
      2. Hardcoded topic templates (network, rag) as last-resort.
      3. Generic label-extraction fallback.
    """
    parsed = _parse_diagram_description(diagram_description)
    if parsed:
        return parsed

    # Topic-specific hardcoded templates removed. The fallback is
    # purely topic-agnostic: extract labels from the slide title +
    # bullets + diagram_description and chain them.

    # Generic fallback: extract labels from bullets/title
    labels = _unique_labels(
        [
            _label_from_text(title, max_words=3, max_chars=24),
            *[_label_from_text(b, max_words=3, max_chars=24) for b in bullets[:4]],
            _label_from_text(diagram_description, max_words=3, max_chars=24),
        ],
        needed=4,
    )[:6]

    node_ids = [f"N{i}" for i in range(len(labels))]
    shapes = ["rectangle", "oval", "hexagon", "diamond", "rectangle", "oval"]
    lines = ["direction: right"]
    for i, (nid, lbl) in enumerate(zip(node_ids, labels)):
        safe_label = lbl.replace('"', '')
        lines.append(f'{nid}: "{safe_label}"')
        lines.append(f"{nid}.shape: {shapes[i % len(shapes)]}")

    # Main chain
    for i in range(len(node_ids) - 1):
        lines.append(f"{node_ids[i]} -> {node_ids[i+1]}")
    # One skip connection for complexity
    if len(node_ids) >= 4:
        lines.append(f"{node_ids[0]} -> {node_ids[-2]}")

    lines.append("style.font-size: 32")
    return "\n".join(lines)


def _d2_structure_likely_broken(code: str) -> bool:
    """Detect the recurring LLM failure pattern in generated D2.

    Sonnet/Gemini occasionally emit:
      - multiple top-level `style.fill` declarations (only one applies, the
        rest are wasted and signal the LLM forgot the surrounding container)
      - dotted references like `network.input -> processes.forward` without
        ever declaring a `network: { ... }` or `processes: { ... }` block.

    D2 is lenient and auto-creates stub containers/nodes with default
    labels (just the IDs), so the renderer succeeds and produces a generic
    diagram instead of the intended one. We detect that pattern up front
    so callers can substitute a cleaner D2 source.
    """
    if not code:
        return True
    lines = [l for l in code.splitlines() if l.strip()]
    inside_brace = 0
    top_level_style_fill = 0
    declared_containers: set[str] = set()
    for ln in lines:
        stripped = ln.strip()
        opens = ln.count("{")
        closes = ln.count("}")
        if inside_brace == 0:
            if stripped.startswith("style.fill"):
                top_level_style_fill += 1
            m_container = re.match(r"^([\w\-]+)\s*:[^{]*\{\s*$", stripped)
            if m_container:
                declared_containers.add(m_container.group(1))
        inside_brace = max(0, inside_brace + opens - closes)
    if top_level_style_fill > 1:
        return True
    referenced_containers: set[str] = set()
    for ln in lines:
        if "->" not in ln and "<-" not in ln:
            continue
        edge_part = ln.split(":", 1)[0]
        for tok in re.split(r"\s*->\s*|\s*<-\s*", edge_part):
            tok = tok.strip()
            if "." in tok:
                referenced_containers.add(tok.split(".", 1)[0])
    undefined_refs = referenced_containers - declared_containers
    return len(undefined_refs) >= 2


def _generate_text_to_diagram(
    *,
    query: str,
    slide_number: int,
    out_dir: Path,
    title: str,
    bullets: list[str],
    evidence_text: str,
    diagram_description: str,
    diagram_code: str,
) -> tuple[Path | None, str | None]:
    """Generate a diagram by rendering D2 code (LLM-first, heuristic fallback)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    render_path = out_dir / f"slide_{slide_number:02d}_diagram_render.png"
    out_path = out_dir / f"slide_{slide_number:02d}_diagram_text2diagram.png"

    # Prefer LLM-generated D2; parse the description only if no code was returned.
    d2_code = _extract_d2_block(diagram_code)
    if not d2_code:
        # Empty LLM output — fall back to the deterministic parse so the
        # slide still renders something instead of disappearing.
        d2_code = _parse_diagram_description(diagram_description)
    elif _d2_structure_likely_broken(d2_code):
        # LLM emitted structurally broken D2 (multiple top-level styles, or
        # dotted refs to never-declared parent containers). The renderer
        # would still produce a generic diagram with stub labels — swap it
        # for the parsed description, which preserves the intended labels.
        parsed = _parse_diagram_description(diagram_description)
        if parsed:
            d2_code = parsed
    if d2_code:
        d2_code = _resolve_placeholder_labels(
            d2_code,
            title=title or query,
            bullets=bullets,
            diagram_description=diagram_description,
        )

    errors: list[str] = []

    # Try D2 from model output.
    if d2_code:
        ok, reason = _render_d2_png(d2_code=d2_code, out_path=render_path)
        if ok:
            _compose_diagram_canvas(
                rendered_path=render_path,
                out_path=out_path,
                slide_number=slide_number,
                title=title or query,
                bullets=bullets,
            )
            _cleanup_file(render_path)
            return out_path, "generated:text_to_diagram:d2"
        errors.append(f"D2: {reason}")

    # Heuristic D2 fallback.
    d2_heuristic = _build_d2_from_text(
        query=query, title=title, bullets=bullets,
        evidence_text=evidence_text, diagram_description=diagram_description,
    )
    if d2_heuristic:
        ok, reason = _render_d2_png(d2_code=d2_heuristic, out_path=render_path)
        if ok:
            _compose_diagram_canvas(
                rendered_path=render_path, out_path=out_path,
                slide_number=slide_number, title=title or query, bullets=bullets,
            )
            _cleanup_file(render_path)
            return out_path, "generated:text_to_diagram:d2_heuristic"
        errors.append(f"D2_heuristic: {reason}")

    if _diagram_renderer_strict():
        reason_text = "; ".join(errors) if errors else "no_valid_diagram_code"
        raise RuntimeError(f"Diagram render failed (strict): {reason_text}")
    return None, None



def fetch_image(
    *,
    query: str,
    slide_number: int,
    out_dir: Path,
    provider: str,
    max_candidates: int,
    avoid_urls: set[str] | None = None,
    title: str = "",
    bullets: list[str] | None = None,
    evidence_text: str = "",
    diagram_description: str = "",
    diagram_code: str = "",
) -> tuple[Path | None, str | None]:
    """Fetch one suitable image from configured provider(s), with fallbacks and dedupe inputs."""
    if provider == "none":
        return None, None
    try:
        bullets = bullets or []
        providers = [provider]
        if provider == "auto":
            providers = ["diagram"]

        for current_provider in providers:
            if current_provider == "diagram":
                # Preferred path: text->diagram renderers.
                path, src = _generate_text_to_diagram(
                    query=query,
                    slide_number=slide_number,
                    out_dir=out_dir,
                    title=title,
                    bullets=bullets,
                    evidence_text=evidence_text,
                    diagram_description=diagram_description,
                    diagram_code=diagram_code,
                )
                if path is not None:
                    return path, src
                return None, None
            if current_provider == "placeholder":
                out_path = out_dir / f"slide_{slide_number:02d}_placeholder_{_safe_name(query)}.png"
                result_path = create_placeholder_image(out_path, title or query, query)
                return result_path, "generated:placeholder"
        return None, None
    except RuntimeError:
        raise
    except (URLError, TimeoutError, ValueError):
        return None, None
    except Exception:
        return None, None


def create_placeholder_image(out_path: Path, title: str, query: str) -> Path:
    """Create a visual placeholder image when no web image is available."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1280, 720), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle([(40, 40), (1240, 680)], fill=(245, 249, 255))
    draw.text((80, 90), f"Illustrative image placeholder: {title}", fill=(32, 56, 94), font=font)
    draw.text((80, 140), f"Query: {query}", fill=(55, 81, 123), font=font)
    draw.line([(200, 420), (520, 320), (790, 460), (1040, 300)], fill=(79, 129, 189), width=6)
    img.save(out_path)
    return out_path
