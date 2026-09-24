"""LLM JSON generation and RAG retrieval helpers (semantic + lexical)."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, quote_plus
from urllib.request import Request, urlopen

from config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from models import PresentationRequest, SlideContent, SlideOutline


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


@dataclass
class ChunkItem:
    chunk_id: str
    source: str
    text: str
    tokens: set[str]


@dataclass
class ChunkHit:
    chunk_id: str
    source: str
    score: float
    text: str


def _normalize_space(text: str) -> str:
    """Collapse repeated whitespace and trim leading/trailing spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    """Tokenize text into normalized terms used by retrieval/scoring logic."""
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 3 and tok not in STOPWORDS]


def _compact_text(text: str, *, max_words: int = 14, max_chars: int = 110) -> str:
    """Normalize and compact text to a bounded word/character length.

    Always truncate on whole-word boundaries so the output never ends
    mid-word. Trailing connectors like trailing prepositions are trimmed
    so subtitles/bullets read as complete thoughts.
    """
    clean = _normalize_space(str(text))
    if not clean:
        return ""
    words = clean.split()
    if len(words) > max_words:
        clean = " ".join(words[:max(1, max_words)]).rstrip(",;:")
    if len(clean) > max_chars:
        clean = clean[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
    # Trim dangling connector words that would make the subtitle look cut off.
    _trailing_connectors = {
        "a", "an", "and", "the", "of", "to", "in", "on", "at", "by", "for",
        "with", "from", "into", "through", "via", "over", "under", "about",
        "as", "or", "but", "is", "are", "was", "were", "be", "been", "being",
        "that", "which", "who", "whom", "whose",
    }
    parts = clean.split()
    while parts and parts[-1].lower().rstrip(",;:.") in _trailing_connectors:
        parts.pop()
    clean = " ".join(parts).rstrip(",;:")
    return _normalize_space(clean)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _try_parse_json_payload(text: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(text)
    except Exception:
        return None
    if isinstance(loaded, dict):
        return loaded
    if isinstance(loaded, list):
        if all(isinstance(item, dict) for item in loaded):
            return {"slides": loaded}
    return None


def _looks_like_slide_object(obj: dict[str, Any]) -> bool:
    """Return whether looks like slide object."""
    keys = {str(k).strip().lower() for k in obj.keys()}
    if "slide_number" in keys and "title" in keys:
        return True
    if "slide_number" in keys and "goal" in keys:
        return True
    if "title" in keys and ("goal" in keys or "objective" in keys):
        return True
    return False


def _extract_top_level_objects(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "{":
            i += 1
            continue
        start = i
        depth = 1
        i += 1
        while i < n and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth != 0:
            continue
        candidate = text[start:i]
        parsed = _try_parse_json_payload(candidate)
        if parsed is None:
            parsed = _try_parse_json_payload(_remove_trailing_commas(candidate))
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    stripped = _strip_code_fences(text)
    direct = _try_parse_json_payload(stripped)
    if direct is not None:
        return direct

    # Try extracting a top-level JSON array before falling back to first object.
    array_starts = [idx for idx, ch in enumerate(stripped) if ch == "["]
    for start in array_starts:
        depth = 0
        for idx in range(start, len(stripped)):
            ch = stripped[idx]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : idx + 1]
                    parsed = _try_parse_json_payload(candidate)
                    if parsed is not None:
                        return parsed
                    repaired = _try_parse_json_payload(_remove_trailing_commas(candidate))
                    if repaired is not None:
                        return repaired
                    break

    objects = _extract_top_level_objects(stripped)
    if objects:
        # Gemini sometimes emits one JSON object per line for outline slides.
        slide_like = [obj for obj in objects if _looks_like_slide_object(obj)]
        if len(slide_like) >= 2:
            return {"slides": slide_like}
        return objects[0]
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        repaired = _try_parse_json_payload(_remove_trailing_commas(stripped[first : last + 1]))
        if repaired is not None:
            return repaired

    # Last-resort: JSON was truncated mid-stream (model hit max_new_tokens).
    # Try various suffixes to close the open object, then fall back to trimming
    # back to the last complete key-value pair (before the last comma).
    if first >= 0:
        s = stripped[first:]
        for suffix in ('"}', '"]}', '"}}}', '"}}', '}'):
            try:
                result = json.loads(s + suffix)
                if isinstance(result, dict) and result:
                    return result
            except Exception:
                pass
        repaired_fragment = _repair_truncated_json_fragment(s)
        if repaired_fragment is not None:
            return repaired_fragment
        # Trim to last comma (drops the incomplete last field) then close.
        last_comma = s.rfind(",")
        if last_comma > 1:
            candidate = _remove_trailing_commas(s[:last_comma] + "}")
            try:
                result = json.loads(candidate)
                if isinstance(result, dict) and result:
                    return result
            except Exception:
                pass

    return None


def _repair_truncated_json_fragment(text: str) -> dict[str, Any] | None:
    """Attempt to repair a truncated top-level JSON object."""
    start = text.find("{")
    if start < 0:
        return None

    fragment = text[start:]
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in fragment:
        if escape:
            out.append(ch)
            escape = False
            continue

        if in_string:
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch in "\r\n\t":
                out.append(" ")
                continue
            if ord(ch) < 32:
                continue
            out.append(ch)
            continue

        if ch == '"':
            out.append(ch)
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            out.append(ch)
            continue
        if ch in "}]":
            if not stack:
                break
            expected = "}" if stack[-1] == "{" else "]"
            if ch != expected:
                continue
            stack.pop()
            out.append(ch)
            if not stack:
                candidate = "".join(out)
                parsed = _try_parse_json_payload(candidate)
                if parsed is not None:
                    return parsed
                repaired = _try_parse_json_payload(_remove_trailing_commas(candidate))
                if repaired is not None:
                    return repaired
            continue
        if ord(ch) < 32:
            if ch in "\r\n\t":
                out.append(" ")
            continue
        out.append(ch)

    candidate = "".join(out).rstrip(", ")
    if not candidate:
        return None
    if escape:
        candidate += "\\"
    if in_string:
        candidate += '"'
    while stack:
        candidate = _remove_trailing_commas(candidate.rstrip(", "))
        opener = stack.pop()
        candidate += "}" if opener == "{" else "]"

    parsed = _try_parse_json_payload(candidate)
    if parsed is not None:
        return parsed
    repaired = _try_parse_json_payload(_remove_trailing_commas(candidate))
    if repaired is not None:
        return repaired

    last_comma = candidate.rfind(",")
    if last_comma > 1:
        trimmed = _remove_trailing_commas(candidate[:last_comma] + "}")
        trimmed_parsed = _try_parse_json_payload(trimmed)
        if trimmed_parsed is not None:
            return trimmed_parsed
    return None


def _is_gemini_transient_error(message: str) -> bool:
    """Return whether is gemini transient error."""
    msg = (message or "").lower()
    transient_markers = (
        "request failed (503)",
        "request failed (429)",
        "service unavailable",
        "\"status\": \"unavailable\"",
        "high demand",
        "temporarily unavailable",
        "timed out",
        "connection reset",
        "remote end closed connection",
    )
    return any(marker in msg for marker in transient_markers)


def _ci_get(obj: dict[str, Any], *keys: str) -> Any:
    lookup = {str(k).strip().lower(): v for k, v in obj.items()}
    for key in keys:
        if key.lower() in lookup:
            return lookup[key.lower()]
    return None


def _find_outline_list(payload: dict[str, Any]) -> list[Any] | None:
    if _looks_like_slide_object(payload):
        return [payload]

    direct = _ci_get(payload, "slides")
    if isinstance(direct, list):
        return direct

    for key in ("outline", "presentation_outline", "items", "sections", "deck", "presentation"):
        val = _ci_get(payload, key)
        if isinstance(val, list):
            return val
        if isinstance(val, dict):
            nested = _find_outline_list(val)
            if nested:
                return nested

    def _walk(value: Any) -> list[Any] | None:
        if isinstance(value, list):
            if value and all(isinstance(x, dict) for x in value):
                first = value[0]
                has_outline_shape = any(
                    k in {str(key).strip().lower() for key in first.keys()}
                    for k in ("slide_number", "number", "title", "slide_title", "goal", "objective")
                )
                if has_outline_shape:
                    return value
            for item in value:
                got = _walk(item)
                if got:
                    return got
            return None
        if isinstance(value, dict):
            for nested in value.values():
                got = _walk(nested)
                if got:
                    return got
            return None
        return None

    return _walk(payload)


def _compact_node_label(raw_label: str, *, max_words: int = 4, max_chars: int = 24) -> str:
    """Compact a D2 node label so it fits inside diagram boxes."""
    label = _normalize_space(str(raw_label))
    if not label:
        return ""
    words = re.findall(r"[A-Za-z0-9+\-_/]+", label)
    if not words:
        return ""
    label = " ".join(words[: max(1, int(max_words))])
    if len(label) > max_chars:
        label = label[: max_chars].rsplit(" ", 1)[0].strip()
    return _normalize_space(label) or "Step"


def _humanize_diagram_id(raw_id: str, *, fallback: str = "Group") -> str:
    """Turn an internal D2 id into a human-readable container label."""
    clean = _normalize_space(str(raw_id))
    if not clean:
        return fallback
    clean = clean.replace(".", " ").replace("_", " ").replace("-", " ")
    clean = re.sub(r"([a-z])([A-Z])", r"\1 \2", clean)
    clean = re.sub(r"([A-Za-z])(\d+)", r"\1 \2", clean)
    words = [w for w in clean.split() if w]
    if not words:
        return fallback
    normalized: list[str] = []
    for word in words[:4]:
        low = word.lower()
        if low in {"grp", "group", "cluster", "container"}:
            normalized.append("Group")
        elif low in {"stage", "phase", "step"}:
            normalized.append(low.capitalize())
        elif word.isupper():
            normalized.append(word)
        else:
            normalized.append(word.capitalize())
    return _normalize_space(" ".join(normalized)) or fallback


def _normalize_d2_brace_line(line: str) -> str | None:
    """Preserve valid D2 containers/maps and repair missing group labels."""
    ln = line.strip().rstrip(",")
    if not ln:
        return None
    if re.fullmatch(r"\}+", ln):
        return ln

    inline_m = re.match(
        r'^([A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*"?([^"{][^{}]*?)"?\s*\{\s*(.+?)\s*\}\s*$',
        ln,
    )
    if inline_m:
        node_id = inline_m.group(1)
        raw_label = inline_m.group(2)
        props = inline_m.group(3).strip()
        short_label = _compact_node_label(raw_label, max_words=5, max_chars=30)
        return f'{node_id}: "{short_label}" {{ {props} }}'

    named_container_m = re.match(
        r'^([A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*"?([^"{][^{}]*?)"?\s*\{\s*$',
        ln,
    )
    if named_container_m:
        group_id = named_container_m.group(1)
        raw_label = named_container_m.group(2)
        label = _compact_text(raw_label, max_words=4, max_chars=32) or _humanize_diagram_id(group_id)
        return f'{group_id}: "{label}" {{'

    unnamed_container_m = re.match(
        r'^([A-Za-z_][A-Za-z0-9_.-]*)\s*(?::\s*)?\{\s*$',
        ln,
    )
    if unnamed_container_m:
        group_id = unnamed_container_m.group(1)
        return f'{group_id}: "{_humanize_diagram_id(group_id)}" {{'

    return None


def _bullet_candidates_from_text(text: str, *, max_items: int = 8) -> list[dict[str, str]]:
    """Extract concise bullet/explanation pairs from free text."""
    clean = _normalize_space(str(text or ""))
    if not clean:
        return []
    parts = re.split(r"(?:\n+|[.;!?]\s+)", str(text).replace("\\n", "\n"))
    out: list[dict[str, str]] = []
    seen_local: set[str] = set()
    for part in parts:
        raw = _normalize_space(part.lstrip("-*0123456789. "))
        if not raw:
            continue
        bullet = _compact_text(raw, max_words=24, max_chars=200)
        explanation = _compact_text(raw, max_words=20, max_chars=170)
        if not bullet:
            continue
        key = bullet.lower()
        if key in seen_local:
            continue
        seen_local.add(key)
        out.append({"bullet": bullet, "explanation": explanation})
        if len(out) >= max_items:
            break
    return out


def _extract_diagram_description(data: dict[str, Any]) -> str:
    """Extract a useful diagram description from mixed model outputs."""
    direct = _normalize_space(
        str(
            _ci_get(
                data,
                "diagram_description",
                "diagram_prompt",
                "diagram_spec",
                "image_plan",
                "image_prompt",
            )
            or ""
        )
    )
    if direct:
        return direct
    top_level_parts: list[str] = []
    for key in ("nodes", "arrows", "steps", "relations", "flow"):
        value = _ci_get(data, key)
        if isinstance(value, list):
            items = [_normalize_space(str(v)) for v in value if _normalize_space(str(v))]
            if items:
                top_level_parts.append(f"{key}: " + "; ".join(items))
        elif isinstance(value, str):
            cleaned = _normalize_space(value)
            if cleaned:
                top_level_parts.append(f"{key}: {cleaned}")
    if top_level_parts:
        merged = _normalize_space(" | ".join(top_level_parts))
        if merged:
            return merged
    plan = _ci_get(data, "diagram_plan")
    if isinstance(plan, dict):
        parts: list[str] = []
        for key in ("description", "prompt", "layout", "sequence"):
            value = _normalize_space(str(plan.get(key, "")))
            if value:
                parts.append(value)
        for key in ("nodes", "arrows", "steps"):
            value = plan.get(key)
            if isinstance(value, list):
                items = [_normalize_space(str(v)) for v in value if _normalize_space(str(v))]
                if items:
                    parts.append(f"{key}: " + "; ".join(items))
        merged = _normalize_space(" | ".join(parts))
        if merged:
            return merged
    return ""


def _extract_diagram_description_from_raw_text(text: str) -> str:
    """Recover a usable diagram description from malformed raw model text."""
    stripped = _normalize_space(_strip_code_fences(text))
    if not stripped:
        return ""

    parsed = _extract_first_json_object(stripped)
    if isinstance(parsed, dict):
        recovered = _extract_diagram_description(parsed)
        if recovered:
            return recovered

    matches: list[str] = []
    for key in ("diagram_description", "description", "diagram", "nodes", "arrows", "steps"):
        found = re.findall(rf'"{key}"\s*:\s*"([^"]+)"', stripped, flags=re.IGNORECASE)
        for item in found:
            clean = _normalize_space(item)
            if clean:
                matches.append(clean)
    if matches:
        merged = _normalize_space(" | ".join(dict.fromkeys(matches)))
        if merged:
            return merged

    quoted = re.findall(r'"([^"]+)"', stripped)
    if quoted:
        pieces = []
        for item in quoted[:10]:
            clean = _normalize_space(item)
            if clean and len(clean) > 2:
                pieces.append(clean)
        if pieces:
            return _normalize_space(" | ".join(dict.fromkeys(pieces)))

    return stripped


def _extract_diagram_code(data: dict[str, Any]) -> str:
    """Extract D2/Mermaid/DOT code from model output and sanitize it for rendering."""
    direct = str(
        _ci_get(
            data,
            "diagram_code",
            "mermaid_code",
            "diagram_mermaid",
            "dot_code",
        )
        or ""
    )
    raw = direct
    if not raw:
        plan = _ci_get(data, "diagram_plan")
        if isinstance(plan, dict):
            raw = str(
                _ci_get(
                    plan,
                    "diagram_code",
                    "mermaid_code",
                    "dot_code",
                )
                or ""
            )
    if not raw:
        return ""

    text = str(raw).replace("\\n", "\n").strip()
    if text.startswith("```"):
        text = _strip_code_fences(text)
    # Keep only one block and keep it compact to reduce truncation/failure risk.
    lines = [ln.rstrip() for ln in text.splitlines() if _normalize_space(ln)]
    if not lines:
        return ""
    if len(lines) > 26:
        lines = lines[:26]
    code = "\n".join(lines)
    if len(code) > 1300:
        code = code[:1300].rsplit("\n", 1)[0].strip()
    low = code.lower()
    if "flowchart" in low or low.startswith("graph "):
        return code
    if low.startswith("digraph") or low.startswith("graph "):
        return code
    if "->" in code and any(
        marker in low
        for marker in (
            "direction:",
            "style:",
            "classes:",
            "shape:",
            "class:",
            "near:",
        )
    ):
        return code
    return ""


class LLMService:
    _PIPELINE_CACHE: dict[str, tuple[Any, Any]] = {}
    _D2_FORMATTER_CACHE: dict[str, Any] = {}

    def __init__(
        self,
        *,
        provider: str = "huggingface",
        model_id: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        adapter_path: str | None = None,
    ) -> None:
        self.provider = _normalize_space(provider).lower() or "huggingface"
        self.model_id = model_id
        self.max_new_tokens = max(256, int(max_new_tokens))
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self._tokenizer: Any | None = None
        self._generator: Any | None = None
        self._gemini_api_key: str | None = None
        self._gemini_api_base = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self._anthropic_api_key: str | None = None
        self._anthropic_client: Any | None = None
        self._openai_api_key: str | None = None
        self._hf_adapter_path: str | None = None

        if self.provider == "huggingface":
            self._hf_adapter_path = _normalize_space(adapter_path) if adapter_path is not None else (
                _normalize_space(os.environ.get("HF_ADAPTER_PATH", ""))
                or _normalize_space(os.environ.get("LORA_ADAPTER_PATH", ""))
                or None
            )
            if self._hf_adapter_path == "":
                self._hf_adapter_path = None
            self._tokenizer, self._generator = self._get_or_create_pipeline(
                model_id,
                adapter_path=self._hf_adapter_path,
            )
        elif self.provider == "gemini":
            self._gemini_api_key = (
                _normalize_space(os.environ.get("GEMINI_API_KEY", ""))
                or _normalize_space(os.environ.get("GOOGLE_API_KEY", ""))
                or None
            )
            if not self._gemini_api_key:
                raise RuntimeError(
                    "LLM provider is gemini, but GEMINI_API_KEY / GOOGLE_API_KEY is missing."
                )
        elif self.provider == "anthropic":
            self._anthropic_api_key = _normalize_space(os.environ.get("ANTHROPIC_API_KEY", "")) or None
            if not self._anthropic_api_key:
                raise RuntimeError(
                    "LLM provider is anthropic, but ANTHROPIC_API_KEY is missing."
                )
        elif self.provider == "openai":
            self._openai_api_key = _normalize_space(os.environ.get("OPENAI_API_KEY", "")) or None
            if not self._openai_api_key:
                raise RuntimeError(
                    "LLM provider is openai, but OPENAI_API_KEY is missing."
                )
        else:
            raise RuntimeError(f"Unsupported llm provider: {self.provider}")

    @classmethod
    def _get_or_create_pipeline(
        cls,
        model_id: str,
        *,
        adapter_path: str | None = None,
    ) -> tuple[Any, Any]:
        """Return or create pipeline."""
        clean_adapter = _normalize_space(adapter_path or "") or None
        cache_key = f"{model_id}@@{clean_adapter or ''}"
        cached = cls._PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except Exception as exc:
            raise RuntimeError("Could not import transformers/torch.") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=model_dtype,
            device_map="auto",
        )
        if clean_adapter:
            try:
                from peft import PeftModel
            except Exception as exc:
                raise RuntimeError(
                    "HF_ADAPTER_PATH is set, but peft is not installed."
                ) from exc
            model = PeftModel.from_pretrained(model, clean_adapter)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            # Clear checkpoint sampling defaults when using deterministic decoding.
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(generation_config, attr):
                    setattr(generation_config, attr, None)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        cls._PIPELINE_CACHE[cache_key] = (tokenizer, generator)
        return tokenizer, generator

    def _generate_json_huggingface(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        """Generate json huggingface."""
        if self._tokenizer is None or self._generator is None:
            raise RuntimeError("Hugging Face generator not initialized.")
        token_budget = max(128, int(max_new_tokens))
        tokenizer = self._tokenizer
        generator = self._generator
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = user_prompt
        else:
            prompt = user_prompt

        # Gemma 3 1B has an 8192-token context window. Guard against prompts
        # that exceed the window minus the generation budget — if the tokenizer
        # silently truncates mid-sentence the model responds with refusals.
        try:
            model_max = int(getattr(tokenizer, "model_max_length", 8192) or 8192)
            model_max = min(model_max, 8192)
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_len = int(prompt_ids["input_ids"].shape[-1])
            safety_margin = 64
            max_budget_without_extra_truncation = max(256, model_max - min(prompt_len, model_max - 320) - safety_margin)
            effective_budget = min(token_budget, max_budget_without_extra_truncation)
            if effective_budget < token_budget:
                print(
                    "    HuggingFace JSON generation: "
                    f"reducing max_new_tokens from {token_budget} to {effective_budget} "
                    "to preserve prompt context."
                )
                token_budget = effective_budget
            input_limit = max(256, model_max - token_budget - safety_margin)
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_limit)
            prompt = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
        except Exception:
            pass  # fall through to generator with original prompt

        generation_args: dict[str, Any] = {
            "max_new_tokens": token_budget,
            "return_full_text": False,
            "pad_token_id": getattr(tokenizer, "eos_token_id", None),
            "truncation": True,
        }
        if self.temperature > 0.0:
            generation_args["do_sample"] = True
            generation_args["temperature"] = self.temperature
            generation_args["top_p"] = min(max(self.top_p, 0.01), 1.0)
        else:
            generation_args["do_sample"] = False

        outputs = generator(prompt, **generation_args)
        if not outputs:
            raise RuntimeError("LLM returned no outputs.")
        generated = str(outputs[0].get("generated_text", ""))
        parsed = _extract_first_json_object(generated)
        if parsed is None:
            excerpt = _normalize_space(_strip_code_fences(generated))[:280]
            raise RuntimeError(f"LLM returned invalid JSON. Excerpt: {excerpt}")
        return parsed

    def _generate_text_huggingface(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Generate plain text with the Hugging Face local model."""
        if self._tokenizer is None or self._generator is None:
            raise RuntimeError("Hugging Face generator not initialized.")
        tokenizer = self._tokenizer
        generator = self._generator
        token_budget = max(96, int(max_new_tokens))

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = user_prompt
        else:
            prompt = user_prompt

        try:
            model_max = int(getattr(tokenizer, "model_max_length", 8192) or 8192)
            model_max = min(model_max, 8192)
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_len = int(prompt_ids["input_ids"].shape[-1])
            safety_margin = 64
            effective_budget = min(token_budget, max(192, model_max - min(prompt_len, model_max - 256) - safety_margin))
            input_limit = max(256, model_max - effective_budget - safety_margin)
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_limit)
            prompt = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
            token_budget = effective_budget
        except Exception:
            pass

        outputs = generator(
            prompt,
            max_new_tokens=token_budget,
            return_full_text=False,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
            truncation=True,
            do_sample=False,
        )
        if not outputs:
            raise RuntimeError("LLM returned no plain-text outputs.")
        return _normalize_space(str(outputs[0].get("generated_text", "")))

    def _generate_json_gemini(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        """Generate json gemini."""
        if not self._gemini_api_key:
            raise RuntimeError("Gemini API key is not configured.")
        token_budget = max(128, int(max_new_tokens))
        model = quote(self.model_id, safe="._-")
        endpoint = f"{self._gemini_api_base}/models/{model}:generateContent?key={quote_plus(self._gemini_api_key)}"
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": max(0.0, float(self.temperature)),
                "topP": min(max(float(self.top_p), 0.01), 1.0),
                "maxOutputTokens": token_budget,
                "responseMimeType": "application/json",
            },
        }
        req = Request(
            endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=180) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            details = ""
            try:
                details = exc.read().decode("utf-8", errors="replace")
            except Exception:
                details = str(exc)
            raise RuntimeError(f"Gemini request failed ({exc.code}): {details[:500]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        try:
            obj = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"Gemini returned non-JSON response: {raw[:320]}") from exc

        prompt_feedback = obj.get("promptFeedback") or {}
        block_reason = _normalize_space(str(prompt_feedback.get("blockReason", "")))
        if block_reason:
            raise RuntimeError(f"Gemini blocked prompt: {block_reason}")

        generated_text = ""
        finish_reason = ""
        for candidate in obj.get("candidates", []) or []:
            finish_reason = _normalize_space(str((candidate or {}).get("finishReason", "")))
            content = candidate.get("content") or {}
            for part in content.get("parts", []) or []:
                text = _normalize_space(str((part or {}).get("text", "")))
                if text:
                    generated_text = text
                    break
            if generated_text:
                break

        if not generated_text:
            raise RuntimeError(f"Gemini returned empty content: {raw[:320]}")

        parsed = _extract_first_json_object(generated_text)
        if parsed is None:
            excerpt = _normalize_space(_strip_code_fences(generated_text))[:320]
            if finish_reason.upper() in {"MAX_TOKENS", "LENGTH"}:
                raise RuntimeError(f"Gemini JSON truncated at max tokens ({finish_reason}). Excerpt: {excerpt}")
            raise RuntimeError(f"Gemini returned invalid JSON ({finish_reason or 'unknown'}). Excerpt: {excerpt}")
        return parsed

    def _generate_json_openai(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        """Call OpenAI Chat Completions and return parsed JSON.

        Direct HTTP — no SDK dependency. Forces JSON-object response_format
        so the model can't drift into prose. Mirrors the Anthropic branch's
        retry semantics for transient errors.
        """
        if not self._openai_api_key:
            raise RuntimeError("OpenAI API key is not configured.")
        url = "https://api.openai.com/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model_id,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "response_format": {"type": "json_object"},
            "max_tokens": int(max_new_tokens),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._openai_api_key}",
            "Content-Type": "application/json",
        }
        import requests
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text[:320]}")
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        finish_reason = _normalize_space(
            str(data.get("choices", [{}])[0].get("finish_reason", ""))
        )
        parsed = _extract_first_json_object(content)
        if parsed is None:
            excerpt = _normalize_space(_strip_code_fences(content))[:320]
            if finish_reason.lower() in {"length", "max_tokens"}:
                raise RuntimeError(
                    f"OpenAI JSON truncated at max tokens ({finish_reason}). Excerpt: {excerpt}"
                )
            raise RuntimeError(
                f"OpenAI returned invalid JSON ({finish_reason or 'unknown'}). Excerpt: {excerpt}"
            )
        return parsed

    def _get_anthropic_client(self) -> Any:
        """Lazy-init Anthropic SDK client; cached on the instance."""
        if self._anthropic_client is not None:
            return self._anthropic_client
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from exc
        self._anthropic_client = anthropic.Anthropic(api_key=self._anthropic_api_key)
        return self._anthropic_client

    def _generate_json_anthropic(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        """Call Claude and return parsed JSON.

        Uses the "assistant prefill with `{`" trick so Claude is forced to
        emit a JSON object right away (no markdown fences, no preamble).
        """
        if not self._anthropic_api_key:
            raise RuntimeError("Anthropic API key is not configured.")
        client = self._get_anthropic_client()
        token_budget = max(256, int(max_new_tokens))
        # These model variants reject assistant prefill and temperature.
        # Restore the opening JSON brace after generation instead.
        _mid = (self.model_id or "").lower()
        no_prefill = "opus" in _mid or "sonnet-4-6" in _mid
        messages = [{"role": "user", "content": user_prompt}]
        if not no_prefill:
            messages.append({"role": "assistant", "content": "{"})
        create_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": token_budget,
            "system": system_prompt,
            "messages": messages,
        }
        if not no_prefill:
            create_kwargs["temperature"] = max(0.0, float(self.temperature))
        try:
            response = client.messages.create(**create_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Anthropic request failed: {exc}") from exc

        # Stitch the prefill back onto the generated text so the parser sees
        # a complete JSON object starting from `{`. Models that don't support
        # prefill already return full JSON, so don't double-prepend.
        text = "" if no_prefill else "{"
        for block in getattr(response, "content", []) or []:
            block_text = getattr(block, "text", None)
            if block_text:
                text += block_text

        finish_reason = _normalize_space(str(getattr(response, "stop_reason", "")))
        parsed = _extract_first_json_object(text)
        if parsed is None:
            excerpt = _normalize_space(_strip_code_fences(text))[:320]
            if finish_reason.lower() in {"max_tokens", "length"}:
                raise RuntimeError(
                    f"Anthropic JSON truncated at max tokens ({finish_reason}). Excerpt: {excerpt}"
                )
            raise RuntimeError(
                f"Anthropic returned invalid JSON ({finish_reason or 'unknown'}). Excerpt: {excerpt}"
            )
        return parsed

    def generate_json(self, *, system_prompt: str, user_prompt: str, max_new_tokens: int | None = None) -> dict[str, Any]:
        """Generate a JSON response from the configured LLM provider with retries/repair."""
        token_budget = max(128, int(max_new_tokens or self.max_new_tokens))
        if self.provider == "openai":
            transient_tries = int(_normalize_space(os.environ.get("OPENAI_TRANSIENT_TRIES", "5")) or "5")
            transient_tries = max(1, min(transient_tries, 10))
            last_error: Exception | None = None
            for attempt in range(1, transient_tries + 1):
                try:
                    return self._generate_json_openai(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_new_tokens=token_budget,
                    )
                except RuntimeError as exc:
                    last_error = exc
                    msg = str(exc).lower()
                    transient = any(k in msg for k in ("rate_limit", "429", "timeout", "connection", "500", "502", "503", "504"))
                    if not transient or attempt >= transient_tries:
                        raise
                    delay_s = min(60.0, 2.0 * (2 ** (attempt - 1)))
                    print(f"    OpenAI transient error (attempt {attempt}/{transient_tries}), retrying in {delay_s:.0f}s...")
                    time.sleep(delay_s)
            raise RuntimeError(f"OpenAI JSON generation failed: {last_error}")
        if self.provider == "anthropic":
            # Single-shot retry loop on transient errors (429/overloaded).
            transient_tries = int(_normalize_space(os.environ.get("ANTHROPIC_TRANSIENT_TRIES", "5")) or "5")
            transient_tries = max(1, min(transient_tries, 10))
            last_error: Exception | None = None
            for attempt in range(1, transient_tries + 1):
                try:
                    return self._generate_json_anthropic(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_new_tokens=token_budget,
                    )
                except RuntimeError as exc:
                    last_error = exc
                    msg = str(exc).lower()
                    transient = any(k in msg for k in ("overloaded", "rate_limit", "429", "timeout", "connection"))
                    if not transient or attempt >= transient_tries:
                        raise
                    delay_s = min(60.0, 2.0 * (2 ** (attempt - 1)))
                    print(f"    Anthropic transient error (attempt {attempt}/{transient_tries}), retrying in {delay_s:.0f}s...")
                    time.sleep(delay_s)
            raise RuntimeError(f"Anthropic JSON generation failed: {last_error}")
        if self.provider == "gemini":
            # Use progressively larger output budgets to recover from truncated JSON responses.
            try:
                retry_cap = int(_normalize_space(os.environ.get("GEMINI_RETRY_MAX_TOKENS", "8192")))
            except Exception:
                retry_cap = 8192
            retry_cap = max(4096, min(retry_cap, 20000))
            budget_cap = max(token_budget, min(retry_cap, int(max(self.max_new_tokens * 4, token_budget * 4))))
            budgets = [
                token_budget,
                max(token_budget + 600, int(token_budget * 1.8)),
                max(token_budget + 1400, int(token_budget * 2.8)),
                max(token_budget + 2400, int(token_budget * 3.8)),
            ]
            budgets = [max(128, min(int(b), budget_cap)) for b in budgets]
            # Keep order, remove accidental duplicates after clamping.
            seen_budgets: set[int] = set()
            ordered_budgets: list[int] = []
            for b in budgets:
                if b in seen_budgets:
                    continue
                seen_budgets.add(b)
                ordered_budgets.append(b)
            last_error: Exception | None = None
            base_prompt = user_prompt
            for attempt, budget in enumerate(ordered_budgets, start=1):
                prompt = base_prompt
                if attempt > 1:
                    prompt = (
                        f"{base_prompt}\n\n"
                        "STRICT OUTPUT RULES (retry):\n"
                        "- Return exactly one complete JSON object.\n"
                        "- No markdown fences, no prose, no comments.\n"
                        "- Ensure syntactically valid JSON.\n"
                        "- Keep all strings concise to avoid truncation.\n"
                        "- Avoid long explanations; short factual wording only.\n"
                    )
                transient_tries = int(_normalize_space(os.environ.get("GEMINI_TRANSIENT_TRIES", "10")) or "10")
                transient_tries = max(4, min(transient_tries, 20))
                max_delay_s = float(_normalize_space(os.environ.get("GEMINI_MAX_DELAY_S", "120")) or "120")
                for transient_try in range(1, transient_tries + 1):
                    try:
                        return self._generate_json_gemini(
                            system_prompt=system_prompt,
                            user_prompt=prompt,
                            max_new_tokens=budget,
                        )
                    except RuntimeError as exc:
                        msg = str(exc).lower()
                        last_error = exc
                        if _is_gemini_transient_error(msg):
                            if transient_try >= transient_tries:
                                break
                            delay_s = min(max_delay_s, 2.0 * (2 ** (transient_try - 1)))
                            print(f"    Gemini overloaded (attempt {transient_try}/{transient_tries}), retrying in {delay_s:.0f}s...")
                            time.sleep(delay_s)
                            continue
                        if not any(
                            key in msg
                            for key in (
                                "invalid json",
                                "truncated",
                                "empty content",
                                "max tokens",
                            )
                        ):
                            raise
                        break
            if last_error is not None:
                raise RuntimeError(f"Gemini JSON generation failed after retries: {last_error}") from last_error
            raise RuntimeError("Gemini JSON generation failed with unknown error.")
        # HuggingFace models (Gemma 1B) generate verbose JSON that can exceed
        # the default budget. Retry with slightly larger budgets, but keep the
        # caller-requested budget as the starting point so we do not needlessly
        # truncate the prompt and drift off-topic.
        try:
            hf_retry_cap = int(_normalize_space(os.environ.get("HF_RETRY_MAX_TOKENS", "4096")) or "4096")
        except Exception:
            hf_retry_cap = 4096
        budget_cap = max(token_budget, min(hf_retry_cap, 6000))
        budgets = [
            token_budget,
            max(token_budget + 320, int(token_budget * 1.3)),
            max(token_budget + 800, int(token_budget * 1.7)),
        ]
        ordered_budgets: list[int] = []
        seen_budgets: set[int] = set()
        for budget in budgets:
            clamped = max(128, min(int(budget), budget_cap))
            if clamped in seen_budgets:
                continue
            seen_budgets.add(clamped)
            ordered_budgets.append(clamped)

        last_error: Exception | None = None
        base_prompt = user_prompt
        for attempt, budget in enumerate(ordered_budgets, start=1):
            prompt = base_prompt
            if attempt > 1:
                prompt = (
                    f"{base_prompt}\n\n"
                    "STRICT OUTPUT RULES (retry):\n"
                    "- Return exactly one complete JSON object.\n"
                    "- No markdown fences, no prose, no comments.\n"
                    "- Keep every string concise.\n"
                    "- If a field becomes long, summarize it instead of listing many items.\n"
                    "- Ensure all quotes, brackets, and braces are closed.\n"
                )
            try:
                return self._generate_json_huggingface(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    max_new_tokens=budget,
                )
            except RuntimeError as exc:
                last_error = exc
                msg = str(exc).lower()
                retryable = any(
                    marker in msg
                    for marker in (
                        "invalid json",
                        "no outputs",
                        "truncated",
                        "unterminated",
                    )
                )
                if not retryable or attempt >= len(ordered_budgets):
                    raise
        if last_error is not None:
            raise RuntimeError(f"Hugging Face JSON generation failed after retries: {last_error}") from last_error
        raise RuntimeError("Hugging Face JSON generation failed with unknown error.")

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        """Parse a boolean environment flag with a safe default fallback."""
        raw = _normalize_space(os.environ.get(name, ""))
        if not raw:
            return default
        low = raw.lower()
        if low in {"1", "true", "yes", "y", "on"}:
            return True
        if low in {"0", "false", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def _extract_d2_code(text: str, *, max_lines: int = 44, max_chars: int = 2400) -> str:
        """Extract compact D2 code from mixed text/code payloads."""
        if not text:
            return ""

        code = _strip_code_fences(str(text)).replace("\\n", "\n")

        # Restore common LaTeX symbols whose escape sequences were lost in JSON.
        _LATEX_SURVIVORS = [
            (r"\\?nabla\b", "∇"),
            (r"\\?partial\b", "∂"),
            (r"\\?theta\b", "θ"),
            (r"\\?alpha\b", "α"),
            (r"\\?beta\b", "β"),
            (r"\\?gamma\b", "γ"),
            (r"\\?delta\b", "δ"),
            (r"\\?epsilon\b", "ε"),
            (r"\\?lambda\b", "λ"),
            (r"\\?mu\b", "μ"),
            (r"\\?sigma\b", "σ"),
            (r"\\?phi\b", "φ"),
            (r"\\?omega\b", "ω"),
            (r"\\?sum\b", "Σ"),
            (r"\\?prod\b", "∏"),
            (r"\\?circ\b", "∘"),
            (r"\\?cdot\b", "·"),
            (r"\\?times\b", "×"),
            (r"\\?leftarrow\b", "←"),
            (r"\\?rightarrow\b", "→"),
            (r"\\?to\b(?![a-zA-Z])", "→"),
            # Survivors after backslash stripping (abla ← \nabla, heta ← \theta, …)
            (r"\babla\b", "∇"),
            (r"\bheta\b", "θ"),
            (r"\blpha\b", "α"),
            (r"\beta\b", "β"),
            (r"\bamma\b", "γ"),
            (r"\bartial\b", "∂"),
            (r"\bambda\b", "λ"),
            (r"\bigma\b", "σ"),
        ]
        for pattern, replacement in _LATEX_SURVIVORS:
            code = re.sub(pattern, replacement, code)

        raw_lines = [ln.rstrip() for ln in code.splitlines() if _normalize_space(ln)]
        lines: list[str] = []
        brace_depth = 0
        for raw in raw_lines:
            ln = raw.strip()
            low = ln.lower()
            # Drop layout hints that often create unreadable overlaps in generated diagrams.
            if any(tok in low for tok in ("above(", "below(", "left(", "right(")):
                continue

            # Preserve valid D2 containers/maps and repair missing group labels.
            if "{" in ln or "}" in ln:
                brace_line = _normalize_d2_brace_line(ln)
                if brace_line is None:
                    continue
                open_count = brace_line.count("{")
                close_count = brace_line.count("}")
                if open_count == 0 and close_count > brace_depth:
                    close_count = brace_depth
                    if close_count <= 0:
                        continue
                    brace_line = "}" * close_count
                if close_count > open_count and brace_depth <= 0:
                    continue
                ln = brace_line
                low = ln.lower()

            # (e) Drop dangling keys — empty map keys from flattened structures (e.g. "word:" with nothing after).
            if re.match(r'^[A-Za-z0-9_.-]+:\s*$', ln):
                continue

            # (e2) Drop nodes that are placeholders the model left behind when
            # it couldn't actually express the content — `latex`, `formula`,
            # `equation`, `eq` with no meaningful label. These show up as
            # disconnected boxes labelled "formula" floating next to a diagram.
            _PLACEHOLDER_IDS = {"latex", "formula", "equation", "eq", "math", "expr"}
            placeholder_match = re.match(
                r'^([a-z_][a-z0-9_]*)\s*(?::\s*"?([^"{}]*?)"?)?\s*$', low
            )
            if placeholder_match:
                pid = placeholder_match.group(1)
                plabel = (placeholder_match.group(2) or "").strip()
                if pid in _PLACEHOLDER_IDS and (not plabel or plabel in _PLACEHOLDER_IDS):
                    continue

            # (f) Normalize font-size → style.font-size at line start.
            if re.match(r'^font-size:', ln):
                ln = re.sub(r'^font-size:', 'style.font-size:', ln)

            # (a) Normalize shape aliases to supported D2 shapes.
            shape_m = re.match(r'^([A-Za-z0-9_.-]+\.)?shape:\s*(.+)$', ln, re.IGNORECASE)
            if shape_m:
                prefix = shape_m.group(1) or ""
                shape_val = shape_m.group(2).strip().lower()
                _RECT_ALIASES = {"note", "sticky_note", "sticky-note", "note_card", "card", "box", "process"}
                _OVAL_ALIASES = {"terminator", "start", "end"}
                if shape_val in _RECT_ALIASES:
                    ln = f"{prefix}shape: rectangle"
                elif shape_val in _OVAL_ALIASES:
                    ln = f"{prefix}shape: oval"

            # (b) Normalize invalid edge syntax.
            if "->" in ln or "--" in ln:
                # Replace -- with ->
                ln = ln.replace("--", "->")
                # Remove port suffixes (.south, .north, .east, .west) from node references in edges.
                ln = re.sub(r'\.(south|north|east|west)\b', '', ln)

            # Remove edge labels (`A -> B: label`) to keep arrows clean and avoid text on connectors.
            if "->" in ln and ":" in ln:
                ln = re.sub(r"(\s*->\s*[^:]+?)\s*:\s*.+$", r"\1", ln).rstrip()

            # (d) Validate edge lines — must have non-empty source AND destination.
            if "->" in ln:
                parts = ln.split("->", 1)
                src = parts[0].strip()
                dst = parts[1].strip() if len(parts) > 1 else ""
                if not src or not dst:
                    continue

            # Compact simple node labels so text fits inside diagram boxes.
            if "->" not in ln and ":" in ln:
                m = re.match(r'^([A-Za-z0-9_.-]+)\s*:\s*"?([^"{][^{}]*?)"?\s*$', ln)
                if m:
                    node_id = m.group(1)
                    raw_label = m.group(2)
                    short_label = _compact_node_label(raw_label)
                    ln = f'{node_id}: "{short_label}"'
            if _normalize_space(ln):
                lines.append(ln)
                brace_depth += ln.count("{") - ln.count("}")
                if brace_depth < 0:
                    brace_depth = 0
        if not lines:
            return ""
        if lines[0].strip().lower() in {"d2", "d2:"}:
            lines = lines[1:]
        if not lines:
            return ""
        if brace_depth > 0:
            lines.extend("}" for _ in range(brace_depth))

        # Remove disconnected top-level nodes; retain container members and edge endpoints.
        edge_ids: set[str] = set()
        for ln in lines:
            if "->" in ln:
                parts = ln.split("->")
                for i in range(len(parts) - 1):
                    src = parts[i].strip().split()[-1] if parts[i].strip() else ""
                    dst = parts[i + 1].strip().split()[0] if parts[i + 1].strip() else ""
                    src = src.strip(":").rstrip(",")
                    dst = dst.strip(":").rstrip(",")
                    if src:
                        edge_ids.add(src)
                        edge_ids.add(src.split(".")[0])
                    if dst:
                        edge_ids.add(dst)
                        edge_ids.add(dst.split(".")[0])

        cleaned: list[str] = []
        seen_labels: set[str] = set()
        for ln in lines:
            # Directives / edges / styles / directions are always kept.
            if "->" in ln or ln.startswith(("direction:", "style.", "*", "title:")):
                cleaned.append(ln)
                continue
            # Node declaration: `id: "Label"` at top level (no dot in id).
            node_m = re.match(r'^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*"?([^"{}]*)"?\s*$', ln)
            if node_m:
                nid = node_m.group(1)
                nlabel = _normalize_space(node_m.group(2))
                # Drop orphans — top-level nodes that never appear in an edge.
                if nid not in edge_ids:
                    continue
                # Drop exact duplicate labels (keeps the first mention).
                key = nlabel.lower() if nlabel else ""
                if key and key in seen_labels:
                    continue
                if key:
                    seen_labels.add(key)
            cleaned.append(ln)
        lines = cleaned or lines

        # Stack graphs with 5+ top-level nodes vertically to reduce label overlap.
        # Keep the existing direction for graphs with containers.
        top_level_nodes = sum(
            1 for ln in lines
            if re.match(r'^[A-Za-z_][A-Za-z0-9_-]*\s*:\s*', ln) and "->" not in ln
        )
        has_containers = any("{" in ln for ln in lines) or any("." in ln.split(":")[0] for ln in lines if ":" in ln and "->" not in ln)
        direction_line_idx = next(
            (i for i, ln in enumerate(lines) if ln.strip().lower().startswith("direction:")),
            None,
        )
        want_vertical = top_level_nodes >= 5 and not has_containers
        if direction_line_idx is not None and want_vertical:
            lines[direction_line_idx] = "direction: down"
        elif direction_line_idx is None:
            lines.insert(0, "direction: down" if want_vertical else "direction: right")

        if len(lines) > max(1, int(max_lines)):
            lines = lines[: max(1, int(max_lines))]
        out = "\n".join(lines).strip()
        if len(out) > max(120, int(max_chars)):
            out = out[: max(120, int(max_chars))].rsplit("\n", 1)[0].strip()
        low = out.lower()
        if "flowchart" in low or low.startswith("digraph") or low.startswith("graph "):
            return ""
        if "->" not in out:
            return ""
        if any(marker in low for marker in ("direction:", "shape:", "style:", "classes:", "class:", "near:")):
            return out
        # Accept simple D2 edge lists as long as there are multiple relations.
        if out.count("->") >= 2:
            return out
        return ""

    @classmethod
    def _get_or_create_d2_formatter(cls, model_id: str) -> Any:
        """Return a cached LLM service for D2 normalization.

        Provider resolution order:
          1. `D2_FORMATTER_PROVIDER` env (explicit override)
          2. `anthropic` if `ANTHROPIC_API_KEY` is set
          3. `openai` if `OPENAI_API_KEY` is set
          4. `gemini` otherwise (backward compat)
        """
        explicit = _normalize_space(os.environ.get("D2_FORMATTER_PROVIDER", "")).lower()
        if explicit in {"anthropic", "openai", "gemini"}:
            provider = explicit
        elif _normalize_space(os.environ.get("ANTHROPIC_API_KEY", "")):
            provider = "anthropic"
        elif _normalize_space(os.environ.get("OPENAI_API_KEY", "")):
            provider = "openai"
        else:
            provider = "gemini"
        if provider == "anthropic":
            fallback_model = "claude-sonnet-4-5"
        elif provider == "openai":
            fallback_model = "gpt-4.1-mini"
        else:
            fallback_model = "gemini-3-flash-preview"
        key = f"{provider}:{_normalize_space(model_id) or fallback_model}"
        cached = cls._D2_FORMATTER_CACHE.get(key)
        if cached is not None:
            return cached
        formatter = LLMService(
            provider=provider,
            model_id=_normalize_space(model_id) or fallback_model,
            max_new_tokens=1200,
            temperature=0.0,
            top_p=0.9,
        )
        cls._D2_FORMATTER_CACHE[key] = formatter
        return formatter

    # -- D2 few-shot examples (used by the formatter prompt) --
    # Based on real D2 patterns from d2lang.com — containers, edge labels,
    # styling, wildcards, nested references, LaTeX formulas.
    _D2_FEW_SHOTS = {
        # 0. Classes + chained edges — the cleanest D2 pattern from the tour
        # (d2lang.com/tour/classes). Reusable styles via a `classes` block,
        # connection chaining `a -> b -> c` instead of line-per-edge. This
        # produces tight, consistent visuals and keeps the file short.
        "classes_chained": (
            "direction: down\n"
            "\n"
            "classes: {\n"
            "  stage: {\n"
            "    style.stroke-dash: 3\n"
            '    style.fill: "aliceblue"\n'
            "  }\n"
            "  data: {\n"
            "    shape: cylinder\n"
            '    style.fill: "mintcream"\n'
            "  }\n"
            "  check: {\n"
            "    shape: diamond\n"
            '    style.fill: "lemonchiffon"\n'
            "  }\n"
            "}\n"
            "\n"
            "source: Raw Signal {\n"
            "  shape: oval\n"
            "}\n"
            "\n"
            "pipeline: {\n"
            "  class: stage\n"
            "  filter: Filter\n"
            "  extract: Extract\n"
            "  store: Store {\n"
            "    class: data\n"
            "  }\n"
            "  filter -> extract -> store\n"
            "}\n"
            "\n"
            "verify: Validate {\n"
            "  class: check\n"
            "}\n"
            "\n"
            "result: Output {\n"
            "  shape: oval\n"
            "}\n"
            "\n"
            "source -> pipeline.filter\n"
            "pipeline.store -> verify -> result\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # Compact D2 example: two containers with three nodes each, stacked vertically.
        "compact_pipeline": (
            "direction: down\n"
            "\n"
            "ingest: Input Stage {\n"
            '  style.fill: "aliceblue"\n'
            "  style.stroke-dash: 3\n"
            "  raw: Raw Input {\n"
            "    shape: oval\n"
            "  }\n"
            "  clean: Preprocess\n"
            "  feat: Features {\n"
            "    shape: cylinder\n"
            "  }\n"
            "  raw -> clean\n"
            "  clean -> feat\n"
            "}\n"
            "\n"
            "compute: Compute Stage {\n"
            '  style.fill: "mintcream"\n'
            "  style.stroke-dash: 3\n"
            "  model: Model Core {\n"
            "    shape: hexagon\n"
            "  }\n"
            "  check: Validate {\n"
            "    shape: diamond\n"
            "  }\n"
            "  out: Output {\n"
            "    shape: oval\n"
            "  }\n"
            "  model -> check\n"
            "  check -> out\n"
            "  check -> model\n"
            "}\n"
            "\n"
            "ingest.feat -> compute.model\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 1. Transformer architecture — containers with internal edges + cross-group connections
        "transformer": (
            "direction: right\n"
            "\n"
            "title: Transformer Architecture {\n"
            "  shape: text\n"
            "  near: top-center\n"
            "  style.font-size: 32\n"
            "}\n"
            "\n"
            "encoder: Encoder Stack {\n"
            "  style.stroke-dash: 2\n"
            '  *.style.fill: "lightskyblue"\n'
            "  emb: Input Embedding\n"
            "  mha: Multi-Head Attention {\n"
            "    shape: hexagon\n"
            "  }\n"
            "  ff: Feed Forward\n"
            "  norm: Layer Norm {\n"
            "    shape: diamond\n"
            "  }\n"
            "  emb -> mha\n"
            "  mha -> ff\n"
            "  ff -> norm\n"
            "  emb -> norm\n"
            "}\n"
            "\n"
            "decoder: Decoder Stack {\n"
            "  style.stroke-dash: 2\n"
            '  *.style.fill: "peachpuff"\n'
            "  masked: Masked Attention {\n"
            "    shape: hexagon\n"
            "  }\n"
            "  cross: Cross Attention {\n"
            "    shape: hexagon\n"
            "  }\n"
            "  ff2: Feed Forward\n"
            "  out: Output Probs {\n"
            "    shape: diamond\n"
            "  }\n"
            "  masked -> cross\n"
            "  cross -> ff2\n"
            "  ff2 -> out\n"
            "}\n"
            "\n"
            "encoder.norm -> decoder.cross\n"
            "decoder.out -> output\n"
            "\n"
            "output: Softmax Output {\n"
            "  shape: oval\n"
            '  style.fill: "honeydew"\n'
            "}\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 2. RAG pipeline — groups with cylinders, nested edges, feedback
        "rag_pipeline": (
            "direction: down\n"
            "\n"
            "user: User Query {\n"
            "  shape: oval\n"
            '  style.fill: "aliceblue"\n'
            "}\n"
            "\n"
            "retrieval: Retrieval Stage {\n"
            '  style.fill: "lavenderblush"\n'
            "  style.stroke-dash: 3\n"
            "  enc: Encoder\n"
            "  idx: Vector Index {\n"
            "    shape: cylinder\n"
            "  }\n"
            "  kb: Knowledge Base {\n"
            "    shape: cylinder\n"
            "  }\n"
            "  enc -> idx\n"
            "  kb -> idx\n"
            "}\n"
            "\n"
            "generation: Generation Stage {\n"
            '  style.fill: "honeydew"\n'
            "  ctx: Context Builder\n"
            "  llm: LLM Generator {\n"
            "    shape: hexagon\n"
            "  }\n"
            "  ctx -> llm\n"
            "}\n"
            "\n"
            "answer: Answer {\n"
            "  shape: oval\n"
            '  style.fill: "lemonchiffon"\n'
            "}\n"
            "\n"
            "user -> retrieval.enc\n"
            "retrieval.idx -> generation.ctx\n"
            "user -> generation.ctx\n"
            "generation.llm -> answer\n"
            "answer -> user\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 3. Classification metrics — fan-out groups like restaurants/diners pattern
        "classification_metrics": (
            "direction: right\n"
            "\n"
            "inputs: Model Outputs {\n"
            "  style.stroke-dash: 2\n"
            '  *.style.fill: "lavender"\n'
            "  pred: Predictions\n"
            "  truth: True Labels\n"
            "}\n"
            "\n"
            "confusion: Confusion Matrix {\n"
            "  style.stroke-dash: 2\n"
            '  *.style.fill: "mistyrose"\n'
            "  tp: True Positives {\n"
            "    shape: diamond\n"
            "  }\n"
            "  fp: False Positives\n"
            "  fn: False Negatives\n"
            "  tn: True Negatives {\n"
            "    shape: diamond\n"
            "  }\n"
            "}\n"
            "\n"
            "metrics: Metrics {\n"
            "  style.stroke-dash: 2\n"
            '  *.style.fill: "honeydew"\n'
            "  prec: Precision {\n"
            "    shape: hexagon\n"
            "  }\n"
            "  rec: Recall {\n"
            "    shape: hexagon\n"
            "  }\n"
            "  f1: F1 Score {\n"
            "    shape: cylinder\n"
            "  }\n"
            "  prec -> f1\n"
            "  rec -> f1\n"
            "}\n"
            "\n"
            "inputs.pred -> confusion.tp\n"
            "inputs.pred -> confusion.fp\n"
            "inputs.truth -> confusion.tp\n"
            "inputs.truth -> confusion.fn\n"
            "confusion.tp -> metrics.prec\n"
            "confusion.fp -> metrics.prec\n"
            "confusion.tp -> metrics.rec\n"
            "confusion.fn -> metrics.rec\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 4. Training loop with equation — LaTeX formula node
        "training_with_equation": (
            "direction: right\n"
            "\n"
            "loss_eq: |latex\n"
            "  L = -\\\\frac{1}{N}\\\\sum_{i=1}^{N} y_i \\\\log(\\\\hat{y}_i)\n"
            "|\n"
            "\n"
            "data: Data {\n"
            '  style.fill: "mintcream"\n'
            "  raw: Raw Data {\n"
            "    shape: cylinder\n"
            "  }\n"
            "  split: Train/Test {\n"
            "    shape: diamond\n"
            "  }\n"
            "  raw -> split\n"
            "}\n"
            "\n"
            "training: Training Loop {\n"
            '  style.fill: "papayawhip"\n'
            "  style.stroke-dash: 3\n"
            "  model: Model {\n"
            "    shape: hexagon\n"
            "  }\n"
            "  eval: Evaluate {\n"
            "    shape: diamond\n"
            "  }\n"
            "  tune: Tune Hyperparams {\n"
            "    shape: oval\n"
            "  }\n"
            "  model -> eval\n"
            "  eval -> tune\n"
            "  tune -> model\n"
            "}\n"
            "\n"
            "deploy: Deploy {\n"
            "  shape: hexagon\n"
            '  style.fill: "honeydew"\n'
            "}\n"
            "\n"
            "data.split -> training.model\n"
            "data.split -> training.eval\n"
            "training.eval -> deploy\n"
            "loss_eq -> training.eval\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 5. LLM inference pipeline — inspired by LangUnits/Experiment pattern
        "llm_inference": (
            "direction: right\n"
            "\n"
            "title: LLM Inference Pipeline {\n"
            "  shape: text\n"
            "  near: top-center\n"
            "  style.font-size: 32\n"
            "}\n"
            "\n"
            "input: Input Processing {\n"
            '  style.fill: "aliceblue"\n'
            "  prompt: User Prompt\n"
            "  tokenizer: Tokenizer {\n"
            "    shape: hexagon\n"
            "  }\n"
            "  prompt -> tokenizer\n"
            "}\n"
            "\n"
            "model: LLM Core {\n"
            '  style.fill: "papayawhip"\n'
            "  style.stroke-dash: 3\n"
            "  embed: Embeddings {\n"
            "    shape: cylinder\n"
            "  }\n"
            "  attn: Self-Attention {\n"
            "    shape: hexagon\n"
            "  }\n"
            "  ffn: Feed Forward\n"
            "  head: LM Head {\n"
            "    shape: diamond\n"
            "  }\n"
            "  embed -> attn\n"
            "  attn -> ffn\n"
            "  ffn -> head\n"
            "}\n"
            "\n"
            "output: Output {\n"
            '  style.fill: "honeydew"\n'
            "  decode: Decode Tokens\n"
            "  response: Response {\n"
            "    shape: oval\n"
            "  }\n"
            "  decode -> response\n"
            "}\n"
            "\n"
            "input.tokenizer -> model.embed\n"
            "model.head -> output.decode\n"
            "output.response -> input.prompt\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 6. Decision flowchart — winning strategy pattern
        "decision_flow": (
            "direction: down\n"
            "\n"
            "title: Decision Process {\n"
            "  shape: text\n"
            "  near: top-center\n"
            "  style.font-size: 32\n"
            "}\n"
            "\n"
            "start: Start {\n"
            "  shape: oval\n"
            '  style.fill: "honeydew"\n'
            "}\n"
            "\n"
            "check: Evaluate Condition {\n"
            "  shape: diamond\n"
            '  style.fill: "papayawhip"\n'
            "}\n"
            "\n"
            "pathA: Path A {\n"
            '  style.fill: "aliceblue"\n'
            "  procA: Process A {\n"
            "    shape: hexagon\n"
            "  }\n"
            "}\n"
            "\n"
            "pathB: Path B {\n"
            '  style.fill: "mistyrose"\n'
            "  procB: Process B {\n"
            "    shape: hexagon\n"
            "  }\n"
            "}\n"
            "\n"
            "merge: Merge Results {\n"
            "  shape: diamond\n"
            '  style.fill: "lavenderblush"\n'
            "}\n"
            "\n"
            "end: Final Output {\n"
            "  shape: oval\n"
            '  style.fill: "honeydew"\n'
            "}\n"
            "\n"
            "start -> check\n"
            "check -> pathA.procA\n"
            "check -> pathB.procB\n"
            "pathA.procA -> merge\n"
            "pathB.procB -> merge\n"
            "merge -> end\n"
            "end -> start\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 7. Autoencoder with skip connections — bottleneck pattern
        "autoencoder": (
            "direction: down\n"
            "\n"
            "encoder: Encoder {\n"
            '  style.fill: "aliceblue"\n'
            "  style.stroke-dash: 3\n"
            "  inp: Input Data\n"
            "  enc1: Encoder L1\n"
            "  enc2: Encoder L2\n"
            "  inp -> enc1\n"
            "  enc1 -> enc2\n"
            "}\n"
            "\n"
            "latent: Latent Space {\n"
            "  shape: diamond\n"
            '  style.fill: "papayawhip"\n'
            "}\n"
            "\n"
            "kl: |latex\n"
            "  D_{KL}(q(z|x) \\\\| p(z))\n"
            "| {\n"
            "  shape: hexagon\n"
            '  style.fill: "mistyrose"\n'
            "}\n"
            "\n"
            "decoder: Decoder {\n"
            '  style.fill: "honeydew"\n'
            "  style.stroke-dash: 3\n"
            "  dec2: Decoder L2\n"
            "  dec1: Decoder L1\n"
            "  out: Reconstruction\n"
            "  dec2 -> dec1\n"
            "  dec1 -> out\n"
            "}\n"
            "\n"
            "loss: Recon Loss {\n"
            "  shape: oval\n"
            '  style.fill: "mistyrose"\n'
            "}\n"
            "\n"
            "encoder.enc2 -> latent\n"
            "latent -> kl\n"
            "latent -> decoder.dec2\n"
            "encoder.enc1 -> decoder.dec1\n"
            "decoder.out -> loss\n"
            "kl -> loss\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 8. RL agent loop — cycle with replay buffer
        "rl_cycle": (
            "direction: right\n"
            "\n"
            "agent: Agent {\n"
            '  style.fill: "aliceblue"\n'
            "  policy: Policy Network {\n"
            "    shape: hexagon\n"
            "  }\n"
            "  value: Value Function {\n"
            "    shape: oval\n"
            "  }\n"
            "}\n"
            "\n"
            "env: Environment {\n"
            '  style.fill: "mistyrose"\n'
            "  state: State\n"
            "  reward: Reward {\n"
            "    shape: diamond\n"
            "  }\n"
            "}\n"
            "\n"
            "memory: Replay Buffer {\n"
            "  shape: cylinder\n"
            '  style.fill: "lavenderblush"\n'
            "}\n"
            "\n"
            "agent.policy -> env.state\n"
            "env.state -> env.reward\n"
            "env.reward -> memory\n"
            "env.state -> agent.value\n"
            "memory -> agent.policy\n"
            "agent.value -> agent.policy\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 9. Pure cycle — flat closed loop, NO containers at all.
        # Use for `cycle` and `state-machine` signatures.
        "pure_cycle": (
            "direction: right\n"
            "\n"
            "collect: Collect Data {\n"
            "  shape: cylinder\n"
            '  style.fill: "aliceblue"\n'
            "}\n"
            "train: Train Model {\n"
            "  shape: hexagon\n"
            '  style.fill: "lavender"\n'
            "}\n"
            "validate: Validate {\n"
            "  shape: diamond\n"
            '  style.fill: "lemonchiffon"\n'
            "}\n"
            "deploy: Deploy {\n"
            "  shape: oval\n"
            '  style.fill: "honeydew"\n'
            "}\n"
            "monitor: Monitor Drift {\n"
            "  shape: rectangle\n"
            '  style.fill: "mistyrose"\n'
            "}\n"
            "\n"
            "collect -> train: feed\n"
            "train -> validate: evaluate\n"
            "validate -> deploy: if good\n"
            "validate -> train: if poor\n"
            "deploy -> monitor: live data\n"
            "monitor -> collect: retrain\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 10. Flat comparison — side-by-side nodes, NO containers.
        # Use for `comparison` and `matrix` signatures.
        "comparison_flat": (
            "direction: down\n"
            "\n"
            "hard_svm: Hard Margin SVM {\n"
            "  shape: rectangle\n"
            '  style.fill: "aliceblue"\n'
            "}\n"
            "soft_svm: Soft Margin SVM {\n"
            "  shape: rectangle\n"
            '  style.fill: "mistyrose"\n'
            "}\n"
            "\n"
            "sep_data: Separable Data {\n"
            "  shape: cylinder\n"
            '  style.fill: "honeydew"\n'
            "}\n"
            "noisy_data: Noisy Data {\n"
            "  shape: cylinder\n"
            '  style.fill: "honeydew"\n'
            "}\n"
            "\n"
            "no_slack: No Slack {\n"
            "  shape: diamond\n"
            '  style.fill: "lemonchiffon"\n'
            "}\n"
            "slack_var: Slack Variable {\n"
            "  shape: diamond\n"
            '  style.fill: "lemonchiffon"\n'
            "}\n"
            "\n"
            "result: SVM Decision {\n"
            "  shape: hexagon\n"
            '  style.fill: "lavender"\n'
            "}\n"
            "\n"
            "sep_data -> hard_svm: fits\n"
            "noisy_data -> soft_svm: tolerates\n"
            "hard_svm -> no_slack: enforces\n"
            "soft_svm -> slack_var: allows\n"
            "no_slack -> result: strict\n"
            "slack_var -> result: flexible\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 11. LLM Experiment — vars, colors, style.multiple, animated edges
        "llm_experiment": (
            "direction: right\n"
            "\n"
            "models: Model Variants {\n"
            '  style.fill: "lavender"\n'
            "  gpt: GPT-4\n"
            "  claude: Claude\n"
            "  llama: LLaMA {\n"
            "    style.multiple: true\n"
            "  }\n"
            "}\n"
            "\n"
            "prompts: Prompting Strategies {\n"
            '  style.fill: "lightskyblue"\n'
            "  zero: Zero-Shot\n"
            "  few: Few-Shot\n"
            "  cot: Chain of Thought {\n"
            "    shape: hexagon\n"
            "  }\n"
            "}\n"
            "\n"
            "eval: Evaluation {\n"
            '  style.fill: "lightgray"\n'
            "  dataset: Test Dataset {\n"
            "    shape: cylinder\n"
            "  }\n"
            "  metrics: Metrics {\n"
            "    shape: diamond\n"
            "  }\n"
            "  dataset -> metrics\n"
            "}\n"
            "\n"
            "results: Best Config {\n"
            "  shape: oval\n"
            '  style.fill: "lightgreen"\n'
            "}\n"
            "\n"
            "models -> eval.dataset\n"
            "prompts -> models\n"
            "eval.metrics -> results\n"
            "results -> prompts\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 12. Flat fan-out — one source → N parallel variants → one result.
        # NO containers. Use when slide shows multiple options/types/kernels
        # that all feed into a single outcome.
        "flat_fanout": (
            "direction: down\n"
            "\n"
            "inputData: Input Data {\n"
            "  shape: cylinder\n"
            '  style.fill: "aliceblue"\n'
            "}\n"
            "\n"
            "rbfKernel: RBF Kernel {\n"
            "  shape: hexagon\n"
            '  style.fill: "lavender"\n'
            "}\n"
            "polyKernel: Polynomial Kernel {\n"
            "  shape: hexagon\n"
            '  style.fill: "papayawhip"\n'
            "}\n"
            "linKernel: Linear Kernel {\n"
            "  shape: hexagon\n"
            '  style.fill: "mintcream"\n'
            "}\n"
            "\n"
            "classResult: Classification {\n"
            "  shape: diamond\n"
            '  style.fill: "lemonchiffon"\n'
            "}\n"
            "\n"
            "inputData -> rbfKernel: Gaussian\n"
            "inputData -> polyKernel: degree d\n"
            "inputData -> linKernel: dot product\n"
            "rbfKernel -> classResult: curved boundary\n"
            "polyKernel -> classResult: poly boundary\n"
            "linKernel -> classResult: linear boundary\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 13. Star topology — one central concept → N radiating properties.
        # NO containers. Use when slide is about properties/effects of one thing.
        "star_topology": (
            "direction: right\n"
            "\n"
            "svCore: Support Vector {\n"
            "  shape: hexagon\n"
            '  style.fill: "lavender"\n'
            "}\n"
            "\n"
            "marginWidth: Margin Width {\n"
            "  shape: oval\n"
            '  style.fill: "aliceblue"\n'
            "}\n"
            "decBoundary: Decision Boundary {\n"
            "  shape: rectangle\n"
            '  style.fill: "mintcream"\n'
            "}\n"
            "genError: Generalization Error {\n"
            "  shape: oval\n"
            '  style.fill: "honeydew"\n'
            "}\n"
            "sparsity: Sparsity {\n"
            "  shape: oval\n"
            '  style.fill: "papayawhip"\n'
            "}\n"
            "alphaWeight: Alpha Weights {\n"
            "  shape: diamond\n"
            '  style.fill: "mistyrose"\n'
            "}\n"
            "\n"
            "svCore -> marginWidth: maximizes\n"
            "svCore -> decBoundary: defines\n"
            "svCore -> genError: bounds\n"
            "svCore -> sparsity: causes\n"
            "svCore -> alphaWeight: nonzero only\n"
            "\n"
            "style.font-size: 56\n"
        ),

        # 14. Layers stack — sequential flat nodes, direction down, NO containers.
        # Use for layer-by-layer, step-by-step, or depth-based concepts.
        "layers_stack": (
            "direction: down\n"
            "\n"
            "rawInput: Raw Input {\n"
            "  shape: oval\n"
            '  style.fill: "aliceblue"\n'
            "}\n"
            "featureLayer: Feature Extraction {\n"
            "  shape: rectangle\n"
            '  style.fill: "lavender"\n'
            "}\n"
            "kernelMap: Kernel Mapping {\n"
            "  shape: hexagon\n"
            '  style.fill: "papayawhip"\n'
            "}\n"
            "optimStep: Optimization {\n"
            "  shape: diamond\n"
            '  style.fill: "lemonchiffon"\n'
            "}\n"
            "decOutput: Decision Output {\n"
            "  shape: oval\n"
            '  style.fill: "honeydew"\n'
            "}\n"
            "\n"
            "rawInput -> featureLayer: extract\n"
            "featureLayer -> kernelMap: transform\n"
            "kernelMap -> optimStep: dot product\n"
            "optimStep -> decOutput: classify\n"
            "\n"
            "style.font-size: 56\n"
        ),
    }

    def _format_diagram_code(
        self,
        *,
        slide_number: int,
        teaching_scenario: str,
        structure_plan: str,
        bullet_plan: list[dict[str, str]],
        diagram_description: str,
        diagram_code: str,
        image_plan: str,
        slide_title: str = "",
        diagram_theme: str = "",
        rag_entries_text: str = "",
    ) -> str:
        """Convert a diagram_description (plain text) into valid D2 code via an LLM formatter.

        This is the ONLY place where D2 code is generated. The scenario step
        produces a textual diagram_description; this method turns it into code
        by giving the formatter complete, compilable D2 few-shot examples.
        """
        if not self._env_flag("D2_FORMAT_WITH_GEMINI", True):
            return self._extract_d2_code(diagram_code) if diagram_code else ""

        # Provider selection follows _get_or_create_d2_formatter. We also now
        # accept OpenAI as a second-tier fallback (direct HTTP, no SDK needed),
        # so having *any* of these keys is enough to proceed.
        has_any_key = bool(
            _normalize_space(os.environ.get("GEMINI_API_KEY", ""))
            or _normalize_space(os.environ.get("GOOGLE_API_KEY", ""))
            or _normalize_space(os.environ.get("ANTHROPIC_API_KEY", ""))
            or _normalize_space(os.environ.get("OPENAI_API_KEY", ""))
        )
        if not has_any_key:
            return self._extract_d2_code(diagram_code) if diagram_code else ""

        if _normalize_space(os.environ.get("ANTHROPIC_API_KEY", "")):
            _d2_provider_default = "anthropic"
        elif _normalize_space(os.environ.get("OPENAI_API_KEY", "")):
            _d2_provider_default = "openai"
        else:
            _d2_provider_default = "gemini"
        _d2_provider = _normalize_space(
            os.environ.get("D2_FORMATTER_PROVIDER", _d2_provider_default)
        ).lower()
        if _d2_provider == "anthropic":
            default_model = "claude-sonnet-4-5"
        elif _d2_provider == "openai":
            default_model = "gpt-4.1-mini"
        else:
            default_model = "gemini-3-flash-preview"
        model_id = (
            _normalize_space(os.environ.get("D2_FORMATTER_MODEL_ID", ""))
            or _normalize_space(os.environ.get("D2_GEMINI_MODEL_ID", ""))
            or default_model
        )
        # 1200 was enough for tiny linear graphs but consistently truncated
        # diagrams with containers + nested children + Unicode formula labels —
        # the model would stop mid-string and the JSON wouldn't parse. 6000
        # has comfortable headroom for a rich container diagram with styles.
        try:
            max_tokens = int(_normalize_space(os.environ.get("D2_GEMINI_MAX_NEW_TOKENS", "6000")) or "6000")
        except Exception:
            max_tokens = 6000
        max_tokens = max(256, min(max_tokens, 12000))

        bullets = []
        for row in bullet_plan[:4]:
            b = _normalize_space(str(row.get("bullet", "")))
            e = _normalize_space(str(row.get("explanation", "")))
            if b and e:
                bullets.append(f"- {b}: {e}")
            elif b:
                bullets.append(f"- {b}")
        bullets_text = "\n".join(bullets) if bullets else "- Explain core concept"

        # Choose which example to show first — it has the most influence on output.
        _sig = (structure_plan or diagram_description or "").lower()
        if any(k in _sig for k in ("cycle", "state-machine", "loop", "feedback", "circular", "iterative", "repeat", "retrain")):
            _primary_ex = "pure_cycle"
        elif any(k in _sig for k in ("comparison", "matrix", "side-by-side", " vs ", "versus", "contrast", "tradeoff", "trade-off", "compare", "difference between")):
            _primary_ex = "comparison_flat"
        elif any(k in _sig for k in ("fan-out", "fanout", "variants", "types of", "choices", "alternatives", "multiple paths")):
            _primary_ex = "flat_fanout"
        elif any(k in _sig for k in ("properties of", "effects of", "characteristics", "role of", "impact of", "central concept", "hub")):
            _primary_ex = "star_topology"
        elif any(k in _sig for k in ("layer", "depth", "tier", "stacked", "sequential steps")):
            _primary_ex = "layers_stack"
        elif any(k in _sig for k in ("decision", "branch", "condition", "threshold", "split", "route")):
            _primary_ex = "decision_flow"
        elif any(k in _sig for k in ("training", "loss", "gradient", "backprop", "optimize", "learning rate")):
            _primary_ex = "training_with_equation"
        elif any(k in _sig for k in ("hierarchy", "tree", "taxonomy", "ontology")):
            _primary_ex = "compact_pipeline"
        else:
            _primary_ex = None  # no bias — LLM chooses the best structure

        # Build ordered example list: primary first, then 3 structurally
        # different companions. Cap at 4 total so the prompt stays compact
        # — sending all 16 examples caused OpenAI to return empty responses.
        _COMPANION_POOL = [
            "pure_cycle", "comparison_flat", "flat_fanout",
            "star_topology", "decision_flow", "layers_stack",
            "training_with_equation", "transformer",
        ]
        if _primary_ex:
            _shot_keys = [_primary_ex]
            for _k in _COMPANION_POOL:
                if _k != _primary_ex and len(_shot_keys) < 4:
                    _shot_keys.append(_k)
        else:
            # No keyword match — pick 4 structurally diverse examples
            _shot_keys = ["transformer", "pure_cycle", "flat_fanout", "comparison_flat"]
        _ordered_shots = {k: self._D2_FEW_SHOTS[k] for k in _shot_keys if k in self._D2_FEW_SHOTS}
        examples_block = ""
        for i, (name, code) in enumerate(_ordered_shots.items(), 1):
            label = name.replace("_", " ").title()
            examples_block += f"\nEXAMPLE {i} — {label}:\n{code}\n"
        _sig_label = _sig[:50] or "flow"

        system_prompt = (
            "You are a D2 diagram code generator for educational slides.\n"
            "Return valid JSON with one key: `diagram_code`.\n"
            "The value must be ONLY D2 code (no markdown fences, no explanation).\n"
            f"Diagram signature detected: {_sig_label}\n"
            "\n"
            "D2 SYNTAX REFERENCE:\n"
            "- direction: `direction: right` or `direction: down`\n"
            "- node with shape: `node_id: Human Label { shape: oval }`\n"
            "- Human Label MUST be a semantic concept, not the shape name or a placeholder ID.\n"
            "  BAD: `a: rectangle { shape: rectangle }`, `x1: diamond { shape: diamond }`, `n1: node1 { shape: oval }`\n"
            "  GOOD: `loss: Loss Function { shape: hexagon }`, `gate: Decision Gate { shape: diamond }`\n"
            "CRITICAL — SHAPE ATTRIBUTE SYNTAX:\n"
            "  The shape keyword MUST always be written as `shape: oval` (with `shape:` prefix).\n"
            "  NEVER write the shape name bare on its own line inside a node block — it becomes a child node!\n"
            "  BAD:  `bio: Biological Neuron { oval }`       — 'oval' renders as a separate child box\n"
            "  BAD:  `inp: Weighted Inputs {\\n  hexagon\\n }` — 'hexagon' renders as a separate child box\n"
            "  GOOD: `bio: Biological Neuron { shape: oval }` — oval is the node's shape, no child box\n"
            "  GOOD: `bio: Biological Neuron {\\n  shape: oval\\n}` — same, multiline form\n"
            "- container: `group_id: Human Group Name {\\n  child: Label { shape: oval }\\n}`\n"
            "  CRITICAL: containers MUST have a human-readable name after the colon.\n"
            "  BAD: `grp1 { ... }` or `group1: { ... }` — these show as 'grp1' on slide.\n"
            "  GOOD: `grp1: Input Stage { ... }` — shows 'Input Stage' as the group header.\n"
            "- container style: `style.fill: \"aliceblue\"`, `style.stroke-dash: 3`\n"
            "- edge: `a -> b: label`\n"
            "- nested edge: `group.child -> other.child: label`\n"
            "- shape types: rectangle, oval, diamond, hexagon, cylinder, cloud, text\n"
            "- math labels: Unicode only inside double quotes — NO LaTeX backslashes.\n"
            "- ANY label with `(`, `)`, `=`, `/` MUST be double-quoted.\n"
            "- global font: last line must be `style.font-size: 56`\n"
            "\n"
            "SHAPE GUIDE:\n"
            "- rectangle = process/data, oval = start/end, diamond = decision\n"
            "- hexagon = computation, cylinder = storage, cloud = external\n"
            "\n"
            "STRUCTURE — match the diagram signature:\n"
            "- `cycle` / `state-machine` / `loop`: flat closed loop, NO containers at all. "
            "Top-level nodes only, arrows form a ring with at least one branch or shortcut.\n"
            "- `comparison` / `matrix` / `vs`: flat nodes, NO containers. "
            "Two or three columns of items all feeding into one shared result node.\n"
            "- `flow` / `pipeline`: 2-3 containers each named after a stage, "
            "cross-stage arrows via dotted paths `stage.node -> stage2.node`.\n"
            "- `hierarchy` / `tree` / `stack`: containers one level deep, "
            "root at top, children fanning down with labeled edges.\n"
            "- DEFAULT for most teaching/mechanism/architecture slides: prefer 2-3 named containers. "
            "Use flat layouts only when the content is clearly a cycle, comparison, matrix, or simple fan-out.\n"
            "- RULE: containers must have ≥ 2 children and a human name. "
            "Single-child containers are forbidden.\n"
            "- Complexity target by signature:\n"
            "  * cycle/comparison/state-machine: 5-7 nodes, 6-10 edges, keep it clean.\n"
            "  * flow/pipeline/hierarchy/tree/stack: 7-10 nodes, 8-14 edges, richer internal structure.\n"
            "  * where the slide mentions phases, modules, data stores, transforms, metrics, or feedback, represent them explicitly instead of collapsing them.\n"
            "- Prefer richer diagrams whenever the topic supports them: show intermediate processing, side branches, storage/evaluation nodes, and one feedback or skip path.\n"
            "- Use ≥ 3 different shapes.\n"
            "- Pastel fills: aliceblue, honeydew, lavender, mistyrose, papayawhip, "
            "peachpuff, lemonchiffon, mintcream — different color per group.\n"
            "- When the structure is flow/pipeline/hierarchy/tree/stack, prefer 2-3 named groups and at least one labeled cross-group relation.\n"
            "- Max 52 lines. Every node in ≥ 1 edge.\n"
            "\n"
            "FORBIDDEN NODE IDs (D2 reserved keywords — cause compile errors):\n"
            "- `left`, `right`, `top`, `bottom`, `north`, `south`, `east`, `west`, `near`\n"
            "- `class`, `classes`, `shape`, `label`, `style`, `direction`, `icon`, "
            "`source`, `target`, `layers`, `scenarios`, `steps`, `vars`, `level`, `width`, `height`\n"
            "- Avoid suffixed forms `class1`, `left1` — use `leaf_a`, `branch_yes` instead.\n"
            "\n"
            "FORBIDDEN GENERIC IDs — these make diagrams look identical across all slides:\n"
            "- NEVER use: `group1`, `group2`, `group3`, `stage1`, `stage2`, `stage3`,\n"
            "  `phase1`, `phase2`, `block1`, `block2`, `section1`, `section2`\n"
            "- Use content-derived IDs: e.g. `softMargin`, `kernelTrick`, `dualProblem`,\n"
            "  `inputLayer`, `featureSpace`, `decisionBoundary` — NEVER generic.\n"
            "\n"
            "CONTAINER LABEL RULE (CRITICAL):\n"
            "  Every container MUST be written as `id: Semantic Name { ... }` — the label\n"
            "  after the colon is what the viewer sees on the diagram.\n"
            "  BAD: `grp1 { ... }`           → renders as 'grp1' (ID exposed as label)\n"
            "  BAD: `group2 { ... }`          → renders as 'group2'\n"
            "  BAD: `inputStage: { ... }`     → empty label, renders as 'inputStage'\n"
            "  GOOD: `inputStage: Input Processing { ... }` → renders as 'Input Processing'\n"
            "  GOOD: `modelCore: Neural Network { ... }`    → renders as 'Neural Network'\n"
            "  The semantic name MUST reflect the actual content of that group (not 'Group 1').\n"
            "\n"
            "CHOOSE THE RIGHT STRUCTURE for the content:\n"
            "- Cycle/iterative process → flat closed loop (pure_cycle example)\n"
            "- A vs B comparison → flat two-column nodes (comparison_flat example)\n"
            "- One concept → many effects/types → flat fanout (flat_fanout example)\n"
            "- One central entity with properties → star/radial (star_topology example)\n"
            "- Sequential layers/steps with no grouping → vertical stack (layers_stack example)\n"
            "- Decision with branches → diamond + paths (decision_flow example)\n"
            "- Complex pipeline with logical stage groupings → containers (transformer example)\n"
            "Do NOT default to 3-container horizontal pipelines for every slide.\n"
            "\nThe examples below show all available structures — pick the one that best fits:\n"
            f"{examples_block}"
        )
        # Build a context block carrying forward every piece of info the
        # earlier 2 prompts decided, so the D2 code stays consistent with the
        # bullets, the teacher scenario, and the RAG-grounded facts.
        context_lines = []
        if slide_title:
            context_lines.append(f"Slide title: {slide_title}")
        if diagram_theme:
            context_lines.append(f"Diagram theme: {diagram_theme}")
        if teaching_scenario:
            context_lines.append(f"Teaching scenario: {teaching_scenario}")
        if structure_plan:
            context_lines.append(f"Structure plan: {structure_plan}")
        context_lines.append(f"Diagram description: {diagram_description}")
        context_lines.append(f"Bullet points:\n{bullets_text}")
        if rag_entries_text:
            rag_trim = _normalize_space(rag_entries_text)
            if len(rag_trim) > 1200:
                rag_trim = rag_trim[:1200] + "…"
            context_lines.append(f"RAG facts to honor:\n{rag_trim}")
        context_block = "\n\n".join(context_lines)

        user_prompt = (
            "Generate D2 code for this diagram.\n\n"
            f"{context_block}\n\n"
            "Output JSON with key `diagram_code` containing ONLY the D2 lines.\n"
            "The diagram MUST illustrate the diagram_theme above and visually echo the bullets.\n"
            "Copy the exact syntax pattern from the examples above. Do not invent new syntax."
        )
        # Use OpenAI directly when selected as the D2 formatter.
        explicit = _normalize_space(os.environ.get("D2_FORMATTER_PROVIDER", "")).lower()
        if explicit == "openai":
            openai_code = self._format_d2_with_openai(
                slide_number=slide_number,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            if openai_code:
                return openai_code
            print(f"  [D2 formatter] slide {slide_number} OpenAI returned empty — no further fallback")
            return self._extract_d2_code(diagram_code) if diagram_code else ""

        # Primary provider (Anthropic / Gemini) with 3-attempt retry — a single
        # empty payload used to leave the slide with no D2 code, after which
        # image_service silently fell back to a generic template identical
        # across every "neural-network-ish" slide.
        import time
        formatter = None
        last_err: Exception | None = None
        attempt_tokens = max_tokens
        for attempt in range(3):
            try:
                if formatter is None:
                    formatter = self._get_or_create_d2_formatter(model_id)
                data = formatter.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=attempt_tokens,
                )
                raw = str(_ci_get(data, "diagram_code", "d2_code", "code") or "")
                normalized = self._extract_d2_code(raw)
                if normalized:
                    return normalized
                last_err = RuntimeError(f"empty diagram_code on attempt {attempt + 1}")
            except Exception as exc:
                last_err = exc
                # escalate budget on truncation errors
                if any(k in str(exc).lower() for k in ("truncat", "max_tokens", "max tokens", "length")):
                    attempt_tokens = min(12000, attempt_tokens + 3000)
            if attempt < 2:
                time.sleep(2 ** attempt)
        print(f"  [D2 formatter] slide {slide_number} primary provider failed after 3 attempts: {last_err}")

        # Secondary provider: OpenAI Chat Completions via raw HTTP (no SDK
        # dependency). Only kicks in when OPENAI_API_KEY is set, so users
        # without one keep the prior behavior.
        openai_code = self._format_d2_with_openai(
            slide_number=slide_number,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
        if openai_code:
            return openai_code

        # Final fallback: any pre-existing code passed in; otherwise empty so
        # image_service parses the diagram_description into per-slide D2.
        return self._extract_d2_code(diagram_code) if diagram_code else ""

    def _format_d2_with_openai(
        self,
        *,
        slide_number: int,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        """Fallback D2 generation via OpenAI Chat Completions (direct HTTP).

        Only runs when OPENAI_API_KEY is set. We don't import the `openai`
        SDK so users can skip installing yet another dependency.
        """
        key = _normalize_space(os.environ.get("OPENAI_API_KEY", ""))
        if not key:
            return ""

        # Model precedence: OPENAI_D2_MODEL_ID, D2_FORMATTER_MODEL_ID,
        # OPENAI_MODEL_ID, then gpt-4-turbo.
        model_id = (
            _normalize_space(os.environ.get("OPENAI_D2_MODEL_ID", ""))
            or _normalize_space(os.environ.get("D2_FORMATTER_MODEL_ID", ""))
            or _normalize_space(os.environ.get("OPENAI_MODEL_ID", ""))
            or "gpt-4-turbo"
        )
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model_id,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        import json as _json
        last_err: Exception | None = None
        import time
        for attempt in range(2):
            try:
                import requests
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code != 200:
                    last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                        time.sleep(2)
                        continue
                    break
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                try:
                    parsed = _json.loads(content)
                except Exception:
                    parsed = {}
                raw = str(_ci_get(parsed, "diagram_code", "d2_code", "code") or "")
                normalized = self._extract_d2_code(raw)
                if normalized:
                    print(f"  [D2 formatter] slide {slide_number} via OpenAI ({model_id}).")
                    return normalized
                last_err = RuntimeError("empty diagram_code in OpenAI response")
            except Exception as exc:
                last_err = exc
            if attempt == 0:
                time.sleep(1.5)
        print(f"  [D2 formatter] slide {slide_number} OpenAI fallback failed: {last_err}")
        return ""


    # Multi-step pipeline methods (Prompt 1 / 2.1 / 2.2)


    def generate_global_plan(self, prompt: str) -> list[dict[str, Any]]:
        """Prompt 1: Generate a global plan with unique idea + diagram theme per slide."""
        data = self.generate_json(
            system_prompt="You are a precise presentation planner. Return valid JSON only.",
            user_prompt=prompt,
            max_new_tokens=min(2400, self.max_new_tokens),
        )
        plan = _ci_get(data, "plan", "slides", "slide_plan")
        if not isinstance(plan, list):
            raise RuntimeError(f"Global plan missing `plan` array. Keys: {list(data.keys())}")
        result = []
        for item in plan:
            if not isinstance(item, dict):
                continue
            result.append({
                "slide_number": int(_ci_get(item, "slide_number", "number") or 0),
                "main_idea": _normalize_space(str(_ci_get(item, "main_idea", "idea") or "")),
                "diagram_theme": _normalize_space(str(_ci_get(item, "diagram_theme", "diagram") or "")),
                "progression": _normalize_space(str(_ci_get(item, "progression", "builds_on") or "")),
            })
        if not result:
            raise RuntimeError("Global plan returned empty slide list.")
        return result

    @staticmethod
    def _parse_layout(raw: Any) -> dict[str, Any]:
        """Validate + clamp the model-generated `layout` object into safe ranges.

        Returns an empty dict if the model didn't provide one — downstream
        Marp export falls back to the PNG-aspect-ratio heuristic.
        """
        if not isinstance(raw, dict):
            return {}
        allowed_kind = {
            "text-dominant",
            "balanced",
            "diagram-dominant",
            "diagram-only",
            "diagram-above-bullets",
        }
        allowed_pos = {"right", "left", "top", "center-bottom"}

        def _clip_int(v: Any, lo: int, hi: int, default: int) -> int:
            try:
                x = int(float(v))
            except Exception:
                return default
            return max(lo, min(hi, x))

        def _clip_float(v: Any, lo: float, hi: float, default: float) -> float:
            try:
                x = float(v)
            except Exception:
                return default
            return max(lo, min(hi, x))

        kind = _normalize_space(str(raw.get("kind", "")).lower())
        if kind not in allowed_kind:
            kind = "balanced"
        pos = _normalize_space(str(raw.get("image_position", "")).lower())
        if pos not in allowed_pos:
            pos = "right"
        accent = _normalize_space(str(raw.get("accent_color", "") or "#005ab4"))
        if not accent.startswith("#") or len(accent) not in (4, 7):
            accent = "#005ab4"
        bg = _normalize_space(str(raw.get("background", "") or "white"))
        if bg != "white" and (not bg.startswith("#") or len(bg) not in (4, 7)):
            bg = "white"

        return {
            "kind": kind,
            "text_ratio": _clip_int(raw.get("text_ratio"), 25, 65, 38),
            "image_position": pos,
            "title_size_em": _clip_float(raw.get("title_size_em"), 1.0, 2.0, 1.5),
            "bullet_size_em": _clip_float(raw.get("bullet_size_em"), 0.8, 1.3, 0.95),
            "bullet_line_height": _clip_float(raw.get("bullet_line_height"), 1.2, 1.8, 1.45),
            "diagram_max_height_pct": _clip_int(raw.get("diagram_max_height_pct"), 50, 100, 85),
            "diagram_max_width_pct": _clip_int(raw.get("diagram_max_width_pct"), 50, 100, 90),
            "accent_color": accent,
            "background": bg,
            "notes": _compact_text(str(raw.get("notes", "")), max_words=25, max_chars=180),
        }

    def _decide_final_layout_with_openai(
        self,
        *,
        prompt: str,
        slide_number: int = 0,
    ) -> dict[str, Any] | None:
        """OpenAI fallback for Prompt 3 when Anthropic fails.

        Direct HTTP, no SDK import — same pattern as the D2 formatter
        OpenAI fallback. Returns the raw JSON dict or None on failure.
        """
        key = _normalize_space(os.environ.get("OPENAI_API_KEY", ""))
        if not key:
            return None
        model_id = (
            _normalize_space(os.environ.get("OPENAI_LAYOUT_MODEL_ID", ""))
            or _normalize_space(os.environ.get("OPENAI_MODEL_ID", ""))
            or "gpt-4o-mini"
        )
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model_id,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": "You are a precise slide layout designer. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        import json as _json
        import time
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                import requests
                resp = requests.post(url, headers=headers, json=payload, timeout=45)
                if resp.status_code != 200:
                    last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                        time.sleep(2)
                        continue
                    break
                content = (
                    resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                parsed = _json.loads(content) if content else {}
                if isinstance(parsed, dict) and parsed:
                    print(
                        f"  [layout-decider] slide {slide_number} recovered via OpenAI ({model_id})."
                    )
                    return parsed
                last_err = RuntimeError("empty response from OpenAI")
            except Exception as exc:
                last_err = exc
            if attempt == 0:
                time.sleep(1.5)
        print(f"  [layout-decider] slide {slide_number} OpenAI fallback failed: {last_err}")
        return None

    def decide_final_layout(
        self,
        *,
        slide_w: int,
        slide_h: int,
        image_w: int,
        image_h: int,
        title: str,
        bullets: list[str],
        draft_layout: dict[str, Any],
        slide_number: int = 0,
    ) -> dict[str, Any]:
        """Prompt 3: pick final layout for one slide given real dimensions.

        Calls Anthropic first; if that errors or returns garbage, falls
        back to OpenAI (when OPENAI_API_KEY is set). All values are
        clamped at the end so a misbehaving model can't break the slide.
        """
        from prompts import final_layout_prompt
        prompt = final_layout_prompt(
            slide_w=slide_w,
            slide_h=slide_h,
            image_w=image_w,
            image_h=image_h,
            title=title,
            bullets=bullets,
            draft_layout=draft_layout,
        )
        data: dict[str, Any] = {}
        anthropic_err: Exception | None = None
        try:
            data = self.generate_json(
                system_prompt="You are a precise slide layout designer. Return valid JSON only.",
                user_prompt=prompt,
                max_new_tokens=400,
            )
        except Exception as exc:
            anthropic_err = exc
            data = {}
        if not isinstance(data, dict) or not data:
            if anthropic_err:
                print(f"  [layout-decider] slide {slide_number} Anthropic failed: {anthropic_err}")
            fallback = self._decide_final_layout_with_openai(
                prompt=prompt, slide_number=slide_number
            )
            if isinstance(fallback, dict) and fallback:
                data = fallback

        def _clip_int(v: Any, lo: int, hi: int, default: int) -> int:
            try:
                x = int(float(v))
            except Exception:
                return default
            return max(lo, min(hi, x))

        def _clip_float(v: Any, lo: float, hi: float, default: float) -> float:
            try:
                x = float(v)
            except Exception:
                return default
            return max(lo, min(hi, x))

        pos = _normalize_space(str(_ci_get(data, "image_position") or "")).lower()
        if pos not in {"right", "left", "top", "bottom"}:
            pos = "right"
        return {
            "image_position": pos,
            "image_size_pct": _clip_int(_ci_get(data, "image_size_pct"), 50, 100, 95),
            "text_ratio": _clip_int(_ci_get(data, "text_ratio"), 30, 60, 42),
            "bullet_size_em": _clip_float(_ci_get(data, "bullet_size_em"), 0.80, 1.15, 1.00),
            "bullet_line_height": _clip_float(_ci_get(data, "bullet_line_height"), 1.20, 1.50, 1.32),
            "title_size_em": _clip_float(_ci_get(data, "title_size_em"), 1.0, 1.8, 1.5),
            "reason": _normalize_space(str(_ci_get(data, "reason") or ""))[:200],
        }

    def generate_slide_detail(self, prompt: str, *, min_bullets: int = 2, max_bullets: int = 3) -> dict[str, Any]:
        """Prompt 2.1: Generate slide detail (bullets, speech, diagram theme) with global plan context."""
        data: dict[str, Any] | None = None
        last_error: Exception | None = None
        base_budget = min(2400, self.max_new_tokens)
        retry_budgets = (
            base_budget,
            min(1600, base_budget),
            min(1200, base_budget),
        )
        # HuggingFace (Gemma 1B) is verbose by nature; enforce compact limits from
        # the very first attempt so it doesn't run out of max_new_tokens mid-JSON.
        _hf_compact_suffix = (
            "\n\nSTRICT OUTPUT RULES:\n"
            "- Return one JSON object only.\n"
            "- Use EXACTLY these top-level keys: teaching_scenario, structure_plan, bullet_plan, diagram_theme, speech_plan, layout.\n"
            f"- bullet_plan must contain {min_bullets} to {max_bullets} items.\n"
            "- Keep all fields compact.\n"
            "- teaching_scenario <= 35 words.\n"
            "- structure_plan <= 18 words.\n"
            "- speech_plan <= 80 words.\n"
            "- layout.notes <= 8 words.\n"
            "- No extra keys, no markdown, no prose outside JSON.\n"
        ) if self.provider == "huggingface" else ""
        for attempt, token_budget in enumerate(retry_budgets, start=1):
            user_prompt = prompt + _hf_compact_suffix
            if attempt > 1:
                user_prompt = (
                    f"{prompt}\n\n"
                    "STRICT RETRY RULES:\n"
                    "- Return one JSON object only.\n"
                    "- Use EXACTLY these top-level keys: teaching_scenario, structure_plan, bullet_plan, diagram_theme, speech_plan, layout.\n"
                    f"- bullet_plan must contain {min_bullets} to {max_bullets} items.\n"
                    "- Keep all fields compact.\n"
                    "- teaching_scenario <= 35 words.\n"
                    "- structure_plan <= 18 words.\n"
                    "- speech_plan <= 80 words.\n"
                    "- layout.notes <= 8 words.\n"
                    "- No extra keys, no markdown, no prose outside JSON.\n"
                )
            try:
                data = self.generate_json(
                    system_prompt="You are a precise pedagogical planner. Return valid JSON only.",
                    user_prompt=user_prompt,
                    max_new_tokens=token_budget,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt == len(retry_budgets):
                    raise
        if data is None:
            raise RuntimeError(f"Slide detail generation failed: {last_error}")
        scenario = _normalize_space(str(_ci_get(data, "teaching_scenario", "scenario") or ""))
        structure_plan = _compact_text(
            str(_ci_get(data, "structure_plan", "structure") or ""), max_words=35, max_chars=240,
        )
        speech_plan = _compact_text(
            str(_ci_get(data, "speech_plan", "speech") or ""), max_words=110, max_chars=760,
        )
        diagram_theme = _normalize_space(str(_ci_get(data, "diagram_theme", "diagram") or ""))

        # Extract bullet_plan
        bullet_plan_raw = _ci_get(data, "bullet_plan", "bullets")
        bullet_plan: list[dict[str, str]] = []
        if isinstance(bullet_plan_raw, list):
            for item in bullet_plan_raw:
                if isinstance(item, dict):
                    b = _compact_text(str(_ci_get(item, "bullet", "line") or ""), max_words=12, max_chars=110)
                    e = _compact_text(str(_ci_get(item, "explanation", "detail") or ""), max_words=20, max_chars=170)
                else:
                    b = _compact_text(str(item), max_words=12, max_chars=110)
                    e = ""
                if b:
                    bullet_plan.append({"bullet": b, "explanation": e})

        # Guardrails
        if len(bullet_plan) < min_bullets:
            for fallback in _bullet_candidates_from_text(scenario) + _bullet_candidates_from_text(speech_plan):
                if len(bullet_plan) >= max_bullets:
                    break
                b = _compact_text(str(fallback.get("bullet", "")), max_words=12, max_chars=110)
                if b and b.lower() not in {x["bullet"].lower() for x in bullet_plan}:
                    bullet_plan.append({"bullet": b, "explanation": ""})

        if not scenario:
            scenario = "Explain the core concept and key takeaway for this slide."
        if not speech_plan:
            speech_plan = scenario

        return {
            "teaching_scenario": scenario,
            "structure_plan": structure_plan,
            "bullet_plan": bullet_plan[:max_bullets],
            "diagram_theme": diagram_theme,
            "speech_plan": speech_plan,
            "layout": self._parse_layout(_ci_get(data, "layout", "visual_layout", "composition")),
        }

    def generate_batch_deck(self, prompt: str, expected_slides: int) -> list[dict[str, Any]]:
        """Generate the ENTIRE deck in one Gemini call (batch mode).

        Returns a list of slide objects with teaching_scenario, bullet_plan,
        diagram_description, diagram_code, speech_plan.
        """
        # Use max tokens to fit all slides (Gemini 2.5 Flash supports up to 65K output tokens).
        budget = max(16000, self.max_new_tokens * 5)
        data = self.generate_json(
            system_prompt="You are a precise pedagogical planner AND D2 diagram generator. Return valid JSON only.",
            user_prompt=prompt,
            max_new_tokens=budget,
        )
        slides = _ci_get(data, "slides", "deck", "items")
        if not isinstance(slides, list):
            raise RuntimeError(f"Batch deck missing `slides` array. Keys: {list(data.keys())}")
        if len(slides) < expected_slides:
            print(f"  Batch returned {len(slides)}/{expected_slides} slides (likely truncated). Retrying with larger budget...")
            bigger_budget = max(32000, budget * 2)
            data = self.generate_json(
                system_prompt="You are a precise pedagogical planner AND D2 diagram generator. Return valid JSON only.",
                user_prompt=(
                    f"{prompt}\n\n"
                    "STRICT RETRY: Return ALL slides. Keep diagram_code compact (max 30 lines each). "
                    "Keep bullet explanations short. Avoid repetition."
                ),
                max_new_tokens=bigger_budget,
            )
            slides = _ci_get(data, "slides", "deck", "items")
            if not isinstance(slides, list):
                raise RuntimeError("Batch retry missing `slides` array.")
        if len(slides) < expected_slides:
            print(f"  Batch still returned only {len(slides)}/{expected_slides} slides — using what we got.")

        normalized: list[dict[str, Any]] = []
        for item in slides[:expected_slides]:
            if not isinstance(item, dict):
                continue
            scenario = _normalize_space(str(_ci_get(item, "teaching_scenario", "scenario") or ""))
            structure_plan = _compact_text(
                str(_ci_get(item, "structure_plan", "structure") or ""), max_words=35, max_chars=240,
            )
            speech_plan = _compact_text(
                str(_ci_get(item, "speech_plan", "speech") or ""), max_words=110, max_chars=760,
            )
            diagram_desc = _normalize_space(str(_ci_get(item, "diagram_description", "description") or ""))
            raw_diagram_code = str(_ci_get(item, "diagram_code", "d2_code", "code") or "")

            bullet_plan_raw = _ci_get(item, "bullet_plan", "bullets")
            bullet_plan: list[dict[str, str]] = []
            if isinstance(bullet_plan_raw, list):
                for row in bullet_plan_raw:
                    if isinstance(row, dict):
                        b = _compact_text(str(_ci_get(row, "bullet", "line") or ""), max_words=12, max_chars=110)
                        e = _compact_text(str(_ci_get(row, "explanation", "detail") or ""), max_words=20, max_chars=170)
                    else:
                        b = _compact_text(str(row), max_words=12, max_chars=110)
                        e = ""
                    if b:
                        bullet_plan.append({"bullet": b, "explanation": e})

            diagram_code = self._extract_d2_code(raw_diagram_code) if raw_diagram_code else ""

            normalized.append({
                "slide_number": int(_ci_get(item, "slide_number", "number") or (len(normalized) + 1)),
                "main_idea": _normalize_space(str(_ci_get(item, "main_idea", "idea") or "")),
                "teaching_scenario": scenario or "Explain the core concept for this slide.",
                "structure_plan": structure_plan,
                "bullet_plan": bullet_plan[:3],
                "diagram_description": diagram_desc,
                "diagram_code": diagram_code,
                "speech_plan": speech_plan or scenario,
                "image_plan": diagram_desc,
            })
        return normalized

    def generate_slide_combined(self, prompt: str, *, min_bullets: int = 2, max_bullets: int = 3) -> dict[str, Any]:
        """Combined single-call slide generator: bullets + diagram_description + diagram_code.

        Used to stay under free-tier rate limits (1 call per slide vs 3).
        """
        data = self.generate_json(
            system_prompt="You are a precise pedagogical planner AND D2 diagram generator. Return valid JSON only.",
            user_prompt=prompt,
            max_new_tokens=min(3200, self.max_new_tokens),
        )
        scenario = _normalize_space(str(_ci_get(data, "teaching_scenario", "scenario") or ""))
        structure_plan = _compact_text(
            str(_ci_get(data, "structure_plan", "structure") or ""), max_words=35, max_chars=240,
        )
        speech_plan = _compact_text(
            str(_ci_get(data, "speech_plan", "speech") or ""), max_words=110, max_chars=760,
        )
        diagram_desc = _normalize_space(str(_ci_get(data, "diagram_description", "description") or ""))
        raw_diagram_code = str(_ci_get(data, "diagram_code", "d2_code", "code") or "")

        # Extract bullet_plan
        bullet_plan_raw = _ci_get(data, "bullet_plan", "bullets")
        bullet_plan: list[dict[str, str]] = []
        if isinstance(bullet_plan_raw, list):
            for item in bullet_plan_raw:
                if isinstance(item, dict):
                    b = _compact_text(str(_ci_get(item, "bullet", "line") or ""), max_words=12, max_chars=110)
                    e = _compact_text(str(_ci_get(item, "explanation", "detail") or ""), max_words=20, max_chars=170)
                else:
                    b = _compact_text(str(item), max_words=12, max_chars=110)
                    e = ""
                if b:
                    bullet_plan.append({"bullet": b, "explanation": e})

        # Backfill bullets if needed
        if len(bullet_plan) < min_bullets:
            for fallback in _bullet_candidates_from_text(scenario) + _bullet_candidates_from_text(speech_plan):
                if len(bullet_plan) >= max_bullets:
                    break
                b = _compact_text(str(fallback.get("bullet", "")), max_words=12, max_chars=110)
                if b and b.lower() not in {x["bullet"].lower() for x in bullet_plan}:
                    bullet_plan.append({"bullet": b, "explanation": ""})

        # Fail LOUD when Gemini returned empty / malformed payload — previously
        # this silently filled in a placeholder ("Explain the core concept...")
        # which made downstream diagrams fall back to the generic heuristic
        # template, producing identical visuals across slides.
        if not bullet_plan and not diagram_desc and not raw_diagram_code:
            raise RuntimeError(
                "generate_slide_combined: Gemini returned empty payload "
                "(no bullets, no diagram_description, no diagram_code). "
                "Likely rate-limit, timeout, or JSON parse failure."
            )

        if not scenario:
            scenario = "Explain the core concept and key takeaway for this slide."
        if not speech_plan:
            speech_plan = scenario

        # Clean D2 code (same extraction as the D2 formatter does)
        diagram_code = self._extract_d2_code(raw_diagram_code) if raw_diagram_code else ""

        return {
            "teaching_scenario": scenario,
            "structure_plan": structure_plan,
            "bullet_plan": bullet_plan[:max_bullets],
            "diagram_description": diagram_desc,
            "diagram_code": diagram_code,
            "speech_plan": speech_plan,
            "image_plan": diagram_desc,
            "layout": self._parse_layout(_ci_get(data, "layout", "visual_layout", "composition")),
        }

    def generate_diagram_concept(self, prompt: str) -> str:
        """Prompt 2.2: Generate diagram concept (nodes, arrows, shapes) — no code."""
        try:
            data = self.generate_json(
                system_prompt="You are a diagram designer. Return valid JSON only.",
                user_prompt=prompt,
                max_new_tokens=min(1200, self.max_new_tokens),
            )
            desc = _extract_diagram_description(data)
        except RuntimeError:
            if self.provider != "huggingface":
                raise
            print("    Diagram concept JSON failed; retrying with plain-text fallback...")
            raw = self._generate_text_huggingface(
                system_prompt="You are a diagram designer. Return one short plain-text diagram description only.",
                user_prompt=(
                    f"{prompt}\n\n"
                    "Fallback output rules:\n"
                    "- Do not return JSON.\n"
                    "- Do not return code.\n"
                    "- Write 1 concise paragraph describing nodes and arrows.\n"
                    "- Keep it under 120 words.\n"
                ),
                max_new_tokens=min(360, self.max_new_tokens),
            )
            desc = _extract_diagram_description_from_raw_text(raw)
        if not desc:
            raise RuntimeError("Diagram concept returned empty diagram_description.")
        return _compact_text(desc, max_words=170, max_chars=1200)

    def generate_prolog_from_text(self, text: str) -> tuple[list[str], list[str]]:
        """V2_LLM variant: ask the LLM to extract Prolog facts + rules from *text*.

        Returns (facts, rules) — same shape as prolog_extractor.extract_prolog().
        The LLM extraction is richer (coreference, implicit relations) but costs
        one extra API call per slide; compare with the spaCy version for ablation.
        """
        system = (
            "You are a knowledge-extraction assistant. "
            "Given a block of educational text, extract all meaningful relationships "
            "as Prolog facts and rules. "
            "Return ONLY valid JSON with keys 'facts' (list of strings) and "
            "'rules' (list of strings). No markdown, no prose outside JSON."
        )
        # Two few-shot examples anchor the output format and atom style.
        user = (
            "Extract Prolog facts and rules from educational text.\n\n"
            "Format rules:\n"
            "- Facts: predicate(subject, object). — lowercase atoms, underscores only.\n"
            "- Rules: conclusion(X,Y) :- condition(X,Y). — use Prolog variables (uppercase).\n"
            "- Max 15 facts and 5 rules. Atoms: lowercase, alphanumeric + underscore only.\n"
            "- Return JSON: {\"facts\": [...], \"rules\": [...]}\n\n"
            "--- EXAMPLE 1 ---\n"
            "TEXT: John owns a car. Mary drives the car. "
            "If someone owns a car then they can drive it.\n"
            "OUTPUT: {\"facts\": [\"owns(john, car).\", \"drives(mary, car).\"], "
            "\"rules\": [\"can_drive(X, car) :- owns(X, car).\"]}\n\n"
            "--- EXAMPLE 2 ---\n"
            "TEXT: A support vector machine finds the optimal hyperplane. "
            "The hyperplane maximizes the margin between classes. "
            "If a kernel is radial then it maps data to higher dimensions.\n"
            "OUTPUT: {\"facts\": [\"finds(svm, hyperplane).\", "
            "\"maximizes(hyperplane, margin).\", \"separates(margin, classes).\"], "
            "\"rules\": [\"maps_to_higher_dim(K, data) :- radial_kernel(K).\"]}\n\n"
            "--- NOW EXTRACT ---\n"
            f"TEXT: {text}\n"
            "OUTPUT:"
        )
        data = self.generate_json(
            system_prompt=system,
            user_prompt=user,
            max_new_tokens=min(800, self.max_new_tokens),
        )
        facts = [str(x) for x in (data.get("facts") or []) if x][:15]
        rules = [str(x) for x in (data.get("rules") or []) if x][:5]
        # Ensure trailing dot on each entry
        facts = [f if f.endswith(".") else f + "." for f in facts]
        rules = [r if r.endswith(".") else r + "." for r in rules]
        return facts, rules

    def generate_outline(self, req: PresentationRequest, prompt: str) -> list[SlideOutline]:
        """Generate and validate a full slide outline from the user request.

        AUTO mode (`req.slides_count == 0`) accepts 5..15 slides; TARGET mode
        (`req.slides_count > 0`) defaults to the exact requested count so
        experiment runs stay comparable. Set `ALLOW_OUTLINE_SLIDE_DRIFT=1`
        to restore the older target-2..target+3 density-based behavior.
        """
        allow_target_drift = _normalize_space(
            os.environ.get("ALLOW_OUTLINE_SLIDE_DRIFT", "")
        ).lower() in {"1", "true", "yes", "on"}
        if int(req.slides_count) <= 0:
            min_slides, max_slides = 5, 15
        else:
            target = max(3, int(req.slides_count))
            if allow_target_drift:
                min_slides = max(3, target - 2)
                max_slides = min(20, target + 3)
            else:
                min_slides = target
                max_slides = target
        last_error: Exception | None = None
        base_prompt = prompt
        for attempt in range(1, 4):
            user_prompt = base_prompt
            if attempt > 1:
                user_prompt = (
                    f"{base_prompt}\n\n"
                    "STRICT RETRY RULES:\n"
                    "- Return one JSON object only.\n"
                    f"- Must include key `slides` as an array with length between {min_slides} and {max_slides}.\n"
                    "- Each slide item must include: slide_number, type, title, goal.\n"
                    "- Do not return a single slide object.\n"
                    "- No markdown fences, no prose outside JSON.\n"
                )
            data = self.generate_json(
                system_prompt="You are a precise presentation planner. Return valid JSON only.",
                user_prompt=user_prompt,
                max_new_tokens=min(2400, self.max_new_tokens),
            )
            raw_slides = _find_outline_list(data)
            if not isinstance(raw_slides, list):
                keys = ", ".join(sorted(str(k) for k in data.keys()))
                last_error = RuntimeError(f"Outline output missing `slides` list. Top-level keys: [{keys}]")
                continue
            if len(raw_slides) < min_slides:
                last_error = RuntimeError(
                    f"Outline must contain at least {min_slides} slides, got {len(raw_slides)}."
                )
                continue
            if len(raw_slides) > max_slides:
                raw_slides = raw_slides[:max_slides]

            slides: list[SlideOutline] = []
            parse_failed = False
            for idx, raw in enumerate(raw_slides, start=1):
                if not isinstance(raw, dict):
                    last_error = RuntimeError(f"Outline slide {idx} is not a JSON object.")
                    parse_failed = True
                    break
                number = int(_ci_get(raw, "slide_number", "number", "index") or idx)
                slide_type = _normalize_space(str(_ci_get(raw, "type", "slide_type") or "content")).lower()
                title = _normalize_space(str(_ci_get(raw, "title", "slide_title", "heading") or ""))
                goal = _normalize_space(str(_ci_get(raw, "goal", "objective", "description") or ""))
                if not title:
                    last_error = RuntimeError(f"Outline slide {idx} missing title.")
                    parse_failed = True
                    break
                if not goal:
                    goal = _normalize_space(f"Explain the key idea of: {title}")
                if slide_type not in {"title", "content", "section", "conclusion"}:
                    slide_type = "content"
                slides.append(
                    SlideOutline(
                        slide_number=number,
                        type=slide_type,
                        title=title,
                        goal=goal,
                    )
                )
            if parse_failed:
                continue

            slides.sort(key=lambda s: s.slide_number)
            for idx, slide in enumerate(slides, start=1):
                slide.slide_number = idx
            slides[0].type = "title"
            slides[-1].type = "conclusion"
            return slides

        if last_error is not None:
            raise RuntimeError(f"Outline generation failed after retries: {last_error}") from last_error
        raise RuntimeError("Outline generation failed with unknown error.")

    def generate_slide_content(
        self,
        slide_number: int,
        prompt: str,
        *,
        min_bullets: int = 3,
        max_bullets: int = 5,
        fallback_subtitle: str = "",
    ) -> SlideContent:
        """Generate and validate content payload for a single slide."""
        max_bullets = max(1, int(max_bullets))
        min_bullets = max(1, min(int(min_bullets), max_bullets))
        base_prompt = prompt
        last_error: Exception | None = None
        for attempt in range(1, 4):
            user_prompt = base_prompt
            if attempt > 1:
                user_prompt = (
                    f"{base_prompt}\n\n"
                    "STRICT RETRY RULES:\n"
                    "- Return one complete JSON object only.\n"
                    "- Include: title, subtitle, bullets, voiceover, image_suggestion, layout_hint.\n"
                    "- `bullets` must be a JSON array with required count.\n"
                    "- Keep voiceover concise and valid JSON-safe text.\n"
                    "- No markdown fences, no prose outside JSON.\n"
                )
            data = self.generate_json(
                system_prompt="You are an evidence-grounded slide writer. Return valid JSON only.",
                user_prompt=user_prompt,
                max_new_tokens=min(2400, self.max_new_tokens),
            )
            title = _compact_text(str(_ci_get(data, "title", "slide_title", "heading") or ""), max_words=8, max_chars=62)
            subtitle = _compact_text(str(_ci_get(data, "subtitle", "sub_title", "tagline") or ""), max_words=7, max_chars=56)
            voiceover = _normalize_space(
                str(_ci_get(data, "voiceover", "audio_script", "speaker_notes", "narration") or "")
            )
            image_suggestion = _normalize_space(
                str(_ci_get(data, "image_suggestion", "image_prompt", "image_description") or "")
            )
            layout_hint = _normalize_space(str(_ci_get(data, "layout_hint", "layout") or "")).lower() or "text_only"
            bullets_raw = _ci_get(data, "bullets", "key_points", "bullet_points", "points")
            if isinstance(bullets_raw, str):
                bullets = [
                    _normalize_space(line.lstrip("-*0123456789. ").strip())
                    for line in bullets_raw.splitlines()
                    if _normalize_space(line.lstrip("-*0123456789. ").strip())
                ]
            elif isinstance(bullets_raw, list):
                bullets = [_normalize_space(str(item)) for item in bullets_raw if _normalize_space(str(item))]
            else:
                last_error = RuntimeError(f"Slide {slide_number}: `bullets` must be a list/string.")
                continue

            compacted: list[str] = []
            seen_bullets: set[str] = set()
            for bullet in bullets:
                b = _compact_text(bullet, max_words=24, max_chars=200)
                if not b:
                    continue
                key = b.lower()
                if key in seen_bullets:
                    continue
                seen_bullets.add(key)
                compacted.append(b)
            bullets = compacted

            bullets = bullets[:max_bullets]
            if len(bullets) < min_bullets:
                last_error = RuntimeError(f"Slide {slide_number}: expected at least {min_bullets} bullet points.")
                continue
            if not title:
                last_error = RuntimeError(f"Slide {slide_number}: missing title.")
                continue
            if not voiceover:
                last_error = RuntimeError(f"Slide {slide_number}: missing voiceover.")
                continue
            if not subtitle:
                subtitle = _compact_text(fallback_subtitle, max_words=7, max_chars=56)
            if layout_hint not in {"title", "text_only", "text_left_image_right", "section_divider", "conclusion"}:
                layout_hint = "text_only"
            return SlideContent(
                slide_number=slide_number,
                title=title,
                bullets=bullets,
                voiceover=voiceover,
                image_suggestion=image_suggestion,
                layout_hint=layout_hint,
                subtitle=subtitle,
            )

        if last_error is not None:
            raise RuntimeError(f"Slide {slide_number}: generation failed after retries: {last_error}") from last_error
        raise RuntimeError(f"Slide {slide_number}: generation failed with unknown error.")

    def generate_slide_scenario(
        self,
        slide_number: int,
        prompt: str,
        *,
        min_bullets: int = 2,
        max_bullets: int = 3,
    ) -> dict[str, Any]:
        """Generate a slide-level pedagogical scenario plan grounded in RAG entries."""
        max_bullets = max(1, int(max_bullets))
        min_bullets = max(1, min(int(min_bullets), max_bullets))
        base_prompt = prompt
        last_error: Exception | None = None

        for attempt in range(1, 4):
            user_prompt = base_prompt
            if attempt > 1:
                user_prompt = (
                    f"{base_prompt}\n\n"
                    "STRICT RETRY RULES:\n"
                    "- Return one complete JSON object only.\n"
                    "- Include keys: teaching_scenario, structure_plan, bullet_plan, diagram_description, diagram_code, image_plan, speech_plan.\n"
                    "- `bullet_plan` must be an array of objects with `bullet` and `explanation`.\n"
                    "- No markdown fences, no prose outside JSON.\n"
                )
            data = self.generate_json(
                system_prompt="You are a precise pedagogical planner grounded in RAG evidence. Return valid JSON only.",
                user_prompt=user_prompt,
                max_new_tokens=min(2400, self.max_new_tokens),
            )
            scenario = _normalize_space(str(_ci_get(data, "teaching_scenario", "scenario", "slide_scenario") or ""))
            structure_plan = _compact_text(
                str(_ci_get(data, "structure_plan", "slide_structure", "organization_plan") or ""),
                max_words=35,
                max_chars=240,
            )
            speech_plan = _compact_text(
                str(_ci_get(data, "speech_plan", "voiceover_plan", "speech_script", "narration_plan") or ""),
                max_words=110,
                max_chars=760,
            )
            bullet_plan_raw = _ci_get(data, "bullet_plan", "teaching_plan", "idea_plan", "line_plan")
            bullet_plan: list[dict[str, str]] = []
            if isinstance(bullet_plan_raw, list):
                for item in bullet_plan_raw:
                    bullet = ""
                    explanation = ""
                    if isinstance(item, dict):
                        bullet = _compact_text(
                            str(_ci_get(item, "bullet", "line", "idea", "point") or ""),
                            max_words=14,
                            max_chars=110,
                        )
                        explanation = _compact_text(
                            str(_ci_get(item, "explanation", "detail", "teaching_note", "why_it_matters") or "")
                            ,
                            max_words=20,
                            max_chars=170,
                        )
                    else:
                        bullet = _compact_text(str(item), max_words=14, max_chars=110)
                    if bullet:
                        bullet_plan.append({"bullet": bullet, "explanation": explanation})
            elif isinstance(bullet_plan_raw, dict):
                for key in ("bullets", "items", "lines", "points"):
                    value = _ci_get(bullet_plan_raw, key)
                    if isinstance(value, list):
                        for item in value:
                            bullet = _compact_text(str(item), max_words=14, max_chars=110)
                            if bullet:
                                bullet_plan.append({"bullet": bullet, "explanation": ""})
                    elif isinstance(value, str):
                        bullet_plan.extend(_bullet_candidates_from_text(value))
            elif isinstance(bullet_plan_raw, str):
                lines = [ln.strip() for ln in bullet_plan_raw.splitlines() if ln.strip()]
                for ln in lines:
                    clean = _normalize_space(ln.lstrip("-*0123456789. "))
                    if not clean:
                        continue
                    if ":" in clean:
                        left, right = clean.split(":", 1)
                        bullet = _compact_text(left, max_words=24, max_chars=200)
                        explanation = _compact_text(right, max_words=20, max_chars=170)
                    else:
                        bullet = _compact_text(clean, max_words=24, max_chars=200)
                        explanation = ""
                    if bullet:
                        bullet_plan.append({"bullet": bullet, "explanation": explanation})
            # Deduplicate while preserving order.
            deduped: list[dict[str, str]] = []
            seen: set[str] = set()
            for item in bullet_plan:
                key = item["bullet"].lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            bullet_plan = deduped[:max_bullets]

            # Backfill incomplete bullet plans from other generated fields to prevent hard failure.
            if len(bullet_plan) < min_bullets:
                fallback_rows: list[dict[str, str]] = []
                fallback_rows.extend(_bullet_candidates_from_text(structure_plan))
                fallback_rows.extend(_bullet_candidates_from_text(scenario))
                fallback_rows.extend(_bullet_candidates_from_text(speech_plan))
                for row in fallback_rows:
                    if len(bullet_plan) >= max_bullets:
                        break
                    b = _compact_text(str(row.get("bullet", "")), max_words=14, max_chars=110)
                    e = _compact_text(str(row.get("explanation", "")), max_words=20, max_chars=170)
                    if not b:
                        continue
                    key = b.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    bullet_plan.append({"bullet": b, "explanation": e})

            # Last-resort guardrail: never block the pipeline due to one malformed scenario slide.
            if len(bullet_plan) < min_bullets:
                guardrails = [
                    "Core concept for this slide",
                    "How the idea works step by step",
                    "Key takeaway for students",
                ]
                for text in guardrails:
                    if len(bullet_plan) >= min_bullets:
                        break
                    b = _compact_text(text, max_words=24, max_chars=200)
                    key = b.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    bullet_plan.append({"bullet": b, "explanation": ""})

            if len(bullet_plan) < min_bullets:
                last_error = RuntimeError(
                    f"Slide {slide_number}: scenario bullet_plan must contain at least {min_bullets} lines."
                )
                continue

            diagram_description = _extract_diagram_description(data)
            image_plan = _compact_text(
                str(_ci_get(data, "image_plan", "image_prompt", "image_suggestion") or ""),
                max_words=25,
                max_chars=170,
            )
            if not image_plan:
                image_plan = diagram_description
            if not diagram_description:
                diagram_description = image_plan
            diagram_description = _compact_text(diagram_description, max_words=80, max_chars=560)
            # Scenario may or may not include raw diagram_code. Either way,
            # the D2 formatter generates proper D2 from diagram_description.
            diagram_code = _extract_diagram_code(data)
            diagram_code = self._format_diagram_code(
                slide_number=slide_number,
                teaching_scenario=scenario,
                structure_plan=structure_plan,
                bullet_plan=bullet_plan,
                diagram_description=diagram_description,
                diagram_code=diagram_code,
                image_plan=image_plan,
            )

            if not scenario:
                exp = [_normalize_space(item.get("explanation", "")) for item in bullet_plan]
                exp = [x for x in exp if x]
                if exp:
                    scenario = _normalize_space(" ".join(exp[:2]))
            if not scenario:
                bullets_only = [_normalize_space(item.get("bullet", "")) for item in bullet_plan]
                bullets_only = [x for x in bullets_only if x]
                if bullets_only:
                    scenario = _normalize_space(f"Teach this slide through: {'; '.join(bullets_only[:3])}.")
            if not scenario and structure_plan:
                scenario = structure_plan
            if not scenario and speech_plan:
                scenario = speech_plan
            if not scenario and diagram_description:
                scenario = _normalize_space(f"Use the diagram to explain: {diagram_description}")
            if not scenario and image_plan:
                scenario = _normalize_space(f"Use the visual to explain: {image_plan}")
            if not scenario:
                # Final guardrail so one malformed slide cannot stop the whole deck generation.
                scenario = "Explain the core concept and key takeaway for this slide."
            scenario = _compact_text(scenario, max_words=55, max_chars=380)
            if not scenario:
                last_error = RuntimeError(f"Slide {slide_number}: missing teaching_scenario in scenario plan.")
                continue
            if not speech_plan:
                exp = [_normalize_space(item.get("explanation", "")) for item in bullet_plan]
                exp = [x for x in exp if x]
                if exp:
                    speech_plan = _compact_text(" ".join(exp[:3]), max_words=110, max_chars=760)
                else:
                    speech_plan = scenario

            return {
                "slide_number": int(slide_number),
                "teaching_scenario": scenario,
                "structure_plan": structure_plan,
                "bullet_plan": bullet_plan,
                "diagram_description": diagram_description,
                "diagram_code": diagram_code,
                "image_plan": image_plan,
                "speech_plan": speech_plan,
            }

        if last_error is not None:
            raise RuntimeError(f"Slide {slide_number}: scenario generation failed after retries: {last_error}") from last_error
        raise RuntimeError(f"Slide {slide_number}: scenario generation failed with unknown error.")


def load_chunks(chunks_path: Path) -> list[ChunkItem]:
    """Load chunked source text used as the RAG knowledge base."""
    rows: list[ChunkItem] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            text = _normalize_space(str(item.get("text", "")))
            if not text:
                continue
            tokens = set(_tokenize(text))
            if not tokens:
                continue
            chunk_id = _normalize_space(str(item.get("chunk_id", ""))) or f"chunk_{line_no:06d}"
            source = _normalize_space(str(item.get("source", "")))
            rows.append(ChunkItem(chunk_id=chunk_id, source=source, text=text, tokens=tokens))
    if not rows:
        raise RuntimeError(f"No usable chunk rows found in {chunks_path}")
    return rows


def retrieve_chunks_lexical(chunks: list[ChunkItem], query: str, top_k: int) -> list[ChunkHit]:
    """Retrieve top chunk matches using token-overlap lexical scoring."""
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        q_tokens = {tok for tok in query.lower().split() if _normalize_space(tok)}
    if not q_tokens:
        return []

    hits: list[ChunkHit] = []
    for row in chunks:
        overlap = len(row.tokens.intersection(q_tokens))
        if overlap <= 0:
            continue
        precision = overlap / max(1.0, len(q_tokens))
        coverage = overlap / max(1.0, len(row.tokens))
        score = (0.85 * precision) + (0.15 * coverage)
        hits.append(ChunkHit(chunk_id=row.chunk_id, source=row.source, score=float(score), text=row.text))
    hits.sort(key=lambda it: it.score, reverse=True)
    return hits[: max(1, int(top_k))]


class SemanticRetriever:
    _MODEL_CACHE: dict[str, tuple[Any, Any, Any]] = {}

    def __init__(self, *, chunks: list[ChunkItem], model_id: str, batch_size: int = 32) -> None:
        self.chunks = chunks
        self.model_id = model_id
        self.batch_size = max(1, int(batch_size))
        self._tokenizer, self._model, self._device = self._get_or_create_model(model_id)
        self._chunk_embeddings = self._encode_texts([c.text for c in chunks], batch_size=self.batch_size)

    @classmethod
    def _get_or_create_model(cls, model_id: str) -> tuple[Any, Any, Any]:
        """Return or create model."""
        cached = cls._MODEL_CACHE.get(model_id)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("Could not import transformers/torch for semantic retrieval.") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        cls._MODEL_CACHE[model_id] = (tokenizer, model, device)
        return tokenizer, model, device

    def _encode_texts(self, texts: list[str], *, batch_size: int) -> Any:
        import torch
        import torch.nn.functional as F

        if not texts:
            return torch.empty((0, 1), dtype=torch.float32)

        vectors: list[Any] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                outputs = self._model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                emb = summed / counts
                emb = F.normalize(emb, p=2, dim=1)
                vectors.append(emb.cpu())
        return torch.cat(vectors, dim=0)

    def search(self, query: str, top_k: int) -> list[ChunkHit]:
        import torch

        if not self.chunks:
            return []
        query_vec = self._encode_texts([query], batch_size=1)
        if query_vec.shape[0] == 0:
            return []
        scores = torch.matmul(self._chunk_embeddings, query_vec[0])
        k = min(max(1, int(top_k)), scores.shape[0])
        vals, idxs = torch.topk(scores, k=k)

        hits: list[ChunkHit] = []
        for score, idx in zip(vals.tolist(), idxs.tolist()):
            chunk = self.chunks[int(idx)]
            hits.append(
                ChunkHit(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    score=float(score),
                    text=chunk.text,
                )
            )
        return hits


def retrieve_chunks(
    *,
    chunks: list[ChunkItem],
    query: str,
    top_k: int,
    rag_mode: str,
    semantic_retriever: SemanticRetriever | None = None,
) -> list[ChunkHit]:
    """Dispatch retrieval to semantic or lexical mode based on configuration."""
    mode = _normalize_space(rag_mode).lower() or "semantic"
    if mode == "semantic":
        if semantic_retriever is None:
            raise RuntimeError("Semantic RAG requires a SemanticRetriever instance.")
        return semantic_retriever.search(query=query, top_k=top_k)
    if mode == "lexical":
        return retrieve_chunks_lexical(chunks=chunks, query=query, top_k=top_k)
    raise RuntimeError(f"Unsupported rag_mode: {rag_mode}")


def build_evidence_text(hits: list[ChunkHit], max_chars: int) -> str:
    """Serialize retrieved chunks into compact evidence text for prompting."""
    if not hits:
        return ""
    rows: list[str] = []
    for hit in hits:
        excerpt = _normalize_space(hit.text)
        if len(excerpt) > 450:
            excerpt = excerpt[:447].rstrip() + "..."
        rows.append(f"[chunk_id={hit.chunk_id}; source={hit.source or 'unknown'}; score={hit.score:.4f}] {excerpt}")
    out = "\n\n".join(rows).strip()
    limit = max(500, int(max_chars))
    if len(out) > limit:
        out = out[: limit - 3].rstrip() + "..."
    return out


def make_slide_query(req: PresentationRequest, outline: SlideOutline) -> str:
    """Build the retrieval query for one slide from topic, title, and goal."""
    return _normalize_space(f"{req.topic} {outline.title} {outline.goal}")
