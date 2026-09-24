"""
Prolog extractor — converts natural-language text to Prolog facts and rules
using spaCy dependency parsing (no LLM required).

Pipeline:
  text → spaCy doc → SVO triples + if-then patterns → Prolog facts / rules

Intended use (V2_spacy diagram mode):
    facts, rules = extract_prolog(bullet_text)
    # facts: ["learns(student, concepts).", ...]
    # rules: ["can_apply(X, Y) :- learns(X, Y)."]

The extracted KB is:
  1. Passed to diagram_concept_prompt_v2() so the LLM generates diagrams
     grounded in explicit relational structure rather than free text.
  2. Saved in scenario_plan.json for later querying / educational exercises
     (see prolog_runner.py).
"""

from __future__ import annotations

import re
from typing import Optional

# spaCy is loaded lazily so importing this module is free if you never call it.
_nlp: Optional[object] = None
_MODEL = "en_core_web_sm"


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy  # type: ignore
            _nlp = spacy.load(_MODEL)
        except ModuleNotFoundError as exc:
            missing = exc.name or "a spaCy dependency"
            raise RuntimeError(
                f"spaCy dependency '{missing}' is missing. "
                "Run: bash scripts/setup_pod.sh "
                "or install manually with: "
                "pip install 'click>=8.1,<9' && python -m spacy download "
                f"{_MODEL}"
            ) from exc
        except OSError:
            raise RuntimeError(
                f"spaCy model '{_MODEL}' not found. "
                "Run: bash scripts/setup_pod.sh "
                f"or: python -m spacy download {_MODEL}"
            )
    return _nlp


# token normalization

def _norm(token) -> str:
    """Lemmatize, lowercase, replace spaces with underscores."""
    return re.sub(r"[^a-z0-9_]", "", token.lemma_.lower().replace(" ", "_").replace("-", "_"))


def _norm_span(span) -> str:
    """Normalize a multi-token span (e.g. noun chunks)."""
    return re.sub(r"[^a-z0-9_]", "", span.lemma_.lower().replace(" ", "_").replace("-", "_"))


# compound / noun-phrase head resolution

def _head_label(token) -> str:
    """Return a readable Prolog atom for a token, incorporating immediate compound modifiers."""
    parts = [c for c in token.lefts if c.dep_ in ("compound", "amod") and c.i < token.i]
    parts.append(token)
    raw = "_".join(_norm(t) for t in parts if _norm(t))
    return raw or _norm(token)


# sentence-level extraction

def _extract_from_sentence(sent) -> tuple[list[str], list[str]]:
    """Return (facts, rules) extracted from a single sentence."""
    facts: list[str] = []
    rules: list[str] = []
    lower_texts = [t.text.lower() for t in sent]

    # if … then … → rule
    if "if" in lower_texts:
        rule = _try_ifthen_rule(sent)
        if rule:
            rules.append(rule)
            return facts, rules  # handled as rule, skip SVO

    # simple SVO fact
    for token in sent:
        if token.dep_ != "ROOT":
            continue
        subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass", "csubj")]
        objects = [w for w in token.rights if w.dep_ in ("dobj", "attr", "pobj", "nsubjpass", "oprd", "acomp")]

        # also collect prepositional objects one level deeper
        for child in token.rights:
            if child.dep_ == "prep":
                objects += [w for w in child.rights if w.dep_ == "pobj"]

        if not subjects:
            continue

        verb = _norm(token)
        if not verb:
            continue

        for subj in subjects[:2]:  # cap at 2 subjects per verb
            s_label = _head_label(subj)
            if not s_label:
                continue
            if objects:
                for obj in objects[:2]:  # cap at 2 objects
                    o_label = _head_label(obj)
                    if o_label and s_label != o_label:
                        facts.append(f"{verb}({s_label}, {o_label}).")
            else:
                # intransitive: unary fact
                facts.append(f"{verb}({s_label}).")

    return facts, rules


def _try_ifthen_rule(sent) -> str | None:
    """Try to parse an if-then sentence into a Prolog rule `head :- body.`"""
    text = sent.text
    low = text.lower()

    split_then = None
    if " then " in low:
        idx = low.index(" then ")
        split_then = idx + len(" then ")
    elif ", " in low and "if" in low:
        idx = low.index("if")
        comma_idx = low.find(",", idx)
        if comma_idx != -1:
            split_then = comma_idx + 2

    if split_then is None:
        return None

    if_part = text[:split_then].strip().lstrip("If ").lstrip("if ")
    then_part = text[split_then:].strip().rstrip(".")

    nlp = _get_nlp()
    cond_facts, _ = _extract_from_sentence(next(nlp(if_part).sents))
    concl_facts, _ = _extract_from_sentence(next(nlp(then_part).sents))

    if cond_facts and concl_facts:
        head = concl_facts[0].rstrip(".")
        body = cond_facts[0].rstrip(".")
        return f"{head} :- {body}."
    return None


# public API

def extract_prolog(text: str) -> tuple[list[str], list[str]]:
    """Extract Prolog facts and rules from *text* using spaCy (no LLM).

    Returns:
        facts: list of strings like "learns(student, concept)."
        rules: list of strings like "can_apply(X, Y) :- learns(X, Y)."
    """
    if not text or not text.strip():
        return [], []

    nlp = _get_nlp()
    doc = nlp(text)

    all_facts: list[str] = []
    all_rules: list[str] = []

    for sent in doc.sents:
        f, r = _extract_from_sentence(sent)
        all_facts.extend(f)
        all_rules.extend(r)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_facts = [x for x in all_facts if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]
    seen.clear()
    unique_rules = [x for x in all_rules if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]

    return unique_facts, unique_rules


def prolog_kb_text(facts: list[str], rules: list[str]) -> str:
    """Format facts + rules as a compact Prolog KB string for prompt injection."""
    lines: list[str] = []
    if facts:
        lines.append("% Facts")
        lines.extend(facts)
    if rules:
        lines.append("% Rules")
        lines.extend(rules)
    return "\n".join(lines)


def prolog_kb_summary(facts: list[str], rules: list[str], max_items: int = 12) -> str:
    """Return a shortened KB (first max_items entries) to keep prompts concise."""
    all_items = facts + rules
    shown = all_items[:max_items]
    suffix = f"\n% … and {len(all_items) - max_items} more" if len(all_items) > max_items else ""
    return "\n".join(shown) + suffix
