"""
prolog_runner.py — Educational Prolog exercise demo.

Loads the Prolog KB extracted by the diagram pipeline (saved in scenario_plan.json),
builds a combined knowledge base, and runs demonstrative queries.

Two execution backends (tried in order):
  1. pyswip   — Python bindings for SWI-Prolog (pip install pyswip)
  2. built-in — pure-Python forward-chaining on ground facts (subset of Prolog)

Usage (standalone):
    python prolog_runner.py --scenario data/scenario_plan.json

Usage (from code):
    from prolog_runner import run_demo
    run_demo(scenario_path="data/scenario_plan.json")
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# KB loading

def load_kb_from_scenario(scenario_path: str | Path) -> tuple[list[str], list[str]]:
    """Collect all Prolog facts + rules from every slide entry in scenario_plan.json."""
    data = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
    all_facts: list[str] = []
    all_rules: list[str] = []
    seen: set[str] = set()

    entries = data if isinstance(data, list) else data.get("slides", [data])
    for entry in entries:
        for f in entry.get("prolog_facts", []):
            if f and f not in seen:
                all_facts.append(f)
                seen.add(f)
        for r in entry.get("prolog_rules", []):
            if r and r not in seen:
                all_rules.append(r)
                seen.add(r)
    return all_facts, all_rules


# Pure-Python forward-chaining engine (fallback)

def _parse_fact(s: str) -> tuple[str, list[str]] | None:
    """Parse 'pred(a, b).' → ('pred', ['a', 'b'])"""
    m = re.match(r"(\w+)\(([^)]*)\)\.$", s.strip())
    if not m:
        return None
    pred = m.group(1)
    args = [a.strip() for a in m.group(2).split(",")]
    return pred, args


def _parse_rule(s: str) -> tuple[tuple[str, list[str]], tuple[str, list[str]]] | None:
    """Parse 'head(a,b) :- body(a,b).' → (head_parsed, body_parsed)"""
    m = re.match(r"(.+?)\s*:-\s*(.+?)\.$", s.strip())
    if not m:
        return None
    head = _parse_fact(m.group(1).strip() + ".")
    body = _parse_fact(m.group(2).strip() + ".")
    if head and body:
        return head, body
    return None


def _forward_chain(facts: list[str], rules: list[str]) -> list[tuple[str, list[str]]]:
    """One-pass forward chaining: derive new facts from ground rules."""
    known: list[tuple[str, list[str]]] = []
    for f in facts:
        parsed = _parse_fact(f)
        if parsed:
            known.append(parsed)

    derived: list[tuple[str, list[str]]] = []
    for r in rules:
        parsed_rule = _parse_rule(r)
        if not parsed_rule:
            continue
        (h_pred, h_args), (b_pred, b_args) = parsed_rule
        for k_pred, k_args in known:
            if k_pred == b_pred and len(k_args) == len(b_args):
                # Ground substitution: map body vars to known args
                subst: dict[str, str] = {}
                match = True
                for bv, ka in zip(b_args, k_args):
                    if bv[0].isupper():  # Prolog variable
                        subst[bv] = ka
                    elif bv != ka:
                        match = False
                        break
                if match:
                    new_args = [subst.get(a, a) for a in h_args]
                    entry = (h_pred, new_args)
                    if entry not in known and entry not in derived:
                        derived.append(entry)
    return derived


class SimplePrologEngine:
    """Minimal pure-Python Prolog engine for ground-fact querying."""

    def __init__(self, facts: list[str], rules: list[str]):
        self.base: list[tuple[str, list[str]]] = []
        for f in facts:
            p = _parse_fact(f)
            if p:
                self.base.append(p)
        self.derived = _forward_chain(facts, rules)
        self.all_facts = self.base + self.derived

    def query(self, predicate: str) -> list[list[str]]:
        """Return all argument lists matching *predicate*."""
        return [args for pred, args in self.all_facts if pred == predicate]

    def query_subject(self, subject: str) -> list[tuple[str, str]]:
        """Return all (predicate, object) pairs where subject appears as first arg."""
        return [
            (pred, args[1] if len(args) > 1 else "")
            for pred, args in self.all_facts
            if args and args[0] == subject
        ]

    def all_predicates(self) -> list[str]:
        return sorted({pred for pred, _ in self.all_facts})

    def all_entities(self) -> list[str]:
        entities: set[str] = set()
        for _, args in self.all_facts:
            entities.update(args)
        return sorted(entities)


# pyswip backend

def _try_pyswip(facts: list[str], rules: list[str]) -> bool:
    """Try to assert KB into SWI-Prolog via pyswip. Returns True if available."""
    try:
        from pyswip import Prolog  # type: ignore
        prolog = Prolog()
        for f in facts:
            clean = f.rstrip(".")
            prolog.assertz(clean)
        for r in rules:
            clean = r.rstrip(".")
            prolog.assertz(clean)
        return True
    except ImportError:
        return False
    except Exception:
        return False


# demo

def run_demo(
    scenario_path: str | Path,
    max_entities: int = 8,
    max_derived: int = 10,
) -> dict[str, Any]:
    """Run the educational Prolog demo and return a summary dict.

    The demo shows:
      1. The loaded KB (facts + rules)
      2. All unique entities (→ potential diagram nodes)
      3. Derived facts after forward chaining (→ implicit relationships)
      4. Sample queries: what does entity X do?
    """
    facts, rules = load_kb_from_scenario(scenario_path)

    print("=" * 60)
    print("PROLOG KNOWLEDGE BASE — extracted from slide content")
    print("=" * 60)
    print(f"\nFacts ({len(facts)}):")
    for f in facts:
        print(f"  {f}")
    if rules:
        print(f"\nRules ({len(rules)}):")
        for r in rules:
            print(f"  {r}")

    engine = SimplePrologEngine(facts, rules)
    entities = engine.all_entities()[:max_entities]
    derived = engine.derived[:max_derived]
    predicates = engine.all_predicates()

    print(f"\n{'─'*60}")
    print("DERIVED FACTS (forward chaining over rules):")
    if derived:
        for pred, args in derived:
            print(f"  {pred}({', '.join(args)}).")
    else:
        print("  (no rules to fire)")

    print(f"\n{'─'*60}")
    print(f"ALL ENTITIES in KB (potential diagram nodes): {', '.join(entities)}")
    print(f"ALL PREDICATES (potential edge labels):       {', '.join(predicates)}")

    print(f"\n{'─'*60}")
    print("SAMPLE QUERIES — what relationships does each entity have?")
    for entity in entities[:5]:
        rels = engine.query_subject(entity)
        if rels:
            print(f"\n  ?- X('{entity}', Y).")
            for pred, obj in rels:
                print(f"     → {pred}('{entity}', '{obj}').")

    summary = {
        "facts": facts,
        "rules": rules,
        "entities": engine.all_entities(),
        "predicates": predicates,
        "derived": [{"pred": p, "args": a} for p, a in derived],
    }

    has_swi = _try_pyswip(facts, rules)
    print(f"\n{'─'*60}")
    print(f"SWI-Prolog backend (pyswip): {'available ✓' if has_swi else 'not installed — using built-in engine'}")
    if not has_swi:
        print("  To enable full Prolog: pip install pyswip  (requires SWI-Prolog installed)")

    print("=" * 60)
    return summary


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prolog KB demo from scenario_plan.json")
    parser.add_argument("--scenario", required=True, help="Path to scenario_plan.json")
    parser.add_argument("--max-entities", type=int, default=8)
    parser.add_argument("--max-derived", type=int, default=10)
    args = parser.parse_args()

    if not Path(args.scenario).exists():
        print(f"Error: scenario file not found: {args.scenario}", file=sys.stderr)
        sys.exit(1)

    run_demo(
        scenario_path=args.scenario,
        max_entities=args.max_entities,
        max_derived=args.max_derived,
    )


if __name__ == "__main__":
    main()
