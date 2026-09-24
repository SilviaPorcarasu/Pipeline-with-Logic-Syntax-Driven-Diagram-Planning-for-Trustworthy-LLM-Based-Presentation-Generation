"""Diversity / similarity metrics for the educational-slides pipeline.

Implemented metrics:

  Lexical level:
    - distinct_n(texts, n)            — OpenAI evals-style ratio of unique
                                        n-grams over total tokens.
    - sacrebleu_score(hyp, refs)      — corpus-BLEU via the `sacrebleu` lib.

  Syntactic level:
    - dependency_ted(s1, s2)          — Tree Edit Distance between two
                                        spaCy dependency trees.
    - syntactic_similarity(s1, s2)    — TED normalized to (0, 1].

  Semantic level (cross-run diversity):
    - pairwise_semantic_cosine(decks) — mean cosine distance between every
                                        pair of runs' bullet embeddings.

  Correctness:
    - llm_judge_score(...)            — LLM-as-judge correctness evaluation
                                        for text concept and diagram concept.

The lexical metrics need only stdlib + `sacrebleu`. The syntactic metric
needs `spacy` + `en_core_web_sm`. The semantic/LLM metrics are lazy-loaded.
"""

from __future__ import annotations

from functools import lru_cache
import os
import re
from typing import Iterable



# Lexical: Distinct-N (OpenAI ChatGPT evals style)
#   https://github.com/openai/evals/blob/main/evals/metrics/diversity.py
#   https://aclanthology.org/N16-1014/

def distinct_n(texts: Iterable[str], n: int) -> float:
    """Distinct-N: unique n-grams / total tokens, across a corpus.

    Higher = more lexical diversity. Designed to be applied across a
    *collection* (e.g. all bullets from one run, or all bullets across
    runs); a single short sentence will trivially score near 1.0.
    """
    all_tokens: list[str] = []
    all_ngrams: set[tuple[str, ...]] = set()

    for text in texts:
        tokens = str(text).strip().split()
        all_tokens.extend(tokens)
        for i in range(len(tokens) - n + 1):
            all_ngrams.add(tuple(tokens[i:i + n]))

    if not all_tokens:
        return 0.0
    return len(all_ngrams) / len(all_tokens)



# Lexical: sacreBLEU
#   https://huggingface.co/docs/evaluate/package_reference/sacrebleu
#   https://aclanthology.org/W18-6319/

def sacrebleu_score(hypothesis: str, references: list[str]) -> float:
    """Single-sentence BLEU against a list of references via sacrebleu.

    Returns the sacreBLEU score (0-100). Lower = the hypothesis differs
    more from every reference (higher diversity vs. those references).
    """
    import sacrebleu  # noqa: WPS433 — lazy on purpose

    score = sacrebleu.corpus_bleu([hypothesis], [references])
    return float(score.score)


def sacrebleu_corpus(hypotheses: list[str], references_per_hyp: list[list[str]]) -> float:
    """Corpus-BLEU when you have N hypotheses and one reference set per hyp.

    `references_per_hyp[i]` is the list of references for `hypotheses[i]`.
    sacrebleu wants references transposed: ref_set[k] is the k-th reference
    aligned with every hypothesis. We pad short reference lists with "" so
    the transpose is rectangular.
    """
    import sacrebleu

    if not hypotheses:
        return 0.0
    max_refs = max((len(r) for r in references_per_hyp), default=0) or 1
    ref_sets: list[list[str]] = [[] for _ in range(max_refs)]
    for refs in references_per_hyp:
        padded = list(refs) + [""] * (max_refs - len(refs))
        for k, r in enumerate(padded):
            ref_sets[k].append(r)
    score = sacrebleu.corpus_bleu(hypotheses, ref_sets)
    return float(score.score)



# Syntactic: Dependency Tree Edit Distance (TED)
#   https://aclanthology.org/W19-4006/

class _DepNode:
    """Tiny tree node carrying just the dependency label."""

    __slots__ = ("label", "children")

    def __init__(self, label: str) -> None:
        self.label = label
        self.children: list[_DepNode] = []


@lru_cache(maxsize=1)
def _load_spacy():
    """Load `en_core_web_sm` once per process. Surfaces a clear error
    when the model is missing instead of spaCy's stack trace."""
    import spacy  # noqa: WPS433 — lazy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model `en_core_web_sm` is not installed. Run:\n"
            "    python -m spacy download en_core_web_sm"
        ) from exc


def _build_tree(token) -> _DepNode:
    node = _DepNode(token.dep_)
    for child in token.children:
        node.children.append(_build_tree(child))
    return node


def _ted(a: _DepNode | None, b: _DepNode | None) -> int:
    """Recursive tree edit distance over labeled trees.

    Faithful port of the algorithm specified by the advisor. Cost 0 when
    the two node labels match, 1 otherwise; children aligned via DP with
    insert / delete / replace operations.
    """
    if a is None and b is None:
        return 0
    if a is None:
        return 1 + sum(_ted(None, child) for child in b.children)
    if b is None:
        return 1 + sum(_ted(child, None) for child in a.children)

    cost = 0 if a.label == b.label else 1

    m, n = len(a.children), len(b.children)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + _ted(a.children[i - 1], None)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + _ted(None, b.children[j - 1])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + _ted(a.children[i - 1], None),
                dp[i][j - 1] + _ted(None, b.children[j - 1]),
                dp[i - 1][j - 1] + _ted(a.children[i - 1], b.children[j - 1]),
            )

    return cost + dp[m][n]


def _sentence_to_tree(doc) -> _DepNode | None:
    """Build the dependency tree for the first sentence of a spaCy Doc.

    Empty input -> None so `_ted(None, ...)` handles it cleanly.
    """
    sents = list(doc.sents)
    if not sents:
        return None
    return _build_tree(sents[0].root)


def _tree_size(node: _DepNode | None) -> int:
    """Return the number of nodes in a dependency tree."""
    if node is None:
        return 0
    return 1 + sum(_tree_size(child) for child in node.children)


def dependency_ted(s1: str, s2: str) -> int:
    """Tree edit distance between two sentences' dependency parses."""
    nlp = _load_spacy()
    return _ted(_sentence_to_tree(nlp(s1)), _sentence_to_tree(nlp(s2)))


def normalized_dependency_ted(s1: str, s2: str) -> float:
    """Length-normalized dependency TED in [0, 1].

    The raw tree edit distance is divided by the total number of nodes in the
    two dependency trees so values stay comparable across longer texts.
    Higher values still indicate greater structural diversity.
    """
    nlp = _load_spacy()
    t1 = _sentence_to_tree(nlp(s1))
    t2 = _sentence_to_tree(nlp(s2))
    denom = max(_tree_size(t1) + _tree_size(t2), 1)
    return float(_ted(t1, t2)) / float(denom)


def syntactic_similarity(s1: str, s2: str) -> tuple[float, int]:
    """Return (similarity, distance) where similarity = 1 / (1 + TED).

    similarity ∈ (0, 1]; 1.0 means identical syntactic structure.
    """
    distance = dependency_ted(s1, s2)
    return 1.0 / (1.0 + distance), distance



# Aggregations used by the eval driver

def deck_distinct_ns(bullets: list[str]) -> dict[str, float]:
    """Distinct-1, Distinct-2, Distinct-3 over one deck's bullets."""
    return {f"distinct_{n}": distinct_n(bullets, n) for n in (1, 2, 3)}


def pairwise_self_bleu(decks: list[list[str]]) -> float:
    """Self-BLEU across N runs: each deck-as-string vs. the others.

    Lower self-BLEU = more diversity across runs. Joins each deck's
    bullets into one document so we get one BLEU per (run, others) pair
    and average them.
    """
    if len(decks) < 2:
        return 0.0
    docs = [" ".join(b) for b in decks]
    scores = []
    for i, hyp in enumerate(docs):
        refs = [docs[j] for j in range(len(docs)) if j != i]
        scores.append(sacrebleu_score(hyp, refs))
    return sum(scores) / len(scores)


def pairwise_mean_ted(decks: list[list[str]]) -> float:
    """Mean dependency TED across paired bullets from N runs.

    Pairs slide-i bullet-j of run A with slide-i bullet-j of run B; if
    decks have different shapes, only paired indices are compared.
    """
    if len(decks) < 2:
        return 0.0
    distances: list[int] = []
    for i in range(len(decks)):
        for j in range(i + 1, len(decks)):
            for a, b in zip(decks[i], decks[j]):
                if a and b:
                    distances.append(dependency_ted(a, b))
    return sum(distances) / len(distances) if distances else 0.0



# Grounding: legacy embedding helpers + NLI / hallucination evaluation

class _TfidfEncoder:
    """TF-IDF fallback encoder used when sentence-transformers/torch are unavailable.

    Fits a fresh vectorizer on each batch (always called with all texts at once
    in this codebase), so the feature space is consistent within each metric call.
    Cosine similarities are lexical rather than semantic but still informative for
    relative comparisons across runs.
    """

    def encode(self, texts, convert_to_numpy: bool = True):
        import numpy as np  # noqa: WPS433
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: WPS433

        texts = list(texts)
        if len(texts) < 2:
            texts = texts + [""]
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), sublinear_tf=True)
        mat = vec.fit_transform(texts).toarray().astype(np.float32)
        return mat


@lru_cache(maxsize=4)
def _load_sentence_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load a SentenceTransformer once per process.

    Falls back to a TF-IDF encoder if sentence_transformers cannot be imported
    (e.g. torch/transformers version mismatch on the pod). The fallback produces
    lexical rather than semantic similarity — fix the environment with:
        pip install "transformers==4.46.3" "sentence-transformers==3.3.1"
    """
    import logging  # noqa: WPS433
    import sys  # noqa: WPS433
    import warnings  # noqa: WPS433

    try:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            return SentenceTransformer(model_name)
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            f"[metrics] sentence_transformers unavailable ({exc});\n"
            "  falling back to TF-IDF cosine similarity (lexical, not semantic).\n"
            "  Fix: pip install 'transformers==4.46.3' 'sentence-transformers==3.3.1'",
            file=sys.stderr,
        )
        return _TfidfEncoder()


def _l2_normalize(vec):
    import numpy as np  # noqa: WPS433

    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _default_nli_model_name() -> str:
    """Return the default NLI model used for chunk-grounding checks."""
    return str(
        os.environ.get(
            "NLI_MODEL_ID",
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        )
    ).strip()


def _default_nli_threshold() -> float:
    """Return the entailment threshold used by hallucination detection."""
    raw = str(os.environ.get("NLI_ENTAILMENT_THRESHOLD", "0.5")).strip()
    try:
        return float(raw)
    except Exception:
        return 0.5


def _resolve_nli_label_indices(config) -> dict[str, int]:
    """Map model-specific NLI labels to entailment / neutral / contradiction."""
    id2label = getattr(config, "id2label", {}) or {}
    out: dict[str, int] = {}
    for idx, label in id2label.items():
        name = str(label).strip().lower()
        if "entail" in name:
            out["entailment"] = int(idx)
        elif "contrad" in name or "refut" in name:
            out["contradiction"] = int(idx)
        elif "neutral" in name:
            out["neutral"] = int(idx)
    num_labels = int(getattr(config, "num_labels", 0) or 0)
    if len(out) < 3 and num_labels == 3:
        out.setdefault("entailment", 0)
        out.setdefault("neutral", 1)
        out.setdefault("contradiction", 2)
    missing = {"entailment", "neutral", "contradiction"} - set(out)
    if missing:
        raise RuntimeError(f"Could not resolve NLI labels for model config: missing {sorted(missing)}")
    return out


@lru_cache(maxsize=2)
def _load_nli_bundle(model_name: str | None = None):
    """Load and cache the NLI tokenizer/model bundle once per process."""
    import torch  # noqa: WPS433
    from transformers import (  # noqa: WPS433
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    chosen = str(model_name or _default_nli_model_name()).strip()
    tokenizer = AutoTokenizer.from_pretrained(chosen)
    model = AutoModelForSequenceClassification.from_pretrained(chosen)
    model.eval()
    return tokenizer, model, _resolve_nli_label_indices(model.config), torch


def nli_grounding_scores(
    hypothesis: str,
    premises: list[str],
    *,
    model_name: str | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """Score one generated text against a list of retrieved chunks with NLI.

    The returned dict contains:
      - best_entailment: strongest chunk→text support
      - mean_entailment: average entailment over slide chunks
      - best_contradiction: strongest contradiction signal
      - hallucination: 1 when unsupported / contradicted, else 0
      - factual_precision: 1 - hallucination
    """
    hyp = str(hypothesis or "").strip()
    slide_chunks = [str(p or "").strip() for p in premises if str(p or "").strip()]
    if not hyp or not slide_chunks:
        return {
            "best_entailment": 0.0,
            "mean_entailment": 0.0,
            "best_contradiction": 0.0,
            "hallucination": 1.0 if hyp else 0.0,
            "factual_precision": 0.0 if hyp else 1.0,
        }

    tokenizer, model, label_idx, torch = _load_nli_bundle(model_name)
    cutoff = _default_nli_threshold() if threshold is None else float(threshold)
    inputs = tokenizer(
        slide_chunks,
        [hyp] * len(slide_chunks),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu()

    entail_vals = probs[:, label_idx["entailment"]].tolist()
    contra_vals = probs[:, label_idx["contradiction"]].tolist()
    best_entail = max(float(v) for v in entail_vals)
    mean_entail = sum(float(v) for v in entail_vals) / len(entail_vals)
    best_contra = max(float(v) for v in contra_vals)
    hallucination = 1.0 if (best_contra > best_entail or best_entail < cutoff) else 0.0
    return {
        "best_entailment": best_entail,
        "mean_entailment": mean_entail,
        "best_contradiction": best_contra,
        "hallucination": hallucination,
        "factual_precision": 1.0 - hallucination,
    }


def _empty_grounding_summary(per_slide: list[dict] | None = None) -> dict[str, float]:
    """Return the zero-valued grounding summary shape used across the repo."""
    return {
        "mean_entailment": 0.0,
        "mean_contradiction": 0.0,
        "hallucination_rate": 0.0,
        "factual_precision": 0.0,
        "n_texts_scored": 0.0,
        "per_slide": per_slide or [],
    }


def _aggregate_texts_vs_chunks_nli(
    texts_per_slide: list[list[str]],
    chunks_per_slide: list[list[str]],
    *,
    model_name: str | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """Aggregate slide-aligned generated texts vs retrieved chunks with NLI."""
    rows: list[dict[str, float]] = []
    per_slide: list[dict] = []

    total_slides = max(len(texts_per_slide), len(chunks_per_slide))
    for idx in range(total_slides):
        slide_texts = texts_per_slide[idx] if idx < len(texts_per_slide) else []
        slide_chunks = chunks_per_slide[idx] if idx < len(chunks_per_slide) else []
        slide_items: list[dict] = []
        for text in slide_texts:
            clean = str(text or "").strip()
            if not clean or not slide_chunks:
                continue
            score = nli_grounding_scores(
                clean,
                slide_chunks,
                model_name=model_name,
                threshold=threshold,
            )
            slide_items.append({"text": clean, **score})
            rows.append(score)
        if slide_items:
            per_slide.append(
                {
                    "slide_number": idx + 1,
                    "n_chunks": float(len([c for c in slide_chunks if str(c or "").strip()])),
                    "n_texts_scored": float(len(slide_items)),
                    "mean_entailment": sum(item["best_entailment"] for item in slide_items) / len(slide_items),
                    "mean_contradiction": sum(item["best_contradiction"] for item in slide_items) / len(slide_items),
                    "hallucination_rate": sum(item["hallucination"] for item in slide_items) / len(slide_items),
                    "factual_precision": sum(item["factual_precision"] for item in slide_items) / len(slide_items),
                    "items": slide_items,
                }
            )
        else:
            per_slide.append(
                {
                    "slide_number": idx + 1,
                    "n_chunks": float(len([c for c in slide_chunks if str(c or "").strip()])),
                    "n_texts_scored": 0.0,
                    "mean_entailment": 0.0,
                    "mean_contradiction": 0.0,
                    "hallucination_rate": 0.0,
                    "factual_precision": 0.0,
                    "items": [],
                }
            )

    if not rows:
        return _empty_grounding_summary(per_slide)

    return {
        "mean_entailment": sum(r["best_entailment"] for r in rows) / len(rows),
        "mean_contradiction": sum(r["best_contradiction"] for r in rows) / len(rows),
        "hallucination_rate": sum(r["hallucination"] for r in rows) / len(rows),
        "factual_precision": sum(r["factual_precision"] for r in rows) / len(rows),
        "n_texts_scored": float(len(rows)),
        "per_slide": per_slide,
    }


def grounding_scores(
    bullet: str,
    chunks: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, float]:
    """Legacy embedding-based grounding helper retained for compatibility.

    Returns a dict with `max_cosine`, `mean_cosine`, `min_euclidean`,
    `mean_euclidean`. Higher cosine and lower euclidean indicate stronger
    lexical/embedding alignment. New evaluation code uses NLI instead.

    Both metrics are computed on L2-normalized embeddings so euclidean
    stays in the interpretable [0, 2] range (independent of the encoder's
    raw embedding magnitude). The relationship is
    `||a-b||_2 = sqrt(2 - 2*cos(a, b))` — both metrics encode the same
    geometry, but advisors typically want to see both side-by-side.
    """
    import numpy as np  # noqa: WPS433

    if not bullet or not chunks:
        return {"max_cosine": 0.0, "mean_cosine": 0.0,
                "min_euclidean": 0.0, "mean_euclidean": 0.0}
    encoder = _load_sentence_encoder(model_name)
    embs = encoder.encode([bullet, *chunks], convert_to_numpy=True)
    bullet_v = _l2_normalize(embs[0])
    chunk_vs = [_l2_normalize(v) for v in embs[1:]]
    cos = [float(np.dot(bullet_v, c)) for c in chunk_vs]
    # Euclidean on normalized vectors — range [0, 2], directly comparable
    # across encoders.
    euc = [float(np.linalg.norm(bullet_v - c)) for c in chunk_vs]
    return {
        "max_cosine": max(cos),
        "mean_cosine": sum(cos) / len(cos),
        "min_euclidean": min(euc),
        "mean_euclidean": sum(euc) / len(euc),
    }


def deck_grounding(
    bullets_per_slide: list[list[str]],
    chunks_per_slide: list[list[str]],
    *,
    model_name: str | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """Aggregate slide-by-slide bullet grounding against retrieved chunks.

    The score is computed with NLI entailment over each bullet and the set of
    chunks retrieved for the same slide. Hallucination is flagged whenever the
    best entailment score is below the threshold or a contradiction score
    exceeds the best entailment score.
    """
    out = _aggregate_texts_vs_chunks_nli(
        bullets_per_slide,
        chunks_per_slide,
        model_name=model_name,
        threshold=threshold,
    )
    out["n_bullets_scored"] = out.get("n_texts_scored", 0.0)
    return out


def chunk_overlap_bleu(
    generated_per_slide: list[str],
    chunks_per_slide: list[list[str]],
) -> dict[str, float]:
    """sacreBLEU between generated text and the RAG chunks that backed it.

    Advisor's plan, page 3 (3): compare each generated text stream to the
    *original* chunks. This measures lexical fidelity to the source —
    LOW BLEU means the model paraphrased (good for teaching style), HIGH
    BLEU means it copied chunk text verbatim. Pairs with the embedding
    grounding metric: cosine ≈ semantic alignment, BLEU ≈ literal overlap.

    `generated_per_slide[i]` is the model's text for slide i+1 (bullets
    joined with " / ", or the speech_plan, etc.). `chunks_per_slide[i]`
    is that slide's RAG chunks. Returns mean and max BLEU over the slides
    plus how many slides actually had both inputs.
    """
    import sacrebleu  # noqa: WPS433 — lazy

    scores: list[float] = []
    for gen, refs in zip(generated_per_slide, chunks_per_slide):
        gen = (gen or "").strip()
        refs = [r for r in (refs or []) if r and r.strip()]
        if not gen or not refs:
            continue
        score = sacrebleu.corpus_bleu([gen], [refs])
        scores.append(float(score.score))
    if not scores:
        return {"mean_bleu_vs_chunks": 0.0, "max_bleu_vs_chunks": 0.0,
                "n_slides_scored": 0}
    return {
        "mean_bleu_vs_chunks": sum(scores) / len(scores),
        "max_bleu_vs_chunks": max(scores),
        "n_slides_scored": float(len(scores)),
    }


def pairwise_semantic_cosine(
    decks: list[list[str]],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, float]:
    """Mean cosine *distance* between every pair of runs (cross-run diversity).

    Complements `pairwise_self_bleu` (lexical) with a semantic measure.
    Each deck's bullets are joined into one document, encoded with the
    sentence encoder, and all pairwise cosine similarities are computed.

    Returns:
        mean_cosine_sim   — mean similarity across pairs  (higher = more similar)
        mean_cosine_dist  — 1 - mean_cosine_sim           (higher = more diverse)
        n_pairs           — number of run pairs compared
    """
    import numpy as np  # noqa: WPS433

    if len(decks) < 2:
        return {"mean_cosine_sim": 0.0, "mean_cosine_dist": 0.0, "n_pairs": 0}

    encoder = _load_sentence_encoder(model_name)
    docs = [" ".join(b) for b in decks]
    embs = [_l2_normalize(encoder.encode([d], convert_to_numpy=True)[0]) for d in docs]

    sims: list[float] = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sims.append(float(np.dot(embs[i], embs[j])))

    mean_sim = sum(sims) / len(sims)
    return {
        "mean_cosine_sim": mean_sim,
        "mean_cosine_dist": 1.0 - mean_sim,
        "n_pairs": float(len(sims)),
    }



# Correctness: LLM-as-judge

_LLM_JUDGE_SYSTEM = (
    "You are an expert evaluator for educational presentations. "
    "You assess how well a generated slide matches its intended learning goal. "
    "Return ONLY valid JSON — no markdown, no prose outside JSON."
)

_LLM_JUDGE_PROMPT = """\
Evaluate the following slide content for correctness and concept coverage.

Slide title: {title}
Learning goal: {goal}
RAG context (source facts): {rag_context}

Generated bullets:
{bullets}

Diagram description:
{diagram_description}

Score each dimension from 0.0 to 1.0:
- concept_coverage: do the bullets cover the key concepts in the goal and RAG context?
- factual_accuracy: are the statements factually correct and free of hallucinations?
- absence_of_repetition: are the bullets diverse (penalise near-identical bullets)?
- diagram_relevance: does the diagram description match the slide topic (not a generic flowchart)?

Return JSON:
{{
  "concept_coverage": <float>,
  "factual_accuracy": <float>,
  "absence_of_repetition": <float>,
  "diagram_relevance": <float>,
  "overall": <average of the four>,
  "comments": "<one sentence>"
}}"""

_DIAGRAM_SET_JUDGE_SYSTEM = (
    "You are an expert evaluator for educational presentation diagrams. "
    "You assess whether a set of generated diagram descriptions is conclusive, "
    "coherent, and pedagogically useful for a shared presentation scenario. "
    "Return ONLY valid JSON — no markdown, no prose outside JSON."
)

_DIAGRAM_SET_JUDGE_PROMPT = """\
Evaluate the following set of diagram descriptions generated for a single presentation scenario.

Shared slide sequence and learning goals:
{slides_context}

Generated diagram descriptions:
{diagram_set}

Score each dimension from 0.0 to 1.0:
- diagram_conclusiveness: do the diagrams clearly communicate the intended concepts and relations?
- diagram_coherence: do the diagrams form a coherent visual set across the presentation?
- diagram_specificity: are the diagrams specific to the slide content rather than generic flowcharts?
- diagram_groundedness: do the diagrams remain aligned with the stated slide goals?

Return JSON:
{{
  "diagram_conclusiveness": <float>,
  "diagram_coherence": <float>,
  "diagram_specificity": <float>,
  "diagram_groundedness": <float>,
  "overall": <average of the four>,
  "comments": "<one sentence>"
}}"""


def llm_judge_score(
    *,
    title: str,
    goal: str,
    bullets: list[str],
    diagram_description: str,
    rag_context: str,
    llm_service,
) -> dict[str, float]:
    """LLM-as-judge correctness score for one slide.

    Uses the pipeline's existing LLMService so no extra API key is needed.
    Returns a dict with keys: concept_coverage, factual_accuracy,
    absence_of_repetition, diagram_relevance, overall, comments.
    Falls back to zeros on parse error so it never breaks the eval run.
    """
    bullets_text = "\n".join(f"- {b}" for b in bullets if b)
    prompt = _LLM_JUDGE_PROMPT.format(
        title=title or "(no title)",
        goal=goal or "(no goal)",
        rag_context=(rag_context or "")[:800],
        bullets=bullets_text or "(none)",
        diagram_description=(diagram_description or "")[:400],
    )
    try:
        data = llm_service.generate_json(
            system_prompt=_LLM_JUDGE_SYSTEM,
            user_prompt=prompt,
            max_new_tokens=400,
        )
        keys = ("concept_coverage", "factual_accuracy",
                "absence_of_repetition", "diagram_relevance")
        scores = {k: float(data.get(k, 0.0)) for k in keys}
        # Recompute overall as mean in case LLM rounded differently
        scores["overall"] = sum(scores[k] for k in keys) / len(keys)
        scores["comments"] = str(data.get("comments", ""))
        return scores
    except Exception:
        return {
            "concept_coverage": 0.0, "factual_accuracy": 0.0,
            "absence_of_repetition": 0.0, "diagram_relevance": 0.0,
            "overall": 0.0, "comments": "evaluation_failed",
        }


def deck_llm_judge(
    slides: list[dict],
    llm_service,
) -> dict[str, float]:
    """Aggregate llm_judge_score over all slides in a deck.

    `slides` is a list of dicts with keys: title, goal, bullets (list[str]),
    diagram_description, rag_context — matching the scenario_manifest format.
    Returns mean scores over all slides plus n_slides_scored and per-slide rows.
    """
    keys = ("concept_coverage", "factual_accuracy",
            "absence_of_repetition", "diagram_relevance", "overall")
    rows: list[dict] = []
    for slide in slides:
        row = llm_judge_score(
            title=slide.get("title", ""),
            goal=slide.get("goal", ""),
            bullets=slide.get("bullets", []),
            diagram_description=slide.get("diagram_description", ""),
            rag_context=slide.get("rag_context", ""),
            llm_service=llm_service,
        )
        row["slide_number"] = slide.get("slide_number")
        row["title"] = slide.get("title", "")
        rows.append(row)
    if not rows:
        return {k: 0.0 for k in keys} | {"n_slides_scored": 0, "per_slide": []}
    out = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    out["n_slides_scored"] = float(len(rows))
    out["per_slide"] = rows
    return out


def diagram_set_llm_judge(
    slides: list[dict],
    llm_service,
) -> dict[str, float]:
    """LLM-as-judge over the concatenated diagram descriptions of one deck.

    Uses the shared slide titles/goals as context and evaluates only the
    diagram descriptions, not bullets or rendered images.
    """
    keys = (
        "diagram_conclusiveness",
        "diagram_coherence",
        "diagram_specificity",
        "diagram_groundedness",
    )
    rows = []
    for idx, slide in enumerate(slides, start=1):
        title = str(slide.get("title", "") or "").strip()
        goal = str(slide.get("goal", "") or "").strip()
        desc = str(slide.get("diagram_description", "") or "").strip()
        if not desc:
            continue
        rows.append(
            {
                "idx": idx,
                "title": title or f"Slide {idx}",
                "goal": goal or "(no goal)",
                "diagram_description": desc,
            }
        )
    if not rows:
        return {k: 0.0 for k in keys} | {"overall": 0.0, "comments": "no_diagrams", "n_diagrams_scored": 0.0}

    slides_context = "\n".join(
        f"{row['idx']}. {row['title']} — goal: {row['goal']}"
        for row in rows
    )
    diagram_set = "\n\n".join(
        f"[Slide {row['idx']}] {row['diagram_description']}"
        for row in rows
    )
    prompt = _DIAGRAM_SET_JUDGE_PROMPT.format(
        slides_context=slides_context[:3000],
        diagram_set=diagram_set[:7000],
    )
    try:
        data = llm_service.generate_json(
            system_prompt=_DIAGRAM_SET_JUDGE_SYSTEM,
            user_prompt=prompt,
            max_new_tokens=400,
        )
        scores = {k: float(data.get(k, 0.0)) for k in keys}
        scores["overall"] = sum(scores[k] for k in keys) / len(keys)
        scores["comments"] = str(data.get("comments", ""))
        scores["n_diagrams_scored"] = float(len(rows))
        return scores
    except Exception:
        return {
            "diagram_conclusiveness": 0.0,
            "diagram_coherence": 0.0,
            "diagram_specificity": 0.0,
            "diagram_groundedness": 0.0,
            "overall": 0.0,
            "comments": "evaluation_failed",
            "n_diagrams_scored": float(len(rows)),
        }



# Grounding: bullets / diagrams ↔ slide description  (thesis §3 formulas)


def _mean_aligned_cosine(
    texts_a: list[str],
    texts_b: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """(1/N) Σ_i cos(e(a_i), e(b_i)) over aligned, non-empty pairs."""
    import numpy as np

    pairs = [(a, b) for a, b in zip(texts_a, texts_b)
             if (a or "").strip() and (b or "").strip()]
    if not pairs:
        return 0.0
    all_texts = [p[0] for p in pairs] + [p[1] for p in pairs]
    embs = _load_sentence_encoder(model_name).encode(all_texts, convert_to_numpy=True)
    n = len(pairs)
    return sum(
        float(np.dot(_l2_normalize(embs[i]), _l2_normalize(embs[n + i])))
        for i in range(n)
    ) / n


def grounding_bullet_vs_slide_desc(
    bullets_per_slide: list[list[str]],
    slide_descs: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """G_{s-b} = (1/N_b) Σ_{i,j} cos(e(b_{ij}), e(s_i)).

    Mean cosine similarity of every bullet with its own slide's description.
    """
    import numpy as np

    bullets: list[str] = []
    paired: list[str] = []
    for slide_bs, s_i in zip(bullets_per_slide, slide_descs):
        if not (s_i or "").strip():
            continue
        for b in slide_bs:
            if (b or "").strip():
                bullets.append(b)
                paired.append(s_i)
    if not bullets:
        return 0.0
    unique_descs = list(dict.fromkeys(paired))
    desc_idx = {d: i for i, d in enumerate(unique_descs)}
    all_texts = bullets + unique_descs
    embs = _load_sentence_encoder(model_name).encode(all_texts, convert_to_numpy=True)
    nb = len(bullets)
    b_norms = [_l2_normalize(embs[k]) for k in range(nb)]
    d_norms = [_l2_normalize(embs[nb + k]) for k in range(len(unique_descs))]
    return sum(
        float(np.dot(b_norms[k], d_norms[desc_idx[paired[k]]]))
        for k in range(nb)
    ) / nb


def grounding_concat_bullets_vs_slide_desc(
    bullets_per_slide: list[list[str]],
    slide_descs: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """G_{s-b_all} = (1/N_S) Σ_i cos(e(B_i), e(s_i)), B_i = concat of all bullets on slide i."""
    B_texts = [" ".join(b for b in bs if (b or "").strip()) for bs in bullets_per_slide]
    return _mean_aligned_cosine(B_texts, slide_descs, model_name=model_name)


def grounding_diagram_vs_slide_desc(
    diagram_descs: list[str],
    slide_descs: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """G_{s-d} = (1/N_S) Σ_i cos(e(d_i), e(s_i))."""
    return _mean_aligned_cosine(diagram_descs, slide_descs, model_name=model_name)


def grounding_prologue_vs_slide_desc(
    prologue_descs: list[str],
    slide_descs: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """G_{s-d_Prologue} = (1/N_S) Σ_i cos(e(d^Prologue_i), e(s_i))."""
    return _mean_aligned_cosine(prologue_descs, slide_descs, model_name=model_name)


def cosine_prologue_vs_diagram(
    prologue_descs: list[str],
    diagram_descs: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """Mean cos(e(d^Prologue_i), e(d_i)) — alignment between Prologue rep and diagram description."""
    return _mean_aligned_cosine(prologue_descs, diagram_descs, model_name=model_name)



# Within-deck diversity (bullets vs bullets, diagrams vs diagrams)


def within_deck_diversity(
    texts: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    with_ted: bool = True,
    with_bleu: bool = True,
) -> dict[str, float]:
    """All-pairs diversity metrics for a flat list of texts from one presentation.

    Pairwise cosine *distance* (1 − sim): higher → more semantically diverse.
    Self-BLEU: lower → more lexically diverse.
    Mean dep-TED: higher → more syntactically diverse.

    Returns mean, max, std for each; plus n_pairs.
    """
    import numpy as np

    texts = [t for t in texts if (t or "").strip()]
    n = len(texts)
    zero = {
        "mean_cosine_dist": 0.0, "max_cosine_dist": 0.0, "std_cosine_dist": 0.0,
        "mean_self_bleu": 0.0,
        "mean_dep_ted": 0.0, "max_dep_ted": 0.0,
        "n_pairs": 0,
    }
    if n < 2:
        return zero

    # Cosine distances — batch encode once
    embs = _load_sentence_encoder(model_name).encode(texts, convert_to_numpy=True)
    norms = [_l2_normalize(embs[i]) for i in range(n)]
    cosine_dists = [
        1.0 - float(np.dot(norms[i], norms[j]))
        for i in range(n) for j in range(i + 1, n)
    ]

    out: dict[str, float] = {
        "mean_cosine_dist": float(np.mean(cosine_dists)),
        "max_cosine_dist": float(np.max(cosine_dists)),
        "std_cosine_dist": float(np.std(cosine_dists)),
    }

    if with_bleu:
        bleu_scores = [
            sacrebleu_score(texts[i], [texts[j] for j in range(n) if j != i])
            for i in range(n)
        ]
        out["mean_self_bleu"] = float(np.mean(bleu_scores))
    else:
        out["mean_self_bleu"] = 0.0

    if with_ted:
        ted_vals = [
            dependency_ted(texts[i], texts[j])
            for i in range(n) for j in range(i + 1, n)
        ]
        out["mean_dep_ted"] = float(np.mean(ted_vals))
        out["max_dep_ted"] = float(np.max(ted_vals))
    else:
        out["mean_dep_ted"] = 0.0
        out["max_dep_ted"] = 0.0

    out["n_pairs"] = len(cosine_dists)
    return out


def _humanize_identifier(identifier: str) -> str:
    """Convert an internal node id into a readable fallback label."""
    raw = str(identifier or "").strip().split(".")[-1]
    raw = raw.replace("_", " ")
    raw = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw)
    return re.sub(r"\s+", " ", raw).strip() or "Concept"


def _extract_node_labels(diagram_text: str) -> dict[str, str]:
    """Extract node-id -> label mappings from textual diagram serializations."""
    text = str(diagram_text or "")
    labels: dict[str, str] = {}

    nodes_match = re.search(r"nodes:\s*(.*?)(?:\|\s*arrows:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if nodes_match:
        nodes_blob = nodes_match.group(1)
        for alias, spec in re.findall(r"([A-Za-z_][\w.\-]*)\s*\(([^()]+)\)", nodes_blob):
            parts = [p.strip() for p in spec.split(",") if p.strip()]
            if parts:
                labels[str(alias).strip()] = parts[-1]

    for alias, label in re.findall(r'^\s*([A-Za-z_][\w.\-]*)\s*:\s*"([^"]+)"', text, flags=re.MULTILINE):
        labels.setdefault(str(alias).strip(), str(label).strip())

    return labels


def _edge_to_sentence(src: str, dst: str, relation: str, labels: dict[str, str]) -> str:
    """Convert one diagram edge into a canonical relation sentence."""
    src_label = labels.get(src, _humanize_identifier(src))
    dst_label = labels.get(dst, _humanize_identifier(dst))
    rel = re.sub(r"\s+", " ", str(relation or "").strip()).strip(" .,:;")
    if rel:
        if rel and rel[0].isupper():
            rel = rel[0].lower() + rel[1:]
        return f"{src_label} {rel} {dst_label}."
    return f"{src_label} leads to {dst_label}."


def diagram_relation_sentences(diagram_text: str) -> list[str]:
    """Extract canonical relation sentences from `arrows:` or D2 `->` edges."""
    text = str(diagram_text or "").strip()
    if not text:
        return []

    labels = _extract_node_labels(text)
    arrows_match = re.search(r"arrows:\s*(.*?)(?:\|\s*[A-Za-z_ ]+:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    search_space = arrows_match.group(1) if arrows_match else text
    sentences: list[str] = []
    for src, dst, rel in re.findall(
        r"([A-Za-z_][\w.\-]*)\s*->\s*([A-Za-z_][\w.\-]*)(?:\s*:\s*([^,\n|]+))?",
        search_space,
    ):
        sentence = _edge_to_sentence(src, dst, rel, labels)
        if sentence:
            sentences.append(sentence)
    return list(dict.fromkeys(sentences))


def diagram_grounding(
    diagram_descriptions_per_slide: list[str],
    chunks_per_slide: list[list[str]],
    *,
    model_name: str | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """Ground diagram relations against retrieved chunks using NLI.

    Each diagram is converted into a set of relation sentences extracted from
    its directed edges (e.g. ``A leads to B.''). These sentences are then
    scored slide-by-slide against the retrieved chunks of the same slide.
    """
    relations_per_slide: list[list[str]] = []
    for desc in diagram_descriptions_per_slide:
        desc_text = str(desc or "").strip()
        relations = diagram_relation_sentences(desc_text)
        relations_per_slide.append(relations or ([desc_text] if desc_text else []))

    out = _aggregate_texts_vs_chunks_nli(
        relations_per_slide,
        chunks_per_slide,
        model_name=model_name,
        threshold=threshold,
    )
    out["n_diagrams_scored"] = out.get("n_texts_scored", 0.0)
    out["n_relations_scored"] = out.get("n_texts_scored", 0.0)
    for slide in out.get("per_slide", []):
        slide["n_relations_scored"] = slide.get("n_texts_scored", 0.0)
    return out
