"""Retrieval metrics for the USC semantic search step.

All functions take normalized citation strings (e.g. "18 U.S.C. § 1030").
Use `normalize_citation` to canonicalize free-form citations before comparing.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Sequence


_CITATION_RE = re.compile(
    r"(?P<title>\d+[A-Za-z]?)\s*U\.?\s*S\.?\s*C\.?\s*(?:§|sec(?:tion)?|s\.)?\s*"
    r"(?P<section>\d+[A-Za-z\-]*)",
    re.IGNORECASE,
)


def normalize_citation(citation: str) -> str:
    """Return citations as 'TITLE U.S.C. § SECTION'. Empty string if unparseable."""
    if not citation:
        return ""
    m = _CITATION_RE.search(citation.replace("§", "§"))
    if not m:
        return citation.strip()
    return f"{m.group('title')} U.S.C. § {m.group('section')}"


def _norm_set(items: Iterable[str]) -> List[str]:
    return [normalize_citation(c) for c in items if c]


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = set(_norm_set(relevant))
    top = _norm_set(retrieved)[:k]
    if not top:
        return 0.0
    hits = sum(1 for c in top if c in rel)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(_norm_set(relevant))
    if not rel:
        return 0.0
    top = set(_norm_set(retrieved)[:k])
    return len(top & rel) / len(rel)


def hit_rate_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1.0 if at least one relevant doc appears in top-k, else 0.0."""
    rel = set(_norm_set(relevant))
    top = _norm_set(retrieved)[:k]
    return 1.0 if any(c in rel for c in top) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(_norm_set(relevant))
    for i, c in enumerate(_norm_set(retrieved), start=1):
        if c in rel:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(_norm_set(relevant))
    if not rel:
        return 0.0
    hits = 0
    score = 0.0
    for i, c in enumerate(_norm_set(retrieved), start=1):
        if c in rel:
            hits += 1
            score += hits / i
    return score / len(rel)


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance_grades: Dict[str, float],
    k: int,
) -> float:
    """nDCG with arbitrary graded relevance.

    `relevance_grades` maps normalized citation -> grade (e.g. primary=3, secondary=2, distractor=0).
    Unknown citations get grade 0.
    """
    if k <= 0:
        return 0.0
    grades = {normalize_citation(c): g for c, g in relevance_grades.items()}
    top = _norm_set(retrieved)[:k]
    dcg = sum((grades.get(c, 0.0)) / math.log2(i + 2) for i, c in enumerate(top))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


# ---------- aggregate over a dataset ----------

DEFAULT_GRADES = {"primary": 3.0, "secondary": 2.0, "distractor_avoid": 0.0}


def evaluate_run(
    runs: List[Dict],
    ks: Sequence[int] = (1, 3, 5, 10),
    grades: Dict[str, float] = None,
) -> Dict[str, float]:
    """Aggregate metrics over a list of {case_id, retrieved, expected_statutes} dicts.

    `expected_statutes` is the dict from ground_truth.json with keys primary/secondary/distractor_avoid.
    """
    grades = grades or DEFAULT_GRADES
    per_k = defaultdict(list)
    mrr_scores: List[float] = []
    map_scores: List[float] = []
    distractor_rates: List[float] = []

    for r in runs:
        retrieved = r["retrieved"]
        exp = r["expected_statutes"]
        relevant = list(exp.get("primary", [])) + list(exp.get("secondary", []))
        distractors = set(_norm_set(exp.get("distractor_avoid", [])))

        graded: Dict[str, float] = {}
        for c in exp.get("primary", []):
            graded[c] = grades["primary"]
        for c in exp.get("secondary", []):
            graded.setdefault(c, grades["secondary"])

        for k in ks:
            per_k[f"precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            per_k[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            per_k[f"hit_rate@{k}"].append(hit_rate_at_k(retrieved, relevant, k))
            per_k[f"ndcg@{k}"].append(ndcg_at_k(retrieved, graded, k))

        mrr_scores.append(reciprocal_rank(retrieved, relevant))
        map_scores.append(average_precision(retrieved, relevant))

        if distractors:
            top_norm = set(_norm_set(retrieved)[: max(ks)])
            distractor_rates.append(len(top_norm & distractors) / len(distractors))

    summary = {name: mean(vals) for name, vals in per_k.items()}
    summary["mrr"] = mean(mrr_scores) if mrr_scores else 0.0
    summary["map"] = mean(map_scores) if map_scores else 0.0
    if distractor_rates:
        summary["distractor_rate"] = mean(distractor_rates)
    summary["n_cases"] = len(runs)
    return summary
