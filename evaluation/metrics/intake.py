"""Metrics for the Case Intake agent's structured JSON output."""
from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REQUIRED_KEYS = {
    "case_type",
    "legal_domain",
    "primary_issue",
    "summary",
    "key_facts",
    "relevant_entities",
    "locations",
    "dates",
    "federal_hooks",
    "missing_info_questions",
    "search_queries",
}
ALLOWED_CASE_TYPES = {"criminal", "civil", "administrative", "unclear"}


def parse_loose_json(raw: str) -> Tuple[bool, Any]:
    if not isinstance(raw, str):
        return True, raw
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*```$", "", s).strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    try:
        return True, json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return True, json.loads(m.group(0))
            except Exception:
                return False, None
        return False, None


def schema_validity(output: Any) -> Dict[str, Any]:
    """Returns granular schema scoring, not just a boolean."""
    if not isinstance(output, dict):
        return {"valid_json_object": False, "missing_keys": list(REQUIRED_KEYS), "extra_keys": [], "score": 0.0}
    keys = set(output.keys())
    missing = REQUIRED_KEYS - keys
    extra = keys - REQUIRED_KEYS
    case_type_ok = output.get("case_type") in ALLOWED_CASE_TYPES
    list_keys = [
        "key_facts",
        "relevant_entities",
        "locations",
        "dates",
        "federal_hooks",
        "missing_info_questions",
        "search_queries",
    ]
    list_ok = all(isinstance(output.get(k, None), list) for k in list_keys)
    score = (
        (1 - len(missing) / len(REQUIRED_KEYS)) * 0.6
        + (1.0 if case_type_ok else 0.0) * 0.2
        + (1.0 if list_ok else 0.0) * 0.2
    )
    return {
        "valid_json_object": True,
        "missing_keys": sorted(missing),
        "extra_keys": sorted(extra),
        "case_type_in_enum": case_type_ok,
        "list_fields_typed": list_ok,
        "score": score,
    }


def classification_correct(predicted: str, expected: str) -> bool:
    return (predicted or "").strip().lower() == (expected or "").strip().lower()


def domain_match(predicted: str, expected_options: Sequence[str]) -> bool:
    """Match if predicted == option, or substring either direction, OR if a meaningful
    content token overlaps (e.g. 'Drug Enforcement' shares 'drug' with 'Drug Trafficking')."""
    p = (predicted or "").strip().lower()
    if not p:
        return False
    STOP = {"and", "the", "of", "for", "a", "in", "law", "crime", "crimes", "fraud"}
    p_toks = {t for t in re.split(r"[^a-z0-9]+", p) if t and t not in STOP and len(t) >= 4}
    for opt in expected_options:
        ol = opt.lower()
        if p == ol or ol in p or p in ol:
            return True
        opt_toks = {t for t in re.split(r"[^a-z0-9]+", ol) if t and t not in STOP and len(t) >= 4}
        if p_toks & opt_toks:
            return True
    return False


def _tokens(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t]


_STOPWORDS = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "by", "at", "from", "into"}


def _content_tokens(s: str) -> set:
    return {t for t in _tokens(s) if t not in _STOPWORDS and len(t) >= 3}


def _bag_overlap(a: str, b: str) -> float:
    """Overlap normalised by the SHORTER side. A short GT hook 'transmission to foreign agent'
    fully overlapping a longer prediction 'transmitted to foreign contact' scores high.
    Stopwords/short tokens removed. Light prefix stemming (5-char prefix) avoids inflection mismatch."""
    ta = _content_tokens(a)
    tb = _content_tokens(b)
    if not ta or not tb:
        return 0.0
    # prefix-stem to handle transmit/transmission/transmitted
    pa = {t[:5] for t in ta}
    pb = {t[:5] for t in tb}
    return len(pa & pb) / min(len(pa), len(pb))


def hooks_f1(predicted: Iterable[str], expected: Iterable[str], overlap_threshold: float = 0.5) -> Dict[str, float]:
    """Soft F1 over federal_hooks. A predicted hook 'matches' an expected one if their token
    Jaccard-like overlap >= threshold (avoids penalizing benign paraphrase)."""
    p = list(predicted or [])
    e = list(expected or [])
    if not p and not e:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not p:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not e:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    matched_e = set()
    tp = 0
    for ph in p:
        for i, eh in enumerate(e):
            if i in matched_e:
                continue
            if _bag_overlap(ph, eh) >= overlap_threshold:
                matched_e.add(i)
                tp += 1
                break
    precision = tp / len(p)
    recall = tp / len(e)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def search_query_keyword_coverage(queries: Iterable[str], keywords: Iterable[str]) -> float:
    """Fraction of expected keywords that appear (case-insensitive substring) across the generated queries."""
    kws = [k.lower() for k in keywords]
    if not kws:
        return 0.0
    blob = " ".join((q or "").lower() for q in queries)
    return sum(1 for k in kws if k in blob) / len(kws)


def evaluate_intake_run(runs: List[Dict]) -> Dict[str, float]:
    """Aggregate over [{case_id, predicted (dict|str), expected (gt case dict)}]."""
    schema_scores: List[float] = []
    json_validity: List[float] = []
    case_type_acc: List[float] = []
    domain_acc: List[float] = []
    hooks_f1_scores: List[float] = []
    hooks_p: List[float] = []
    hooks_r: List[float] = []
    kw_cov: List[float] = []

    for r in runs:
        ok, parsed = parse_loose_json(r["predicted"]) if isinstance(r["predicted"], str) else (True, r["predicted"])
        json_validity.append(1.0 if ok else 0.0)
        if not ok or not isinstance(parsed, dict):
            schema_scores.append(0.0)
            case_type_acc.append(0.0)
            domain_acc.append(0.0)
            hooks_f1_scores.append(0.0)
            hooks_p.append(0.0)
            hooks_r.append(0.0)
            kw_cov.append(0.0)
            continue
        s = schema_validity(parsed)
        schema_scores.append(s["score"])
        case_type_acc.append(1.0 if classification_correct(parsed.get("case_type", ""), r["expected"]["expected_case_type"]) else 0.0)
        domain_acc.append(1.0 if domain_match(parsed.get("legal_domain", ""), r["expected"]["expected_legal_domain"]) else 0.0)
        f1 = hooks_f1(parsed.get("federal_hooks", []), r["expected"]["expected_federal_hooks"])
        hooks_f1_scores.append(f1["f1"])
        hooks_p.append(f1["precision"])
        hooks_r.append(f1["recall"])
        kw_cov.append(
            search_query_keyword_coverage(parsed.get("search_queries", []), r["expected"]["expected_search_query_keywords"])
        )

    return {
        "n_cases": len(runs),
        "json_validity": mean(json_validity) if json_validity else 0.0,
        "schema_score": mean(schema_scores) if schema_scores else 0.0,
        "case_type_accuracy": mean(case_type_acc) if case_type_acc else 0.0,
        "legal_domain_accuracy": mean(domain_acc) if domain_acc else 0.0,
        "federal_hooks_f1": mean(hooks_f1_scores) if hooks_f1_scores else 0.0,
        "federal_hooks_precision": mean(hooks_p) if hooks_p else 0.0,
        "federal_hooks_recall": mean(hooks_r) if hooks_r else 0.0,
        "search_query_keyword_coverage": mean(kw_cov) if kw_cov else 0.0,
    }
