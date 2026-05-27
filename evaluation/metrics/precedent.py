"""Metrics for the legal precedent search agent.

We can't verify holdings without a labeled corpus, so we evaluate:
  - source-trust precision (URL whitelist)
  - opinion vs non-opinion detection (regex on title/snippet/url)
  - dedup correctness (no near-duplicate case names)
  - court-tier breakdown (SCOTUS / Circuit / District / unknown)
  - schema validity
"""
from __future__ import annotations

import re
from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlparse

from .intake import parse_loose_json


TRUSTED_DOMAINS = {
    "law.cornell.edu",
    "supreme.justia.com",
    "law.justia.com",
    "cases.justia.com",
    "courtlistener.com",
    "www.courtlistener.com",
    "supremecourt.gov",
    "www.supremecourt.gov",
    "uscourts.gov",
    "casetext.com",
    "leagle.com",
    "openjurist.org",
    "scholar.google.com",
}

NON_OPINION_PATTERNS = re.compile(
    r"\b(brief|amicus|petition for cert|cert petition|docket|press release|news|law review)\b",
    re.IGNORECASE,
)

OPINION_SIGNAL_RE = re.compile(
    r"\bv\.\b|\bU\.?\s*S\.?\b|\bS\.?\s*Ct\.?\b|\bF\.?\s*3?d\b|\bF\.?\s*Supp\.?\b",
    re.IGNORECASE,
)

CIRCUIT_RE = re.compile(r"\b(\d+(?:st|nd|rd|th)\s+Cir(?:cuit)?\.?|D\.C\.\s*Cir|Federal\s+Circuit)\b", re.IGNORECASE)
SCOTUS_RE = re.compile(r"\b(Supreme Court|U\.?\s*S\.?\s*Supreme|SCOTUS|U\.?S\.?\s+\d+|\d+\s+U\.?S\.?\b)\b", re.IGNORECASE)
DISTRICT_RE = re.compile(r"\bD\.\s*[A-Z][a-z]+\.|\bF\.?\s*Supp\.?\b", re.IGNORECASE)

REQUIRED_PRECEDENT_KEYS = {"name", "court_year", "citation", "holding", "relevance", "url"}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


def is_trusted(url: str) -> bool:
    d = _domain(url)
    if not d:
        return False
    return d in TRUSTED_DOMAINS or any(d.endswith("." + t) for t in TRUSTED_DOMAINS)


def looks_like_opinion(name: str, citation: str, url: str) -> bool:
    blob = " ".join([name or "", citation or "", url or ""])
    if NON_OPINION_PATTERNS.search(blob):
        return False
    return bool(OPINION_SIGNAL_RE.search(blob))


def court_tier(name: str, citation: str, url: str) -> str:
    blob = " ".join([name or "", citation or "", url or ""])
    if SCOTUS_RE.search(blob) or "supreme" in (url or "").lower():
        return "scotus"
    if CIRCUIT_RE.search(blob):
        return "circuit"
    if DISTRICT_RE.search(blob):
        return "district"
    return "unknown"


def _norm_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def dedup_correctness(precedents: List[Dict]) -> Dict[str, Any]:
    names = [_norm_name(p.get("name", "")) for p in precedents]
    seen = Counter(names)
    dupes = sum(1 for c in seen.values() if c > 1)
    return {"n_precedents": len(precedents), "duplicate_groups": dupes, "is_unique": dupes == 0}


def precedent_schema_validity(precedents: List[Dict]) -> Dict[str, float]:
    if not precedents:
        return {"valid_share": 1.0, "n": 0}  # empty list is allowed per task spec
    ok = 0
    for p in precedents:
        if isinstance(p, dict) and REQUIRED_PRECEDENT_KEYS.issubset(p.keys()):
            ok += 1
    return {"valid_share": ok / len(precedents), "n": len(precedents)}


def no_guessing_compliance(precedents: List[Dict]) -> Dict[str, float]:
    """If court_year or citation is non-empty, it should be supported by some textual signal."""
    if not precedents:
        return {"score": 1.0, "n": 0}
    ok = 0
    for p in precedents:
        cy = (p.get("court_year") or "").strip()
        ct = (p.get("citation") or "").strip()
        if not cy and not ct:
            ok += 1
            continue
        blob = " ".join([p.get("name", ""), p.get("url", ""), ct])
        if (cy and re.search(re.escape(cy[-4:]), blob)) or (ct and OPINION_SIGNAL_RE.search(blob + " " + ct)):
            ok += 1
    return {"score": ok / len(precedents), "n": len(precedents)}


def evaluate_precedent_run(runs: List[Dict]) -> Dict[str, float]:
    """runs: [{case_id, precedent_output (dict|str)}]"""
    trust_p, opinion_p, schema_v, dedup_ok, noguess = [], [], [], [], []
    tier_counts = Counter()
    n_with_any = 0

    for r in runs:
        out = r["precedent_output"]
        if isinstance(out, str):
            ok, parsed = parse_loose_json(out)
            out = parsed if ok else {"precedents": [], "notes": ["parse failed"]}
        precedents = (out or {}).get("precedents", []) or []
        schema_v.append(precedent_schema_validity(precedents)["valid_share"])
        dedup_ok.append(1.0 if dedup_correctness(precedents)["is_unique"] else 0.0)
        noguess.append(no_guessing_compliance(precedents)["score"])
        if precedents:
            n_with_any += 1
            trust_p.append(mean(1.0 if is_trusted(p.get("url", "")) else 0.0 for p in precedents))
            opinion_p.append(mean(1.0 if looks_like_opinion(p.get("name", ""), p.get("citation", ""), p.get("url", "")) else 0.0 for p in precedents))
            for p in precedents:
                tier_counts[court_tier(p.get("name", ""), p.get("citation", ""), p.get("url", ""))] += 1

    summary = {
        "n_cases": len(runs),
        "cases_with_precedents": n_with_any,
        "schema_validity": mean(schema_v) if schema_v else 0.0,
        "dedup_correct_share": mean(dedup_ok) if dedup_ok else 0.0,
        "no_guessing_compliance": mean(noguess) if noguess else 0.0,
        "trusted_source_precision": mean(trust_p) if trust_p else 0.0,
        "opinion_page_precision": mean(opinion_p) if opinion_p else 0.0,
    }
    total = sum(tier_counts.values()) or 1
    summary["court_tier_share"] = {k: tier_counts[k] / total for k in ("scotus", "circuit", "district", "unknown")}
    return summary
