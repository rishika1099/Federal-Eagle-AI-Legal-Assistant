"""Deterministic mock pipeline so the eval harness is runnable WITHOUT API keys
or the chroma DB. Lets you verify the metric code end-to-end before you spend money.

Each function returns the same shape the real agent would, but seeded from the
ground-truth labels so most metrics score >0 (not perfect, by design).
"""
from __future__ import annotations

import json
import random
from typing import Dict, List

from ..metrics.cost_latency import CaseRunMetrics, StageMetric


def mock_intake(case: Dict, seed: int = 0) -> Dict:
    rng = random.Random(seed + hash(case["id"]) % 9999)
    # drop one expected hook to simulate imperfect recall
    hooks = list(case["expected_federal_hooks"])
    if rng.random() < 0.4 and hooks:
        hooks = hooks[:-1]
    return {
        "case_type": case["expected_case_type"],
        "legal_domain": case["expected_legal_domain"][0],
        "primary_issue": case["name"],
        "summary": case["scenario"][:300],
        "key_facts": [s.strip() for s in case["scenario"].split(".") if s.strip()][:8],
        "relevant_entities": ["Defendant", "Victim", "Federal Agency"],
        "locations": ["unknown"],
        "dates": ["unknown"],
        "federal_hooks": hooks,
        "missing_info_questions": [],
        "search_queries": case["expected_search_query_keywords"][:5],
    }


def mock_usc_retrieval(case: Dict, k: int = 5, seed: int = 0) -> List[str]:
    rng = random.Random(seed + hash(case["id"]) % 9999)
    primary = list(case["expected_statutes"].get("primary", []))
    secondary = list(case["expected_statutes"].get("secondary", []))
    distractors = list(case["expected_statutes"].get("distractor_avoid", []))
    # drop the lowest-ranked primary 30% of the time
    if rng.random() < 0.3 and len(primary) > 1:
        primary = primary[:-1]
    rng.shuffle(secondary)
    out = primary + secondary[: max(0, k - len(primary) - 1)]
    if rng.random() < 0.5 and distractors:
        out.append(distractors[0])
    return out[:k]


def mock_usc_top_statutes(case: Dict, k: int = 5, seed: int = 0) -> List[Dict]:
    cites = mock_usc_retrieval(case, k=k, seed=seed)
    out = []
    for c in cites:
        out.append({
            "citation": c,
            "title": c.split(" ")[0],
            "title_name": "CRIMES AND CRIMINAL PROCEDURE",
            "chapter": "?",
            "chapter_title": "?",
            "section_number": c.split("§")[-1].strip() if "§" in c else "?",
            "section_title": case["name"],
            "excerpt": f"Mock excerpt for {c} covering {case['name'].lower()}.",
            "content": f"Mock excerpt for {c} covering {case['name'].lower()}.",
            "why_relevant_hint": "Matches user's described conduct.",
        })
    return out


def mock_precedents(case: Dict, seed: int = 0) -> Dict:
    rng = random.Random(seed + hash(case["id"]) % 9999)
    if rng.random() < 0.25:
        return {"precedents": [], "notes": ["No high-confidence opinion pages found."]}
    return {
        "precedents": [
            {
                "name": "United States v. Doe",
                "court_year": "U.S. Supreme Court 2019",
                "citation": "586 U.S. ___",
                "holding": f"Addresses interpretation of {case['expected_legal_domain'][0]} under federal law.",
                "relevance": "Cited for the federal hook present in user facts.",
                "url": "https://supreme.justia.com/cases/federal/us/586/12345/",
            }
        ],
        "notes": [],
    }


def mock_drafter(case: Dict, upstream_top_statutes: List[Dict], precedents: Dict, seed: int = 0) -> Dict:
    intake = mock_intake(case, seed)
    return {
        "summary": {
            "case_type": case["expected_case_type"],
            "primary_issue": case["name"],
            "federal_hook": case["expected_federal_hooks"][:3],
            "confidence": "medium",
            "assumptions": [],
        },
        "statutes": [
            {
                "citation": s["citation"],
                "title": s["title"],
                "title_name": s["title_name"],
                "section_title": s["section_title"],
                "why_relevant": f"Conduct matches elements of {s['section_title']}.",
                "elements": ["Element A", "Element B", "Element C"],
                "excerpt": s["excerpt"],
            }
            for s in upstream_top_statutes[:3]
        ],
        "elements_analysis": [
            {
                "citation": s["citation"],
                "checklist": [
                    {"element": "Element A", "status": "met", "supporting_facts": ["fact 1"]},
                    {"element": "Element B", "status": "unknown", "supporting_facts": []},
                ],
            }
            for s in upstream_top_statutes[:2]
        ],
        "precedents": precedents.get("precedents", []),
        "next_steps": ["Preserve evidence", "Document timeline", "Coordinate with counsel"],
        "clarifying_questions": [],
        "draft_document": {
            "document_type": "Internal Memo",
            "content": (
                "BACKGROUND\n1. " + case["scenario"][:200] + "\n\n"
                "RISK SUMMARY\n2. Conduct may implicate federal statutes listed below.\n\n"
                "RELEVANT STATUTES (CITATIONS ONLY)\n"
                + "\n".join(f"{i+3}. {s['citation']}" for i, s in enumerate(upstream_top_statutes[:3]))
                + "\n\nRECOMMENDED ACTIONS\n6. Preserve [EVIDENCE] and contact [DISTRICT] counsel.\n"
            ),
        },
        "disclaimer": "Educational use only. Not legal advice.",
    }


def run_mock_full_pipeline(case: Dict, seed: int = 0):
    """Returns (intake, top_statutes, precedents, drafter, run_metrics)."""
    intake = mock_intake(case, seed)
    top_statutes = mock_usc_top_statutes(case, seed=seed)
    precedents = mock_precedents(case, seed)
    drafter = mock_drafter(case, top_statutes, precedents, seed)

    rm = CaseRunMetrics(case_id=case["id"])
    rm.add(StageMetric(stage="case_intake", duration_s=1.1 + 0.2 * (seed % 5), prompt_tokens=420, completion_tokens=180, model="gpt-4o-mini", cost_usd=(420 * 0.15 + 180 * 0.60) / 1_000_000))
    rm.add(StageMetric(stage="usc_retrieval", duration_s=0.6, prompt_tokens=380, completion_tokens=300, model="gpt-4o-mini", cost_usd=(380 * 0.15 + 300 * 0.60) / 1_000_000))
    rm.add(StageMetric(stage="precedent_search", duration_s=2.3, prompt_tokens=520, completion_tokens=260, model="gpt-4o-mini", cost_usd=(520 * 0.15 + 260 * 0.60) / 1_000_000))
    rm.add(StageMetric(stage="drafter", duration_s=2.8, prompt_tokens=1200, completion_tokens=900, model="gpt-4o-mini", cost_usd=(1200 * 0.15 + 900 * 0.60) / 1_000_000))
    return intake, top_statutes, precedents, drafter, rm
