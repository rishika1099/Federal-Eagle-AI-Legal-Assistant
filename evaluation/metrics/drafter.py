"""Metrics for the final Drafter agent's JSON output, plus RAGAS-style groundedness."""
from __future__ import annotations

import re
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .intake import parse_loose_json
from .retrieval import normalize_citation


REQUIRED_TOP_KEYS = {
    "summary",
    "statutes",
    "elements_analysis",
    "precedents",
    "next_steps",
    "clarifying_questions",
    "draft_document",
    "disclaimer",
}
SUMMARY_REQUIRED = {"case_type", "primary_issue", "federal_hook", "confidence", "assumptions"}
STATUTE_REQUIRED = {"citation", "title", "title_name", "section_title", "why_relevant", "elements", "excerpt"}
DRAFT_DOC_REQUIRED = {"document_type", "content"}
CONFIDENCE_ENUM = {"low", "medium", "high"}
CASE_TYPE_ENUM = {"criminal", "civil", "administrative", "unclear"}


def drafter_schema_validity(output: Any) -> Dict[str, Any]:
    if not isinstance(output, dict):
        return {"valid_json_object": False, "score": 0.0}
    keys = set(output.keys())
    missing_top = REQUIRED_TOP_KEYS - keys
    pieces: List[float] = [1 - len(missing_top) / len(REQUIRED_TOP_KEYS)]

    summary = output.get("summary") or {}
    if isinstance(summary, dict):
        missing_s = SUMMARY_REQUIRED - set(summary.keys())
        pieces.append(1 - len(missing_s) / len(SUMMARY_REQUIRED))
        pieces.append(1.0 if summary.get("case_type") in CASE_TYPE_ENUM else 0.0)
        pieces.append(1.0 if summary.get("confidence") in CONFIDENCE_ENUM else 0.0)
    else:
        pieces.extend([0.0, 0.0, 0.0])

    statutes = output.get("statutes") or []
    if isinstance(statutes, list) and statutes:
        sub = []
        for st in statutes:
            if not isinstance(st, dict):
                sub.append(0.0)
                continue
            sub.append(1 - len(STATUTE_REQUIRED - set(st.keys())) / len(STATUTE_REQUIRED))
        pieces.append(mean(sub))
    else:
        pieces.append(0.0)

    dd = output.get("draft_document") or {}
    if isinstance(dd, dict):
        pieces.append(1 - len(DRAFT_DOC_REQUIRED - set(dd.keys())) / len(DRAFT_DOC_REQUIRED))
    else:
        pieces.append(0.0)

    return {
        "valid_json_object": True,
        "missing_top_keys": sorted(missing_top),
        "score": mean(pieces),
    }


# ---------- citation faithfulness (no hallucinated statutes) ----------

def citation_faithfulness(
    drafter_statutes: Iterable[Dict],
    upstream_top_statutes: Iterable[Dict],
) -> Dict[str, float]:
    """Fraction of drafter-output citations that appear in upstream usc_section_task.top_statutes.

    Hallucination = drafter cites something the retriever never returned.
    """
    upstream = {normalize_citation(s.get("citation", "")) for s in upstream_top_statutes or []}
    drafter = [normalize_citation(s.get("citation", "")) for s in drafter_statutes or []]
    drafter = [c for c in drafter if c]
    if not drafter:
        return {"faithfulness": 0.0, "n_drafter_citations": 0, "n_hallucinated": 0}
    grounded = sum(1 for c in drafter if c in upstream)
    return {
        "faithfulness": grounded / len(drafter),
        "n_drafter_citations": len(drafter),
        "n_hallucinated": len(drafter) - grounded,
    }


def excerpt_substring_grounding(
    drafter_statutes: Iterable[Dict],
    upstream_top_statutes: Iterable[Dict],
    min_overlap_ratio: float = 0.7,
) -> Dict[str, float]:
    """For each drafter excerpt, check whether a long substring (>= min_overlap_ratio of its length)
    appears in the corresponding upstream excerpt/content. Catches paraphrase-as-quote.
    """
    upstream_by_cite = {
        normalize_citation(s.get("citation", "")): (s.get("excerpt") or s.get("content") or "")
        for s in upstream_top_statutes or []
    }
    drafter_list = list(drafter_statutes or [])
    if not drafter_list:
        return {"excerpt_grounding": 0.0, "n_checked": 0}
    grounded = 0
    checked = 0
    for s in drafter_list:
        cite = normalize_citation(s.get("citation", ""))
        excerpt = (s.get("excerpt") or "").strip()
        if not cite or not excerpt:
            continue
        src = upstream_by_cite.get(cite, "")
        if not src:
            continue
        checked += 1
        # crude n-gram check
        ngrams = _ngrams(excerpt, n=8)
        if not ngrams:
            continue
        hit = sum(1 for ng in ngrams if ng in src)
        if hit / len(ngrams) >= min_overlap_ratio:
            grounded += 1
    return {"excerpt_grounding": (grounded / checked) if checked else 0.0, "n_checked": checked}


def _ngrams(text: str, n: int = 8) -> List[str]:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < n:
        return [" ".join(toks)] if toks else []
    return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]


# ---------- draft-document quality heuristics ----------

MARKDOWN_RE = re.compile(r"(?m)(^\s*[-*•]\s)|(\*\*[^*]+\*\*)|(^#+\s)|(`[^`]+`)")
PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z _/]+\]")


def draft_quality(draft_document: Dict) -> Dict[str, Any]:
    """Heuristic check of the plain-text draft format the prompt asked for."""
    if not isinstance(draft_document, dict):
        return {"score": 0.0, "reasons": ["draft_document not an object"]}
    content = draft_document.get("content", "") or ""
    doctype = draft_document.get("document_type", "") or ""
    reasons: List[str] = []

    has_markdown = bool(MARKDOWN_RE.search(content))
    if has_markdown:
        reasons.append("markdown formatting present")
    has_allcaps_heading = bool(re.search(r"(?m)^[A-Z][A-Z /()&]{4,}$", content))
    if not has_allcaps_heading:
        reasons.append("no ALL-CAPS heading detected")
    has_numbered = bool(re.search(r"(?m)^\s*\d+\.\s", content))
    if not has_numbered:
        reasons.append("no numbered paragraphs detected")
    has_placeholder = bool(PLACEHOLDER_RE.search(content))
    repeats_title = doctype and content.strip().lower().startswith(doctype.strip().lower())
    if repeats_title:
        reasons.append("content repeats document_type as first line")
    has_disclaimer_in_content = "disclaimer" in content.lower() or "not legal advice" in content.lower()
    if has_disclaimer_in_content:
        reasons.append("disclaimer leaked into draft content")

    pieces = [
        1.0 if not has_markdown else 0.0,
        1.0 if has_allcaps_heading else 0.0,
        1.0 if has_numbered else 0.0,
        1.0 if has_placeholder else 0.5,
        1.0 if not repeats_title else 0.0,
        1.0 if not has_disclaimer_in_content else 0.0,
    ]
    return {
        "score": mean(pieces),
        "has_markdown": has_markdown,
        "has_allcaps_heading": has_allcaps_heading,
        "has_numbered_paragraphs": has_numbered,
        "has_placeholders": has_placeholder,
        "repeats_title": repeats_title,
        "disclaimer_leaked": has_disclaimer_in_content,
        "reasons": reasons,
        "char_length": len(content),
        "word_count": len(content.split()),
    }


# ---------- RAGAS-style groundedness (token-overlap heuristic, no LLM) ----------

def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def _token_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def context_precision(
    retrieved_contexts: Sequence[str],
    relevant_contexts: Sequence[str],
) -> float:
    """Fraction of retrieved contexts that are relevant (ground truth)."""
    if not retrieved_contexts:
        return 0.0
    rel = [_token_set(r) for r in relevant_contexts]
    hits = 0
    for c in retrieved_contexts:
        tc = _token_set(c)
        if not tc:
            continue
        if any((len(tc & r) / max(1, len(tc | r))) > 0.2 for r in rel):
            hits += 1
    return hits / len(retrieved_contexts)


def context_recall(answer_facts: Sequence[str], retrieved_contexts: Sequence[str]) -> float:
    """Fraction of answer facts that are supported by some retrieved context (token overlap)."""
    if not answer_facts:
        return 0.0
    src = " ".join(retrieved_contexts).lower()
    supported = sum(1 for f in answer_facts if _bag_overlap(f, src) >= 0.3)
    return supported / len(answer_facts)


def faithfulness(answer_claims: Sequence[str], retrieved_contexts: Sequence[str]) -> float:
    """Fraction of claims in the answer that are entailed by retrieved context (token overlap proxy)."""
    if not answer_claims:
        return 0.0
    src = " ".join(retrieved_contexts).lower()
    return mean(1.0 if _bag_overlap(c, src) >= 0.25 else 0.0 for c in answer_claims)


def answer_relevance(question: str, answer: str) -> float:
    """Token-overlap proxy for how on-topic the answer is given the question."""
    return _bag_overlap(question, answer)


def _bag_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


# ---------- elements-analysis structure checks ----------

ELEM_STATUS_ENUM = {"met", "unknown", "not_met"}


def elements_analysis_structure(elements_analysis: List[Dict], allowed_citations: Iterable[str]) -> Dict[str, float]:
    if not isinstance(elements_analysis, list) or not elements_analysis:
        return {"valid_share": 0.0, "status_enum_share": 0.0, "citation_grounded_share": 0.0, "n_blocks": 0}
    allowed = {normalize_citation(c) for c in allowed_citations}
    valid = 0
    status_ok = 0
    status_total = 0
    grounded = 0
    for block in elements_analysis:
        if not isinstance(block, dict):
            continue
        cite_ok = normalize_citation(block.get("citation", "")) in allowed
        checklist = block.get("checklist") or []
        if isinstance(checklist, list) and checklist and all(isinstance(x, dict) for x in checklist):
            valid += 1
            for item in checklist:
                status_total += 1
                if item.get("status") in ELEM_STATUS_ENUM:
                    status_ok += 1
        if cite_ok:
            grounded += 1
    n = len(elements_analysis)
    return {
        "n_blocks": n,
        "valid_share": valid / n,
        "status_enum_share": (status_ok / status_total) if status_total else 0.0,
        "citation_grounded_share": grounded / n,
    }


# ---------- aggregate ----------

def evaluate_drafter_run(runs: List[Dict]) -> Dict[str, float]:
    """runs: list of {case_id, drafter_output (str|dict), upstream_top_statutes, intake_question}"""
    schema_scores, faith_scores, hall_counts, excerpt_scores = [], [], [], []
    draft_quality_scores = []
    el_valid_shares, el_status_shares, el_grounded_shares = [], [], []
    ragas_faith, ragas_ans_rel, ragas_ctx_p, ragas_ctx_r = [], [], [], []

    for r in runs:
        out = r["drafter_output"]
        if isinstance(out, str):
            ok, parsed = parse_loose_json(out)
            if not ok or not isinstance(parsed, dict):
                schema_scores.append(0.0)
                continue
            out = parsed
        s = drafter_schema_validity(out)
        schema_scores.append(s["score"])

        f = citation_faithfulness(out.get("statutes", []), r.get("upstream_top_statutes", []))
        faith_scores.append(f["faithfulness"])
        hall_counts.append(f["n_hallucinated"])
        e = excerpt_substring_grounding(out.get("statutes", []), r.get("upstream_top_statutes", []))
        excerpt_scores.append(e["excerpt_grounding"])

        draft_quality_scores.append(draft_quality(out.get("draft_document", {}))["score"])

        ea = elements_analysis_structure(
            out.get("elements_analysis", []),
            [s.get("citation", "") for s in out.get("statutes", [])],
        )
        el_valid_shares.append(ea["valid_share"])
        el_status_shares.append(ea["status_enum_share"])
        el_grounded_shares.append(ea["citation_grounded_share"])

        # RAGAS-style proxies
        ctxs = [s.get("excerpt") or s.get("content") or "" for s in r.get("upstream_top_statutes", [])]
        gt_ctxs = [s.get("excerpt") or "" for s in r.get("relevant_upstream", [])] or ctxs
        claims = [
            (out.get("summary") or {}).get("primary_issue", ""),
            *((out.get("summary") or {}).get("federal_hook", []) or []),
            *[st.get("why_relevant", "") for st in out.get("statutes", [])],
        ]
        ragas_faith.append(faithfulness([c for c in claims if c], ctxs))
        ragas_ans_rel.append(answer_relevance(r.get("intake_question", ""), (out.get("summary") or {}).get("primary_issue", "")))
        ragas_ctx_p.append(context_precision(ctxs, gt_ctxs))
        ragas_ctx_r.append(context_recall([c for c in claims if c], ctxs))

    return {
        "n_cases": len(runs),
        "schema_score": _safe_mean(schema_scores),
        "citation_faithfulness": _safe_mean(faith_scores),
        "hallucinated_citations_per_case": _safe_mean(hall_counts),
        "excerpt_grounding": _safe_mean(excerpt_scores),
        "draft_quality_score": _safe_mean(draft_quality_scores),
        "elements_block_validity": _safe_mean(el_valid_shares),
        "elements_status_enum_share": _safe_mean(el_status_shares),
        "elements_citation_grounded": _safe_mean(el_grounded_shares),
        "ragas_faithfulness": _safe_mean(ragas_faith),
        "ragas_answer_relevance": _safe_mean(ragas_ans_rel),
        "ragas_context_precision": _safe_mean(ragas_ctx_p),
        "ragas_context_recall": _safe_mean(ragas_ctx_r),
    }


def _safe_mean(xs: List[float]) -> float:
    return mean(xs) if xs else 0.0
