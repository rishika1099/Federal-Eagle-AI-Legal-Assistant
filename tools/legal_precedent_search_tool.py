# tools/legal_precedent_search_tool.py
import os
import re
from typing import Optional
from dotenv import load_dotenv
from crewai.tools import tool
from tavily import TavilyClient

load_dotenv()

LEGAL_SOURCES = [
    "supremecourt.gov",
    "uscourts.gov",
    "law.cornell.edu",
    "courtlistener.com",
    "justia.com",
    "law.justia.com",
    "cases.justia.com",
    "findlaw.com",
]

# Strong "this is a case/opinion" markers
_CASE_SIGNAL_RE = re.compile(
    r"\bv\.\b|S\. Ct\.|F\.\d+d|F\. Supp\.|U\.S\.\s*\d+|WL\s*\d+|Fed\.\s*Appx\.",
    re.IGNORECASE,
)

# Usually NOT opinions (often commentary, journals, filings, etc.)
_BAD_HINT_RE = re.compile(
    r"\b(brief|amicus|petition|docket|appendix|motion|certiorari|"
    r"law\s+review|journal|article|note|commentary|practice\s+guide|"
    r"outline|syllabus|analysis|blog|newsletter)\b",
    re.IGNORECASE,
)

# SupremeCourt.gov DocketPDF is filings (nearly never the opinion page)
_SCOTUS_DOCKETPDF_RE = re.compile(r"supremecourt\.gov/docketpdf/", re.IGNORECASE)

# URL paths that are very likely opinions/case pages
_OPINION_PATH_RE = re.compile(
    r"(law\.cornell\.edu/supremecourt/text/)"
    r"|(\bcourtlistener\.com/opinion\b)"
    r"|(\blaw\.justia\.com/cases\b)"
    r"|(\bcases\.justia\.com\b)",
    re.IGNORECASE,
)

def _extract_source_type(url: str) -> str:
    if not url:
        return "Unknown"
    u = url.lower()
    if "supremecourt.gov" in u:
        return "U.S. Supreme Court"
    if "courtlistener.com" in u:
        return "CourtListener"
    if "law.cornell.edu" in u:
        return "Cornell LII"
    if "law.justia.com" in u or "cases.justia.com" in u or "justia.com" in u:
        return "Justia"
    if "findlaw.com" in u:
        return "FindLaw"
    if "uscourts.gov" in u:
        return "U.S. Courts"
    return "Legal Source"

def _guess_case_name(title: str) -> str:
    if not title:
        return ""
    m = re.search(r"([A-Z][A-Za-z.\s]+ v\. [A-Z][A-Za-z.\s]+)", title)
    return m.group(1).strip() if m else title.strip()

def _is_pdf(url: str) -> bool:
    return (url or "").lower().split("?")[0].endswith(".pdf")

def _looks_bad(title: str, url: str, snippet: str) -> bool:
    blob = f"{title} {url} {snippet}"
    return bool(_BAD_HINT_RE.search(blob))

def _is_opinionish_url(url: str) -> bool:
    return bool(_OPINION_PATH_RE.search(url or ""))

def _opinionish_boost(url: str) -> int:
    u = (url or "").lower()
    boost = 0
    if "law.cornell.edu/supremecourt/text/" in u:
        boost += 6
    if "law.cornell.edu/supct/pdf/" in u:
        boost += 4
    if "courtlistener.com/opinion" in u:
        boost += 6
    if "law.justia.com/cases" in u:
        boost += 5
    if "cases.justia.com" in u:
        boost += 3
    if "uscourts.gov" in u:
        boost += 1
    return boost

def _extract_statute_hint(query: str) -> Optional[str]:
    """
    If user query contains a statute-like token, capture a simple hint:
    - "18 u.s.c. 1030" or "18 usc 1030" -> "1030"
    - "18 u.s.c. § 1343" -> "1343"
    """
    if not query:
        return None
    q = query.lower()
    m = re.search(r"\b\d+\s*u\.?\s*s\.?\s*c\.?\s*§?\s*(\d{3,5})\b", q)
    if m:
        return m.group(1)
    # Also allow raw section numbers if user wrote "§1030"
    m2 = re.search(r"§\s*(\d{3,5})\b", q)
    if m2:
        return m2.group(1)
    return None

def _relevance_boost_for_statute(statute_num: Optional[str], title: str, snippet: str) -> int:
    """
    If the query includes a statute number, prefer results that also mention it
    or a common name (CFAA for 1030, etc.).
    """
    if not statute_num:
        return 0
    blob = f"{title} {snippet}".lower()

    # direct match
    if statute_num in blob:
        return 3

    # a couple of high-value aliases (expand later if you want)
    if statute_num == "1030" and ("cfaa" in blob or "computer fraud and abuse" in blob):
        return 2

    return 0

def _pdf_allowed(url: str, title: str, snippet: str) -> bool:
    """
    Allow PDFs only if they plausibly contain opinions / official case text.
    Filter out filings and obvious non-opinion docs.
    """
    if not _is_pdf(url):
        return True

    if _SCOTUS_DOCKETPDF_RE.search(url or ""):
        return False

    if _looks_bad(title, url, snippet):
        return False

    u = (url or "").lower()

    # Known good opinion PDF sources
    if "law.cornell.edu/supct/pdf" in u:
        return True
    if "courtlistener.com" in u:
        return True
    if "cases.justia.com" in u:
        return True
    if "uscourts.gov" in u:
        return True

    # Otherwise: allow but will rank lower
    return True

def search_legal_precedents_raw(query: str) -> list[dict]:
    """
    Raw Tavily search (callable from Python).
    Returns lightweight case-law hits.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    query = (query or "").strip()
    if not query:
        return []

    query = query[:220]
    max_results = int(os.getenv("TAVILY_MAX_RESULTS", "10"))
    search_depth = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")

    statute_hint = _extract_statute_hint(query)

    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_domains=LEGAL_SOURCES,
            search_depth=search_depth,
        )

        raw_results = response.get("results", []) or []
        out: list[dict] = []

        for item in raw_results:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("content") or "").strip()[:900]

            if not url or not title:
                continue

            # Reject obvious commentary/non-opinion pages early
            if _looks_bad(title, url, snippet):
                continue

            if not _pdf_allowed(url, title, snippet):
                continue

            case_marker_hit = bool(_CASE_SIGNAL_RE.search(f"{title} {snippet}"))
            opinion_url_hint = _is_opinionish_url(url)

            # If it doesn't look like a case AND doesn't look like an opinion URL, drop it
            # (prevents index pages and generic “US Supreme Court cases” pages)
            if not case_marker_hit and not opinion_url_hint:
                continue

            score = 0

            # Case-likeness
            if case_marker_hit:
                score += 4

            # Opinion URL paths are very important
            score += _opinionish_boost(url)

            # Statute relevance guardrail
            score += _relevance_boost_for_statute(statute_hint, title, snippet)

            # PDFs get a small penalty unless known-good source
            if _is_pdf(url):
                u = url.lower()
                known_good_pdf = (
                    ("law.cornell.edu/supct/pdf" in u)
                    or ("courtlistener.com" in u)
                    or ("cases.justia.com" in u)
                )
                if not known_good_pdf:
                    score -= 1

            out.append({
                "name_guess": _guess_case_name(title),
                "title": title,
                "snippet": snippet,
                "url": url,
                "source": _extract_source_type(url),

                # extra signals (harmless; the agent can ignore, but they help)
                "is_pdf": _is_pdf(url),
                "case_marker_hit": case_marker_hit,
                "opinion_url_hint": opinion_url_hint,
                "score": score,
            })

        # Sort best first
        out.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Strip internal score before returning to the agent (optional)
        for x in out:
            x.pop("score", None)

        return out

    except Exception:
        return []

@tool("Legal Precedent Search Tool")
def search_legal_precedents(query: str) -> list[dict]:
    """
    CrewAI tool wrapper for precedent search.
    """
    return search_legal_precedents_raw(query)
