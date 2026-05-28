# tools/usc_sections_search_tool.py
import os
import re
from dotenv import load_dotenv
from crewai.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Optional

load_dotenv()

_PERSIST_DIR = os.getenv("PERSIST_DIRECTORY_PATH", "./chroma_db")
_COLLECTION = os.getenv("USC_COLLECTION_NAME", "usc_complete")
_TOP_K = int(os.getenv("USC_SEARCH_TOP_K", "8"))
_LEXICAL_K = int(os.getenv("USC_LEXICAL_K", "5"))

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectordb: Optional[Chroma] = None

# Citation parsing for direct lookup AND for excerpt repair (below)
_CITATION_RE = re.compile(
    r"(\d+[A-Za-z]?)\s*U\.?\s*S\.?\s*C\.?\s*(?:§|sec(?:tion)?|s\.)?\s*(\d+[A-Za-z\-]*)",
    re.IGNORECASE,
)

# Common-name phrase -> canonical citation(s).  When the user query contains
# one of these phrases (case-insensitive substring), the corresponding
# citation(s) are prepended to the merged results via direct-metadata lookup.
# This is the "hard route" for statutes whose section_title is generic
# (e.g. 21 U.S.C. § 841 = "Prohibited acts A"); without this, MiniLM
# ranks more-topically-titled but less-central sections above them.
_QUERY_TO_CITATIONS: dict[str, list[str]] = {
    "drug trafficking": ["21 U.S.C. § 841", "21 U.S.C. § 846"],
    "narcotics trafficking": ["21 U.S.C. § 841", "21 U.S.C. § 846"],
    "controlled substance": ["21 U.S.C. § 841", "21 U.S.C. § 846"],
    "drug conspiracy": ["21 U.S.C. § 846"],
    "drug importation": ["21 U.S.C. § 952", "21 U.S.C. § 960"],
    "wire fraud": ["18 U.S.C. § 1343"],
    "mail fraud": ["18 U.S.C. § 1341"],
    "bank fraud": ["18 U.S.C. § 1344"],
    "bank robbery": ["18 U.S.C. § 2113"],
    "identity theft": ["18 U.S.C. § 1028", "18 U.S.C. § 1028A"],
    "aggravated identity theft": ["18 U.S.C. § 1028A"],
    "computer fraud": ["18 U.S.C. § 1030"],
    "cfaa": ["18 U.S.C. § 1030"],
    "computer fraud and abuse act": ["18 U.S.C. § 1030"],
    "kidnapping": ["18 U.S.C. § 1201"],
    "ransom": ["18 U.S.C. § 1202", "18 U.S.C. § 875"],
    "money laundering": ["18 U.S.C. § 1956", "18 U.S.C. § 1957"],
    "structuring": ["31 U.S.C. § 5324"],
    "currency transaction report": ["31 U.S.C. § 5313"],
    "tax evasion": ["26 U.S.C. § 7201"],
    "false tax return": ["26 U.S.C. § 7206"],
    "employment tax": ["26 U.S.C. § 7202"],
    "espionage": ["18 U.S.C. § 793", "18 U.S.C. § 794"],
    "classified information": ["18 U.S.C. § 798"],
    "securities fraud": ["15 U.S.C. § 78j"],
    "hobbs act": ["18 U.S.C. § 1951"],
}


def _alias_matched_citations(query: str) -> list[str]:
    """Return the citations (in order) for any alias phrase contained in the query."""
    q = (query or "").lower()
    out: list[str] = []
    seen: set[str] = set()
    for phrase, cites in _QUERY_TO_CITATIONS.items():
        if phrase in q:
            for c in cites:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
    return out


def _lookup_by_citation_string(vectordb, citation: str) -> list:
    """Fetch the doc for a specific 'NN U.S.C. § MM' string."""
    m = _CITATION_RE.search(citation)
    if not m:
        return []
    return _direct_citation_lookup(vectordb, f"{m.group(1)} U.S.C. § {m.group(2)}")


def _normalize_citation(citation: str) -> str:
    m = _CITATION_RE.search(citation or "")
    if not m:
        return (citation or "").strip()
    return f"{m.group(1)} U.S.C. § {m.group(2)}"


def _pick_excerpt(content: str, max_chars: int = 600) -> str:
    """Return a contiguous substring up to max_chars, preferring the start of the
    statute body ('§ NNN.') over the document header lines added by the builder."""
    if not content:
        return ""
    text = content.replace("\n", " ")
    m = re.search(r"§\s*\d+[A-Za-z\-]*\.", text)
    start = m.start() if m else 0
    return text[start : start + max_chars].strip()


def repair_drafter_excerpts(drafter_output: Dict, usc_top_statutes: List[Dict]) -> Dict:
    """Deterministic post-processor: replace each drafter statute's excerpt with a
    verbatim contiguous substring of the upstream USC content.

    The drafter agent is instructed to copy excerpts verbatim, but in practice it
    paraphrases — a hallucination risk in legal output. This swaps in the real
    text. No extra LLM call. Statutes whose citation can't be matched upstream
    are left unchanged and marked __repaired__=False.
    """
    if not isinstance(drafter_output, dict):
        return drafter_output
    statutes = drafter_output.get("statutes")
    if not isinstance(statutes, list):
        return drafter_output

    upstream_by_cite: Dict[str, Dict] = {}
    for s in usc_top_statutes or []:
        cite = _normalize_citation(s.get("citation", ""))
        if cite:
            upstream_by_cite[cite] = s

    repaired = 0
    for st in statutes:
        if not isinstance(st, dict):
            continue
        src = upstream_by_cite.get(_normalize_citation(st.get("citation", "")))
        if not src:
            st["__repaired__"] = False
            continue
        new_excerpt = _pick_excerpt(src.get("content") or src.get("excerpt") or "")
        if new_excerpt:
            st["excerpt"] = new_excerpt
            st["__repaired__"] = True
            repaired += 1
        else:
            st["__repaired__"] = False
    drafter_output["__excerpts_repaired__"] = repaired
    return drafter_output


def _get_vectordb() -> Chroma:
    global _embeddings, _vectordb

    if not os.path.exists(_PERSIST_DIR):
        raise FileNotFoundError(
            f"Vector database not found at '{_PERSIST_DIR}'. "
            f"Please run 'python usc_vectordb_builder.py' first to create it."
        )

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    if _vectordb is None:
        _vectordb = Chroma(
            collection_name=_COLLECTION,
            persist_directory=_PERSIST_DIR,
            embedding_function=_embeddings
        )

    return _vectordb


def _doc_to_result(doc, query: str) -> Dict:
    md = doc.metadata or {}
    content = (doc.page_content or "").strip()
    excerpt = content.replace("\n", " ")[:600]
    section_number = md.get("section") or md.get("Section") or md.get("section_number")
    return {
        "query": query,
        "citation": md.get("citation", "Unknown"),
        "section_number": section_number,
        "section_title": md.get("section_title", "Unknown"),
        "title": str(md.get("title", "Unknown")),
        "title_name": md.get("title_name", "Unknown Title"),
        "chapter": md.get("chapter"),
        "chapter_title": md.get("chapter_title", "Unknown Chapter"),
        "content": content,
        "excerpt": excerpt,
    }


def _direct_citation_lookup(vectordb: Chroma, query: str) -> List:
    """If the query contains a citation like '21 U.S.C. § 841', fetch that statute by metadata."""
    m = _CITATION_RE.search(query)
    if not m:
        return []
    title_num, section_num = m.group(1), m.group(2)
    try:
        # Chroma exposes the underlying collection; use a metadata-filtered get
        coll = vectordb._collection
        res = coll.get(where={"$and": [{"title": {"$eq": title_num}}, {"section": {"$eq": section_num}}]}, limit=1)
        out = []
        for content, md in zip(res.get("documents", []), res.get("metadatas", [])):
            class _D:
                pass
            d = _D()
            d.page_content = content
            d.metadata = md
            out.append(d)
        return out
    except Exception:
        return []


def _lexical_search(vectordb: Chroma, query: str, k: int) -> List:
    """Substring-based fallback: scan all documents for content containing every meaningful query token.
    Scores by token-hit count + a bonus for hits in section_title.
    """
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", query.lower()) if len(t) >= 3]
    if not tokens:
        return []
    try:
        coll = vectordb._collection
        # Pull everything once; for 4-5k USC sections this is cheap (~15MB)
        res = coll.get()
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
    except Exception:
        return []

    scored = []
    for content, md in zip(docs, metas):
        lc = (content or "").lower()
        st = (md or {}).get("section_title", "").lower()
        hits = sum(1 for t in tokens if t in lc)
        if hits == 0:
            continue
        title_hits = sum(1 for t in tokens if t in st)
        score = hits + 3 * title_hits  # weight section-title matches heavily
        scored.append((score, content, md))
    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    for score, content, md in scored[:k]:
        class _D:
            pass
        d = _D()
        d.page_content = content
        d.metadata = md
        out.append(d)
    return out


@tool("USC Sections Search Tool")
def search_usc_sections(query: str) -> List[Dict]:
    """
    Hybrid retrieval across the locally-built USC Chroma vector database.
    Combines, in order, deduped by citation:
      (1) direct citation lookup if the query looks like 'NN U.S.C. § MM',
      (2) MiniLM semantic search on the original query,
      (3) lexical (token-overlap) fallback so statutes whose section_title
          does not contain the topical word can still be surfaced.
    """
    query = (query or "").strip()
    if not query:
        return []

    vectordb = _get_vectordb()

    merged: List = []
    seen_citations = set()

    def _add(doc):
        cite = (doc.metadata or {}).get("citation", "")
        if cite and cite not in seen_citations:
            seen_citations.add(cite)
            merged.append(doc)

    # (1) Direct citation lookup if the query contains a 'NN U.S.C. § MM' token
    for d in _direct_citation_lookup(vectordb, query):
        _add(d)

    # (2) Alias-based hard route: well-known query phrases pin specific
    # statutes to the top, regardless of how MiniLM scores them. Covers
    # the case where a statute's section_title is generic (e.g. § 841).
    for cite in _alias_matched_citations(query):
        for d in _lookup_by_citation_string(vectordb, cite):
            _add(d)

    # (3) Semantic search on original query
    for d in vectordb.similarity_search(query, k=_TOP_K):
        _add(d)

    # (4) Lexical fallback
    for d in _lexical_search(vectordb, query, k=_LEXICAL_K):
        _add(d)

    if not merged:
        return []

    cap = _TOP_K + _LEXICAL_K
    return [_doc_to_result(d, query) for d in merged[:cap]]
