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


# Small static query expansion. For each canonical topic, we add variant phrases
# that the section_title or content might use. Keeps recall up without changing
# the embedding model.
_QUERY_EXPANSIONS: Dict[str, List[str]] = {
    "wire fraud": ["scheme to defraud", "interstate communications fraud"],
    "drug trafficking": ["controlled substance", "manufacture distribute dispense", "import controlled substance"],
    "money laundering": ["financial transaction proceeds", "structuring transactions", "currency transaction report"],
    "identity theft": ["aggravated identity theft", "identification documents", "false identification"],
    "kidnapping": ["interstate transportation victim", "ransom demand"],
    "bank robbery": ["federally insured", "robbery extortion bank"],
    "tax evasion": ["attempt evade defeat tax", "false return", "willful failure file"],
    "computer fraud": ["unauthorized access", "protected computer", "exceed authorized access"],
    "espionage": ["national defense information", "gathering transmitting defense"],
}


def _expand_queries(query: str) -> List[str]:
    """Return [query, ...expansions]. Expansions are added only when a topical
    keyword is detected, keeping noise low for off-topic queries."""
    qs = [query]
    lq = query.lower()
    for keyword, variants in _QUERY_EXPANSIONS.items():
        if keyword in lq:
            qs.extend(variants)
    return qs


@tool("USC Sections Search Tool")
def search_usc_sections(query: str) -> List[Dict]:
    """
    Hybrid retrieval across the locally-built USC Chroma vector database.
    Combines, in order, deduped by citation:
      (1) direct citation lookup if the query looks like 'NN U.S.C. § MM',
      (2) MiniLM semantic search on the original query,
      (3) MiniLM semantic search on each topical expansion variant,
      (4) lexical (token-overlap) fallback.
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

    # (1) Direct citation lookup, highest priority
    for d in _direct_citation_lookup(vectordb, query):
        _add(d)

    # (2) + (3) Semantic search on the original query, then on each expansion
    for q in _expand_queries(query):
        for d in vectordb.similarity_search(q, k=_TOP_K):
            _add(d)

    # (4) Lexical fallback
    for d in _lexical_search(vectordb, query, k=_LEXICAL_K):
        _add(d)

    if not merged:
        return []

    cap = _TOP_K + _LEXICAL_K
    return [_doc_to_result(d, query) for d in merged[:cap]]
