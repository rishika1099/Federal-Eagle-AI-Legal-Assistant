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
_TOP_K = int(os.getenv("USC_SEARCH_TOP_K", "5"))
_LEXICAL_K = int(os.getenv("USC_LEXICAL_K", "5"))

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectordb: Optional[Chroma] = None

# Citation parsing for direct lookup
_CITATION_RE = re.compile(
    r"(\d+[A-Za-z]?)\s*U\.?\s*S\.?\s*C\.?\s*(?:§|sec(?:tion)?)?\s*(\d+[A-Za-z\-]*)",
    re.IGNORECASE,
)


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
    Combines: (1) direct citation lookup if the query looks like 'NN U.S.C. § MM',
              (2) MiniLM semantic search,
              (3) lexical (token-overlap) fallback so statutes whose section_title
                  doesn't contain the topical word can still be surfaced.
    Results are merged in order, deduped by citation.
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

    # (1) Direct citation lookup — highest priority
    for d in _direct_citation_lookup(vectordb, query):
        _add(d)

    # (2) Semantic search
    for d in vectordb.similarity_search(query, k=_TOP_K):
        _add(d)

    # (3) Lexical fallback
    for d in _lexical_search(vectordb, query, k=_LEXICAL_K):
        _add(d)

    if not merged:
        return []

    return [_doc_to_result(d, query) for d in merged[: _TOP_K + _LEXICAL_K]]
