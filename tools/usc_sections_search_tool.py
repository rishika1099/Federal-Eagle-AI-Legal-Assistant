# tools/usc_sections_search_tool.py
import os
from dotenv import load_dotenv
from crewai.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Optional

load_dotenv()

_PERSIST_DIR = os.getenv("PERSIST_DIRECTORY_PATH", "./chroma_db")
_COLLECTION = os.getenv("USC_COLLECTION_NAME", "usc_complete")
_TOP_K = int(os.getenv("USC_SEARCH_TOP_K", "5"))

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectordb: Optional[Chroma] = None

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


@tool("USC Sections Search Tool")
def search_usc_sections(query: str) -> List[Dict]:
    """
    Semantic search across the locally-built USC Chroma vector database.
    Returns top-k statute matches with normalized metadata and a short excerpt.
    """
    query = (query or "").strip()
    if not query:
        return []

    vectordb = _get_vectordb()
    docs = vectordb.similarity_search(query, k=_TOP_K)

    if not docs:
        return []

    results: List[Dict] = []

    for doc in docs:
        md = doc.metadata or {}
        content = (doc.page_content or "").strip()

        excerpt = content.replace("\n", " ")
        excerpt = excerpt[:600]

        # robust key handling (depends on how builder stored it)
        section_number = md.get("section") or md.get("Section") or md.get("section_number")

        results.append({
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
        })

    return results
