"""rag.py - PDF ingestion + ChromaDB vector store (Acceptance criterion 7)."""
import hashlib
import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSIST_DIR = "chroma_db"
SECTIONS = ["abstract", "introduction", "related work", "background", "method",
            "methodology", "approach", "experiments", "results", "discussion",
            "limitations", "conclusion", "conclusions"]
HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?(" + "|".join(SECTIONS) + r")\b.{0,40}$",
    re.IGNORECASE | re.MULTILINE,
)


@lru_cache(maxsize=1)
def get_embeddings():
    # Local, free embedding model (downloads ~90MB on first run)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _tag_sections(chunks):
    """Heuristic: carry the latest detected heading forward as chunk metadata."""
    current = "front matter"
    for c in chunks:
        m = HEADING_RE.search(c.page_content)
        if m:
            current = m.group(1).lower()
        c.metadata["section"] = current
    return chunks


def ingest_pdf(path: str, file_bytes: bytes):
    """Load -> split -> embed -> store in Chroma. Returns (vectorstore, n_chunks)."""
    paper_id = hashlib.md5(file_bytes).hexdigest()[:12]
    vs = Chroma(
        collection_name=f"paper_{paper_id}",
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )
    existing = vs._collection.count()
    if existing:  # already ingested earlier
        return vs, existing
    pages = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = _tag_sections(splitter.split_documents(pages))
    vs.add_documents(chunks)
    return vs, len(chunks)


def retrieve(vs, query: str, k: int = 4, section: str | None = None):
    """Similarity search, optionally restricted to a section."""
    if section:
        docs = vs.similarity_search(query, k=k, filter={"section": section})
        if docs:
            return docs
    return vs.similarity_search(query, k=k)


def retrieve_multi(vs, queries, k: int = 3):
    """Run several queries and de-duplicate the results."""
    seen, out = set(), []
    for q in queries:
        for d in retrieve(vs, q, k=k):
            if d.page_content not in seen:
                seen.add(d.page_content)
                out.append(d)
    return out


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('section', '?')} | p.{d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )