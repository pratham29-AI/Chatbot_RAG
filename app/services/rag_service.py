"""
RAG Service — document ingestion and semantic retrieval via FAISS.

Responsibilities:
  - Load PDF or plain-text files and split them into overlapping chunks.
  - Generate embeddings with OpenAI and store them in a FAISS index.
  - Persist the index to disk so documents survive application restarts.
  - Expose a similarity-search method that returns ranked, scored chunks.
  - Maintain a lightweight JSON metadata file alongside the index so the API
    can list all indexed documents without re-loading the full index.
"""

import json
import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

# ── paths ────────────────────────────────────────────────────────────────────
INDEX_DIR = Path(settings.faiss_index_path)
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "documents.json"


class RAGService:
    """Singleton-style service that owns the FAISS vector store."""

    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._store: Optional[FAISS] = None
        self._doc_meta: list[dict] = []  # [{id, filename, pages, chunks}]
        self._load_persisted_index()

    # ── public API ────────────────────────────────────────────────────────────

    def has_documents(self) -> bool:
        return self._store is not None and len(self._doc_meta) > 0

    def list_documents(self) -> list[dict]:
        return list(self._doc_meta)

    def add_document(self, file_path: str, filename: str) -> dict:
        """
        Parse, chunk, embed, and index a file.

        Returns a summary dict: {filename, chunk_count, page_count}.
        Raises ValueError for unsupported file types.
        """
        docs = self._load_file(file_path, filename)
        chunks = self._splitter.split_documents(docs)

        if not chunks:
            raise ValueError("File appears to be empty or could not be parsed.")

        # Tag every chunk with its source filename (page is already in metadata
        # for PDFs via PyPDFLoader; add it for text files too)
        for i, chunk in enumerate(chunks):
            chunk.metadata.setdefault("source", filename)
            chunk.metadata.setdefault("page", 0)
            chunk.metadata["chunk_index"] = i

        if self._store is None:
            self._store = FAISS.from_documents(chunks, self._embeddings)
        else:
            self._store.add_documents(chunks)

        page_count = max(c.metadata.get("page", 0) for c in chunks) + 1
        entry = {
            "filename": filename,
            "chunk_count": len(chunks),
            "page_count": page_count,
        }
        self._doc_meta.append(entry)
        self._persist()
        return entry

    def search(self, query: str) -> list[dict]:
        """
        Return up to `retrieval_top_k` chunks whose similarity score exceeds
        `retrieval_score_threshold`.

        Each result is a dict:
          {content, source, page, chunk_index, score}
        """
        if self._store is None:
            return []

        results = self._store.similarity_search_with_relevance_scores(
            query, k=settings.retrieval_top_k
        )

        filtered = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0) + 1,  # 1-based for display
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "score": round(score, 4),
            }
            for doc, score in results
            if score >= settings.retrieval_score_threshold
        ]
        # highest score first
        return sorted(filtered, key=lambda x: x["score"], reverse=True)

    def clear(self) -> None:
        """Remove all indexed documents and delete the persisted index."""
        self._store = None
        self._doc_meta = []
        if INDEX_FILE.exists():
            INDEX_FILE.unlink(missing_ok=True)
            # FAISS also writes an index.pkl alongside index.faiss
            pkl = INDEX_DIR / "index.pkl"
            pkl.unlink(missing_ok=True)
        META_FILE.unlink(missing_ok=True)

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_file(self, file_path: str, filename: str) -> list[Document]:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(
                f"Unsupported file type '{ext}'. Only PDF and TXT are accepted."
            )
        return loader.load()

    def _persist(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(INDEX_DIR))
        META_FILE.write_text(json.dumps(self._doc_meta, indent=2), encoding="utf-8")

    def _load_persisted_index(self) -> None:
        if not INDEX_FILE.exists():
            return
        try:
            self._store = FAISS.load_local(
                str(INDEX_DIR),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            if META_FILE.exists():
                self._doc_meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt index — start fresh rather than crash on startup
            self._store = None
            self._doc_meta = []


# Module-level singleton — imported by tools and routers
rag_service = RAGService()
