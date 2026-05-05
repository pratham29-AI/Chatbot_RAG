"""
Documents Router — file upload and index management.

Endpoints
─────────
  POST   /documents/upload  → upload a PDF or TXT file, index into FAISS
  GET    /documents         → list all indexed documents
  DELETE /documents         → clear the entire FAISS index
"""

import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.rag_service import rag_service

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_MB = 50


@router.post("/upload", summary="Upload and index a PDF or TXT file")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or plain-text file to be indexed into the FAISS vector store.

    The file is:
      1. Validated (type + size).
      2. Saved to a temporary path so LangChain document loaders can read it.
      3. Split into overlapping chunks (configured via CHUNK_SIZE / CHUNK_OVERLAP).
      4. Embedded with OpenAI `text-embedding-3-small`.
      5. Indexed into FAISS and persisted to disk.

    Once indexed, all subsequent `/chat` queries will include this document
    as a retrieval source.
    """
    # ── validate file type ────────────────────────────────────────────────────
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Only PDF and TXT are accepted.",
        )

    # ── read and validate size ────────────────────────────────────────────────
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── write to temp file so loaders can read it ─────────────────────────────
    # PyPDFLoader and TextLoader both require a file path, not a byte stream.
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = rag_service.add_document(file_path=tmp_path, filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        os.unlink(tmp_path)  # always clean up the temp file

    return {
        "message": f"'{filename}' indexed successfully.",
        "filename": result["filename"],
        "chunk_count": result["chunk_count"],
        "page_count": result["page_count"],
    }


@router.get("", summary="List all indexed documents")
async def list_documents():
    """Return metadata for every document currently in the FAISS index."""
    docs = rag_service.list_documents()
    return {
        "document_count": len(docs),
        "documents": docs,
    }


@router.delete("", summary="Clear the entire document index")
async def clear_documents():
    """
    Remove all documents from the FAISS index and delete the persisted index
    files from disk.  This action is irreversible — documents must be
    re-uploaded to restore the index.
    """
    rag_service.clear()
    return {"message": "All documents have been removed from the index."}
