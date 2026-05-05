"""
FastAPI application entry point.

Startup sequence:
  1. Validate that required environment variables are set.
  2. Build the LangChain AgentExecutor and attach it to app.state so it is
     shared across requests without being re-created per call.
  3. Register the chat and documents routers.

The AgentExecutor is stateless — it receives conversation history as input on
every call, so a single shared instance is safe under concurrent requests.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import chat, documents
from app.services.agent_service import build_agent_executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build expensive resources once at startup."""
    app.state.agent_executor = build_agent_executor()
    yield
    # Nothing to clean up on shutdown for an in-memory setup


app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "A conversational AI assistant with RAG (document Q&A), "
        "web search, and calculator capabilities.\n\n"
        "Built with FastAPI · LangChain · OpenAI GPT-4o · FAISS · Tavily."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins during development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "model": settings.model_name,
        "embedding_model": settings.embedding_model,
    }


@app.get("/health", tags=["Health"])
async def health():
    from app.services.rag_service import rag_service

    return {
        "status": "ok",
        "indexed_documents": len(rag_service.list_documents()),
    }
