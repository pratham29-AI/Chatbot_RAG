"""
Chat Router — streaming chat endpoint and session management.

Endpoints
─────────
  POST /chat/sessions              → create a new session
  GET  /chat/sessions              → list all active sessions
  GET  /chat/sessions/{id}         → get session info + message history
  DELETE /chat/sessions/{id}       → clear and delete a session
  POST /chat/sessions/{id}/message → send a message, stream the response (SSE)

Streaming format (Server-Sent Events)
──────────────────────────────────────
Each token is sent as:

    data: {"type": "token", "content": "Hello"}\n\n

End of stream:

    data: {"type": "done"}\n\n

Error:

    data: {"type": "error", "content": "..."}\n\n

The SSE format is the web standard for server-push streams and is natively
supported by browsers (EventSource) and easy to consume in curl / Python.
"""

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.agent_service import stream_agent_response
from app.services.session_service import session_service

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── request / response schemas ────────────────────────────────────────────────

class MessageRequest(BaseModel):
    message: str

    model_config = {"json_schema_extra": {"example": {"message": "What is the main topic of the uploaded document?"}}}


# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


# ── session management endpoints ──────────────────────────────────────────────

@router.post("/sessions", summary="Create a new chat session")
async def create_session():
    """
    Creates a new isolated conversation session.
    Returns the `session_id` that must be passed to the message endpoint.
    """
    session = session_service.create_session()
    return {"session_id": session.session_id, "message": "Session created."}


@router.get("/sessions", summary="List all active sessions")
async def list_sessions():
    return {"sessions": session_service.list_sessions()}


@router.get("/sessions/{session_id}", summary="Get session details")
async def get_session(session_id: str):
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    history = [
        {"role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant",
         "content": msg.content}
        for msg in session.messages
    ]
    return {**session.to_dict(), "history": history}


@router.delete("/sessions/{session_id}", summary="Delete a session")
async def delete_session(session_id: str):
    deleted = session_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": f"Session {session_id} deleted."}


# ── main chat endpoint ────────────────────────────────────────────────────────

@router.post(
    "/sessions/{session_id}/message",
    summary="Send a message and receive a streaming response",
    response_description="Server-Sent Events stream of tokens",
)
async def send_message(session_id: str, body: MessageRequest, request: Request):
    """
    Send a user message to the agent and receive a streaming response.

    The agent will autonomously decide whether to:
    - Search the uploaded documents (RAG via FAISS)
    - Search the web (Tavily)
    - Use the calculator
    - Answer directly from conversation context

    The full conversation history for this session is automatically
    included so the agent can reference earlier turns.

    **Response format:** `text/event-stream` (Server-Sent Events)
    """
    # Retrieve or lazily create the session (allows client-supplied IDs)
    session = session_service.get_or_create(session_id)

    executor = request.app.state.agent_executor

    async def event_generator():
        full_response = ""
        try:
            async for token in stream_agent_response(
                executor=executor,
                user_message=body.message,
                chat_history=session.messages,
            ):
                # Detect the sentinel that carries the assembled final answer
                if token.startswith('{"__final__":'):
                    full_response = json.loads(token)["__final__"]
                    break
                yield _sse({"type": "token", "content": token})

            # Persist the exchange AFTER streaming completes
            session.add_user_message(body.message)
            session.add_ai_message(full_response)

            yield _sse({"type": "done"})

        except Exception as exc:
            yield _sse({"type": "error", "content": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering if behind a proxy
        },
    )
