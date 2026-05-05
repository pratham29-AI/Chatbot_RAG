"""
Session Service — in-memory conversation history management.

Each session stores an ordered list of LangChain BaseMessage objects
(HumanMessage / AIMessage).  The service generates UUIDs for new sessions,
enforces a maximum message cap (to avoid runaway token costs), and exposes
helpers that the agent service and routers consume.

Design decision: in-memory store
  Chosen for simplicity in this assessment context.  For production, swap the
  dict for Redis or a database — only `_store` needs to change; callers are
  unaffected.
"""

import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import settings


class Session:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages: list[BaseMessage] = []
        self.created_at: datetime = datetime.now(timezone.utc)
        self.last_active: datetime = datetime.now(timezone.utc)

    def add_user_message(self, content: str) -> None:
        self.messages.append(HumanMessage(content=content))
        self._trim()
        self.last_active = datetime.now(timezone.utc)

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content=content))
        self._trim()
        self.last_active = datetime.now(timezone.utc)

    def _trim(self) -> None:
        """Keep only the most recent N messages to cap token usage."""
        cap = settings.max_session_messages
        if len(self.messages) > cap:
            # Always preserve the oldest context; drop from the middle if needed.
            # Simpler: just drop the oldest pairs.
            self.messages = self.messages[-cap:]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }


class SessionService:
    def __init__(self) -> None:
        self._store: dict[str, Session] = {}

    def create_session(self) -> Session:
        sid = str(uuid.uuid4())
        session = Session(sid)
        self._store[sid] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._store.get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._store:
            session = Session(session_id)
            self._store[session_id] = session
        return self._store[session_id]

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    def list_sessions(self) -> list[dict]:
        return [s.to_dict() for s in self._store.values()]


# Module-level singleton
session_service = SessionService()
