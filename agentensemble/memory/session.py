"""
Session Protocol & Implementations

Stores conversation history for multi-turn agent interactions.
Research: Memoria, VoltAgent, Redis Agent Memory - 40-60% token savings.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Session(Protocol):
    """
    Protocol for session implementations.

    Stores conversation history so agents maintain context across turns.
    """

    session_id: str

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve conversation history. Latest N if limit set."""
        ...

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Append messages to history."""
        ...

    def clear(self) -> None:
        """Clear all messages."""
        ...


class InMemorySession:
    """
    In-memory session for development and testing.

    Not persistent across process restarts.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._messages: List[Dict[str, Any]] = []

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if limit is None:
            return list(self._messages)
        return list(self._messages[-limit:])

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        self._messages.extend(messages)

    def clear(self) -> None:
        self._messages.clear()
