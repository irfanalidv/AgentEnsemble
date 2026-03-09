"""
SQLite Session

Persistent session storage using SQLite. No external dependencies beyond stdlib.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class SQLiteSession:
    """
    SQLite-backed session for persistent conversation history.

    Stores messages in a local SQLite database. Survives process restarts.
    """

    def __init__(
        self,
        session_id: str = "default",
        db_path: Optional[str] = None,
    ):
        """
        Initialize SQLite session.

        Args:
            session_id: Unique session identifier
            db_path: Path to SQLite file (default: :memory: or ./agentensemble_sessions.db)
        """
        self.session_id = session_id
        self._db_path = db_path or str(Path.cwd() / "agentensemble_sessions.db")
        self._init_db()

    def _init_db(self) -> None:
        """Create table if not exists."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_call_id TEXT,
                    tool_calls TEXT,
                    PRIMARY KEY (session_id, seq)
                )
                """
            )

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve conversation history. Latest N if limit set."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT role, content, tool_call_id, tool_calls FROM session_messages "
                "WHERE session_id = ? ORDER BY seq",
                (self.session_id,),
            )
            rows = cur.fetchall()
        out = []
        for r in rows:
            d: Dict[str, Any] = {"role": r["role"], "content": r["content"] or ""}
            if r["tool_call_id"]:
                d["tool_call_id"] = r["tool_call_id"]
            if r["tool_calls"]:
                d["tool_calls"] = json.loads(r["tool_calls"])
            out.append(d)
        if limit is not None:
            return out[-limit:]
        return out

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Append messages to history."""
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM session_messages WHERE session_id = ?",
                (self.session_id,),
            )
            start = cur.fetchone()[0] + 1
            for i, m in enumerate(messages):
                tc = json.dumps(m["tool_calls"]) if m.get("tool_calls") else None
                conn.execute(
                    "INSERT INTO session_messages (session_id, seq, role, content, tool_call_id, tool_calls) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        self.session_id,
                        start + i,
                        m.get("role", "user"),
                        m.get("content", ""),
                        m.get("tool_call_id"),
                        tc,
                    ),
                )

    def clear(self) -> None:
        """Clear all messages for this session."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "DELETE FROM session_messages WHERE session_id = ?",
                (self.session_id,),
            )
