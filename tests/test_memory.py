"""Tests for memory/session."""

import tempfile
from pathlib import Path

import pytest

from agentensemble.memory import InMemorySession, Session, SQLiteSession


class TestInMemorySession:
    """Tests for InMemorySession."""

    def test_add_and_get_messages(self, in_memory_session):
        in_memory_session.add_messages([{"role": "user", "content": "hi"}])
        msgs = in_memory_session.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hi"

    def test_get_messages_with_limit(self, in_memory_session):
        in_memory_session.add_messages([
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
        ])
        msgs = in_memory_session.get_messages(limit=2)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "2"
        assert msgs[1]["content"] == "3"

    def test_clear(self, in_memory_session):
        in_memory_session.add_messages([{"role": "user", "content": "x"}])
        in_memory_session.clear()
        assert len(in_memory_session.get_messages()) == 0

    def test_session_id(self):
        session = InMemorySession(session_id="my-session")
        assert session.session_id == "my-session"


class TestSQLiteSession:
    """Tests for SQLiteSession."""

    def test_add_and_get(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            session = SQLiteSession(session_id="sqlite-test", db_path=path)
            session.add_messages([{"role": "user", "content": "hello"}])
            msgs = session.get_messages()
            assert len(msgs) == 1
            assert msgs[0]["content"] == "hello"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_clear(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            session = SQLiteSession(session_id="sqlite-clear", db_path=path)
            session.add_messages([{"role": "user", "content": "x"}])
            session.clear()
            assert len(session.get_messages()) == 0
        finally:
            Path(path).unlink(missing_ok=True)
