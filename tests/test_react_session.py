"""Tests for ReActAgent session integration."""

import pytest

from agentensemble.agents import ReActAgent
from agentensemble.memory import InMemorySession


class TestReActAgentSession:
    """Tests for ReActAgent with session."""

    def test_session_param_accepted(self):
        session = InMemorySession("test")
        agent = ReActAgent(name="t", tools=[], session=session)
        assert agent.session is session

    def test_session_to_messages_empty(self):
        session = InMemorySession("test")
        agent = ReActAgent(name="t", tools=[], session=session)
        msgs = agent._session_to_messages()
        assert msgs == []

    def test_session_to_messages_with_history(self):
        session = InMemorySession("test")
        session.add_messages([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        agent = ReActAgent(name="t", tools=[], session=session)
        msgs = agent._session_to_messages()
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == "hi"
        assert msgs[1].content == "hello"
