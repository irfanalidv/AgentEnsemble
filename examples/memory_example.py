"""
Memory / Session Example

Multi-turn conversation with ReActAgent and InMemorySession.
Use case: Chatbots that remember context.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from agentensemble.agents import ReActAgent
from agentensemble.memory import InMemorySession
from agentensemble.tools import SearchTool


async def main():
    session = InMemorySession(session_id="chat-1")
    agent = ReActAgent(
        name="chatbot",
        tools=[SearchTool()],
        session=session,
        max_iterations=3,
    )

    # Turn 1
    print("User: What is the capital of France?")
    r1 = await agent.arun("What is the capital of France?")
    print(f"Agent: {r1['result'][:150]}...")
    print(f"Session messages: {len(session.get_messages())}\n")

    # Turn 2 - agent has context from turn 1
    print("User: What about its population?")
    r2 = await agent.arun("What about its population?")
    print(f"Agent: {r2['result'][:200]}...")
    print(f"Session messages: {len(session.get_messages())}")


if __name__ == "__main__":
    print("\n🎭 Memory / Multi-Turn Example\n")
    asyncio.run(main())
    print("\n✅ Done!")
