"""
Streaming Example

Real-time event streaming for agent execution.
Use case: Show "Thinking...", "Searching...", etc. in UI.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from agentensemble.agents import ReActAgent
from agentensemble.tools import SearchTool


async def main():
    agent = ReActAgent(
        name="streaming_agent",
        tools=[SearchTool()],
        max_iterations=3,
    )

    print("Streaming events:\n")
    async for event in agent.astream("What is 2 + 2?"):
        t = event.get("type", "")
        if t == "thinking":
            print("  💭 Thinking...")
        elif t == "tool_start":
            print(f"  🔧 Calling {event.get('name', '?')}...")
        elif t == "tool_end":
            r = event.get("result", "")[:80]
            print(f"  ✓ Result: {r}...")
        elif t == "done":
            print(f"\n  ✅ Done: {event.get('result', '')[:200]}")
            print(f"  Metadata: {event.get('metadata', {})}")


if __name__ == "__main__":
    print("\n🎭 Streaming Example\n")
    asyncio.run(main())
    print("\n✅ Done!")
