"""
ReAct Agent with LLM-Powered Tool Selection

Demonstrates:
- LLM decides when to use tools vs. answer directly
- Native function calling (no THOUGHT/ACTION text parsing)
- Sync run() and async arun()
- @function_tool for custom tools

Usage:
    PYTHONPATH=. python examples/react_llm_example.py
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from agentensemble.agents import ReActAgent
from agentensemble.tools import SearchTool, function_tool


@function_tool(description="Get current weather for a city")
def get_weather(city: str) -> str:
    """Fetch weather data."""
    return f"Weather in {city}: 72°F, sunny"


async def main():
    agent = ReActAgent(
        name="research_agent",
        tools=[SearchTool(), get_weather],
        max_iterations=5,
    )

    # Query that needs search
    print("=== Query requiring search ===")
    result = await agent.arun("Who won the 2024 Nobel Prize in Physics?")
    print(result["result"][:400] + "..." if len(result["result"]) > 400 else result["result"])
    print(f"Metadata: {result['metadata']}\n")

    # Query that doesn't need search (LLM answers directly)
    print("=== Query requiring no search ===")
    result = await agent.arun("What is 2 + 2?")
    print(result["result"])
    print(f"Metadata: {result['metadata']}\n")

    # Capital of France
    print("=== Capital of France ===")
    result = await agent.arun("What is the capital of France?")
    print(result["result"])
    print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    asyncio.run(main())
