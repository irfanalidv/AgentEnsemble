"""
AgentEnsemble Quickstart - Real-World Use Cases

Simple, practical examples for common agent tasks.
"""

from agentensemble import HybridAgent, ReActAgent
from agentensemble.tools import SearchTool
from agentensemble.orchestration import Ensemble


def example_web_search():
    """Example 1: Simple web search with an agent"""
    print("=" * 60)
    print("Example 1: Web Search Agent")
    print("=" * 60)
    
    # Create search tool (free, no API key needed)
    search = SearchTool()
    
    # Create agent
    agent = ReActAgent(
        name="search_agent",
        tools=[search],
        max_iterations=3
    )
    
    # Search for information
    result = agent.run("What is Python programming?")
    print(f"Query: 'What is Python programming?'")
    print(f"Result: {result['result'][:200]}...")
    print(f"Iterations: {result['metadata']['iterations']}")
    print()


def example_research_agent():
    """Example 2: Research agent for finding information"""
    print("=" * 60)
    print("Example 2: Research Agent")
    print("=" * 60)
    
    # Create hybrid agent with search
    agent = HybridAgent(
        name="research_agent",
        tools=[SearchTool()],
        max_iterations=5
    )
    
    # Research a topic
    result = agent.run("Latest developments in AI agents")
    print(f"Query: 'Latest developments in AI agents'")
    print(f"Result: {result['result'][:200]}...")
    print(f"Metadata: {result['metadata']}")
    print()


def example_multi_agent_research():
    """Example 3: Multi-agent research team"""
    print("=" * 60)
    print("Example 3: Multi-Agent Research Team")
    print("=" * 60)
    
    # Create specialized agents
    researcher = ReActAgent(
        name="researcher",
        tools=[SearchTool()],
        max_iterations=3
    )
    
    validator = ReActAgent(
        name="validator",
        tools=[SearchTool()],
        max_iterations=2
    )
    
    # Create ensemble
    ensemble = Ensemble(
        conductor="supervisor",
        agents={
            "researcher": researcher,
            "validator": validator
        }
    )
    
    # Execute research task
    result = ensemble.perform(
        task="Research Python programming language",
        data={"topic": "Python"}
    )
    
    print(f"Task: 'Research Python programming language'")
    print(f"Agents used: {result.get('agents_used', [])}")
    print(f"Results from {len(result.get('results', {}))} agents")
    print()


if __name__ == "__main__":
    print("\n🎭 AgentEnsemble - Real-World Examples\n")
    
    example_web_search()
    example_research_agent()
    example_multi_agent_research()
    
    print("=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)
    print("\n💡 Real-world use cases:")
    print("  • Web search and information retrieval")
    print("  • Research and fact-finding")
    print("  • Multi-agent collaboration")
    print("  • Task automation with AI agents")
