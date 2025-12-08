"""
AgentEnsemble Structured Output Example

Demonstrates structured output using LangChain's structured output feature.
Based on: https://docs.langchain.com/oss/python/langchain/structured-output
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional

# Check if structured output is available
try:
    from agentensemble.agents import StructuredAgent
    from agentensemble.tools import SearchTool
    from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
    STRUCTURED_AVAILABLE = True
except ImportError:
    STRUCTURED_AVAILABLE = False
    print("⚠️  Structured output requires langchain>=1.1")
    print("   Install with: pip install 'langchain>=1.1'")


def example_structured_output_pydantic():
    """Example: Using Pydantic model for structured output"""
    if not STRUCTURED_AVAILABLE:
        print("Skipping: Structured output not available")
        return
    
    print("=" * 70)
    print("Example 1: Structured Output with Pydantic Model")
    print("=" * 70)
    
    # Define structured output schema
    class ContactInfo(BaseModel):
        """Contact information for a person."""
        name: str = Field(description="The name of the person")
        email: str = Field(description="The email address of the person")
        phone: str = Field(description="The phone number of the person")
    
    # Create agent with structured output
    agent = StructuredAgent(
        name="contact_extractor",
        tools=[SearchTool()] if STRUCTURED_AVAILABLE else [],
        response_format=ContactInfo  # Uses available method (create_agent or with_structured_output)
    )
    
    # Extract structured data
    result = agent.run(
        "Extract contact info from: John Doe, [email protected], (555) 123-4567"
    )
    
    print(f"\nNatural language result: {result['result']}")
    print(f"\nStructured response:")
    print(result['structured_response'])
    print()


def example_structured_output_tool_strategy():
    """Example: Using ToolStrategy for structured output"""
    if not STRUCTURED_AVAILABLE:
        print("Skipping: Structured output not available")
        return
    
    print("=" * 70)
    print("Example 2: Structured Output with ToolStrategy")
    print("=" * 70)
    
    # Define structured output schema
    class ProductReview(BaseModel):
        """Analysis of a product review."""
        rating: Optional[int] = Field(
            description="The rating of the product", 
            ge=1, 
            le=5,
            default=None
        )
        sentiment: Literal["positive", "negative", "neutral"] = Field(
            description="The sentiment of the review"
        )
        key_points: list[str] = Field(
            description="The key points of the review. Lowercase, 1-3 words each."
        )
    
    if not TOOL_STRATEGY_AVAILABLE:
        print("⚠️  ToolStrategy requires langchain>=1.1")
        print("   Using basic Pydantic model instead")
        response_format = ProductReview
    else:
        response_format = ToolStrategy(ProductReview)
    
    # Create agent with ToolStrategy (or basic Pydantic model)
    agent = StructuredAgent(
        name="review_analyzer",
        tools=[SearchTool()],
        response_format=response_format
    )
    
    # Analyze review
    result = agent.run(
        "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"
    )
    
    print(f"\nNatural language result: {result['result'][:200]}...")
    print(f"\nStructured response:")
    print(f"  Rating: {result['structured_response'].rating}")
    print(f"  Sentiment: {result['structured_response'].sentiment}")
    print(f"  Key points: {result['structured_response'].key_points}")
    print()


def example_structured_output_with_search():
    """Example: Structured output with search tool"""
    if not STRUCTURED_AVAILABLE:
        print("Skipping: Structured output not available")
        return
    
    print("=" * 70)
    print("Example 3: Structured Output with Search Tool")
    print("=" * 70)
    
    # Define structured output schema
    class ResearchSummary(BaseModel):
        """Summary of research findings."""
        topic: str = Field(description="The main topic researched")
        key_findings: list[str] = Field(description="Key findings from the research")
        sources_count: int = Field(description="Number of sources found")
        confidence: Literal["low", "medium", "high"] = Field(
            description="Confidence level in the findings"
        )
    
    # Create agent with structured output and search
    agent = StructuredAgent(
        name="researcher",
        tools=[SearchTool()],
        response_format=ResearchSummary,
        system_prompt="You are a research assistant. Use search tools to find information and provide structured summaries."
    )
    
    # Research topic
    result = agent.run(
        "Research the latest developments in AI agents in 2024"
    )
    
    print(f"\nNatural language result: {result['result'][:200]}...")
    if result.get('structured_response'):
        print(f"\nStructured response:")
        print(f"  Topic: {result['structured_response'].topic}")
        print(f"  Key findings: {result['structured_response'].key_findings}")
        print(f"  Sources: {result['structured_response'].sources_count}")
        print(f"  Confidence: {result['structured_response'].confidence}")
    print()


if __name__ == "__main__":
    print("\n🎭 AgentEnsemble Structured Output Examples\n")
    print("Based on: https://docs.langchain.com/oss/python/langchain/structured-output\n")
    
    example_structured_output_pydantic()
    example_structured_output_tool_strategy()
    example_structured_output_with_search()
    
    print("=" * 70)
    print("✅ Structured output examples completed!")
    print("=" * 70)
    print("\n💡 Key features:")
    print("  • Automatic schema validation")
    print("  • Provider-native or tool-calling strategies")
    print("  • Error handling and retry mechanisms")
    print("  • Works with all LangChain-supported models")

