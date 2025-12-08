"""
AgentEnsemble RAG Example - Document Q&A

Real-world use case: Answer questions from documents using RAG.
"""

from agentensemble import HybridAgent
from agentensemble.tools import RAGTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_document_qa():
    """Example: Question answering from documents"""
    print("=" * 60)
    print("Example: Document Question Answering with RAG")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("MISTRAL_API_KEY"):
        print("⚠️  MISTRAL_API_KEY not found in .env")
        print("   RAG features require Mistral AI API key")
        print("   Create .env file with: MISTRAL_API_KEY=your-key")
        return
    
    # Initialize RAG tool
    rag_tool = RAGTool()
    
    # Index documents (e.g., from URLs)
    print("\n1. Indexing documents...")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ]
    rag_tool.index_documents(urls)
    print("✅ Documents indexed!")
    
    # Create agent with RAG tool
    agent = HybridAgent(
        name="rag_agent",
        tools=[rag_tool],
        max_iterations=5
    )
    
    # Ask questions
    print("\n2. Asking questions...")
    questions = [
        "What is task decomposition?",
        "How do agents use tools?",
    ]
    
    for question in questions:
        print(f"\n   Q: {question}")
        result = agent.run(question)
        print(f"   A: {result['result'][:150]}...")
    
    print()


if __name__ == "__main__":
    print("\n🎭 AgentEnsemble RAG Example\n")
    print("Use case: Answer questions from indexed documents\n")
    
    example_document_qa()
    
    print("=" * 60)
    print("✅ RAG example completed!")
    print("=" * 60)
    print("\n💡 Real-world use cases:")
    print("  • Document Q&A systems")
    print("  • Knowledge base search")
    print("  • Technical documentation search")
    print("  • Research paper analysis")
