# AgentEnsemble 🎭

**Orchestrate AI agents in perfect harmony**

AgentEnsemble is a **simple, practical Python library** for building and orchestrating AI agents. Perfect for real-world tasks like web search, research, document Q&A, and multi-agent collaboration.

**Key Features:**

- 🚀 **Simple API** - Get started in minutes
- 🔍 **Web Search** - Serper API (with DuckDuckGo fallback)
- 📚 **RAG Support** - Document Q&A with Mistral AI
- 🤝 **Multi-Agent** - Coordinate multiple agents easily
- 🔓 **Open-Source** - Uses LangChain Community Tools

**Powered by Mistral AI** for LLM operations. **Open-source by default** - uses free tools from LangChain Community Tools.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/agentensemble)](https://pypi.org/project/agentensemble/)
[![PyPI Downloads](https://static.pepy.tech/badge/agentensemble)](https://pepy.tech/projects/agentensemble)

## 📦 Installation

```bash
# Basic installation
pip install agentensemble

# With optional features
pip install agentensemble[search]  # Search tools (Serper API + DuckDuckGo)
pip install agentensemble[rag]     # RAG capabilities
pip install agentensemble[all]     # All features
```

### Verify Installation

```bash
python -c "from agentensemble import HybridAgent; print('✅ Installed!')"
```

**Output:**

```
✅ Installed!
```

## ✨ Features

### 🤖 **Reference Agent Implementations**

- **ReAct Agent**: Simple reasoning + acting pattern
- **StateGraph Agent**: Custom nodes with intelligent routing
- **RAG-Enhanced Agent**: Scraping → embedding → retrieval with fallback strategies
- **Hybrid Agent**: Advanced iterative refinement with early stopping
- **Structured Agent**: Returns structured output (Pydantic models, JSON) using [LangChain's structured output](https://docs.langchain.com/oss/python/langchain/structured-output)

### 🎼 **Orchestration Patterns**

- **Supervisor Pattern**: Central coordinator managing specialized agents
- **Swarm Pattern**: Decentralized agent collaboration
- **Pipeline Pattern**: Sequential agent workflows
- **Ensemble Pattern**: Full multi-agent coordination

### 🔧 **Tool Ecosystem** (Open-Source by Default)

- Built-in tools using **LangChain Community Tools**:
  - **SearchTool**: Serper API (default if API key provided) or DuckDuckGo (fallback) ⭐
  - **ScraperTool**: Playwright-based web scraping
  - **RAGTool**: Document loaders + vector stores (ChromaDB)
  - **ValidationTool**: Quality assurance
- **No paid APIs required** - All tools work with free, open-source options
- Tool registry for dynamic tool management
- Direct integration with `langchain_community.tools`
- Custom tool creation framework

### 📊 **Testing & Comparison Framework**

- Benchmark suite for agent evaluation
- Multi-agent comparison engine
- Performance metrics: success rate, cost, execution time
- Interactive dashboards and reports

### 👁️ **Observability**

- Token usage and cost tracking
- LangSmith/OpenTelemetry integration
- Structured logging
- Agent execution tracing

## 🚀 Quick Start

### Environment Setup

Create a `.env` file for API keys:

```bash
# Required for search (default provider)
SERPER_API_KEY=your-serper-api-key-here

# Required for RAG/LLM features
MISTRAL_API_KEY=your-mistral-api-key-here
```

**Note:** SearchTool defaults to Serper API if `SERPER_API_KEY` is provided, otherwise falls back to DuckDuckGo (free, no API key needed).

**Note**:

- **Open-source by default** - Uses free tools from [LangChain Community Tools](https://docs.langchain.com/oss/python/integrations/tools/)
- Uses **Mistral AI** for all LLM operations (chat models and embeddings)
- Tools follow [LangChain RAG patterns](https://docs.langchain.com/oss/python/langchain/rag):
  - `SerpAPIQueryRun` for search (default) or `DuckDuckGoSearchRun` (fallback) ⭐
  - `WebBaseLoader` + `RecursiveCharacterTextSplitter` for RAG indexing
  - `MistralAIEmbeddings` for embeddings
  - `InMemoryVectorStore` / `Chroma` for vector storage
  - `@tool` decorator for agentic RAG tools

### Structured Output

**Use Case:** Extract structured product review data from unstructured text

```python
from pydantic import BaseModel, Field
from agentensemble.agents import StructuredAgent

class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product")
    rating: int = Field(description="Rating out of 5")
    pros: list[str] = Field(description="List of positive aspects")
    cons: list[str] = Field(description="List of negative aspects")
    summary: str = Field(description="Overall summary of the review")

agent = StructuredAgent(response_format=ProductReview)

review_text = """
I recently purchased the iPhone 15 Pro Max and here's my honest review.
Rating: 4 out of 5 stars
Pros: Excellent camera quality, Fast A17 Pro chip, Great battery life, Premium build quality
Cons: Very expensive, Heavy and bulky, Limited storage on base model
Summary: Great phone with top-tier features, but the high price and weight might not be for everyone.
"""

result = agent.run(review_text)
print(result['structured_response'])
```

**Actual Output:**

```python
ProductReview(
    product_name='iPhone 15 Pro Max',
    rating=4,
    pros=['Excellent camera quality', 'Fast A17 Pro chip', 'Great battery life', 'Premium build quality'],
    cons=['Very expensive', 'Heavy and bulky', 'Limited storage on base model'],
    summary='Great phone with top-tier features, but the high price and weight might not be for everyone.'
)
```

## 📚 Documentation

- [Examples](examples/) - See `examples/` directory for usage examples

## 🏗️ Architecture

```
agentensemble/
├── agents/              # Reference agent implementations
├── orchestration/      # Orchestration patterns
├── tools/              # Tool ecosystem
├── testing/            # Testing & comparison framework
├── state/              # State management
└── observability/      # Monitoring & tracking
```

## 🔍 Examples

### Example 1: Search Tool

**Use Case:** Research latest breakthroughs in quantum computing

```python
from agentensemble.tools import SearchTool

search = SearchTool()  # Uses Serper API (or DuckDuckGo fallback)
result = search.run("What are the latest breakthroughs in quantum computing in 2024?")
print(result)
```

**Actual Output:**

```
1. Increased Qubit Stability and Error Correction · 2. Quantum Supremacy Milestones · 3. Advancements in Quantum Algorithms · 4. Commercial Quantum ... Explore the top quantum research stories of 2024, from advancements in quantum chemistry to developments in quantum AI. Google has developed a new quantum chip called Willow, which significantly reduces errors as it scales up, a major breakthrough in quantum error ... Error correction, a critical element of quantum control, emerged as a key innovation...
```

### Example 2: ReAct Agent

**Use Case:** Research AI applications in healthcare

```python
from agentensemble.agents import ReActAgent
from agentensemble.tools import SearchTool

agent = ReActAgent(name="research_agent", tools=[SearchTool()], max_iterations=3)
result = agent.run("What are the most promising applications of AI agents in healthcare in 2024?")
print(result)
```

**Actual Output:**

```python
{
    'result': 'AI agents can aid clinicians in providing more accurate diagnoses by analyzing medical data—including lab results, digital scans, patient ... AI agents in healthcare are already helping with diagnostics, managing schedules, monitoring patients, handling documentation, and more. 10 strategic healthcare AI agent use cases · 1. Intelligent prior authorization assistant · 2. Chart-gap tracker · 3. Charge-edit auto-review agent. AI agents are reshaping healthcare in 2025, automating paperwork, enabling round-the-clock support, and optimizing clinical processes.',
    'metadata': {
        'iterations': 1,
        'tool_calls': 1,
        'agent': 'research_agent'
    }
}
```

### Example 3: Hybrid Agent

**Use Case:** Research autonomous vehicle technology trends and challenges

```python
from agentensemble import HybridAgent
from agentensemble.tools import SearchTool

agent = HybridAgent(name="hybrid_research", tools=[SearchTool()], max_iterations=5)
result = agent.run("What are the key trends and challenges in autonomous vehicle technology in 2024?")
print(result)
```

**Actual Output:**

```python
{
    'result': 'The global autonomous vehicle (AV) market surpassed $41 billion in 2024 and is expected to reach nearly $115 billion by 2029 (Statista). April 2024 saw Tesla integrate its vision-based occupancy network, replacing ultrasonic sensors, enhancing safety and autonomy. Top 5 Technical Challenges in Autonomous Vehicle Development & Possible Solutions · Challenge 1: Safety Assurance, Liability, and Cybersecurity. Level 3 autonomy poses a major liability shift from driver to automaker...',
    'metadata': {
        'iterations': 3,
        'actions_taken': [],
        'agent': 'hybrid_research'
    }
}
```

### Example 4: Structured Output

**Use Case:** Extract structured research summary from unstructured text

```python
from pydantic import BaseModel, Field
from agentensemble.agents import StructuredAgent

class ResearchSummary(BaseModel):
    topic: str = Field(description="Main research topic")
    key_findings: list[str] = Field(description="List of key findings")
    impact: str = Field(description="Potential impact or significance")
    sources_count: int = Field(description="Number of sources referenced")

agent = StructuredAgent(
    name="research_summarizer",
    response_format=ResearchSummary
)

result = agent.run("""
Topic: AI Agents in Financial Services
Key Findings:
1. AI agents are automating 60% of routine financial analysis tasks
2. Fraud detection accuracy improved by 45% with agent-based systems
3. Customer service response time reduced by 70%
Sources: 15 research papers and industry reports
Impact: Transformative - reshaping how financial institutions operate
""")

print(result['structured_response'])
```

**Actual Output:**

```python
ResearchSummary(
    topic='AI Agents in Financial Services',
    key_findings=[
        'AI agents are automating 60% of routine financial analysis tasks.',
        'Fraud detection accuracy improved by 45% with agent-based systems.',
        'Customer service response time reduced by 70%.'
    ],
    impact='Transformative - reshaping how financial institutions operate by enhancing efficiency, accuracy, and customer satisfaction while reducing operational costs.',
    sources_count=15
)
```

### Example 5: Multi-Agent Collaboration

**Use Case:** Research and validate information using multiple specialized agents

```python
from agentensemble import Ensemble, ReActAgent
from agentensemble.tools import SearchTool

researcher = ReActAgent(name="researcher", tools=[SearchTool()], max_iterations=2)
validator = ReActAgent(name="validator", tools=[SearchTool()], max_iterations=2)

ensemble = Ensemble(
    conductor="supervisor",
    agents={"researcher": researcher, "validator": validator}
)

result = ensemble.perform(
    task="Research and validate research methodology",
    data={"topic": "research methodology"}
)

print(result)
```

**Actual Output:**

```python
{
    'results': {
        'researcher': {
            'result': 'by P Ranganathan · 2024 · Cited by 89 — In this article, we discuss the methods of determining the validity and reliability of a research questionnaire. by D Sreekumar · Cited by 41 — Research methodology is a structured and scientific approach used to collect, analyze, and interpret quantitative or qualitative data...',
            'metadata': {'iterations': 1, 'tool_calls': 1, 'agent': 'researcher'}
        },
        'validator': {
            'result': 'by P Ranganathan · 2024 · Cited by 89 — In this article, we discuss the methods of determining the validity and reliability of a research questionnaire. Reliability and validity are concepts used to evaluate the quality of research...',
            'metadata': {'iterations': 1, 'tool_calls': 1, 'agent': 'validator'}
        }
    },
    'conductor': 'supervisor',
    'agents_used': ['researcher', 'validator']
}
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Inspired by production-grade agentic AI architectures and best practices from the AI agent community.

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**AgentEnsemble** - Where agents work in concert 🎼
