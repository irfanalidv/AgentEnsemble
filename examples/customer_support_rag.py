"""
Customer-support RAG: answer questions from a help-center FAQ.

Shows:
- Indexing local text with ``RAGTool.index_texts`` (no web fetch).
- A ``ReActAgent`` that calls the RAG tool and answers from the retrieved passages.
- Plugging in any LangChain ``Embeddings``; here a small hashing bag-of-words embedder, so
  retrieval is real (lexical) and needs no API.

Needs: ``pip install "agentensemble[rag]"``. No API key: runs offline with a rule-based
stand-in LLM. With ``OPENAI_API_KEY`` (or ``MISTRAL_API_KEY``) set, a real model reads the
retrieved passages and writes the answer.

Run: python examples/customer_support_rag.py [--offline]
"""

import math
import re
import zlib
from pathlib import Path

from langchain_core.embeddings import Embeddings
from offline_llm import RuleBasedLLM, choose_llm

from agentensemble import RAGTool, ReActAgent

FAQ_PATH = Path(__file__).parent / "data" / "faq.md"
QUESTIONS = [
    "How long does a refund take?",
    "How much does express shipping cost?",
    "My password reset link stopped working. Why?",
]


class HashingEmbeddings(Embeddings):
    """Bag-of-words vectors via the hashing trick: deterministic, offline, good enough for FAQs."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if len(token) > 2:
                token = token.removesuffix("s")  # crude stemming: "refunds" == "refund"
                idx = zlib.crc32(token.encode()) % self.dim
                vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def load_faq(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Split the markdown FAQ into one passage per '## ' section."""
    texts, metas = [], []
    for block in path.read_text().split("## ")[1:]:
        title, _, body = block.partition("\n")
        texts.append(f"{title.strip()}: {body.strip()}")
        metas.append({"section": title.strip()})
    return texts, metas


def first_passage(question: str, tool_results: list[str]) -> str:
    """Offline answer: quote the best-matching FAQ passage."""
    if not tool_results:
        return "I could not find that in the FAQ."
    match = re.search(r"Content: (.+)", tool_results[-1])
    return match.group(1) if match else tool_results[-1]


def main() -> None:
    rag = RAGTool(
        embeddings=HashingEmbeddings(),
        description="Search the help-center FAQ (refunds, shipping, returns, passwords, warranty).",
    )
    texts, metas = load_faq(FAQ_PATH)
    print(f"Indexed {rag.index_texts(texts, metas)} FAQ passages from {FAQ_PATH.name}\n")

    offline = RuleBasedLLM(plan=lambda q, tools: ("rag", {"question": q}), answer=first_passage)
    agent = ReActAgent(name="support", tools=[rag], llm=choose_llm(offline), max_iterations=3)

    for question in QUESTIONS:
        result = agent.run(question)
        print(f"Q: {question}\nA: {result['result']}\n")


if __name__ == "__main__":
    main()
