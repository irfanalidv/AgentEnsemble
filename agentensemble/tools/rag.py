"""
RAG Tool

Retrieval-Augmented Generation tool following LangChain RAG patterns.
Based on: https://docs.langchain.com/oss/python/langchain/rag

Supports:
- Agentic RAG: Tool-based retrieval for iterative queries
- RAG Chain: Two-step chain with single LLM call
"""

from typing import Any, List, Optional, Tuple

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_mistralai import MistralAIEmbeddings
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.tools import tool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    WebBaseLoader = None
    Chroma = None
    InMemoryVectorStore = None
    MistralAIEmbeddings = None
    Document = None
    RecursiveCharacterTextSplitter = None
    tool = None


class RAGTool:
    """
    RAG tool following LangChain RAG patterns.

    Implements both:
    1. Agentic RAG: Tool-based retrieval (as per LangChain tutorial)
    2. RAG Chain: Two-step chain with single LLM call

    Based on: https://docs.langchain.com/oss/python/langchain/rag
    """

    def __init__(
        self,
        vector_store: Optional[Any] = None,
        embeddings: Optional[Any] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        fallback_strategies: int = 3,
        **kwargs,
    ):
        """
        Initialize RAG tool.

        Args:
            vector_store: Pre-initialized vector store (or None to create one)
            embeddings: Pre-initialized embeddings model (or None to use Mistral)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            fallback_strategies: Number of fallback query variations
            **kwargs: Additional configuration
        """
        self.name = "rag"
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fallback_strategies = fallback_strategies
        self.config = kwargs
        self._text_splitter: Optional[Any] = None

    def _initialize_components(self) -> None:
        """Initialize text splitter and ensure embeddings/vector store exist"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-community, langchain-mistralai, and langchain-text-splitters "
                "are required for RAGTool. Install with: "
                "pip install langchain-community langchain-mistralai langchain-text-splitters"
            )

        # Initialize text splitter (as per LangChain RAG tutorial)
        if not self._text_splitter and RecursiveCharacterTextSplitter:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

        # Initialize embeddings if not provided (using Mistral)
        if not self.embeddings and MistralAIEmbeddings:
            from agentensemble.utils.llm import get_mistral_embeddings

            self.embeddings = get_mistral_embeddings()

        # Initialize vector store if not provided
        if not self.vector_store:
            if InMemoryVectorStore and self.embeddings:
                self.vector_store = InMemoryVectorStore(self.embeddings)

    def index_documents(self, urls: List[str]) -> None:
        """
        Index documents from URLs (following LangChain RAG tutorial pattern).

        This implements the indexing pipeline:
        1. Load documents with WebBaseLoader
        2. Split with RecursiveCharacterTextSplitter
        3. Store in vector store

        Args:
            urls: List of URLs to load and index
        """
        if not LANGCHAIN_AVAILABLE or not WebBaseLoader:
            raise ImportError("langchain-community required for indexing")

        self._initialize_components()

        # Load documents (as per LangChain tutorial)
        documents = []
        for url in urls:
            try:
                loader = WebBaseLoader(web_paths=(url,))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Warning: Failed to load {url}: {e}")

        if not documents:
            return

        # Split documents (as per LangChain tutorial)
        if self._text_splitter:
            all_splits = self._text_splitter.split_documents(documents)
        else:
            all_splits = documents

        # Store in vector store (as per LangChain tutorial)
        if self.vector_store:
            self.vector_store.add_documents(documents=all_splits)

    def create_retrieval_tool(self):
        """
        Create a LangChain tool for retrieval (following LangChain RAG tutorial).

        Returns a tool decorated with @tool that can be used with create_agent.

        Returns:
            LangChain tool instance
        """
        if not LANGCHAIN_AVAILABLE or not tool:
            raise ImportError("langchain required for tool creation")

        if not self.vector_store:
            raise ValueError("Vector store must be initialized. Call index_documents() first.")

        # Create retrieval tool following LangChain RAG tutorial pattern
        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information to help answer a query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        return retrieve_context

    def retrieve(self, query: str, k: int = 2) -> Tuple[str, List[Document]]:
        """
        Retrieve context for a query (following LangChain RAG tutorial pattern).

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            Tuple of (serialized context, retrieved documents)
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call index_documents() first.")

        retrieved_docs = self.vector_store.similarity_search(query, k=k)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def run(self, question: str, urls: Optional[List[str]] = None, **kwargs) -> str:
        """
        Execute RAG query (simple interface for backward compatibility).

        For full LangChain RAG patterns, use:
        - create_retrieval_tool() for agentic RAG
        - retrieve() for direct retrieval

        Args:
            question: Question to answer
            urls: Optional URLs to index first
            **kwargs: Additional parameters

        Returns:
            Retrieved context
        """
        if not LANGCHAIN_AVAILABLE:
            return f"RAG answer to '{question}' [langchain-community required]"

        # If URLs provided, index them first
        if urls:
            self.index_documents(urls)

        # Retrieve context
        if self.vector_store:
            context, docs = self.retrieve(question, k=3)
            return context

        return f"RAG answer to '{question}' [index documents first with index_documents()]"

    def __call__(self, question: str, urls: Optional[List[str]] = None, **kwargs) -> str:
        """Make tool callable"""
        return self.run(question, urls, **kwargs)
