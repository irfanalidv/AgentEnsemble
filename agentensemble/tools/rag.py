"""
RAG Tool

Retrieval-Augmented Generation tool following LangChain RAG patterns.
Based on: https://docs.langchain.com/oss/python/langchain/rag

Supports:
- Agentic RAG: Tool-based retrieval for iterative queries
- RAG Chain: Two-step chain with single LLM call
"""

import logging
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_core.tools import tool
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_mistralai import MistralAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    WebBaseLoader = None  # type: ignore[assignment,misc]
    Chroma = None  # type: ignore[assignment,misc]
    InMemoryVectorStore = None  # type: ignore[assignment,misc]
    MistralAIEmbeddings = None  # type: ignore[assignment,misc]
    Document = None  # type: ignore[assignment,misc]
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment,misc]
    tool = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

#: Default per-request timeout (seconds) for fetching URLs in ``index_documents``.
DEFAULT_FETCH_TIMEOUT = 30.0


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
        vector_store: Any | None = None,
        embeddings: Any | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        fallback_strategies: int = 3,
        fetch_timeout: float = DEFAULT_FETCH_TIMEOUT,
        **kwargs: Any,
    ):
        """
        Initialize RAG tool.

        Args:
            vector_store: Pre-initialized vector store (or None to create one)
            embeddings: Pre-initialized embeddings model (or None to use Mistral)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            fallback_strategies: Number of fallback query variations
            fetch_timeout: Per-request timeout in seconds when loading URLs
            **kwargs: Additional configuration
        """
        self.name = "rag"
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fallback_strategies = fallback_strategies
        self.fetch_timeout = fetch_timeout
        self.config = kwargs
        self._text_splitter: Any | None = None

    def _initialize_components(self) -> None:
        """Initialize text splitter and ensure embeddings/vector store exist"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-community, langchain-mistralai, and langchain-text-splitters "
                "are required for RAGTool. Install with: "
                "pip install langchain-community langchain-mistralai langchain-text-splitters"
            )

        # Initialize text splitter (as per LangChain RAG tutorial)
        if not self._text_splitter and RecursiveCharacterTextSplitter is not None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

        # Initialize embeddings if not provided (OpenAI or Mistral, picked from the environment)
        if not self.embeddings:
            from agentensemble.utils.llm import get_default_embeddings

            self.embeddings = get_default_embeddings()

        # Initialize vector store if not provided
        if not self.vector_store and InMemoryVectorStore is not None and self.embeddings:
            self.vector_store = InMemoryVectorStore(self.embeddings)

    def index_documents(self, urls: list[str]) -> None:
        """
        Index documents from URLs (following LangChain RAG tutorial pattern).

        This implements the indexing pipeline:
        1. Load documents with WebBaseLoader
        2. Split with RecursiveCharacterTextSplitter
        3. Store in vector store

        Args:
            urls: List of URLs to load and index
        """
        if not LANGCHAIN_AVAILABLE or WebBaseLoader is None:
            raise ImportError("langchain-community required for indexing")

        self._initialize_components()

        # Load documents (as per LangChain tutorial)
        documents = []
        for url in urls:
            try:
                loader = WebBaseLoader(
                    web_paths=(url,),
                    requests_kwargs={"timeout": self.fetch_timeout},
                    raise_for_status=True,
                )
                documents.extend(loader.load())
            except Exception as e:  # one bad URL should not abort the whole batch
                logger.warning("Failed to load %s: %s", url, e)

        self._add_documents(documents)

    def index_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """
        Index raw text (FAQ entries, local files, database rows) without any network fetch.

        Args:
            texts: Text passages to index
            metadatas: Optional metadata per passage (e.g. ``{"source": "faq.md"}``)

        Returns:
            Number of chunks added to the vector store
        """
        if not LANGCHAIN_AVAILABLE or Document is None:
            raise ImportError(
                "langchain-core and langchain-text-splitters are required for indexing. "
                "Install with: pip install 'agentensemble[rag]'"
            )
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas must have the same length as texts")
        self._initialize_components()
        metas = metadatas or [{} for _ in texts]
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metas, strict=True)]
        return self._add_documents(docs)

    def _add_documents(self, documents: list[Any]) -> int:
        """Split documents and add them to the vector store. Returns the chunk count."""
        if not documents:
            return 0

        # Split documents (as per LangChain tutorial)
        if self._text_splitter:
            all_splits = self._text_splitter.split_documents(documents)
        else:
            all_splits = documents

        # Store in vector store (as per LangChain tutorial)
        if self.vector_store is None:
            raise RuntimeError(
                "RAGTool has no vector store: pass vector_store= or embeddings=, "
                "or set OPENAI_API_KEY / MISTRAL_API_KEY for default embeddings."
            )
        self.vector_store.add_documents(documents=all_splits)
        return len(all_splits)

    def create_retrieval_tool(self):
        """
        Create a LangChain tool for retrieval (following LangChain RAG tutorial).

        Returns a tool decorated with @tool that can be used with create_agent.

        Returns:
            LangChain tool instance
        """
        if not LANGCHAIN_AVAILABLE or tool is None:
            raise ImportError("langchain required for tool creation")

        if not self.vector_store:
            raise ValueError("Vector store must be initialized. Call index_documents() first.")

        vector_store = self.vector_store

        # Create retrieval tool following LangChain RAG tutorial pattern
        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information to help answer a query."""
            retrieved_docs = vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        return retrieve_context

    def retrieve(self, query: str, k: int = 2) -> tuple[str, list[Document]]:
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

    def run(self, question: str, urls: list[str] | None = None, **kwargs) -> str:
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
            context, _docs = self.retrieve(question, k=3)
            return context

        return f"RAG answer to '{question}' [index documents first with index_documents()]"

    def __call__(self, question: str, urls: list[str] | None = None, **kwargs) -> str:
        """Make tool callable"""
        return self.run(question, urls, **kwargs)
