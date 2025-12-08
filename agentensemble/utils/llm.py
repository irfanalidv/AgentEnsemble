"""
LLM Utilities

Provides helper functions for initializing Mistral AI models.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    ChatMistralAI = None
    MistralAIEmbeddings = None


def get_mistral_api_key() -> str:
    """
    Get Mistral API key from environment.

    Returns:
        Mistral API key

    Raises:
        ValueError: If API key not found
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY not found in environment. "
            "Set it in .env file or environment variables."
        )
    return api_key


def get_mistral_model(
    model: str = "mistral-large-latest",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> ChatMistralAI:
    """
    Initialize Mistral chat model.

    Args:
        model: Model name (default: "mistral-large-latest")
        temperature: Temperature for generation
        api_key: Optional API key (uses MISTRAL_API_KEY env var if not provided)

    Returns:
        ChatMistralAI instance

    Raises:
        ImportError: If langchain-mistralai not installed
        ValueError: If API key not found
    """
    if not MISTRAL_AVAILABLE:
        raise ImportError(
            "langchain-mistralai is required. "
            "Install with: pip install langchain-mistralai"
        )

    if not api_key:
        api_key = get_mistral_api_key()

    return ChatMistralAI(
        model=model,
        temperature=temperature,
        mistral_api_key=api_key,
    )


def get_mistral_embeddings(
    model: str = "mistral-embed",
    api_key: Optional[str] = None,
) -> MistralAIEmbeddings:
    """
    Initialize Mistral embeddings model.

    Args:
        model: Embedding model name (default: "mistral-embed")
        api_key: Optional API key (uses MISTRAL_API_KEY env var if not provided)

    Returns:
        MistralAIEmbeddings instance

    Raises:
        ImportError: If langchain-mistralai not installed
        ValueError: If API key not found
    """
    if not MISTRAL_AVAILABLE:
        raise ImportError(
            "langchain-mistralai is required. "
            "Install with: pip install langchain-mistralai"
        )

    if not api_key:
        api_key = get_mistral_api_key()

    return MistralAIEmbeddings(
        model=model,
        mistral_api_key=api_key,
    )

