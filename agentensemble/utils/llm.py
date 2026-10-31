"""
LLM Utilities

Helper functions for initializing LangChain chat models (OpenAI or Mistral).
"""

import os

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

try:
    from langchain_openai import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ChatOpenAI = None


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
    api_key: str | None = None,
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
            "langchain-mistralai is required. Install with: pip install langchain-mistralai"
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
    api_key: str | None = None,
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
            "langchain-mistralai is required. Install with: pip install langchain-mistralai"
        )

    if not api_key:
        api_key = get_mistral_api_key()

    return MistralAIEmbeddings(
        model=model,
        mistral_api_key=api_key,
    )


def get_openai_model(
    model: str | None = None,
    temperature: float = 0.7,
    api_key: str | None = None,
) -> "ChatOpenAI":
    """
    Initialize an OpenAI chat model.

    Args:
        model: Model name (default: ``OPENAI_MODEL`` env var, else "gpt-4.1-mini")
        temperature: Temperature for generation (ignored for reasoning models)
        api_key: Optional API key (uses OPENAI_API_KEY env var if not provided)

    Raises:
        ImportError: If langchain-openai not installed
        ValueError: If API key not found
    """
    from agentensemble.llm.openai_provider import OpenAILLMProvider

    return OpenAILLMProvider(model=model, temperature=temperature, api_key=api_key)._model


def get_chat_model(temperature: float = 0.7, provider: str | None = None):
    """
    Initialize the default LangChain chat model from the environment.

    Uses ``AGENTENSEMBLE_LLM_PROVIDER`` if set, else OpenAI when ``OPENAI_API_KEY`` is set,
    else Mistral when ``MISTRAL_API_KEY`` is set.
    """
    from agentensemble.llm.factory import resolve_provider_name

    name = (provider or resolve_provider_name() or "").lower()
    if name == "openai":
        return get_openai_model(temperature=temperature)
    if name == "mistral":
        return get_mistral_model(temperature=temperature)
    raise ValueError(
        "No LLM provider configured. Set OPENAI_API_KEY or MISTRAL_API_KEY "
        "(or AGENTENSEMBLE_LLM_PROVIDER), or pass model= explicitly."
    )
