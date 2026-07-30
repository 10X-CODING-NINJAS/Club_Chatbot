"""
LLM Provider Layer for RAG chatbot.

Abstracts communication with multiple LLM providers.
Supports OpenRouter as the primary provider and Ollama as the fallback.
"""
import os
from typing import List, Dict
import requests
from dotenv import load_dotenv

# Load environment variables at startup
load_dotenv()


class ConfigurationError(Exception):
    """Exception raised for missing or invalid configuration."""
    pass


def _get_timeout() -> float:
    """Helper to get and parse the request timeout from environment."""
    timeout_str = os.getenv("REQUEST_TIMEOUT", "30")
    try:
        return float(timeout_str)
    except ValueError:
        return 30.0


def call_openrouter(messages: List[Dict]) -> str:
    """
    Calls the OpenRouter API with the provided messages.

    Args:
        messages: A list of message dictionaries in standard OpenAI format.

    Returns:
        str: The content of the assistant's response.

    Raises:
        ConfigurationError: If the OPENROUTER_API_KEY is missing.
        requests.exceptions.RequestException: On HTTP or network errors.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ConfigurationError("OPENROUTER_API_KEY environment variable is missing.")

    model = os.getenv("OPENROUTER_MODEL")
    if not model:
        raise ConfigurationError("OPENROUTER_MODEL environment variable is missing.")
    timeout = _get_timeout()
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_ollama(messages: List[Dict]) -> str:
    """
    Calls the local Ollama API with the provided messages.

    Args:
        messages: A list of message dictionaries in standard OpenAI format.

    Returns:
        str: The content of the assistant's response.

    Raises:
        requests.exceptions.RequestException: On HTTP or network errors.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    timeout = _get_timeout()

    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"]


def check_ollama_health() -> bool:
    """
    Checks if the Ollama service is healthy and available.

    Returns:
        bool: True if Ollama is accessible, False otherwise.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = f"{base_url}/api/tags"
    timeout = _get_timeout()

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def call_llm_with_fallback(messages: List[Dict]) -> str:
    """
    Calls OpenRouter and falls back to Ollama if OpenRouter fails.

    Args:
        messages: A list of message dictionaries in standard OpenAI format.

    Returns:
        str: The content of the assistant's response.

    Raises:
        RuntimeError: If both providers fail or Ollama fails during fallback.
    """
    print("Trying OpenRouter...")
    try:
        response = call_openrouter(messages)
        print("OpenRouter request successful.")
        return response
    except (ConfigurationError, requests.exceptions.RequestException) as e:
        print(f"[Fallback] OpenRouter failed: {e}")

    print("Switching to Ollama...")
    if check_ollama_health():
        try:
            response = call_ollama(messages)
            print("Ollama request successful.")
            return response
        except requests.exceptions.RequestException as e:
            # Re-raise HTTP and connection errors as RuntimeError
            raise RuntimeError(f"Ollama request failed: {e}") from e
    else:
        raise RuntimeError("Neither provider is available (OpenRouter failed and Ollama is unhealthy).")
