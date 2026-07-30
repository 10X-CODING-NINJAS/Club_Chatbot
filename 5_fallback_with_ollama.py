"""
LLM Provider Layer for RAG chatbot.

Abstracts communication with multiple LLM providers.
Cascade: OpenRouter (Base) -> Mistral (Fallback 1) -> Groq (Fallback 2) -> Ollama (Local)
"""
import os
from typing import List, Dict
import requests
from dotenv import load_dotenv

load_dotenv()


class ConfigurationError(Exception):
    """Exception raised for missing or invalid configuration."""
    pass


def _get_timeout() -> float:
    timeout_str = os.getenv("REQUEST_TIMEOUT", "30")
    try:
        return float(timeout_str)
    except ValueError:
        return 30.0


def call_openrouter(messages: List[Dict]) -> str:
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


def call_mistral(messages: List[Dict]) -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ConfigurationError("MISTRAL_API_KEY environment variable is missing.")

    model = os.getenv("MISTRAL_MODEL", "mistral-large-latest") 
    timeout = _get_timeout()

    url = "https://api.mistral.ai/v1/chat/completions"
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


def call_groq(messages: List[Dict]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ConfigurationError("GROQ_API_KEY environment variable is missing.")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") 
    timeout = _get_timeout()

    # Groq uses an OpenAI-compatible endpoint schema
    url = "https://api.groq.com/openai/v1/chat/completions"
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
    # Tier 1: OpenRouter (Base)
    print("Trying OpenRouter...")
    try:
        response = call_openrouter(messages)
        print("OpenRouter request successful.")
        return response
    except (ConfigurationError, requests.exceptions.RequestException) as e:
        print(f"[Fallback] OpenRouter failed: {e}")

    # Tier 2: Mistral (First Cloud Fallback)
    print("Switching to Mistral...")
    try:
        response = call_mistral(messages)
        print("Mistral request successful.")
        return response
    except (ConfigurationError, requests.exceptions.RequestException) as e:
        print(f"[Fallback] Mistral failed: {e}")

    # Tier 3: Groq (Second Cloud Fallback)
    print("Switching to Groq...")
    try:
        response = call_groq(messages)
        print("Groq request successful.")
        return response
    except (ConfigurationError, requests.exceptions.RequestException) as e:
        print(f"[Fallback] Groq failed: {e}")

    # Tier 4: Ollama (Local Hardware Fallback)
    print("Switching to Ollama...")
    if check_ollama_health():
        try:
            response = call_ollama(messages)
            print("Ollama request successful.")
            return response
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e
    else:
        raise RuntimeError("Complete System Failure: All cloud providers failed, and Ollama is unhealthy.")