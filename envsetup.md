# Tier 1 (Base)
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free

# Tier 2 (First Cloud Fallback)
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_MODEL=mistral-large-latest

# Tier 3 (Second Cloud Fallback)
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

# Tier 4 (Ollama Cloud API Safety Net)
OLLAMA_API_KEY=your_ollama_cloud_key
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=gpt-oss:120b # Or whichever cloud-supported model you pull
REQUEST_TIMEOUT=30