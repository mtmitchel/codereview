"""
API clients for different LLM providers.
"""

from .openrouter import (
    call_openrouter_api,
    fetch_openrouter_models
)

from .ollama import (
    call_ollama_api,
    fetch_ollama_models
)

from .llm_client import (
    LLMClient,
    OpenRouterClient,
    OllamaClient,
    LLMClientFactory,
    register_llm_clients
)

# Re-export exceptions for convenience
from .exceptions import (
    APIError, 
    APITimeoutError, 
    APIAuthError, 
    APIRequestError, 
    InvalidAPIResponseError, 
    OpenRouterAPIError
)
