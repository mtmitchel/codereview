# Token Estimation Helper
TOKEN_CHAR_RATIO = 4
DEFAULT_MAX_OUTPUT_TOKENS = 1024
MIN_TOKEN_BUDGET_FOR_CODE = 100
CONTEXT_SAFETY_FACTOR = 0.95

def estimate_tokens(text, model_name=None):
    """
    Estimate the number of tokens in a text string.
    Uses tiktoken for OpenAI-compatible models when available,
    falls back to character ratio estimation when not.
    
    Args:
        text (str): The text to estimate tokens for
        model_name (str, optional): Model name for model-specific tokenization
        
    Returns:
        int: Estimated token count
    """
    if not text:
        return 0
    
    try:
        import tiktoken
        
        # Select appropriate encoding based on model name
        if model_name:
            model_name = model_name.lower()
            if "gpt" in model_name or "text-davinci" in model_name:
                # OpenAI models
                try:
                    encoding = tiktoken.encoding_for_model(model_name)
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            elif "claude" in model_name:
                encoding = tiktoken.get_encoding("cl100k_base")  # Claude approximation
            elif "llama" in model_name or "mistral" in model_name:
                encoding = tiktoken.get_encoding("cl100k_base")  # Modern open models approximation
            else:
                encoding = tiktoken.get_encoding("cl100k_base")  # Default for newer models
        else:
            encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding
        
        # Return actual token count
        return len(encoding.encode(text))
    
    except (ImportError, Exception) as e:
        # Fallback to character-based estimation if tiktoken is unavailable
        return len(text) // TOKEN_CHAR_RATIO 