import requests
import json
import time
import traceback
import logging # Import logging
from .exceptions import (
    APIError, APITimeoutError, APIAuthError, APIRequestError, 
    InvalidAPIResponseError, ModelFetchError, OpenRouterAPIError
)
from core.constants import (
    OPENROUTER_API_URL, DEFAULT_API_TIMEOUT, 
    YOUR_SITE_URL, YOUR_APP_NAME
)

# Set up logger for this module
logger = logging.getLogger(__name__)

def call_openrouter_api(api_key, model, formatted_prompt, retries=3, delay=5, timeout=DEFAULT_API_TIMEOUT):
    """
    Call the OpenRouter API to get a response from the specified model.
    
    Args:
        api_key (str): OpenRouter API key
        model (str): ID of the model to use
        formatted_prompt (str): Formatted prompt to send to the model
        retries (int): Number of retry attempts
        delay (int): Delay between retries in seconds
        timeout (int): API timeout in seconds
        
    Returns:
        str: Generated response
        
    Raises:
        APITimeoutError: If the request times out after all retries.
        APIAuthError: If authentication fails (401/403).
        APIRequestError: For other HTTP errors or request issues.
        InvalidAPIResponseError: If the response format is unexpected.
        OpenRouterAPIError: For specific errors reported by the API.
        APIError: For other unexpected errors.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_APP_NAME
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": formatted_prompt}],
    }

    last_exception = None # Keep track of the last exception

    for attempt in range(retries):
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=timeout)
            response.raise_for_status() # Raises HTTPError for 4xx/5xx
            result = response.json()

            # Attempt to parse the response - structure varies slightly
            if "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
                first_choice = result["choices"][0]
                if "message" in first_choice and isinstance(first_choice["message"], dict) and "content" in first_choice["message"]:
                    return first_choice["message"]["content"].strip()
                elif "text" in first_choice: # Some models might use 'text'
                    return first_choice["text"].strip()

            # If parsing fails or structure is wrong
            error_detail = f"Unexpected API response format: {json.dumps(result, indent=2)}"
            logger.warning(f"OpenRouter API call failed: {error_detail}") # Log as warning
            raise InvalidAPIResponseError(error_detail)

        except requests.exceptions.Timeout as e:
            last_exception = e
            logger.warning(f"OpenRouter API Timeout (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise APITimeoutError(f"API request timed out after {retries} attempts") from e
        except requests.exceptions.HTTPError as e:
            last_exception = e
            status_code = e.response.status_code
            error_details_str = ""
            try:
                error_details = e.response.json()
                if 'error' in error_details and 'message' in error_details['error']:
                    error_details_str = error_details['error']['message']
                else:
                    error_details_str = json.dumps(error_details)
            except json.JSONDecodeError:
                error_details_str = e.response.text

            log_message = f"OpenRouter API HTTP Error (Attempt {attempt + 1}/{retries}): Status {status_code}, Details: {error_details_str}"
            logger.warning(log_message)

            if status_code in [401, 403]:
                raise APIAuthError(f"API Authentication Failed ({status_code}). Check API Key.", status_code=status_code, details=error_details_str) from e

            if attempt < retries - 1:
                time.sleep(delay)
            else:
                # Raise a more specific error if possible based on details
                raise OpenRouterAPIError(f"API request failed after {retries} attempts", status_code=status_code, details=error_details_str) from e

        except requests.exceptions.RequestException as e:
            last_exception = e
            logger.warning(f"OpenRouter API Request Error (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise APIRequestError(f"API request failed after {retries} attempts: {e}") from e
        except json.JSONDecodeError as e:
            last_exception = e
            resp_text = response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'
            logger.error(f"OpenRouter API JSON Decode Error: {e}. Response text: {resp_text[:200]}...") # Log error
            raise InvalidAPIResponseError(f"Could not parse API response: {e}") from e
        except Exception as e:
            last_exception = e
            logger.error(f"Unexpected error during OpenRouter API call: {e}", exc_info=True) # Log full traceback
            raise APIError(f"An unexpected error occurred during API call: {e}") from e # Generic base class

    # Should not be reachable if exceptions are raised correctly, but as a fallback:
    raise APIError(f"Max retries ({retries}) reached. Last known error: {last_exception}")

def fetch_openrouter_models(api_key, timeout=20):
    """
    Fetch available models from OpenRouter API.
    
    Args:
        api_key (str): OpenRouter API Key.
        timeout (int): API timeout in seconds.
        
    Returns:
        list: List of model data dictionaries (e.g., [{'id': '...', 'context_length': ...}])
        
    Raises:
        APIAuthError: If the API key is missing.
        ModelFetchError: If the request fails or the response is invalid.
    """
    if not api_key:
        logger.error("Attempted to fetch OpenRouter models without an API key.")
        raise APIAuthError("API Key is required to fetch models.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_APP_NAME
    }
    url = "https://openrouter.ai/api/v1/models"
    
    logger.info(f"Fetching models from OpenRouter API {url}")
    
    try:
        # Direct API call with simple error handling
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        logger.debug(f"OpenRouter API response: {str(data)[:200]}...")
        
        if not isinstance(data, dict) or "data" not in data:
            logger.error(f"Unexpected API response format: {data}")
            raise ModelFetchError("OpenRouter API returned unexpected response format")
            
        models = data.get("data", [])
        if not models:
            logger.warning("No models found in OpenRouter API response")
            return []
            
        # Extract and normalize model data
        model_data = []
        for model in models:
            if isinstance(model, dict) and "id" in model:
                model_id = model.get("id")
                context_length = model.get("context_length", 8192)
                
                # Ensure context_length is valid
                if not isinstance(context_length, int) or context_length <= 0:
                    logger.warning(f"Model '{model_id}' has invalid context_length, using default 8192")
                    context_length = 8192
                    
                model_data.append({"id": model_id, "context_length": context_length})
        
        logger.info(f"Found {len(model_data)} models in OpenRouter")
        
        if not model_data:
            logger.warning("No valid models found in OpenRouter API response")
            return []
            
        return sorted(model_data, key=lambda x: x["id"])
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to OpenRouter: {e}", exc_info=True)
        raise ModelFetchError("Could not connect to OpenRouter API. Check your internet connection.")
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout connecting to OpenRouter: {e}", exc_info=True)
        raise ModelFetchError(f"Connection to OpenRouter timed out after {timeout} seconds")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') and hasattr(e.response, 'status_code') else 'unknown'
        if status_code in [401, 403]:
            logger.error(f"Authentication error from OpenRouter API: {e}", exc_info=True)
            raise APIAuthError("Invalid API key for OpenRouter")
        else:
            logger.error(f"HTTP error from OpenRouter API: {e}", exc_info=True)
            raise ModelFetchError(f"OpenRouter API returned HTTP error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from OpenRouter: {e}", exc_info=True)
        raise ModelFetchError("OpenRouter API returned invalid JSON response")
    except Exception as e:
        logger.error(f"Unexpected error fetching OpenRouter models: {e}", exc_info=True)
        raise ModelFetchError(f"Error fetching OpenRouter models: {e}") 