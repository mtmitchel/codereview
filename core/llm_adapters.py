"""
LLM Provider Adapter Pattern implementation for Code Review Tool.

This module implements the Adapter Pattern to provide a consistent interface
for various LLM providers, making it easy to switch between different models
and services without changing client code.
"""

import logging
import time
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, TypeVar

from core.events import EventEmitter
from core.metrics import get_metrics_collector

logger = logging.getLogger("CodeReviewTool.LLMAdapter")

# Type aliases for better readability
LLMResponse = Dict[str, Any]
LLMParams = Dict[str, Any]
T = TypeVar('T')


class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.
    
    This defines the common interface that all LLM provider implementations must follow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM adapter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.name = "base"
        self.metrics = get_metrics_collector()
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    async def generate_completion(self, 
                               prompt: str, 
                               params: Optional[LLMParams] = None) -> LLMResponse:
        """
        Generate a completion from the LLM.
        
        Args:
            prompt: The prompt text
            params: Additional parameters for the LLM
            
        Returns:
            Dictionary containing the LLM response
        """
        pass
    
    @abstractmethod
    async def generate_chat_completion(self, 
                                    messages: List[Dict[str, str]], 
                                    params: Optional[LLMParams] = None) -> LLMResponse:
        """
        Generate a chat completion from the LLM.
        
        Args:
            messages: List of message dictionaries (role, content)
            params: Additional parameters for the LLM
            
        Returns:
            Dictionary containing the LLM response
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this LLM provider.
        
        Returns:
            Dictionary with capability information
        """
        pass
    
    def _record_metrics(self, 
                       api_call_time: float, 
                       prompt_tokens: int = 0, 
                       completion_tokens: int = 0) -> None:
        """
        Record LLM API call metrics.
        
        Args:
            api_call_time: Time spent on the API call
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
        """
        if self.metrics:
            self.metrics.update_llm_metrics(
                api_call_time=api_call_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
    
    def _get_default_params(self, custom_params: Optional[LLMParams] = None) -> LLMParams:
        """
        Get default parameters merged with custom parameters.
        
        Args:
            custom_params: Custom parameters to merge
            
        Returns:
            Merged parameters dictionary
        """
        defaults = {
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        
        if custom_params:
            defaults.update(custom_params)
        
        return defaults


class OllamaAdapter(LLMAdapter, EventEmitter):
    """
    Adapter for the Ollama API.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ollama adapter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        EventEmitter.__init__(self)
        
        self.name = "ollama"
        self.base_url = self.config.get("ollama_base_url", "http://localhost:11434")
        self.model = self.config.get("ollama_model", "codellama")
        
        logger.info(f"Initialized Ollama adapter with model: {self.model}")
    
    async def generate_completion(self, 
                               prompt: str, 
                               params: Optional[LLMParams] = None) -> LLMResponse:
        """
        Generate a completion from Ollama.
        
        Args:
            prompt: The prompt text
            params: Additional parameters for the LLM
            
        Returns:
            Dictionary containing the LLM response
        """
        import aiohttp
        
        # Merge parameters
        merged_params = self._get_default_params(params)
        
        # Build request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": merged_params.get("temperature", 0.7),
            "max_tokens": merged_params.get("max_tokens", 2000),
            "top_p": merged_params.get("top_p", 1.0),
            "stop": merged_params.get("stop", []),
            "stream": False
        }
        
        start_time = time.time()
        prompt_tokens = len(prompt.split())  # Rough estimation
        completion_tokens = 0
        
        try:
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        return {
                            "error": True,
                            "message": f"Ollama API error: {response.status}",
                            "details": error_text
                        }
                    
                    result = await response.json()
                    
            # Calculate metrics
            completion_tokens = len(result.get("response", "").split())  # Rough estimation
            api_call_time = time.time() - start_time
            
            # Record metrics
            self._record_metrics(api_call_time, prompt_tokens, completion_tokens)
            
            # Emit event
            self.emit_event("llm.completion", {
                "provider": "ollama",
                "model": self.model,
                "api_call_time": api_call_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            })
            
            # Format response to match our standard format
            return {
                "content": result.get("response", ""),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "api_call_time": api_call_time,
                "model": self.model,
                "provider": "ollama"
            }
            
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return {
                "error": True,
                "message": f"Error calling Ollama API: {str(e)}",
                "details": str(e)
            }
    
    async def generate_chat_completion(self, 
                                    messages: List[Dict[str, str]], 
                                    params: Optional[LLMParams] = None) -> LLMResponse:
        """
        Generate a chat completion from Ollama.
        
        Args:
            messages: List of message dictionaries (role, content)
            params: Additional parameters for the LLM
            
        Returns:
            Dictionary containing the LLM response
        """
        import aiohttp
        
        # Merge parameters
        merged_params = self._get_default_params(params)
        
        # Convert messages to Ollama format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Build request payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": merged_params.get("temperature", 0.7),
            "max_tokens": merged_params.get("max_tokens", 2000),
            "top_p": merged_params.get("top_p", 1.0),
            "stop": merged_params.get("stop", []),
            "stream": False
        }
        
        start_time = time.time()
        prompt_tokens = sum(len(msg.get("content", "").split()) for msg in formatted_messages)  # Rough estimation
        completion_tokens = 0
        
        try:
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        return {
                            "error": True,
                            "message": f"Ollama API error: {response.status}",
                            "details": error_text
                        }
                    
                    result = await response.json()
            
            # Get the response message
            response_message = result.get("message", {})
            content = response_message.get("content", "")
            
            # Calculate metrics
            completion_tokens = len(content.split())  # Rough estimation
            api_call_time = time.time() - start_time
            
            # Record metrics
            self._record_metrics(api_call_time, prompt_tokens, completion_tokens)
            
            # Emit event
            self.emit_event("llm.chat_completion", {
                "provider": "ollama",
                "model": self.model,
                "api_call_time": api_call_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            })
            
            # Format response to match our standard format
            return {
                "content": content,
                "role": response_message.get("role", "assistant"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "api_call_time": api_call_time,
                "model": self.model,
                "provider": "ollama"
            }
            
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return {
                "error": True,
                "message": f"Error calling Ollama API: {str(e)}",
                "details": str(e)
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the Ollama provider.
        
        Returns:
            Dictionary with capability information
        """
        return {
            "name": "ollama",
            "supports_chat": True,
            "supports_completion": True,
            "supports_streaming": True,
            "max_context_length": 8192,  # Depends on the model
            "local": True,
            "models": [self.model]
        }


class OpenAIAdapter(LLMAdapter, EventEmitter):
    """
    Adapter for the OpenAI API.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenAI adapter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        EventEmitter.__init__(self)
        
        self.name = "openai"
        self.api_key = self.config.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
        self.model = self.config.get("openai_model", "gpt-3.5-turbo")
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided")
        
        logger.info(f"Initialized OpenAI adapter with model: {self.model}")
    
    async def generate_completion(self, 
                               prompt: str, 
                               params: Optional[LLMParams] = None) -> LLMResponse:
        """
        Generate a completion from OpenAI.
        
        Args:
            prompt: The prompt text
            params: Additional parameters for the LLM
            
        Returns:
            Dictionary containing the LLM response
        """
        try:
            import openai
        except ImportError:
            logger.error("OpenAI package not installed. Install with 'pip install openai'")
            return {
                "error": True,
                "message": "OpenAI package not installed. Install with 'pip install openai'"
            }
        
        # Check if API key is set
        if not self.api_key:
            return {
                "error": True,
                "message": "OpenAI API key not provided"
            }
        
        # Merge parameters
        merged_params = self._get_default_params(params)
        
        # Convert completion request to chat for newer models
        if "gpt" in self.model:
            # Use chat endpoint with a single user message for text completion
            messages = [{"role": "user", "content": prompt}]
            return await self.generate_chat_completion(messages, merged_params)
        
        # For legacy models, use completions endpoint
        openai.api_key = self.api_key
        
        start_time = time.time()
        prompt_tokens = 0
        completion_tokens = 0
        
        try:
            # Make API request
            result = await openai.Completion.acreate(
                model=self.model,
                prompt=prompt,
                max_tokens=merged_params.get("max_tokens", 2000),
                temperature=merged_params.get("temperature", 0.7),
                top_p=merged_params.get("top_p", 1.0),
                frequency_penalty=merged_params.get("frequency_penalty", 0.0),
                presence_penalty=merged_params.get("presence_penalty", 0.0),
                stop=merged_params.get("stop", None)
            )
            
            # Extract content
            content = result.choices[0].text.strip()
            
            # Calculate metrics
            prompt_tokens = result.usage.prompt_tokens
            completion_tokens = result.usage.completion_tokens
            api_call_time = time.time() - start_time
            
            # Record metrics
            self._record_metrics(api_call_time, prompt_tokens, completion_tokens)
            
            # Emit event
            self.emit_event("llm.completion", {
                "provider": "openai",
                "model": self.model,
                "api_call_time": api_call_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            })
            
            # Format response to match our standard format
            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "api_call_time": api_call_time,
                "model": self.model,
                "provider": "openai"
            }
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {
                "error": True,
                "message": f"Error calling OpenAI API: {str(e)}",
                "details": str(e)
            }
    
    async def generate_chat_completion(self, 
                                    messages: List[Dict[str, str]], 
                                    params: Optional[LLMParams] = None) -> LLMResponse:
        """
        Generate a chat completion from OpenAI.
        
        Args:
            messages: List of message dictionaries (role, content)
            params: Additional parameters for the LLM
            
        Returns:
            Dictionary containing the LLM response
        """
        try:
            import openai
        except ImportError:
            logger.error("OpenAI package not installed. Install with 'pip install openai'")
            return {
                "error": True,
                "message": "OpenAI package not installed. Install with 'pip install openai'"
            }
        
        # Check if API key is set
        if not self.api_key:
            return {
                "error": True,
                "message": "OpenAI API key not provided"
            }
        
        # Merge parameters
        merged_params = self._get_default_params(params)
        
        openai.api_key = self.api_key
        
        start_time = time.time()
        prompt_tokens = 0
        completion_tokens = 0
        
        try:
            # Make API request
            result = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                max_tokens=merged_params.get("max_tokens", 2000),
                temperature=merged_params.get("temperature", 0.7),
                top_p=merged_params.get("top_p", 1.0),
                frequency_penalty=merged_params.get("frequency_penalty", 0.0),
                presence_penalty=merged_params.get("presence_penalty", 0.0),
                stop=merged_params.get("stop", None)
            )
            
            # Extract content
            response_message = result.choices[0].message
            content = response_message.content.strip()
            
            # Calculate metrics
            prompt_tokens = result.usage.prompt_tokens
            completion_tokens = result.usage.completion_tokens
            api_call_time = time.time() - start_time
            
            # Record metrics
            self._record_metrics(api_call_time, prompt_tokens, completion_tokens)
            
            # Emit event
            self.emit_event("llm.chat_completion", {
                "provider": "openai",
                "model": self.model,
                "api_call_time": api_call_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            })
            
            # Format response to match our standard format
            return {
                "content": content,
                "role": response_message.role,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "api_call_time": api_call_time,
                "model": self.model,
                "provider": "openai"
            }
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {
                "error": True,
                "message": f"Error calling OpenAI API: {str(e)}",
                "details": str(e)
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the OpenAI provider.
        
        Returns:
            Dictionary with capability information
        """
        return {
            "name": "openai",
            "supports_chat": True,
            "supports_completion": True,
            "supports_streaming": True,
            "max_context_length": 16384 if "gpt-4" in self.model else 4096,  # Depends on the model
            "local": False,
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        }


class AnthropicAdapter(LLMAdapter, EventEmitter):
    """
    Adapter for the Anthropic API (Claude models).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Anthropic adapter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        EventEmitter.__init__(self)
        
        self.name = "anthropic"
        self.api_key = self.config.get("anthropic_api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
        self.model = self.config.get("anthropic_model", "claude-2")
        
        if not self.api_key:
            logger.warning("Anthropic API key not provided")
        
        logger.info(f"Initialized Anthropic adapter with model: {self.model}")
    
    async def generate_completion(self, 
                               prompt: str, 
                               params: Optional[LLMParams] = None) -> LLMResponse:
        """
        Generate a completion from Anthropic.
        
        Args:
            prompt: The prompt text
            params: Additional parameters for the LLM
            
        Returns:
            Dictionary containing the LLM response
        """
        try:
            import anthropic
        except ImportError:
            logger.error("Anthropic package not installed. Install with 'pip install anthropic'")
            return {
                "error": True,
                "message": "Anthropic package not installed. Install with 'pip install anthropic'"
            }
        
        # Check if API key is set
        if not self.api_key:
            return {
                "error": True,
                "message": "Anthropic API key not provided"
            }
        
        # Merge parameters
        merged_params = self._get_default_params(params)
        
        # Format prompt according to Anthropic's requirements
        formatted_prompt = f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}"
        
        start_time = time.time()
        prompt_tokens = len(prompt.split())  # Rough estimation
        completion_tokens = 0
        
        try:
            # Initialize client
            client = anthropic.Client(api_key=self.api_key)
            
            # Make API request
            result = client.completion(
                prompt=formatted_prompt,
                model=self.model,
                max_tokens_to_sample=merged_params.get("max_tokens", 2000),
                temperature=merged_params.get("temperature", 0.7),
                top_p=merged_params.get("top_p", 1.0),
                stop_sequences=merged_params.get("stop", [])
            )
            
            # Extract content
            content = result.completion.strip()
            
            # Calculate metrics
            completion_tokens = len(content.split())  # Rough estimation
            api_call_time = time.time() - start_time
            
            # Record metrics
            self._record_metrics(api_call_time, prompt_tokens, completion_tokens)
            
            # Emit event
            self.emit_event("llm.completion", {
                "provider": "anthropic",
                "model": self.model,
                "api_call_time": api_call_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            })
            
            # Format response to match our standard format
            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "api_call_time": api_call_time,
                "model": self.model,
                "provider": "anthropic"
            }
            
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return {
                "error": True,
                "message": f"Error calling Anthropic API: {str(e)}",
                "details": str(e)
            }
    
    async def generate_chat_completion(self, 
                                    messages: List[Dict[str, str]], 
                                    params: Optional[LLMParams] = None) -> LLMResponse:
        """
        Generate a chat completion from Anthropic.
        
        Args:
            messages: List of message dictionaries (role, content)
            params: Additional parameters for the LLM
            
        Returns:
            Dictionary containing the LLM response
        """
        try:
            import anthropic
        except ImportError:
            logger.error("Anthropic package not installed. Install with 'pip install anthropic'")
            return {
                "error": True,
                "message": "Anthropic package not installed. Install with 'pip install anthropic'"
            }
        
        # Check if API key is set
        if not self.api_key:
            return {
                "error": True,
                "message": "Anthropic API key not provided"
            }
        
        # Merge parameters
        merged_params = self._get_default_params(params)
        
        # Format messages according to Anthropic's requirements
        formatted_prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Add system message at the beginning
                formatted_prompt = f"{content}\n{formatted_prompt}"
            elif role == "user":
                formatted_prompt += f"{anthropic.HUMAN_PROMPT} {content}"
            elif role == "assistant":
                formatted_prompt += f"{anthropic.AI_PROMPT} {content}"
        
        # Ensure the prompt ends with AI prompt
        if not formatted_prompt.endswith(anthropic.AI_PROMPT):
            formatted_prompt += anthropic.AI_PROMPT
        
        start_time = time.time()
        prompt_tokens = sum(len(msg.get("content", "").split()) for msg in messages)  # Rough estimation
        completion_tokens = 0
        
        try:
            # Initialize client
            client = anthropic.Client(api_key=self.api_key)
            
            # Make API request
            result = client.completion(
                prompt=formatted_prompt,
                model=self.model,
                max_tokens_to_sample=merged_params.get("max_tokens", 2000),
                temperature=merged_params.get("temperature", 0.7),
                top_p=merged_params.get("top_p", 1.0),
                stop_sequences=merged_params.get("stop", [])
            )
            
            # Extract content
            content = result.completion.strip()
            
            # Calculate metrics
            completion_tokens = len(content.split())  # Rough estimation
            api_call_time = time.time() - start_time
            
            # Record metrics
            self._record_metrics(api_call_time, prompt_tokens, completion_tokens)
            
            # Emit event
            self.emit_event("llm.chat_completion", {
                "provider": "anthropic",
                "model": self.model,
                "api_call_time": api_call_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            })
            
            # Format response to match our standard format
            return {
                "content": content,
                "role": "assistant",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "api_call_time": api_call_time,
                "model": self.model,
                "provider": "anthropic"
            }
            
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return {
                "error": True,
                "message": f"Error calling Anthropic API: {str(e)}",
                "details": str(e)
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the Anthropic provider.
        
        Returns:
            Dictionary with capability information
        """
        return {
            "name": "anthropic",
            "supports_chat": True,
            "supports_completion": True,
            "supports_streaming": False,
            "max_context_length": 100000,  # Claude has very large context
            "local": False,
            "models": ["claude-2", "claude-instant-1"]
        }


class OpenRouterAdapter(LLMAdapter, EventEmitter):
    """
    Adapter for the OpenRouter API, which provides access to multiple models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenRouter adapter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        EventEmitter.__init__(self)
        
        self.name = "openrouter"
        self.api_key = self.config.get("openrouter_api_key", os.environ.get("OPENROUTER_API_KEY", ""))
        self.model = self.config.get("openrouter_model", "openai/gpt-3.5-turbo")
        
        if not self.api_key:
            logger.warning("OpenRouter API key not provided")
        
        logger.info(f"Initialized OpenRouter adapter with model: {self.model}")
    
    async def generate_completion(self, 
                               prompt: str, 
                               params: Optional[LLMParams] = None) -> LLMResponse:
        """
        Generate a completion from OpenRouter.
        
        Args:
            prompt: The prompt text
            params: Additional parameters for the LLM
            
        Returns:
            Dictionary containing the LLM response
        """
        import aiohttp
        
        # Check if API key is set
        if not self.api_key:
            return {
                "error": True,
                "message": "OpenRouter API key not provided"
            }
        
        # Merge parameters
        merged_params = self._get_default_params(params)
        
        # OpenRouter uses chat format, so convert completion to chat
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_chat_completion(messages, merged_params)
    
    async def generate_chat_completion(self, 
                                    messages: List[Dict[str, str]], 
                                    params: Optional[LLMParams] = None) -> LLMResponse:
        """
        Generate a chat completion from OpenRouter.
        
        Args:
            messages: List of message dictionaries (role, content)
            params: Additional parameters for the LLM
            
        Returns:
            Dictionary containing the LLM response
        """
        import aiohttp
        
        # Check if API key is set
        if not self.api_key:
            return {
                "error": True,
                "message": "OpenRouter API key not provided"
            }
        
        # Merge parameters
        merged_params = self._get_default_params(params)
        
        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": merged_params.get("temperature", 0.7),
            "max_tokens": merged_params.get("max_tokens", 2000),
            "top_p": merged_params.get("top_p", 1.0),
            "frequency_penalty": merged_params.get("frequency_penalty", 0.0),
            "presence_penalty": merged_params.get("presence_penalty", 0.0),
            "stop": merged_params.get("stop", None)
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        prompt_tokens = sum(len(msg.get("content", "").split()) for msg in messages)  # Rough estimation
        completion_tokens = 0
        
        try:
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                        return {
                            "error": True,
                            "message": f"OpenRouter API error: {response.status}",
                            "details": error_text
                        }
                    
                    result = await response.json()
            
            # Extract content
            response_message = result.get("choices", [{}])[0].get("message", {})
            content = response_message.get("content", "").strip()
            
            # Get token usage
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", len(content.split()))
            
            # Calculate metrics
            api_call_time = time.time() - start_time
            
            # Record metrics
            self._record_metrics(api_call_time, prompt_tokens, completion_tokens)
            
            # Emit event
            self.emit_event("llm.chat_completion", {
                "provider": "openrouter",
                "model": self.model,
                "api_call_time": api_call_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            })
            
            # Format response to match our standard format
            return {
                "content": content,
                "role": response_message.get("role", "assistant"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "api_call_time": api_call_time,
                "model": self.model,
                "provider": "openrouter"
            }
            
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            return {
                "error": True,
                "message": f"Error calling OpenRouter API: {str(e)}",
                "details": str(e)
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the OpenRouter provider.
        
        Returns:
            Dictionary with capability information
        """
        return {
            "name": "openrouter",
            "supports_chat": True,
            "supports_completion": True,
            "supports_streaming": True,
            "max_context_length": 16384,  # Depends on the model
            "local": False,
            "models": [
                "openai/gpt-3.5-turbo", 
                "openai/gpt-4", 
                "anthropic/claude-2", 
                "google/palm-2",
                "meta/llama-2-70b"
            ]
        }


class LLMAdapterFactory:
    """
    Factory for creating LLM adapters.
    """
    
    @staticmethod
    def create_adapter(provider: str, config: Optional[Dict[str, Any]] = None) -> LLMAdapter:
        """
        Create an adapter for the specified provider.
        
        Args:
            provider: The provider name
            config: Configuration dictionary
            
        Returns:
            LLM adapter instance
            
        Raises:
            ValueError: If the provider is not supported
        """
        config = config or {}
        
        if provider == "ollama":
            return OllamaAdapter(config)
        elif provider == "openai":
            return OpenAIAdapter(config)
        elif provider == "anthropic":
            return AnthropicAdapter(config)
        elif provider == "openrouter":
            return OpenRouterAdapter(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """
        Get a list of available provider names.
        
        Returns:
            List of provider names
        """
        return ["ollama", "openai", "anthropic", "openrouter"]