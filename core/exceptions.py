"""
Core exceptions for the Code Review Tool.

This module defines a hierarchy of exceptions used throughout the application,
enabling more specific error handling and better user feedback.
"""

class CodeReviewToolError(Exception):
    """Base exception for all Code Review Tool errors."""
    pass

class ConfigurationError(CodeReviewToolError):
    """Error in configuration."""
    pass

class FileError(CodeReviewToolError):
    """Error related to file operations."""
    pass

class APIError(CodeReviewToolError):
    """Base class for API-related errors."""
    pass

class APIAuthError(APIError):
    """Authentication error with API."""
    pass

class APITimeoutError(APIError):
    """API call timed out."""
    pass

class APIRequestError(APIError):
    """Error in API request."""
    pass

class InvalidAPIResponseError(APIError):
    """Invalid response from API."""
    pass

class AnalysisError(CodeReviewToolError):
    """Error during code analysis."""
    pass

class ResourceError(CodeReviewToolError):
    """Error related to resource management."""
    pass

class ChunkingError(CodeReviewToolError):
    """Error during code chunking."""
    pass

class CacheError(CodeReviewToolError):
    """Error related to caching."""
    pass
