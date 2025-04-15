"""
Error handling utilities for the Code Review Tool.

This module provides utilities for handling errors in a consistent manner
across the application.
"""

import logging
import traceback
import sys
from typing import Optional, Callable, Dict, Any, Type, List, Tuple

logger = logging.getLogger("CodeReviewTool.ErrorUtils")

class ErrorHandler:
    """
    Utility class for handling errors with contextual information.
    
    This class provides a centralized approach to error handling,
    with support for custom error handlers, error categorization,
    and detailed error reporting.
    """
    
    # Map of error types to their handlers
    _error_handlers: Dict[Type[Exception], Callable] = {}
    
    # Error categories for grouping similar errors
    _error_categories = {
        'configuration': [
            'ConfigValidationError',
            'ValueError',
            'TypeError',
            'KeyError',
        ],
        'file_system': [
            'FileNotFoundError',
            'PermissionError',
            'IOError',
            'OSError',
        ],
        'network': [
            'APIError',
            'APIAuthError',
            'APITimeoutError',
            'APIRequestError',
            'ConnectionError',
            'TimeoutError',
        ],
        'processing': [
            'ChunkingError',
            'ProcessingError',
            'ParallelProcessingError',
            'ThreadingError',
        ],
        'resource': [
            'MemoryError',
            'ResourceExhaustedError',
            'ThreadPoolError',
        ],
    }
    
    @classmethod
    def register_handler(cls, error_type: Type[Exception], handler: Callable):
        """
        Register a handler for a specific error type.
        
        Args:
            error_type: The exception class to handle
            handler: The handler function to call
        """
        cls._error_handlers[error_type] = handler
    
    @classmethod
    def handle_error(cls, error: Exception, context: str = "", reraise: bool = False) -> Optional[Dict[str, Any]]:
        """
        Handle an error using registered handlers or default behavior.
        
        Args:
            error: The exception to handle
            context: Contextual information about where the error occurred
            reraise: Whether to reraise the exception after handling
            
        Returns:
            Optional dictionary with error information
            
        Raises:
            The original exception if reraise is True
        """
        error_type = type(error)
        error_name = error_type.__name__
        
        # Get category for this error
        category = cls._get_error_category(error_type)
        
        # Log the error with context
        log_message = f"{category.upper() if category else 'UNCATEGORIZED'} ERROR"
        if context:
            log_message += f" in {context}"
        log_message += f": {error_name}: {str(error)}"
        
        logger.error(log_message, exc_info=True)
        
        # Check if we have a custom handler for this error type
        for handler_type, handler in cls._error_handlers.items():
            if isinstance(error, handler_type):
                result = handler(error, context)
                if reraise:
                    raise error
                return result
        
        # Default error handling behavior
        error_info = {
            'type': error_name,
            'message': str(error),
            'context': context,
            'category': category,
            'traceback': traceback.format_exc(),
        }
        
        if reraise:
            raise error
            
        return error_info
    
    @classmethod
    def _get_error_category(cls, error_type: Type[Exception]) -> Optional[str]:
        """
        Get the category for an error type.
        
        Args:
            error_type: The exception class
            
        Returns:
            The category name or None if not categorized
        """
        error_name = error_type.__name__
        
        for category, error_names in cls._error_categories.items():
            if error_name in error_names:
                return category
                
        return None
    
    @classmethod
    def is_critical_error(cls, error: Exception) -> bool:
        """
        Determine if an error is critical and should halt processing.
        
        Args:
            error: The exception to check
            
        Returns:
            True if the error is critical, False otherwise
        """
        # Check if it's a critical error type
        critical_types = [
            'APIAuthError',  # Authentication failures
            'ConfigValidationError',  # Configuration errors
            'ThreadPoolError',  # Threading errors
            'MemoryError',  # Out of memory
        ]
        
        error_name = type(error).__name__
        if error_name in critical_types:
            return True
            
        # Check error categorization
        category = cls._get_error_category(type(error))
        critical_categories = ['configuration', 'resource']
        if category in critical_categories:
            return True
            
        return False
    
    @classmethod
    def safe_execute(cls, func: Callable, *args, context: str = "", 
                     error_value: Any = None, **kwargs) -> Any:
        """
        Execute a function safely, handling any exceptions.
        
        Args:
            func: The function to execute
            *args: Arguments to pass to the function
            context: Contextual information about the function
            error_value: Value to return if an error occurs
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The function result or error_value if an error occurs
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            cls.handle_error(e, context=context)
            return error_value
    
    @classmethod
    def safe_wrapper(cls, func: Callable, context: str = "", error_value: Any = None) -> Callable:
        """
        Create a wrapper that executes a function safely.
        
        Args:
            func: The function to wrap
            context: Contextual information about the function
            error_value: Value to return if an error occurs
            
        Returns:
            A wrapped function that handles exceptions
        """
        def wrapper(*args, **kwargs):
            return cls.safe_execute(func, *args, context=context, error_value=error_value, **kwargs)
        return wrapper
    
    @classmethod
    def get_error_statistics(cls, errors: List[Exception]) -> Dict[str, Any]:
        """
        Get statistics about a list of errors.
        
        Args:
            errors: List of exceptions
            
        Returns:
            Dictionary with error statistics
        """
        if not errors:
            return {
                'total': 0,
                'categories': {},
                'types': {},
            }
            
        # Count errors by category and type
        categories = {}
        types = {}
        
        for error in errors:
            error_type = type(error).__name__
            category = cls._get_error_category(type(error)) or 'uncategorized'
            
            categories[category] = categories.get(category, 0) + 1
            types[error_type] = types.get(error_type, 0) + 1
            
        return {
            'total': len(errors),
            'categories': categories,
            'types': types,
        }
        
    @classmethod
    def format_error_for_user(cls, error: Exception, 
                              include_traceback: bool = False) -> Tuple[str, str]:
        """
        Format an error message for display to the user.
        
        Args:
            error: The exception to format
            include_traceback: Whether to include traceback information
            
        Returns:
            Tuple of (title, message) suitable for display to users
        """
        error_type = type(error).__name__
        category = cls._get_error_category(type(error)) or 'Uncategorized'
        
        # Map categories to user-friendly titles
        category_titles = {
            'configuration': 'Configuration Error',
            'file_system': 'File System Error',
            'network': 'Network Error',
            'processing': 'Processing Error',
            'resource': 'Resource Error',
            'uncategorized': 'Error',
        }
        
        title = category_titles.get(category, 'Error')
        
        # Clean up error message (remove sensitive information, etc.)
        message = str(error)
        
        # For network errors, provide more helpful information
        if category == 'network':
            message += "\n\nPlease check your network connection and API settings."
            
        # For file system errors, provide guidance
        elif category == 'file_system':
            message += "\n\nPlease check that all files and directories exist and are accessible."
            
        # For configuration errors, provide help
        elif category == 'configuration':
            message += "\n\nPlease check your configuration settings."
            
        # Include traceback information if requested
        if include_traceback:
            message += f"\n\nError Type: {error_type}\n"
            message += f"Traceback:\n{traceback.format_exc()}"
            
        return title, message
