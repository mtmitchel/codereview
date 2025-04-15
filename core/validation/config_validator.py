"""
Configuration validator for the Code Review Tool.

This module provides functions to validate configuration settings
and ensure they meet the requirements for proper operation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from utils.error_utils import ErrorHandler

logger = logging.getLogger("CodeReviewTool.ConfigValidator")

class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass

class ConfigValidator:
    """
    Validates configuration settings for the Code Review Tool.
    
    This class checks configuration settings for required values,
    valid formats, and reasonable defaults. It provides detailed
    error messages and warnings to help users correct configuration
    issues.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the validator with a configuration dictionary.
        
        Args:
            config: The configuration dictionary to validate
        """
        self.config = config
        self.warnings = []
        self.errors = []
        self._has_pydantic = self._check_pydantic()
        
    def _check_pydantic(self) -> bool:
        """Check if Pydantic is available for advanced validation."""
        try:
            import pydantic
            return True
        except ImportError:
            return False
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate the configuration and return results.
        
        Returns:
            Tuple containing:
                - Boolean indicating if configuration is valid
                - List of warning messages
                - List of error messages
        """
        self.warnings = []
        self.errors = []
        
        # Run all validation checks
        self._validate_paths()
        self._validate_api_settings()
        self._validate_file_types()
        self._validate_exclusion_patterns()
        self._validate_chunking_settings()
        self._validate_performance_settings()
        
        # Configuration is valid if there are no errors
        is_valid = len(self.errors) == 0
        
        return is_valid, self.warnings, self.errors
    
    def _validate_paths(self):
        """Validate directory and file paths in the configuration."""
        # Check input directory
        input_dir = self.config.get('input_dir')
        if not input_dir:
            self.errors.append("Input directory is required but not specified")
            ErrorHandler.handle_error(Exception("Input directory is required but not specified"), context="Config Validation")
        elif not os.path.isdir(input_dir):
            self.errors.append(f"Input directory '{input_dir}' does not exist or is not a directory")
            ErrorHandler.handle_error(Exception(f"Input directory '{input_dir}' does not exist or is not a directory"), context="Config Validation")
        # Check output file
        output_file = self.config.get('output_file')
        if not output_file:
            self.warnings.append("Output file not specified, will use default")
            ErrorHandler.handle_error(Exception("Output file not specified, will use default"), context="Config Validation")
        else:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.isdir(output_dir):
                self.errors.append(f"Output directory '{output_dir}' does not exist")
                ErrorHandler.handle_error(Exception(f"Output directory '{output_dir}' does not exist"), context="Config Validation")
    
    def _validate_api_settings(self):
        """Validate API and model settings."""
        # Check API source
        api_source = self.config.get('api_source')
        if not api_source:
            self.warnings.append("API source not specified, will use default")
            ErrorHandler.handle_error(Exception("API source not specified, will use default"), context="Config Validation")
        if api_source == 'OpenRouter':
            api_key = self.config.get('api_key')
            if not api_key:
                self.errors.append("OpenRouter API key is required but not specified")
                ErrorHandler.handle_error(Exception("OpenRouter API key is required but not specified"), context="Config Validation")
            model = self.config.get('model')
            if not model:
                self.warnings.append("OpenRouter model not specified, will use default")
                ErrorHandler.handle_error(Exception("OpenRouter model not specified, will use default"), context="Config Validation")
        elif api_source == 'Ollama':
            ollama_url = self.config.get('ollama_url')
            if not ollama_url:
                self.warnings.append("Ollama URL not specified, will use default")
                ErrorHandler.handle_error(Exception("Ollama URL not specified, will use default"), context="Config Validation")
            model = self.config.get('model')
            if not model:
                self.warnings.append("Ollama model not specified, will use default")
                ErrorHandler.handle_error(Exception("Ollama model not specified, will use default"), context="Config Validation")
    
    def _validate_file_types(self):
        """Validate file type settings."""
        file_types = self.config.get('file_types')
        if not file_types or not isinstance(file_types, list) or len(file_types) == 0:
            self.errors.append("At least one file type must be specified")
            ErrorHandler.handle_error(Exception("At least one file type must be specified"), context="Config Validation")
        else:
            for ft in file_types:
                if not isinstance(ft, str):
                    self.errors.append(f"File type '{ft}' is not a string")
                    ErrorHandler.handle_error(Exception(f"File type '{ft}' is not a string"), context="Config Validation")
    
    def _validate_exclusion_patterns(self):
        """Validate exclusion pattern settings."""
        patterns = self.config.get('exclude_patterns')
        if patterns is not None and not isinstance(patterns, list):
            self.errors.append("Exclude patterns must be a list")
            ErrorHandler.handle_error(Exception("Exclude patterns must be a list"), context="Config Validation")
    
    def _validate_chunking_settings(self):
        """Validate code chunking settings."""
        chunk_size = self.config.get('chunk_size')
        if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size <= 0):
            self.errors.append("Chunk size must be a positive integer")
            ErrorHandler.handle_error(Exception("Chunk size must be a positive integer"), context="Config Validation")
        overlap = self.config.get('overlap')
        if overlap is not None and (not isinstance(overlap, int) or overlap < 0):
            self.errors.append("Overlap must be a non-negative integer")
            ErrorHandler.handle_error(Exception("Overlap must be a non-negative integer"), context="Config Validation")
    
    def _validate_performance_settings(self):
        """Validate performance-related settings."""
        max_workers = self.config.get('max_workers')
        if max_workers is not None and (not isinstance(max_workers, int) or max_workers < 1):
            self.errors.append("Max workers must be a positive integer")
            ErrorHandler.handle_error(Exception("Max workers must be a positive integer"), context="Config Validation")
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Static method to validate configuration without creating an instance.
        
        Args:
            config: The configuration dictionary to validate
            
        Returns:
            Tuple containing:
                - Boolean indicating if configuration is valid
                - List of warning messages
                - List of error messages
        """
        validator = cls(config)
        return validator.validate()
    
    @classmethod
    def apply_defaults(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values to missing configuration settings.
        
        Args:
            config: The configuration dictionary to update
            
        Returns:
            Updated configuration dictionary with defaults applied
        """
        defaults = {
            'chunk_size': 2000,
            'overlap': 100,
            'chunk_strategy': 'Attempt Full File First',
            'context_length': 8192,
            'max_concurrent_workers': 4,
            'api_timeout': 60,
            'run_flake8': True,
            'run_semgrep': False,
            'enable_security_analysis': True,
            'enable_semantic_analysis': True,
            'enable_dependency_analysis': True,
            'min_severity_to_report': 'low',
            'max_chunk_workers': 4,
            'report_clean_files': True,
            'cache_analysis_results': True,
            'incremental_analysis': False,
            'max_retries': 3,
            'retry_delay': 2,
            'timeout_factor': 1.5,
            'exclude_patterns': ['node_modules', 'venv', '.git', '__pycache__'],
        }
        
        # Create a new dictionary with defaults applied
        result = defaults.copy()
        
        # Update with user-provided values, but handle exclude_patterns specially
        for key, value in config.items():
            if key == 'exclude_patterns' and isinstance(value, list):
                # Ensure node_modules, venv, and .git are in the exclusion patterns
                exclude_patterns = value.copy()
                default_excludes = ['node_modules', 'venv', '.git', '__pycache__']
                for pattern in default_excludes:
                    if pattern not in exclude_patterns:
                        exclude_patterns.append(pattern)
                result[key] = exclude_patterns
            else:
                result[key] = value
        
        return result
    
    @classmethod
    def with_pydantic(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Pydantic for advanced validation if available.
        
        Args:
            config: The configuration dictionary to validate
            
        Returns:
            Updated configuration dictionary with validation applied
            
        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            from pydantic import BaseModel, Field, validator, root_validator
            
            class CodeReviewConfig(BaseModel):
                """Pydantic model for code review configuration."""
                input_dir: str
                output_file: Optional[str] = None
                file_types: List[str]
                exclude_patterns: Optional[List[str]] = []
                chunk_size: int = 2000
                overlap: int = 100
                chunk_strategy: str = "Attempt Full File First"
                context_length: int = 8192
                max_concurrent_workers: int = Field(4, ge=1, le=16)
                api_timeout: int = Field(60, ge=30, le=1800)
                api_source: Optional[str] = None
                model: Optional[str] = None
                api_key: Optional[str] = None
                ollama_url: Optional[str] = None
                run_flake8: bool = True
                run_semgrep: bool = False
                enable_security_analysis: bool = True
                enable_semantic_analysis: bool = True
                enable_dependency_analysis: bool = True
                min_severity_to_report: str = "low"
                max_chunk_workers: int = Field(4, ge=1, le=16)
                report_clean_files: bool = True
                cache_analysis_results: bool = True
                incremental_analysis: bool = False
                max_retries: int = 3
                retry_delay: int = 2
                timeout_factor: float = 1.5
                
                # Validators
                @validator('input_dir')
                def validate_input_dir(cls, v):
                    if not os.path.isdir(v):
                        ErrorHandler.handle_error(Exception(f"Input directory '{v}' does not exist or is not a directory"), context="Config Validation (Pydantic)")
                        raise ValueError(f"Input directory '{v}' does not exist or is not a directory")
                    return v
                
                @validator('output_file')
                def validate_output_file(cls, v):
                    if v is not None:
                        output_dir = os.path.dirname(v)
                        if output_dir and not os.path.isdir(output_dir):
                            ErrorHandler.handle_error(Exception(f"Output directory '{output_dir}' does not exist"), context="Config Validation (Pydantic)")
                            raise ValueError(f"Output directory '{output_dir}' does not exist")
                    return v
                
                @validator('file_types')
                def validate_file_types(cls, v):
                    if len(v) == 0:
                        ErrorHandler.handle_error(Exception("At least one file type must be specified"), context="Config Validation (Pydantic)")
                        raise ValueError("At least one file type must be specified")
                    return v
                
                @root_validator
                def validate_api_settings(cls, values):
                    api_source = values.get('api_source')
                    api_key = values.get('api_key')
                    ollama_url = values.get('ollama_url')
                    model = values.get('model')
                    
                    if api_source == 'OpenRouter' and not api_key:
                        ErrorHandler.handle_error(Exception("OpenRouter API key is required"), context="Config Validation (Pydantic)")
                        raise ValueError("OpenRouter API key is required")
                    
                    if api_source and not model:
                        ErrorHandler.handle_error(Exception(f"{api_source} model is required"), context="Config Validation (Pydantic)")
                        raise ValueError(f"{api_source} model is required")
                    
                    return values
            
            # Validate using Pydantic
            validated = CodeReviewConfig(**config).dict()
            return validated
            
        except ImportError:
            # Pydantic not available, fall back to basic validation
            return cls.apply_defaults(config)
            
        except Exception as e:
            # Handle validation errors
            ErrorHandler.handle_error(e, context="Config Validation (Pydantic)")
            raise ConfigValidationError(f"Configuration validation failed: {str(e)}")
