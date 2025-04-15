"""
Configuration management for Code Review Tool.

This module provides a centralized configuration system using pydantic
for type validation and handling of default values.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union
from pydantic import BaseModel, Field, root_validator

from core.constants import (
    DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_FILE, DEFAULT_OPENROUTER_MODEL,
    DEFAULT_OLLAMA_URL, DEFAULT_OLLAMA_CONTEXT_LEN, DEFAULT_CHUNK_SIZE, 
    DEFAULT_OVERLAP, DEFAULT_API_TIMEOUT, MAX_API_TIMEOUT,
    MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, DEFAULT_BINARY_CHUNK_SIZE
)


class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    default_model: Optional[str] = None
    fallback_model: Optional[str] = None
    simple_model: Optional[str] = None  # For simple tasks (LOW complexity)
    balanced_model: Optional[str] = None  # For general tasks (MEDIUM complexity)
    complex_model: Optional[str] = None  # For complex tasks (HIGH complexity)


class APIConfig(BaseModel):
    """API configuration settings."""
    model_source: str = "OpenRouter"
    api_key: Optional[str] = None
    ollama_url: str = DEFAULT_OLLAMA_URL
    model: Optional[str] = None
    api_timeout: int = DEFAULT_API_TIMEOUT
    api_max_retries: int = 3
    api_retry_delay: int = 2
    max_api_timeout: int = MAX_API_TIMEOUT
    models: ModelConfig = Field(default_factory=ModelConfig)
    enable_model_fallback: bool = True
    use_streaming: bool = False


class ChunkingConfig(BaseModel):
    """Chunking configuration settings."""
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP
    chunk_strategy: str = "Auto-suggest"
    auto_suggest_chunking: bool = True
    model_context_length: int = DEFAULT_OLLAMA_CONTEXT_LEN
    enable_adaptive_chunking: bool = True
    min_chunk_size: int = MIN_CHUNK_SIZE
    max_chunk_size: int = MAX_CHUNK_SIZE
    binary_chunk_size: int = DEFAULT_BINARY_CHUNK_SIZE


class AnalysisConfig(BaseModel):
    """Analysis configuration settings."""
    enable_security_analysis: bool = True
    enable_semantic_analysis: bool = False
    semantic_sensitivity: str = "medium"
    run_flake8: bool = True
    run_semgrep: bool = True
    cross_file_analysis: bool = False
    enable_dependency_analysis: bool = True
    parallel_processing: bool = True
    incremental_analysis: bool = False


class FileConfig(BaseModel):
    """File configuration settings."""
    input_dir: Path = Field(default_factory=lambda: Path(DEFAULT_INPUT_DIR))
    output_file: Path = Field(default_factory=lambda: Path(DEFAULT_OUTPUT_FILE))
    file_types: List[str] = Field(default_factory=list)
    exclude_patterns: List[str] = Field(default_factory=list)
    
    @root_validator(skip_on_failure=True)
    def validate_paths(cls, values):
        """Ensure paths are absolute."""
        if 'input_dir' in values and values['input_dir']:
            input_dir = values['input_dir']
            if not input_dir.is_absolute():
                values['input_dir'] = input_dir.absolute()
        
        if 'output_file' in values and values['output_file']:
            output_file = values['output_file']
            if not output_file.is_absolute():
                # If input_dir is set, make output_file relative to it
                if 'input_dir' in values and values['input_dir']:
                    values['output_file'] = values['input_dir'] / output_file
                else:
                    values['output_file'] = output_file.absolute()
        return values


class ReportingConfig(BaseModel):
    """Reporting configuration settings."""
    max_chunk_workers: int = 4
    min_severity_to_report: str = "low"
    report_clean_files: bool = True
    

class CacheConfig(BaseModel):
    """Cache configuration settings."""
    use_cache: bool = True
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".code_review_cache")
    clear_cache_on_exit: bool = False


class CodeReviewConfig(BaseModel):
    """Complete configuration for code review."""
    api: APIConfig = Field(default_factory=APIConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    files: FileConfig = Field(default_factory=FileConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    prompt_template: Optional[str] = None
    analysis_type: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a flat dictionary for backward compatibility."""
        config_dict = {}
        
        # Add API config
        config_dict["model_source"] = self.api.model_source
        config_dict["api_key"] = self.api.api_key
        config_dict["ollama_url"] = self.api.ollama_url
        config_dict["model"] = self.api.model
        config_dict["api_timeout"] = self.api.api_timeout
        config_dict["api_max_retries"] = self.api.api_max_retries
        config_dict["api_retry_delay"] = self.api.api_retry_delay
        config_dict["max_api_timeout"] = self.api.max_api_timeout
        config_dict["enable_model_fallback"] = self.api.enable_model_fallback
        config_dict["use_streaming"] = self.api.use_streaming
        
        # Add model config
        config_dict["default_model"] = self.api.models.default_model
        config_dict["fallback_model"] = self.api.models.fallback_model
        config_dict["simple_model"] = self.api.models.simple_model
        config_dict["balanced_model"] = self.api.models.balanced_model
        config_dict["complex_model"] = self.api.models.complex_model
        
        # Add chunking config
        config_dict["chunk_size"] = self.chunking.chunk_size
        config_dict["overlap"] = self.chunking.overlap
        config_dict["chunk_strategy"] = self.chunking.chunk_strategy
        config_dict["auto_suggest_chunking"] = self.chunking.auto_suggest_chunking
        config_dict["model_context_length"] = self.chunking.model_context_length
        config_dict["enable_adaptive_chunking"] = self.chunking.enable_adaptive_chunking
        config_dict["min_chunk_size"] = self.chunking.min_chunk_size
        config_dict["max_chunk_size"] = self.chunking.max_chunk_size
        config_dict["binary_chunk_size"] = self.chunking.binary_chunk_size
        
        # Add analysis config
        config_dict["enable_security_analysis"] = self.analysis.enable_security_analysis
        config_dict["enable_semantic_analysis"] = self.analysis.enable_semantic_analysis
        config_dict["semantic_sensitivity"] = self.analysis.semantic_sensitivity
        config_dict["run_flake8"] = self.analysis.run_flake8
        config_dict["run_semgrep"] = self.analysis.run_semgrep
        config_dict["cross_file_analysis"] = self.analysis.cross_file_analysis
        config_dict["enable_dependency_analysis"] = self.analysis.enable_dependency_analysis
        config_dict["parallel_processing"] = self.analysis.parallel_processing
        config_dict["incremental_analysis"] = self.analysis.incremental_analysis
        
        # Add file config
        config_dict["input_dir"] = str(self.files.input_dir)
        config_dict["output_file"] = str(self.files.output_file)
        config_dict["file_types"] = self.files.file_types
        config_dict["exclude_patterns"] = self.files.exclude_patterns
        
        # Add reporting config
        config_dict["max_chunk_workers"] = self.reporting.max_chunk_workers
        config_dict["min_severity_to_report"] = self.reporting.min_severity_to_report
        config_dict["report_clean_files"] = self.reporting.report_clean_files
        
        # Add cache config
        config_dict["use_cache"] = self.cache.use_cache
        config_dict["cache_dir"] = str(self.cache.cache_dir)
        config_dict["clear_cache_on_exit"] = self.cache.clear_cache_on_exit
        
        # Add other config
        config_dict["prompt_template"] = self.prompt_template
        config_dict["analysis_type"] = self.analysis_type
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CodeReviewConfig':
        """Create a CodeReviewConfig from a flat dictionary."""
        # Create nested dictionaries for each config section
        api_config = {}
        models_config = {}
        chunking_config = {}
        analysis_config = {}
        file_config = {}
        reporting_config = {}
        cache_config = {}
        other_config = {}
        
        # Map keys to their respective sections
        key_mapping = {
            # API config
            "model_source": ("api", "model_source"),
            "api_key": ("api", "api_key"),
            "ollama_url": ("api", "ollama_url"),
            "model": ("api", "model"),
            "api_timeout": ("api", "api_timeout"),
            "api_max_retries": ("api", "api_max_retries"),
            "api_retry_delay": ("api", "api_retry_delay"),
            "max_api_timeout": ("api", "max_api_timeout"),
            "enable_model_fallback": ("api", "enable_model_fallback"),
            "use_streaming": ("api", "use_streaming"),
            
            # Model config
            "default_model": ("models", "default_model"),
            "fallback_model": ("models", "fallback_model"),
            "simple_model": ("models", "simple_model"),
            "balanced_model": ("models", "balanced_model"),
            "complex_model": ("models", "complex_model"),
            
            # Chunking config
            "chunk_size": ("chunking", "chunk_size"),
            "overlap": ("chunking", "overlap"),
            "chunk_strategy": ("chunking", "chunk_strategy"),
            "auto_suggest_chunking": ("chunking", "auto_suggest_chunking"),
            "model_context_length": ("chunking", "model_context_length"),
            "enable_adaptive_chunking": ("chunking", "enable_adaptive_chunking"),
            "min_chunk_size": ("chunking", "min_chunk_size"),
            "max_chunk_size": ("chunking", "max_chunk_size"),
            "binary_chunk_size": ("chunking", "binary_chunk_size"),
            
            # Analysis config
            "enable_security_analysis": ("analysis", "enable_security_analysis"),
            "enable_semantic_analysis": ("analysis", "enable_semantic_analysis"),
            "semantic_sensitivity": ("analysis", "semantic_sensitivity"),
            "run_flake8": ("analysis", "run_flake8"),
            "run_semgrep": ("analysis", "run_semgrep"),
            "cross_file_analysis": ("analysis", "cross_file_analysis"),
            "enable_dependency_analysis": ("analysis", "enable_dependency_analysis"),
            "parallel_processing": ("analysis", "parallel_processing"),
            "incremental_analysis": ("analysis", "incremental_analysis"),
            
            # File config
            "input_dir": ("files", "input_dir"),
            "output_file": ("files", "output_file"),
            "file_types": ("files", "file_types"),
            "exclude_patterns": ("files", "exclude_patterns"),
            
            # Reporting config
            "max_chunk_workers": ("reporting", "max_chunk_workers"),
            "min_severity_to_report": ("reporting", "min_severity_to_report"),
            "report_clean_files": ("reporting", "report_clean_files"),
            
            # Cache config
            "use_cache": ("cache", "use_cache"),
            "cache_dir": ("cache", "cache_dir"),
            "clear_cache_on_exit": ("cache", "clear_cache_on_exit"),
            
            # Other config
            "prompt_template": ("other", "prompt_template"),
            "analysis_type": ("other", "analysis_type"),
        }
        
        # Process each key in the dict
        for key, value in config_dict.items():
            if key in key_mapping:
                section, config_key = key_mapping[key]
                
                # Convert string paths to Path objects
                if key in ["input_dir", "output_file", "cache_dir"] and value:
                    value = Path(value)
                    
                # Add to the appropriate section dict
                if section == "api":
                    api_config[config_key] = value
                elif section == "models":
                    models_config[config_key] = value
                elif section == "chunking":
                    chunking_config[config_key] = value
                elif section == "analysis":
                    analysis_config[config_key] = value
                elif section == "files":
                    file_config[config_key] = value
                elif section == "reporting":
                    reporting_config[config_key] = value
                elif section == "cache":
                    cache_config[config_key] = value
                elif section == "other":
                    other_config[config_key] = value
        
        # Create the model config
        model_config = ModelConfig(**models_config)
        
        # Add model config to API config
        api_config["models"] = model_config
        
        # Create the config object
        config = cls(
            api=APIConfig(**api_config),
            chunking=ChunkingConfig(**chunking_config),
            analysis=AnalysisConfig(**analysis_config),
            files=FileConfig(**file_config),
            reporting=ReportingConfig(**reporting_config),
            cache=CacheConfig(**cache_config),
            prompt_template=other_config.get("prompt_template"),
            analysis_type=other_config.get("analysis_type", "general")
        )
        
        return config


def load_config_from_file(config_path: Union[str, Path]) -> CodeReviewConfig:
    """Load configuration from a file."""
    import json
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return CodeReviewConfig.from_dict(config_dict)


def save_config_to_file(config: CodeReviewConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to a file."""
    import json
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    # Convert Path objects to strings
    for key in ["input_dir", "output_file", "cache_dir"]:
        if key in config_dict and isinstance(config_dict[key], Path):
            config_dict[key] = str(config_dict[key])
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
