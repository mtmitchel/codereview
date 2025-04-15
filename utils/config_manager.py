import os
import json
import yaml  # Added YAML support
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from core.validation.config_validator import validate_config  # Import config validator

class ConfigManager:
    """
    Configuration manager for the code review tool.
    Handles loading and saving settings from/to JSON or YAML files.
    """
    
    DEFAULT_CONFIG_PATH = "config/default_config.json"
    USER_CONFIG_PATH = "config/user_config.json"
    
    # Default configuration structure
    DEFAULT_CONFIG = {
        "model_source": "OpenRouter",
        "model": "openai/gpt-3.5-turbo",
        "api_key": "",
        "ollama_url": "http://localhost:11434",
        "api_timeout": 60,
        "model_context_length": 8192,
        
        "chunk_strategy": "Syntax Aware (Python only)",
        "chunk_size": 4000,
        "overlap": 200,
        "auto_suggest_chunking": True,
        "max_workers": 4,
        "use_cache": True,
        "incremental_analysis": False,
        "adaptive_batch_size": True,
        
        "run_flake8": True,
        "run_semgrep": True,
        "enable_security_analysis": True,
        "enable_semantic_analysis": True,
        "enable_dependency_analysis": True,
        "cross_file_analysis": False,
        "semantic_sensitivity": "medium",
        "analysis_type": "general",
        
        "min_severity_to_report": "Low",
        "report_clean_files": True,
        
        "input_dir": "",
        "output_file": "code_review_report.md",
        "file_types": [
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", 
            ".h", ".hpp", ".cs", ".go", ".rb", ".php", ".swift", 
            ".kt", ".rs"
        ],
        "exclude_patterns": [
            ".git/", "node_modules/", "venv/", "env/", "__pycache__/", 
            "*.min.js", "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dylib", 
            "*.dll", "*.exe", "*.bin", "*.dat", "*.db", "*.sqlite", 
            "*.sqlite3"
        ],
        
        "enable_resource_monitoring": True,
        "cpu_percent_max": 80,
        "memory_percent_max": 75,
        "max_worker_count": 8,
        "threads_max": 20,
        
        "ui": {
            "theme": "system",
            "font_size": 10,
            "enable_animations": True,
            "show_line_numbers": True,
            "wrap_text": True,
            "recent_directories": []
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a custom config file
        """
        self.logger = logging.getLogger("ConfigManager")
        
        # Set config paths
        self.user_config_path = Path(config_path or self.USER_CONFIG_PATH)
        self.default_config_path = Path(self.DEFAULT_CONFIG_PATH)
        
        # Load or create default config
        self._ensure_default_config()
        
        # Load configuration
        self.config = self._load_config()
        
        # Validate the configuration
        self.config = validate_config(self.config)
        
    def _ensure_default_config(self) -> None:
        """Ensure the default configuration file exists."""
        if not self.default_config_path.exists():
            # Create directory if it doesn't exist
            self.default_config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write default config
            with open(self.default_config_path, 'w') as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=2)
                
            self.logger.info(f"Created default configuration at {self.default_config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from files, merging default and user configs.
        
        Returns:
            Dict containing the merged configuration
        """
        # Start with the default configuration
        config = self.DEFAULT_CONFIG.copy()
        
        # Try to load the default config file
        if self.default_config_path.exists():
            try:
                config = self._load_config_file(self.default_config_path, config)
                self.logger.info(f"Loaded default configuration from {self.default_config_path}")
            except Exception as e:
                self.logger.warning(f"Error loading default config: {str(e)}")
        
        # Try to load the user config file
        if self.user_config_path.exists():
            try:
                config = self._load_config_file(self.user_config_path, config)
                self.logger.info(f"Loaded user configuration from {self.user_config_path}")
            except Exception as e:
                self.logger.warning(f"Error loading user config: {str(e)}")
        
        return config
    
    def _load_config_file(self, file_path: Path, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load configuration from a file, supporting both JSON and YAML formats.
        
        Args:
            file_path: Path to the config file
            base_config: Base configuration to update
            
        Returns:
            Updated configuration dictionary
        """
        with open(file_path, 'r') as f:
            file_content = f.read()
            
            # Try loading as JSON first
            try:
                loaded_config = json.loads(file_content)
            except json.JSONDecodeError:
                # If JSON fails, try YAML
                try:
                    loaded_config = yaml.safe_load(file_content)
                    if not isinstance(loaded_config, dict):
                        raise ValueError(f"Invalid YAML configuration format in {file_path}")
                except Exception as e:
                    raise ValueError(f"Failed to parse {file_path} as JSON or YAML: {str(e)}")
            
            # Update the base config
            self._deep_update(base_config, loaded_config)
            
        return base_config
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep update a nested dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def save_config(self, file_format: str = 'json') -> None:
        """
        Save the current configuration to the user config file.
        
        Args:
            file_format: Format to save as ('json' or 'yaml')
        """
        try:
            # Create directory if it doesn't exist
            self.user_config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use the specified format extension if not already in path
            output_path = self.user_config_path
            if file_format.lower() == 'yaml' and not str(output_path).lower().endswith(('.yaml', '.yml')):
                output_path = output_path.with_suffix('.yaml')
            elif file_format.lower() == 'json' and not str(output_path).lower().endswith('.json'):
                output_path = output_path.with_suffix('.json')
            
            with open(output_path, 'w') as f:
                if file_format.lower() == 'yaml':
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                else:
                    json.dump(self.config, f, indent=2)
                
            self.logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation for nested keys.
        
        Args:
            key_path: Path to the configuration value (e.g., "api.model")
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value or the default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation for nested keys.
        
        Args:
            key_path: Path to the configuration value (e.g., "api.model")
            value: Value to set
        """
        keys = key_path.split('.')
        target = self.config
        
        # Navigate to the correct nested dictionary
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
            
        # Set the value
        target[keys[-1]] = value
        
        # Validate the configuration after setting
        self.config = validate_config(self.config)
    
    def reset_to_defaults(self) -> None:
        """Reset the configuration to default values."""
        self.config = self.DEFAULT_CONFIG.copy()
        # Validate the configuration after resetting
        self.config = validate_config(self.config)
        self.save_config()
        self.logger.info("Reset configuration to defaults")
        
    def export_config(self, export_path: str, file_format: str = None) -> None:
        """
        Export the current configuration to a file.
        
        Args:
            export_path: Path to export the configuration to
            file_format: Optional format override ('json' or 'yaml')
        """
        try:
            # Determine format from file extension if not specified
            if file_format is None:
                if export_path.lower().endswith(('.yaml', '.yml')):
                    file_format = 'yaml'
                else:
                    file_format = 'json'
                    
            with open(export_path, 'w') as f:
                if file_format.lower() == 'yaml':
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                else:
                    json.dump(self.config, f, indent=2)
                
            self.logger.info(f"Exported configuration to {export_path}")
        except Exception as e:
            self.logger.error(f"Error exporting config: {str(e)}")
            
    def import_config(self, import_path: str) -> None:
        """
        Import configuration from a file.
        
        Args:
            import_path: Path to import the configuration from
        """
        try:
            # Load the config from file
            imported_config = {}
            with open(import_path, 'r') as f:
                if import_path.lower().endswith(('.yaml', '.yml')):
                    imported_config = yaml.safe_load(f)
                else:
                    imported_config = json.load(f)
                
            # Validate the imported config
            if not isinstance(imported_config, dict):
                raise ValueError("Invalid configuration format")
                
            # Update the current config
            self.config = imported_config
            
            # Validate the configuration
            self.config = validate_config(self.config)
            
            # Save the imported config
            self.save_config()
            
            self.logger.info(f"Imported configuration from {import_path}")
        except Exception as e:
            self.logger.error(f"Error importing config: {str(e)}")
            raise
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """
        Get resource limit configuration.
        
        Returns:
            Dictionary with resource limit configuration
        """
        return {
            'enable_resource_monitoring': self.get('enable_resource_monitoring', True),
            'cpu_percent_max': self.get('cpu_percent_max', 80),
            'memory_percent_max': self.get('memory_percent_max', 75),
            'max_worker_count': self.get('max_worker_count', 8),
            'threads_max': self.get('threads_max', 20),
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration.
        
        Returns:
            Dictionary with LLM configuration
        """
        return {
            'model_source': self.get('model_source', 'OpenRouter'),
            'model': self.get('model', 'openai/gpt-3.5-turbo'),
            'api_key': self.get('api_key', ''),
            'ollama_url': self.get('ollama_url', 'http://localhost:11434'),
            'api_timeout': self.get('api_timeout', 60),
            'model_context_length': self.get('model_context_length', 8192),
        }
    
    def get_exclude_patterns(self) -> List[str]:
        """
        Get the list of exclude patterns.
        
        Returns:
            List of patterns to exclude
        """
        return self.get('exclude_patterns', [])
    
    def set_exclude_patterns(self, patterns: List[str]) -> None:
        """
        Set the list of exclude patterns.
        
        Args:
            patterns: List of patterns to exclude
        """
        self.set('exclude_patterns', patterns)
        
    def get_file_types(self) -> List[str]:
        """
        Get the list of file types to analyze.
        
        Returns:
            List of file extensions
        """
        return self.get('file_types', [])
    
    def add_recent_directory(self, directory: str, max_entries: int = 10) -> None:
        """
        Add a directory to the list of recently used directories.
        
        Args:
            directory: Directory path to add
            max_entries: Maximum number of recent directories to keep
        """
        recent_dirs = self.get('ui.recent_directories', [])
        
        # Remove the directory if it's already in the list
        if directory in recent_dirs:
            recent_dirs.remove(directory)
            
        # Add to the beginning of the list
        recent_dirs.insert(0, directory)
        
        # Trim the list if necessary
        if len(recent_dirs) > max_entries:
            recent_dirs = recent_dirs[:max_entries]
            
        self.set('ui.recent_directories', recent_dirs)
        
    def get_recent_directories(self) -> List[str]:
        """
        Get the list of recently used directories.
        
        Returns:
            List of recently used directories
        """
        return self.get('ui.recent_directories', [])