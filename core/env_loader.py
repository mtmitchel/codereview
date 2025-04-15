"""
Environment loader module for Code Review Tool.

This module handles loading environment variables from .env files
and making them available throughout the application.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_environment_variables():
    """
    Load environment variables from .env files.
    
    Attempts to load from:
    1. .env in the project directory
    2. .env in the parent directory (just in case)
    
    Returns:
        dict: A dictionary of loaded environment variables
    """
    env_vars = {}
    
    try:
        # Get the directory containing this file
        current_dir = Path(__file__).parent.absolute()
        
        # Project directory (parent of core)
        project_dir = current_dir.parent
        
        # Try .env in project directory
        project_env = project_dir / ".env"
        if project_env.exists():
            logger.info(f"Loading environment from {project_env}")
            load_dotenv(project_env)
            logger.info("Environment variables loaded successfully")
        else:
            logger.warning(f"No .env file found at {project_env}")
            
            # Try parent directory as fallback
            parent_env = project_dir.parent / ".env"
            if parent_env.exists():
                logger.info(f"Loading environment from parent directory {parent_env}")
                load_dotenv(parent_env)
                logger.info("Environment variables loaded from parent directory")
            else:
                logger.warning(f"No .env file found at {parent_env}")
        
        # Get all environment variables that might be relevant to the app
        # Starting with common API key patterns
        for key in os.environ:
            if "API_KEY" in key or "OPENROUTER" in key or "OLLAMA" in key:
                # Store without actual values in logs for security
                env_vars[key] = "***" if "key" in key.lower() or "token" in key.lower() else os.environ[key]
                
        return {
            "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
            "OLLAMA_URL": os.environ.get("OLLAMA_URL"),
        }
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}", exc_info=True)
        return {}
