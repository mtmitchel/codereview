"""
Configuration validation module for the Code Review Tool.
"""

import os
import sys

# Ensure the project root is in the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Export main validation classes
from .config_validator import ConfigValidator, ConfigValidationError
