#!/usr/bin/env python3
"""
Smart Exclusion Analyzer for Code Review Tool

This module provides functionality to analyze a codebase and determine
which files are core to functionality and which can be safely excluded.

DEPRECATION NOTICE: This monolithic implementation is being replaced by a
modular package structure in utils.smart_exclusion. Please update your imports
to use the new package instead.
"""

import warnings
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

# Re-export the SmartExclusionAnalyzer from the new modular implementation
from .smart_exclusion.migration import LegacySmartExclusionAnalyzer as SmartExclusionAnalyzer

# Configure logger
logger = logging.getLogger("CodeReviewTool.SmartExclusion")

# Emit deprecation warning when module is imported
warnings.warn(
    "The monolithic smart_exclusion.py module is deprecated. "
    "Please use the new modular implementation from utils.smart_exclusion instead. "
    "This module will be removed in a future release.",
    DeprecationWarning, stacklevel=2
)

# For backward compatibility, export any constants or types needed by existing code
# Re-export these from the SmartExclusionAnalyzer class
DEFAULT_SAFE_EXCLUDES = SmartExclusionAnalyzer.DEFAULT_SAFE_EXCLUDES
HIGH_ENTROPY_THRESHOLD = SmartExclusionAnalyzer.HIGH_ENTROPY_THRESHOLD
MODERATE_ENTROPY_THRESHOLD = SmartExclusionAnalyzer.MODERATE_ENTROPY_THRESHOLD

__all__ = ['SmartExclusionAnalyzer', 'DEFAULT_SAFE_EXCLUDES', 
           'HIGH_ENTROPY_THRESHOLD', 'MODERATE_ENTROPY_THRESHOLD']