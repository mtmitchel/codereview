"""
Language-specific analyzers for the Smart Exclusion system.

This package contains analyzers for different programming languages,
which extract language-specific metadata and importance indicators.
"""

from .python_analyzer import PythonAnalyzer
from .javascript_analyzer import JavaScriptAnalyzer

__all__ = ['PythonAnalyzer', 'JavaScriptAnalyzer']
