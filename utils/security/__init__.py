"""
Security analyzer module for identifying security vulnerabilities in code.

This package provides tools for detecting security issues in various programming languages.
"""

from .vulnerability import SecurityVulnerability
from .analyzer import SecurityAnalyzerPlugin

__all__ = [
    'SecurityVulnerability',
    'SecurityAnalyzerPlugin',
] 