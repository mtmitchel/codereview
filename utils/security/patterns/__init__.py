"""
Security vulnerability detection patterns for different programming languages.
"""

from .python_patterns import PYTHON_PATTERNS
from .javascript_patterns import JAVASCRIPT_PATTERNS
from .rust_patterns import RUST_PATTERNS
from .php_patterns import PHP_PATTERNS
from .java_patterns import JAVA_PATTERNS
from .csharp_patterns import CSHARP_PATTERNS
from .go_patterns import GO_PATTERNS
from .ruby_patterns import RUBY_PATTERNS

__all__ = [
    'PYTHON_PATTERNS',
    'JAVASCRIPT_PATTERNS',
    'RUST_PATTERNS',
    'PHP_PATTERNS',
    'JAVA_PATTERNS',
    'CSHARP_PATTERNS',
    'GO_PATTERNS',
    'RUBY_PATTERNS',
] 