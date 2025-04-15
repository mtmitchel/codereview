"""
Core functionality for code review operations.
"""

from .caching import (
    get_file_hash,
    get_cached_review,
    cache_review,
    clear_cache,
    CACHE_DIR
)

from .token_estimation import (
    estimate_tokens,
    TOKEN_CHAR_RATIO,
    DEFAULT_MAX_OUTPUT_TOKENS,
    MIN_TOKEN_BUDGET_FOR_CODE,
    CONTEXT_SAFETY_FACTOR
)

from .chunking import (
    split_text,
    chunk_python_code_by_structure,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    HAS_GET_SOURCE_SEGMENT
)

from .flake8_utils import (
    run_flake8,
    filter_flake8_output
)

from .prompt_templates import (
    DEFAULT_PROMPT_TEMPLATE,
    CROSS_FILE_PROMPT_TEMPLATE,
    SUMMARIZATION_PROMPT_TEMPLATE,
    PROMPT_FILE_PATH,
    DEFAULT_PROMPT_NAME
)

from .constants import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OLLAMA_CONTEXT_LEN,
    DEFAULT_API_TIMEOUT,
    MODEL_LIST_TIMEOUT,
    OLLAMA_TAGS_TIMEOUT,
    OPENROUTER_API_URL,
    YOUR_SITE_URL,
    YOUR_APP_NAME
)

from .dependencies import (
    ServiceContainer,
    get_container,
    container
)

from .validation.config_validator import ConfigValidator

from .command import (
    Command, CommandResult, ReviewCodeCommand, UpdateReviewConfigCommand,
    # Rest of the imports remain the same
)
