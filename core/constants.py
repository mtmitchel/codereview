import os

# Constants for defaults
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3-8b-instruct"
DEFAULT_INPUT_DIR = os.path.expanduser("~")
DEFAULT_OUTPUT_FILE = os.path.join(os.path.expanduser("~"), "codereviews.md")
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_CONTEXT_LEN = 32768
DEFAULT_PROMPT_NAME = "Default"

# Constants for API timeouts
DEFAULT_API_TIMEOUT = 500  # seconds
MAX_API_TIMEOUT = 600  # seconds (10 minutes maximum)
MODEL_LIST_TIMEOUT = 20  # seconds
OLLAMA_TAGS_TIMEOUT = 10  # seconds

# Constants for chunking
MIN_CHUNK_SIZE = 512  # bytes
MAX_CHUNK_SIZE = 8192  # bytes
DEFAULT_BINARY_CHUNK_SIZE = 1024  # bytes

# Constants for file types
BINARY_FILE_EXTENSIONS = ['.bin', '.lock', '.part.bin', '.o', '.so', '.dylib', '.dll']
TEXT_FILE_EXTENSIONS = ['.py', '.js', '.html', '.css', '.md', '.txt', '.java', '.c', '.cpp', '.h', '.json', '.yaml', '.yml']

# API URLs and information
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
YOUR_SITE_URL = "local-dev"
YOUR_APP_NAME = "Code Reviewer GUI"

# File paths
PROMPT_FILE_PATH = os.path.join(os.path.expanduser("~"), ".code_reviewer_prompts.json")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".code_reviewer_cache")

# Token estimation constants
TOKEN_CHAR_RATIO = 4
DEFAULT_MAX_OUTPUT_TOKENS = 1024
MIN_TOKEN_BUDGET_FOR_CODE = 100
CONTEXT_SAFETY_FACTOR = 0.95

# Common exclusion patterns
DEFAULT_EXCLUSIONS = [
    # Version control
    "**/.git/",
    "**/.svn/",
    "**/.hg/",
    
    # Build and dependency directories
    "**/node_modules/",
    "**/bower_components/",
    "**/vendor/",
    "**/build/",
    "**/dist/",
    "**/target/",  # Rust build directory
    "**/.fingerprint/",  # Rust fingerprint files
    "**/debug/",  # Rust debug build outputs
    "**/release/",  # Rust release build outputs
    "**/__pycache__/",
    "**/*.pyc",
    "**/*.pyo",
    
    # Binary files
    "**/*.png",
    "**/*.jpg", 
    "**/*.jpeg",
    "**/*.gif",
    "**/*.ico",
    "**/*.svg",
    "**/*.ttf",
    "**/*.woff",
    "**/*.pdf",
    "**/*.zip",
    "**/*.gz",
    "**/*.tar",
    "**/*.mp3",
    "**/*.mp4",
    "**/*.wav",
    "**/*.mov"
]
