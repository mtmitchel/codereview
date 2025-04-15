"""
Python-specific security patterns for vulnerability detection.
"""

import re

# Compiled regex patterns for Python code

# SQL Injection patterns
SQL_INJECTION_PATTERNS = [
    (re.compile(r'(?i)(?:execute|executemany|executequery)\s*\(\s*[f]{0,1}[\'"][^\'"\)]*?(?:%s|%d)[^\'"\)]*?[\'"]'), 
     "SQL Injection", "critical", "Possible SQL injection (string formatting). Use parameterized queries."),
    (re.compile(r'(?i)(?:execute|executemany|executequery)\s*\(\s*.*?\.format\('), 
     "SQL Injection", "critical", "Possible SQL injection (string .format()). Use parameterized queries."),
    (re.compile(r'(?i)(?:execute|executemany|executequery)\s*\(\s*f[\'"][^\'"]*\{[^\}]*\}[^\'"]*[\'"]'), 
     "SQL Injection", "critical", "Possible SQL injection (f-string). Use parameterized queries."),
]

# Command Injection patterns
COMMAND_INJECTION_PATTERNS = [
    (re.compile(r'(?i)(?:os\.system|subprocess\.(?:call|run|Popen|check_output)).*?shell\s*=\s*True'), 
     "Insecure Subprocess", "high", "Using shell=True with subprocess can be dangerous if command includes external input."),
    (re.compile(r'(?i)(?:os\.system|subprocess\.(?:call|run|Popen|check_output)).*\+.*(?:user_input|data|param|args|kwargs|request)'), 
     "Command Injection", "critical", "Possible command injection via string concatenation. Sanitize input and avoid shell=True."),
    (re.compile(r'(?i)(?:os\.system|subprocess\.(?:call|run|Popen|check_output)).*format\('), 
     "Command Injection", "critical", "Possible command injection via string formatting. Avoid shell commands with user input."),
    (re.compile(r'(?i)(?:os\.system|subprocess\.(?:call|run|Popen|check_output)).*f[\'"](.*?\{.*?\})'), 
     "Command Injection", "critical", "Possible command injection via f-string. Avoid shell commands with user input."),
]

# Use of dangerous functions
DANGEROUS_FUNCTION_PATTERNS = [
    (re.compile(r'\beval\s*\('), 
     "Use of Eval", "high", "eval() can execute arbitrary code and is dangerous with untrusted input."),
    (re.compile(r'\bexec\s*\('), 
     "Use of Exec", "high", "exec() can execute arbitrary code and is dangerous with untrusted input."),
    (re.compile(r'\bpickle\.(?:load|loads)\s*\('), 
     "Use of Pickle", "high", "Pickle is not secure against erroneous or maliciously constructed data. Avoid deserializing untrusted data."),
    (re.compile(r'(?:yaml|pyyaml)\.(?:load|unsafe_load)\s*\('), 
     "Unsafe YAML Loading", "high", "Using yaml.load() without SafeLoader can execute arbitrary code. Use yaml.safe_load() instead."),
]

# Insecure parsing
INSECURE_PARSING_PATTERNS = [
    (re.compile(r'(?:lxml|xml)\.etree\.ElementTree\.fromstring\(.*?user'), 
     "XXE Vulnerability", "high", "XML parsing without disabling external entities can lead to XXE attacks."),
    (re.compile(r'(?:ElementTree|xml\.etree)\.parse\(.*?user'), 
     "XXE Vulnerability", "high", "XML parsing without disabling external entities can lead to XXE attacks."),
]

# Debug Statements Left
DEBUG_STATEMENT_PATTERNS = [
    (re.compile(r'import\s+pdb\b|pdb\.set_trace\s*\('), 
     "Debug Statement Left", "low", "Debugging statement (pdb.set_trace or import) left in code."),
    (re.compile(r'print\s*\(\s*[\'"]DEBUG|print\s*\(\s*f[\'"]DEBUG'), 
     "Debug Statement Left", "low", "Debug print statement left in code."),
]

# Weak Hashing Algorithms
WEAK_HASHING_PATTERNS = [
    (re.compile(r'\bhashlib\.md5\b'), 
     "Weak Hashing Algorithm", "medium", "MD5 is considered weak for security purposes. Use SHA-256 or stronger."),
    (re.compile(r'\bhashlib\.sha1\b'), 
     "Weak Hashing Algorithm", "medium", "SHA1 is considered weak for security purposes. Use SHA-256 or stronger."),
]

# Insecure Temporary File Creation
INSECURE_TEMP_FILE_PATTERNS = [
    (re.compile(r'\btempfile\.mktemp\b'), 
     "Insecure Temporary File", "medium", "tempfile.mktemp() is insecure due to potential race conditions. Use tempfile.mkstemp() or TemporaryDirectory."),
]

# Use of Assert (can be disabled)
ASSERT_PATTERNS = [
    (re.compile(r'^\s*assert\b'), 
     "Use of Assert", "low", "Assert statements might be disabled in production environments (e.g., with python -O). Do not rely on them for security checks."),
]

# Insecure CORS
INSECURE_CORS_PATTERNS = [
    (re.compile(r'(?i)Access-Control-Allow-Origin[\'"\s]*:[\'"\s]*\*'), 
     "Insecure CORS Configuration", "medium", "Wildcard CORS origin can lead to security vulnerabilities. Specify exact origins instead."),
]

# Insecure cookie settings
INSECURE_COOKIE_PATTERNS = [
    (re.compile(r'(?i)set_cookie\(.*?secure\s*=\s*False'), 
     "Insecure Cookie", "medium", "Setting cookies without the secure flag can expose them to MITM attacks."),
    (re.compile(r'(?i)(?:set_cookie|cookie)\(.*?httponly\s*=\s*False'), 
     "Insecure Cookie", "medium", "Setting cookies without the HttpOnly flag can expose them to XSS attacks."),
]

# JWT without verification
JWT_PATTERNS = [
    (re.compile(r'(?i)jwt\.decode\(.*?verify\s*=\s*False'), 
     "Insecure JWT Verification", "high", "JWT decoding without verification is insecure."),
]

# Hardcoded Credentials
HARDCODED_CREDENTIALS_PATTERNS = [
    (re.compile(r'(?:password|passwd|pwd|apiKey|api_key|secret|token|access_key)\s*=\s*["\'`][A-Za-z0-9_!@#$%^&*()-+={}\[\]|\\:;<>,.?\/]{8,}["\'`]', re.IGNORECASE),
     "Hardcoded Credentials",
     "high",
     "Hardcoded credentials detected. Store secrets in environment variables or a secure vault."
    ),
]

# Combine all patterns
PYTHON_PATTERNS = (
    SQL_INJECTION_PATTERNS + 
    COMMAND_INJECTION_PATTERNS + 
    DANGEROUS_FUNCTION_PATTERNS + 
    INSECURE_PARSING_PATTERNS + 
    DEBUG_STATEMENT_PATTERNS + 
    WEAK_HASHING_PATTERNS + 
    INSECURE_TEMP_FILE_PATTERNS + 
    ASSERT_PATTERNS + 
    INSECURE_CORS_PATTERNS + 
    INSECURE_COOKIE_PATTERNS + 
    JWT_PATTERNS + 
    HARDCODED_CREDENTIALS_PATTERNS
) 