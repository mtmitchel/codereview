"""
Go-specific security patterns for vulnerability detection.
"""

import re

# SQL Injection patterns
SQL_INJECTION_PATTERNS = [
    (re.compile(r'(?i)db\.(?:Query|QueryRow|Exec)\s*\(\s*(?:[\'"][^\'"]*[\'"] \+ |[^,]+\+)'),
     "SQL Injection", "critical", "Possible SQL injection through string concatenation. Use parameterized queries."),
    (re.compile(r'(?i)db\.(?:Query|QueryRow|Exec)\s*\(\s*(?:fmt\.Sprintf\s*\(\s*[\'"][^\'"]*\%[^\'"]*[\'"])'),
     "SQL Injection", "critical", "Possible SQL injection using fmt.Sprintf. Use parameterized queries instead."),
    (re.compile(r'(?i)db\.(?:Query|QueryRow|Exec)\s*\(\s*strings\.(?:Join|Replace)'),
     "SQL Injection", "medium", "Potential SQL injection risk when using string manipulation. Ensure proper parameter usage."),
]

# Command Injection patterns
COMMAND_INJECTION_PATTERNS = [
    (re.compile(r'(?i)(?:exec\.Command|exec\.CommandContext)\s*\(\s*(?:[^,]+\s*\+\s*)'),
     "Command Injection", "critical", "Possible command injection through string concatenation. Avoid shell commands with user input."),
    (re.compile(r'(?i)(?:exec\.Command|exec\.CommandContext)\s*\(\s*(?:fmt\.Sprintf)'),
     "Command Injection", "high", "Potential command injection using fmt.Sprintf. Validate and sanitize all inputs."),
    (re.compile(r'(?i)(?:exec\.Command|exec\.CommandContext)\s*\(\s*(?:os\.Args|flag\.)'),
     "Command Injection", "high", "Command execution with command-line input. Validate all arguments before use."),
    (re.compile(r'(?i)(?:exec\.Command|exec\.CommandContext)\s*\(\s*[\'"](?:bash|sh|cmd|powershell)[\'"]\s*,\s*[\'"]-c[\'"]\s*,'),
     "Shell Command Execution", "high", "Shell command execution is dangerous. Avoid shell interpreter flags like -c."),
]

# Path traversal vulnerabilities
PATH_TRAVERSAL_PATTERNS = [
    (re.compile(r'(?i)(?:os\.Open|os\.OpenFile|ioutil\.ReadFile|ioutil\.WriteFile)\s*\(\s*(?:[^)]+\+)'),
     "Path Traversal", "high", "Possible path traversal in file operations. Validate file paths and avoid user-controlled paths."),
    (re.compile(r'(?i)filepath\.(?:Join|Clean)\s*\(\s*(?:[^,]+,\s*r\.URL\.Query\(\))'),
     "Path Traversal", "high", "Possible path traversal using URL parameters. Use filepath.Clean and validate against allowed paths."),
    (re.compile(r'(?i)(?:http\.ServeFile|http\.FileServer)\s*\(\s*[^,]+\s*,\s*(?:[^)]+\+)'),
     "Path Traversal", "high", "Possible path traversal in web file serving. Validate and restrict file access."),
]

# TLS/Crypto vulnerabilities
CRYPTO_PATTERNS = [
    (re.compile(r'(?i)crypto/md5\.'),
     "Weak Hashing", "high", "MD5 is cryptographically broken. Use SHA-256 or stronger."),
    (re.compile(r'(?i)crypto/sha1\.'),
     "Weak Hashing", "medium", "SHA1 is cryptographically weak. Use SHA-256 or stronger."),
    (re.compile(r'(?i)tls\.Config\s*\{(?:[^}]*,\s*)?InsecureSkipVerify\s*:\s*true'),
     "TLS Verification Disabled", "high", "TLS certificate verification is disabled. This is vulnerable to MITM attacks."),
    (re.compile(r'(?i)MinVersion\s*:\s*tls\.VersionSSL'),
     "Insecure TLS Version", "high", "Using outdated TLS/SSL version. Use TLS 1.2 or higher."),
    (re.compile(r'(?i)rand\.(?:Int|Intn|Float32|Float64)'),
     "Weak Randomness", "medium", "math/rand is not cryptographically secure. Use crypto/rand for security-sensitive operations."),
]

# JSON handling
JSON_PATTERNS = [
    (re.compile(r'(?i)json\.(?:Unmarshal|Decoder\.Decode)\s*\(\s*[^,]+\s*,\s*&map\[string\](?:interface|any)'),
     "Unsafe JSON Deserialization", "low", "Unmarshaling into map[string]interface{} without validation. Consider strict typing."),
    (re.compile(r'(?i)json\.(?:Unmarshal|Decoder\.Decode).*&struct\{'),
     "JSON Type Confusion", "low", "Unmarshaling into anonymous struct. Consider using defined types with validation."),
]

# Template injection vulnerabilities
TEMPLATE_PATTERNS = [
    (re.compile(r'(?i)template\.(?:New|Parse|ParseFiles)\s*\([^)]*\)\.(?:Execute|ExecuteTemplate)\s*\([^,]+,\s*(?:r\.URL|r\.Form|r\.Body)'),
     "Template Injection", "medium", "Potential template injection by executing templates with user input."),
    (re.compile(r'(?i)html/template\.(?:New|Parse|ParseFiles)\s*\([^)]*\)\.(?:Execute|ExecuteTemplate)\s*\([^,]+,\s*(?:r\.URL|r\.Form|r\.Body)'),
     "Template Injection", "low", "Using html/template with user input. Verify proper escaping."),
]

# Insecure HTTP settings
HTTP_PATTERNS = [
    (re.compile(r'(?i)http\.(?:ListenAndServe|Server\s*\{[^}]*\}\.ListenAndServe)\s*\(\s*[^,]+\s*,'),
     "Insecure HTTP", "medium", "Using plain HTTP server. Consider using HTTPS for secure communication."),
    (re.compile(r'(?i)w\.Header\(\)\.(?:Set|Add)\s*\(\s*[\'"](?:Access-Control-Allow-Origin)[\'"]\s*,\s*[\'"]\\*[\'"]\s*\)'),
     "Insecure CORS Configuration", "medium", "Using wildcard CORS origin which may expose your API to unauthorized domains."),
    (re.compile(r'(?i)w\.Header\(\)\.(?:Set|Add)\s*\(\s*[\'"](?:Strict-Transport-Security|X-Content-Type-Options|X-Frame-Options|Content-Security-Policy)[\'"]\s*,'),
     "Security Headers Implementation", "low", "Using security headers is good practice. Verify correct values."),
]

# Context cancelation
CONTEXT_PATTERNS = [
    (re.compile(r'(?i)ctx\s*:=\s*context\.(?:Background|TODO)\(\)'),
     "Missing Timeout/Deadline", "low", "Using background context without timeout/deadline can lead to resource leaks."),
    (re.compile(r'(?i)select\s*\{\s*(?:[^}]*?\s*case[^:]*:[^}]*?){2,}\s*\}'),
     "Select Without Default", "low", "Select without default case can block indefinitely. Consider adding timeout."),
]

# Error handling vulnerabilities
ERROR_PATTERNS = [
    (re.compile(r'(?i)fmt\.(?:Print|Println|Printf)\s*\(\s*err\s*\)'),
     "Error Exposure", "low", "Printing raw error to output. Consider structured error handling."),
    (re.compile(r'(?i)log\.(?:Print|Println|Printf)\s*\(\s*err\s*\)'),
     "Error Logging", "low", "Logging raw error. Consider structured logging with context."),
    (re.compile(r'(?i)http\.Error\s*\(\s*[^,]+\s*,\s*err\.Error\(\)'),
     "Error Information Leakage", "medium", "Exposing raw error messages to HTTP clients. Use sanitized error messages."),
]

# Potential DoS vulnerabilities
DOS_PATTERNS = [
    (re.compile(r'(?i)for\s+[^{]*\{\s*(?:time\.Sleep|runtime\.Gosched)\s*\([^)]+\)\s*\}'),
     "Potential Resource Exhaustion", "low", "Loop with sleep or gosched can potentially consume resources."),
    (re.compile(r'(?i)go\s+func\s*\([^)]*\)\s*\{(?:[^}]*?\s*for\s+[^{]*\{[^}]*?\s*\}){2,}[^}]*?\s*\}\s*\(\s*\)'),
     "Potential Goroutine Leak", "medium", "Nested loops in goroutine may consume excessive resources."),
]

# Combine all patterns
GO_PATTERNS = (
    SQL_INJECTION_PATTERNS +
    COMMAND_INJECTION_PATTERNS +
    PATH_TRAVERSAL_PATTERNS +
    CRYPTO_PATTERNS +
    JSON_PATTERNS +
    TEMPLATE_PATTERNS +
    HTTP_PATTERNS +
    CONTEXT_PATTERNS +
    ERROR_PATTERNS +
    DOS_PATTERNS
) 