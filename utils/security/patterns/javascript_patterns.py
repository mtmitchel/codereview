"""
JavaScript-specific security patterns for vulnerability detection.
"""

import re

# JavaScript security patterns
# Each pattern is a tuple of (regex_pattern, vulnerability_type, severity, message)
JS_BASE_PATTERNS = [
    # Hardcoded Credentials
    (
        re.compile(r'(?:password|passwd|pwd|apiKey|api_key|secret|token|access_key)\s*[:=]\s*["\'`][A-Za-z0-9_!@#$%^&*()-+={}\[\]|\\:;<>,.?\/]{8,}["\'`]', re.IGNORECASE),
        "Hardcoded Credentials",
        "high",
        "Hardcoded credentials detected. Store secrets in environment variables or a secure vault."
    ),
    
    # XSS
    (
        re.compile(r'\.innerHTML\s*=|\.outerHTML\s*=|document\.write\(', re.IGNORECASE),
        "Cross-Site Scripting",
        "high",
        "Potential XSS vulnerability. Use textContent or DOMPurify to sanitize input."
    ),
    
    # Prototype Pollution
    (
        re.compile(r'Object\.assign\(\s*[^,]+\s*,\s*[\w.]+\)|_*proto_*', re.IGNORECASE),
        "Prototype Pollution",
        "medium",
        "Potential prototype pollution vulnerability. Sanitize and validate untrusted object properties."
    ),
    
    # Prototype Modification
    (
        re.compile(r'Object\.prototype\.\w+\s*=', re.IGNORECASE),
        "Prototype Modification",
        "medium",
        "Direct modification of Object prototype can affect all objects and cause unexpected behavior."
    ),
    
    # Eval usage
    (
        re.compile(r'eval\(|Function\(|new Function\(|setTimeout\(\s*["\'`]|setInterval\(\s*["\'`]', re.IGNORECASE),
        "Unsafe Code Execution",
        "high",
        "Unsafe code execution method detected. Avoid using eval() or similar functions with untrusted input."
    ),
    
    # Insecure Regular Expressions
    (
        re.compile(r'[\w.]+\.match\(\s*\/(\.\*|\([^)]*\)\*)\/', re.IGNORECASE),
        "Regular Expression DoS",
        "medium",
        "Potentially vulnerable regular expression detected. Avoid unbounded repetition operators."
    ),
    
    # Timing attacks
    (
        re.compile(r'===|!==|==|!=.*password|token|secret', re.IGNORECASE),
        "Timing Attack",
        "medium",
        "Potential timing attack vulnerability in credential comparison. Use a constant-time comparison function."
    ),
    
    # Insecure HTTP
    (
        re.compile(r'http:\/\/(?!localhost|127\.0\.0\.1)', re.IGNORECASE),
        "Insecure HTTP",
        "medium",
        "Insecure HTTP URL detected. Use HTTPS instead."
    ),
    
    # Insecure Randomness
    (
        re.compile(r'Math\.random\(', re.IGNORECASE),
        "Insecure Randomness",
        "medium",
        "Math.random() is not cryptographically secure. Use crypto.getRandomValues() instead."
    )
]

# DOM-based XSS
DOM_XSS_PATTERNS = [
    (re.compile(r'(?i)(?:innerHTML|outerHTML|document\.write|document\.writeln)\s*=\s*'), 
     "Cross-Site Scripting (XSS)", "high", "Possible XSS vulnerability. Use textContent instead of innerHTML or sanitize input."),
]

# Code Injection
CODE_INJECTION_PATTERNS = [
    (re.compile(r'(?i)(?:eval|setTimeout|setInterval|Function)\s*\(\s*([^\)]+)\)'), 
     "Possible Code Injection", "high", "Evaluating dynamic code can lead to code injection vulnerabilities."),
]

# Insecure JWT usage
JWT_PATTERNS = [
    (re.compile(r'(?i)(?:jwt|jsonwebtoken)\.verify\s*\(\s*[^,]+,\s*[^,]+\s*\)'), 
     "Insecure JWT Verification", "high", "JWT verification without explicit algorithm specification. Specify algorithms to prevent algorithm switching attacks."),
]

# Insecure CORS
CORS_PATTERNS = [
    (re.compile(r'(?i)Access-Control-Allow-Origin.*[\'"]\*[\'"]'), 
     "Insecure CORS Configuration", "medium", "Wildcard CORS origin can lead to security vulnerabilities. Specify exact origins instead."),
]

# NoSQL Injection
NOSQL_INJECTION_PATTERNS = [
    (re.compile(r'(?i)db\.collection\(.*?\)\.find\(\s*\{.*?\$where\s*:'), 
     "NoSQL Injection", "high", "Possible NoSQL injection vulnerability using $where operator."),
    (re.compile(r'(?i)new\s+RegExp\(\s*(?:req|request|user|input)'), 
     "NoSQL Injection", "high", "Creating RegExp from user input can lead to ReDoS attacks."),
]

# Insecure Randomness
RANDOMNESS_PATTERNS = [
    (re.compile(r'(?i)Math\.random\(\)'), 
     "Weak Randomness", "medium", "Math.random() is not cryptographically secure. Use crypto.getRandomValues() instead."),
]

# Path Traversal
PATH_TRAVERSAL_PATTERNS = [
    (re.compile(r'(?i)(?:fs|require\([\'"]fs[\'"]\))\.(?:readFile|readFileSync|createReadStream)\s*\(\s*(?:req|request|user|input|query)'), 
     "Path Traversal", "high", "Possible path traversal vulnerability. Validate file paths."),
]

# Combine all patterns
JAVASCRIPT_PATTERNS = (
    JS_BASE_PATTERNS +
    DOM_XSS_PATTERNS + 
    CODE_INJECTION_PATTERNS + 
    JWT_PATTERNS + 
    CORS_PATTERNS + 
    NOSQL_INJECTION_PATTERNS + 
    RANDOMNESS_PATTERNS + 
    PATH_TRAVERSAL_PATTERNS
) 