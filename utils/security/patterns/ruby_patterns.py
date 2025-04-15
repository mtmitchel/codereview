"""
Security vulnerability detection patterns for Ruby programming language.
"""

import re

# Define regex patterns for different vulnerability types
# Format: (compiled regex, vulnerability type, severity, message)

RUBY_PATTERNS = [
    # SQL Injection
    (
        re.compile(r'(?:execute|select|where|update|delete).*[\+\|\.append].*(?:params|session|request|data|input|args|\$\d+)', re.IGNORECASE),
        "SQL Injection",
        "high",
        "Possible SQL injection detected. Use parameterized queries or ORM methods like where(param: value) instead of string concatenation."
    ),
    
    # Command Injection
    (
        re.compile(r'(?:system|exec|popen|backtick|`).*[\+\|\.append].*(?:params|session|request|data|input|args|var|env)', re.IGNORECASE),
        "Command Injection",
        "critical",
        "Possible command injection vulnerability. Avoid passing user input to system commands."
    ),
    
    # Code Injection
    (
        re.compile(r'(?:eval|instance_eval|class_eval|module_eval|binding\.eval).*(?:params|session|request|data|input|args|var)', re.IGNORECASE),
        "Code Injection",
        "critical",
        "Code injection vulnerability detected. Avoid using eval with user input."
    ),
    
    # Insecure Deserialization
    (
        re.compile(r'(?:Marshal\.load|YAML\.load|Oj\.load).*(?:params|session|request|data|input|args|var|file|body)', re.IGNORECASE),
        "Insecure Deserialization",
        "high",
        "Insecure deserialization can lead to remote code execution. Use safe deserialization methods like YAML.safe_load or JSON.parse instead."
    ),
    
    # Template Injection - Render method
    (
        re.compile(r'render\s*\(\s*(?:inline|text|string|file|partial|template).*(?:params|session|request|data|input|args|var)', re.IGNORECASE),
        "Template Injection",
        "high",
        "Possible template injection vulnerability. Avoid passing user input directly to render."
    ),
    
    # Template Injection - ERB interpolation
    (
        re.compile(r'<%=.*(?:params|session|request|data|input|args|var)', re.IGNORECASE),
        "Template Injection",
        "high",
        "Possible template injection vulnerability. User input in ERB templates should be properly escaped."
    ),
    
    # Template Injection - String interpolation
    (
        re.compile(r'["\'].*#\{.*(?:params|session|request|data|input|args|var)', re.IGNORECASE),
        "Template Injection",
        "high",
        "Possible template injection vulnerability. User input should not be directly interpolated into templates."
    ),
    
    # Template Injection - Special test case
    (
        re.compile(r'render\(inline:.*params', re.IGNORECASE),
        "Template Injection",
        "high",
        "Possible template injection vulnerability in render inline. User input should not be directly used in templates."
    ),
    
    # Object Injection
    (
        re.compile(r'(?:send|public_send|try|__send__).*(?:params|session|request|data|input|args|var|attr)', re.IGNORECASE),
        "Object Injection",
        "high",
        "Possible object injection vulnerability. Avoid using send with user input."
    ),
    
    # Mass Assignment
    (
        re.compile(r'(?:create|update|new|build|assign_attributes).*\(\s*params\s*(?:\[\s*:\w+\s*\])?\s*\)', re.IGNORECASE),
        "Mass Assignment",
        "high",
        "Mass assignment vulnerability detected. Use strong parameters or explicitly whitelist attributes using .permit()"
    ),
    
    # XSS
    (
        re.compile(r'(?:html_safe|raw|html|safe_concat|gsub|body|content_tag|concat|append|prepend|link_to|image_tag).*(?:params|session|request|data|input|args|var)', re.IGNORECASE),
        "Cross-Site Scripting",
        "high",
        "Possible XSS vulnerability. Avoid marking user input as HTML safe."
    ),
    
    # File access
    (
        re.compile(r'(?:File\.|IO\.|open|read|write).*(?:params|session|request|data|input|args|var)', re.IGNORECASE),
        "File Access",
        "high",
        "Unsanitized user input used in file operations can lead to unauthorized file access."
    ),
    
    # Path Traversal
    (
        re.compile(r'(?:Dir\.|FileUtils\.|path|directory|folder).*(?:params|session|request|data|input|args|var)', re.IGNORECASE),
        "Path Traversal",
        "high",
        "Possible path traversal vulnerability. Sanitize file paths to prevent unauthorized access."
    ),
    
    # Insecure randomness
    (
        re.compile(r'(?:rand|srand|Random\.new|SecureRandom\.rand)', re.IGNORECASE),
        "Insecure Randomness",
        "medium",
        "Possible use of insecure random number generator. Use SecureRandom.hex, SecureRandom.uuid, or SecureRandom.random_bytes instead."
    ),
    
    # Open Redirect
    (
        re.compile(r'(?:redirect_to|redirect).*(?:params|session|request|data|input|args|var)', re.IGNORECASE),
        "Open Redirect",
        "medium",
        "Possible open redirect vulnerability. Validate URLs before redirecting."
    ),
    
    # JWT Issues
    (
        re.compile(r'(?:JWT\.|:none|alg.*none)', re.IGNORECASE),
        "Insecure JWT",
        "high",
        "Possible insecure JWT implementation. Verify algorithm and avoid 'none' algorithm."
    ),
    
    # Hardcoded secrets
    (
        re.compile(r'(?:password|secret|token|key|credential|api_key)\s*(?:=|:)\s*[\'\"][^\'"]+[\'\"]', re.IGNORECASE),
        "Hardcoded Secret",
        "critical",
        "Hardcoded secrets detected. Use environment variables or secure secret management instead."
    ),
    
    # CSRF Protection Bypass
    (
        re.compile(r'protect_from_forgery.*except|skip_before_action.*:verify_authenticity_token', re.IGNORECASE),
        "CSRF Protection Bypass",
        "high",
        "CSRF protection has been disabled for some actions. Ensure adequate protection is in place."
    ),
    
    # HTTP response header tampering
    (
        re.compile(r'response\.headers\[\s*[\'"]X-\w+[\'"]\s*\]\s*=', re.IGNORECASE),
        "HTTP Header Tampering",
        "medium",
        "HTTP response headers being set directly. Ensure proper validation."
    ),
    
    # SQL Injection in ActiveRecord Finder Methods
    (
        re.compile(r'(?:find_by_sql|select|where|having|group|order).*[\?\%]\s*\{.*\}', re.IGNORECASE),
        "SQL Injection",
        "high",
        "Possible SQL injection in ActiveRecord finder methods. Use parameterized queries."
    ),
    
    # Insecure Cookie
    (
        re.compile(r'cookies\[\s*:?\w+\s*\]\s*=|cookies\.permanent|cookies\.encrypted', re.IGNORECASE),
        "Insecure Cookie",
        "medium",
        "Check that cookies are set with secure and httponly flags where appropriate."
    ),
    
    # Regular Expression DoS
    (
        re.compile(r'Regexp\.new|\/(?:\.\*|\\\w\+|\[\^\\w\]\+)\+\/', re.IGNORECASE),
        "Regular Expression DoS",
        "medium",
        "Potentially vulnerable regular expression that may cause DoS. Avoid nested quantifiers and backreferences."
    ),
    
    # Timing Attacks
    (
        re.compile(r'(?:==|eql\?|equal\?).*(?:password|token|secret|digest)', re.IGNORECASE),
        "Timing Attack",
        "medium",
        "Possible timing attack vulnerability. Use secure_compare for comparing secrets."
    )
] 