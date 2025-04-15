"""
PHP-specific security patterns for vulnerability detection.
"""

import re

# SQL Injection patterns
SQL_INJECTION_PATTERNS = [
    (re.compile(r'(?i)(?:mysql_query|mysqli_query|pg_query)\s*\(\s*[\'"][^\'"]*(?:\$[a-zA-Z0-9_]+)[^\'"]*[\'"]'), 
     "SQL Injection", "critical", "Possible SQL injection. Use prepared statements with parameterized queries."),
    (re.compile(r'(?i)(?:->query|::query|->exec|::exec)\s*\(\s*[\'"][^\'"]*(?:\$[a-zA-Z0-9_]+|\{)[^\'"]*[\'"]'), 
     "SQL Injection", "critical", "Possible SQL injection in ORM/DB library. Use parameterized queries."),
    (re.compile(r'(?i)(?:->query|::query|->exec|::exec)\s*\(\s*(?:[\'"].*[\'"] \. \$[a-zA-Z0-9_]+)'), 
     "SQL Injection", "critical", "Possible SQL injection through string concatenation. Use parameterized queries."),
]

# Command Injection patterns
COMMAND_INJECTION_PATTERNS = [
    (re.compile(r'(?i)(?:system|exec|passthru|shell_exec|popen|proc_open)\s*\(\s*(?:\$[a-zA-Z0-9_]+|\$_(?:GET|POST|REQUEST|COOKIE))'), 
     "Command Injection", "critical", "Possible command injection with user input. Validate and sanitize all inputs."),
    (re.compile(r'(?i)(?:system|exec|passthru|shell_exec|popen|proc_open)\s*\(\s*[\'"][^\'"]*(?:\$[a-zA-Z0-9_]+|\{)[^\'"]*[\'"]'), 
     "Command Injection", "critical", "Possible command injection with variables in command string. Use escapeshellarg/escapeshellcmd."),
    (re.compile(r'(?i)(?:eval|assert)\s*\(\s*(?:\$[a-zA-Z0-9_]+|\$_(?:GET|POST|REQUEST|COOKIE))'), 
     "Code Injection", "critical", "Possible code injection with user input. Avoid using eval with external data."),
]

# Cross-Site Scripting (XSS) patterns
XSS_PATTERNS = [
    (re.compile(r'(?i)echo\s+(?:\$_(?:GET|POST|REQUEST|COOKIE)|(?:\$[a-zA-Z0-9_]+)\s*=\s*\$_(?:GET|POST|REQUEST|COOKIE))'), 
     "Cross-Site Scripting (XSS)", "high", "Outputting user input directly. Use htmlspecialchars or other context-appropriate escaping."),
    (re.compile(r'(?i)print\s+(?:\$_(?:GET|POST|REQUEST|COOKIE)|(?:\$[a-zA-Z0-9_]+)\s*=\s*\$_(?:GET|POST|REQUEST|COOKIE))'), 
     "Cross-Site Scripting (XSS)", "high", "Outputting user input directly. Use htmlspecialchars or other context-appropriate escaping."),
    (re.compile(r'(?i)(?:<\?=|\<\?php\s+echo)\s+\$(?:_GET|_POST|_REQUEST|_COOKIE)'), 
     "Cross-Site Scripting (XSS)", "high", "Outputting user input directly. Use htmlspecialchars or other context-appropriate escaping."),
]

# File Inclusion vulnerabilities
FILE_INCLUSION_PATTERNS = [
    (re.compile(r'(?i)(?:include|require|include_once|require_once)\s*\(\s*\$(?:[a-zA-Z0-9_]+)'), 
     "File Inclusion", "high", "Dynamic file inclusion can lead to Local/Remote File Inclusion vulnerabilities."),
    (re.compile(r'(?i)(?:include|require|include_once|require_once)\s+[\'"]\.\.\/'), 
     "File Inclusion", "medium", "Including files with relative paths can be risky if path traversal is possible."),
]

# File upload vulnerabilities
FILE_UPLOAD_PATTERNS = [
    (re.compile(r'(?i)move_uploaded_file\s*\(\s*\$_FILES'), 
     "File Upload", "medium", "Ensure proper validation of uploaded files (type, size, name, content)."),
    (re.compile(r'(?i)if\s*\(\s*\$_FILES[^\)]*type[^\)]*==[^\)]*(?:image|application)'), 
     "Weak File Type Validation", "high", "Validating file type using MIME type from client is insecure. Use finfo_file instead."),
    (re.compile(r'(?i)pathinfo\s*\([^\)]*PATHINFO_EXTENSION'), 
     "File Extension Check", "medium", "Checking file extension alone is not sufficient for upload validation."),
]

# Insecure session handling
SESSION_PATTERNS = [
    (re.compile(r'(?i)session\.use_only_cookies\s*=\s*(?:0|false)'), 
     "Insecure Session Settings", "high", "Sessions should use cookies only to prevent session fixation attacks."),
    (re.compile(r'(?i)session\.cookie_secure\s*=\s*(?:0|false)'), 
     "Insecure Session Settings", "medium", "Session cookies should have the secure flag enabled for HTTPS sites."),
    (re.compile(r'(?i)session\.cookie_httponly\s*=\s*(?:0|false)'), 
     "Insecure Session Settings", "medium", "Session cookies should have the HttpOnly flag enabled to prevent XSS access."),
]

# Weak cryptography
CRYPTO_PATTERNS = [
    (re.compile(r'(?i)md5\s*\('), 
     "Weak Hashing", "medium", "MD5 is cryptographically broken. Use password_hash for passwords or secure alternatives."),
    (re.compile(r'(?i)sha1\s*\('), 
     "Weak Hashing", "medium", "SHA1 is cryptographically weak. Use password_hash for passwords or secure alternatives."),
    (re.compile(r'(?i)(?:password_hash|crypt)\s*\([^,]*,\s*PASSWORD_DEFAULT\s*'), 
     "Password Hashing", "low", "Verify you're using the strongest algorithm available (PASSWORD_DEFAULT may change over time)."),
]

# Debugging code
DEBUG_PATTERNS = [
    (re.compile(r'(?i)var_dump\s*\(\s*\$_(?:GET|POST|REQUEST|COOKIE|SESSION|SERVER)'), 
     "Debug Information Leak", "medium", "Exposing sensitive variable data. Remove debugging code in production."),
    (re.compile(r'(?i)print_r\s*\(\s*\$_(?:GET|POST|REQUEST|COOKIE|SESSION|SERVER)'), 
     "Debug Information Leak", "medium", "Exposing sensitive variable data. Remove debugging code in production."),
    (re.compile(r'(?i)error_reporting\s*\(\s*E_ALL\s*\)'), 
     "Debug Setting", "low", "Showing all errors in production can leak sensitive information. Use appropriate error_reporting levels."),
]

# Insecure Deserialization patterns
DESERIALIZATION_PATTERNS = [
    (re.compile(r'(?i)unserialize\s*\(\s*(?:\$[a-zA-Z0-9_]+|\$_(?:GET|POST|REQUEST|COOKIE))'),
     "Insecure Deserialization", "high", "Unserializing user-controlled data can lead to remote code execution. Consider using JSON or other secure alternatives."),
    (re.compile(r'(?i)unserialize\s*\(\s*file_get_contents\s*\(\s*(?:\$[a-zA-Z0-9_]+)'),
     "Insecure Deserialization", "medium", "Unserializing data from potentially untrusted files can lead to code execution. Validate file source and content."),
]

# Code Execution via preg_replace
PREG_REPLACE_PATTERNS = [
    (re.compile(r'(?i)preg_replace\s*\(\s*[\'"][^\'"]*\/e[\'"]'),
     "Code Execution", "critical", "Using the /e modifier in preg_replace can lead to remote code execution. Use preg_replace_callback instead."),
    (re.compile(r'(?i)preg_replace\s*\(\s*\$[a-zA-Z0-9_]+'),
     "Code Execution", "high", "Using dynamic pattern in preg_replace can be dangerous if it contains the /e modifier. Use preg_replace_callback instead."),
]

# Variable Extraction patterns
VARIABLE_EXTRACTION_PATTERNS = [
    (re.compile(r'(?i)extract\s*\(\s*\$_(?:GET|POST|REQUEST|COOKIE)'),
     "Variable Extraction", "high", "Extracting variables from user input can lead to variable overwrite attacks. Avoid extract() with untrusted data."),
    (re.compile(r'(?i)parse_str\s*\(\s*(?:\$[a-zA-Z0-9_]+|\$_(?:GET|POST|REQUEST|COOKIE))'),
     "Variable Extraction", "high", "parse_str() can create variables from user input leading to variable overwrite. Specify the result array as second parameter."),
    (re.compile(r'(?i)extract\s*\(\s*\$[a-zA-Z0-9_]+'),
     "Variable Extraction", "medium", "Using extract() can be dangerous if the array contains untrusted data. Check the source of the data."),
]

# Combine all patterns
PHP_PATTERNS = (
    SQL_INJECTION_PATTERNS + 
    COMMAND_INJECTION_PATTERNS + 
    XSS_PATTERNS + 
    FILE_INCLUSION_PATTERNS + 
    FILE_UPLOAD_PATTERNS + 
    SESSION_PATTERNS + 
    CRYPTO_PATTERNS + 
    DEBUG_PATTERNS +
    DESERIALIZATION_PATTERNS +
    PREG_REPLACE_PATTERNS +
    VARIABLE_EXTRACTION_PATTERNS
) 