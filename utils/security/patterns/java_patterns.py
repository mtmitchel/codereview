"""
Java-specific security patterns for vulnerability detection.
"""

import re

# SQL Injection patterns
SQL_INJECTION_PATTERNS = [
    (
        re.compile(r'(?:executeQuery|executeUpdate|execute|prepareStatement|createStatement)\s*\(\s*(?:[\"\'`]|(?:\+\s*))(?:SELECT|UPDATE|DELETE|INSERT|REPLACE|MERGE)(?:[^\"\'`]*(?:\+\s*)?[\"\'`][^\"\'`]*)+', re.IGNORECASE),
        "SQL Injection",
        "high",
        "Potential SQL injection vulnerability. Use prepared statements with parameterized queries."
    ),
    (
        re.compile(r'(?:executeQuery|executeUpdate|execute|prepareStatement|createStatement)\s*\(\s*(?:[^\),]*\+\s*[^\)]*)\)', re.IGNORECASE),
        "SQL Injection",
        "medium",
        "Potential SQL injection vulnerability. Use prepared statements with parameterized queries."
    ),
    (
        re.compile(r'setString\s*\(\s*\d+\s*,\s*(?:[^\),]*\+\s*[^\)]*)\)', re.IGNORECASE),
        "SQL Injection",
        "medium",
        "Potential SQL injection in parameter binding. Use parameterized queries without string concatenation."
    ),
]

# XML External Entity (XXE) patterns
XXE_PATTERNS = [
    (
        re.compile(r'(?:DocumentBuilderFactory|SAXParserFactory|XMLInputFactory)(?:(?!\.setFeature\(\s*(?:"http://apache\.org/xml/features/disallow-doctype-decl"|"http://xml\.org/sax/features/external-general-entities"|"http://xml\.org/sax/features/external-parameter-entities")|setExpandEntityReferences\s*\(\s*false\s*\))[^\)]*){20,}\.(?:newDocumentBuilder|newSAXParser|createXMLStreamReader)', re.IGNORECASE),
        "XML External Entity (XXE)",
        "high",
        "Potential XXE vulnerability. Disable external entity processing in XML parsers."
    ),
    (
        re.compile(r'(?:DocumentBuilderFactory|SAXParserFactory|XMLInputFactory).*\.setFeature\(\s*(?:"http://apache\.org/xml/features/disallow-doctype-decl"|"http://xml\.org/sax/features/external-general-entities"|"http://xml\.org/sax/features/external-parameter-entities")\s*,\s*false\s*\)', re.IGNORECASE),
        "XML External Entity (XXE)",
        "high",
        "Security features to prevent XXE attacks are explicitly disabled."
    ),
]

# Command Injection patterns
COMMAND_INJECTION_PATTERNS = [
    (
        re.compile(r'(?:Runtime\.getRuntime\(\)|ProcessBuilder)(?:.|\n)*?(?:exec|start)\s*\(\s*(?:[^\"\']*\+\s*|[\"\'`][^\"\'`]*\+[^\"\'`]*[\"\'`])', re.IGNORECASE),
        "Command Injection",
        "high",
        "Potential command injection vulnerability. Avoid using user input in system commands."
    ),
    (
        re.compile(r'(?:Runtime\.getRuntime\(\)|ProcessBuilder)(?:.|\n)*?(?:exec|start)\s*\(\s*(?:String\[\]\s*\{\s*(?:[^}]*,\s*){2,})', re.IGNORECASE),
        "Command Injection",
        "medium",
        "Potential command injection vulnerability. Validate and sanitize all command arguments."
    ),
]

# Deserialization vulnerabilities
DESERIALIZATION_PATTERNS = [
    (
        re.compile(r'(?:ObjectInputStream|XMLDecoder)(?:(?!readObject\(\)|readUTF\(\)|readBoolean\(\)|readInt\(\)|readLong\(\)|readFloat\(\)|readDouble\(\)|readUnsignedByte\(\)|readUnsignedShort\(\))\S)*\.read(?:Object|External)', re.IGNORECASE),
        "Insecure Deserialization",
        "high",
        "Potential insecure deserialization vulnerability. Avoid deserializing untrusted data."
    ),
    (
        re.compile(r'XStream(?:(?!\.denyTypes\(|\.addToBlacklist\(|\.setupDefaultSecurity\()\S)*\.fromXML', re.IGNORECASE),
        "Insecure Deserialization",
        "high",
        "Potential insecure deserialization with XStream. Enable security features before deserializing."
    ),
    (
        re.compile(r'Yaml\.load\(', re.IGNORECASE),
        "Insecure Deserialization",
        "high",
        "Potential insecure YAML deserialization. Use Yaml.safeLoad() or enable SafeConstructor."
    ),
]

# Path traversal vulnerabilities
PATH_TRAVERSAL_PATTERNS = [
    (
        re.compile(r'(?:new\s+File|new\s+FileInputStream|new\s+FileOutputStream|new\s+FileReader|new\s+FileWriter)\s*\(\s*(?:[^\),]*\+\s*[^\)]*)\)', re.IGNORECASE),
        "Path Traversal",
        "high",
        "Potential path traversal vulnerability. Validate file paths and use path canonicalization."
    ),
    (
        re.compile(r'(?:new\s+File|new\s+FileInputStream|new\s+FileOutputStream|new\s+FileReader|new\s+FileWriter)(?:(?!\.getCanonicalPath\(\)|\.normalize\(\))[^\)]*){10,}\)', re.IGNORECASE),
        "Path Traversal",
        "medium",
        "File operations without path validation might lead to path traversal vulnerabilities."
    ),
]

# Insecure randomness patterns
INSECURE_RANDOM_PATTERNS = [
    (
        re.compile(r'new\s+Random\s*\(', re.IGNORECASE),
        "Insecure Randomness",
        "medium",
        "Using java.util.Random for security-sensitive operations. Use SecureRandom instead."
    ),
    (
        re.compile(r'Math\.random\s*\(', re.IGNORECASE),
        "Insecure Randomness",
        "medium",
        "Using Math.random() for security-sensitive operations. Use SecureRandom instead."
    ),
    (
        re.compile(r'SecureRandom(?:(?!\.getInstance\(\"[^\)\"]+\"\))[^\)]*){10,}\.next', re.IGNORECASE),
        "Insecure Randomness",
        "low",
        "SecureRandom is used without explicitly specifying the algorithm. Consider using getInstance() with a specific algorithm."
    ),
]

# Weak cryptography patterns
WEAK_CRYPTO_PATTERNS = [
    (
        re.compile(r'(?:getInstance|getEngine)\s*\(\s*[\"\'`](?:MD5|RC4|DES|3DES|DESede|AES/ECB|RC2|ARCFOUR|BLOWFISH)[\"\'`]', re.IGNORECASE),
        "Weak Cryptography",
        "high",
        "Using weak cryptographic algorithm. Use strong algorithms like AES-GCM with proper key sizes."
    ),
    (
        re.compile(r'KeyPairGenerator(?:(?!\.initialize\s*\(\s*(?:2048|3072|4096)\s*\))[^\)]*){10,}\.generateKeyPair\(\)', re.IGNORECASE),
        "Weak Cryptography",
        "medium",
        "RSA key size may be too small. Use at least 2048 bits for RSA keys."
    ),
    (
        re.compile(r'Cipher\s*\.\s*getInstance\s*\(\s*[\"\'`][^\)\"\'`]*\/CBC\/PKCS5Padding[\"\'`](?:(?!\.init\s*\(\s*Cipher\.ENCRYPT_MODE[^\)]*,\s*[^\)]*IvParameterSpec)[^\)]*){20,}', re.IGNORECASE),
        "Weak Cryptography",
        "medium",
        "Using CBC mode without proper IV handling. Ensure a secure random IV is used for each encryption."
    ),
    (
        re.compile(r'MessageDigest\s*\.\s*getInstance\s*\(\s*[\"\'`](?:SHA-1|MD5)[\"\'`]\s*\)', re.IGNORECASE),
        "Weak Cryptography",
        "high",
        "Using weak hashing algorithm. Use at least SHA-256 for secure hashing."
    ),
]

# Hardcoded secrets patterns
HARDCODED_SECRET_PATTERNS = [
    (
        re.compile(r'(?:password|passwd|pwd|secret|key|token|auth|credentials|api_?key)\s*=\s*[\"\'`][^\"\'\`]{8,}[\"\'`]', re.IGNORECASE),
        "Hardcoded Secret",
        "high",
        "Potential hardcoded secret. Store sensitive information in secure configuration systems."
    ),
    (
        re.compile(r'private\s+static\s+(?:final\s+)?(?:String|char\[\])\s+(?:password|passwd|pwd|secret|key|token|auth|credentials|api_?key)\s*=\s*[\"\'][^\"\',;]{8,}[\"\']', re.IGNORECASE),
        "Hardcoded Secret",
        "high",
        "Hardcoded secret in class variable. Use environment variables or secure vaults instead."
    ),
    (
        re.compile(r'\.(?:authorization|authenticate|setPassword|setCredentials|setApiKey)\s*\(\s*[\"\'`][^\"\'\`]{8,}[\"\'`]\s*\)', re.IGNORECASE),
        "Hardcoded Secret",
        "high",
        "Hardcoded secret in method call. Use environment variables or secure vaults instead."
    ),
]

# Logging sensitive data patterns
SENSITIVE_LOGGING_PATTERNS = [
    (
        re.compile(r'(?:log|logger|System\.out|System\.err)(?:.|\n)*?(?:debug|info|warn|error|println|print|printf)\s*\(\s*(?:[^,\)]*(?:password|passwd|pwd|secret|credential|token|ssn|creditcard|credit_?card|cc_?num)[^,\)]*,|[^,\)]*(?:getPassword|getToken|getCreditCard)\s*\(\s*\)[^,\)]*,)', re.IGNORECASE),
        "Sensitive Data Logging",
        "medium",
        "Potentially logging sensitive information. Avoid logging passwords, tokens, or personal data."
    ),
]

# Insecure cookie patterns
INSECURE_COOKIE_PATTERNS = [
    (
        re.compile(r'(?:Cookie|HttpCookie)(?:(?!\.setSecure\s*\(\s*true\s*\)|\.setHttpOnly\s*\(\s*true\s*\))[^\)]*){20,}\.(?:setValue|setPath|setDomain)', re.IGNORECASE),
        "Insecure Cookie",
        "medium",
        "Cookie created without secure and HttpOnly flags. Set both to true for sensitive cookies."
    ),
    (
        re.compile(r'(?:Cookie|HttpCookie).*\.setSecure\s*\(\s*false\s*\)', re.IGNORECASE),
        "Insecure Cookie",
        "high",
        "Cookie security explicitly disabled with setSecure(false)."
    ),
    (
        re.compile(r'(?:Cookie|HttpCookie).*\.setHttpOnly\s*\(\s*false\s*\)', re.IGNORECASE),
        "Insecure Cookie",
        "high",
        "Cookie HttpOnly flag explicitly disabled with setHttpOnly(false)."
    ),
]

# SSRF Vulnerabilities
SSRF_PATTERNS = [
    (
        re.compile(r'(?:URL|URI|HttpURLConnection)(?:(?!\.getHost\(\)|\.getUserInfo\(\))[^\)]*){10,}\.(?:openConnection|openStream)\(', re.IGNORECASE),
        "Server-Side Request Forgery (SSRF)",
        "high",
        "Potential SSRF vulnerability. Validate and whitelist URLs before making requests."
    ),
    (
        re.compile(r'(?:OkHttpClient|HttpClient|CloseableHttpClient|RestTemplate|WebClient)(?:(?!\.setConnectTimeout\(\)|\.setReadTimeout\(\))[^\)]*){10,}\.(?:newCall|execute|exchange|get|post|put|delete)\(', re.IGNORECASE),
        "Server-Side Request Forgery (SSRF)",
        "medium",
        "HTTP client without proper timeout settings might lead to SSRF vulnerabilities."
    ),
]

# CORS Misconfiguration patterns
CORS_PATTERNS = [
    (
        re.compile(r'response\.(?:setHeader|addHeader)\s*\(\s*[\"\'`](?:Access-Control-Allow-Origin)[\"\'`]\s*,\s*[\"\'`]\*[\"\'`]\s*\)', re.IGNORECASE),
        "CORS Misconfiguration",
        "high",
        "Overly permissive CORS policy allows any origin. Restrict to specific trusted domains."
    ),
    (
        re.compile(r'CorsConfiguration(?:(?!\.setAllowedOrigins\(\)|\\.applyPermitDefaultValues\(\))[^\)]*){10,}', re.IGNORECASE),
        "CORS Misconfiguration",
        "medium",
        "Using CORS configuration without explicitly setting allowed origins may lead to security issues."
    ),
]

# Combine all patterns
JAVA_PATTERNS = (
    *SQL_INJECTION_PATTERNS,
    *XXE_PATTERNS,
    *COMMAND_INJECTION_PATTERNS,
    *DESERIALIZATION_PATTERNS,
    *PATH_TRAVERSAL_PATTERNS,
    *INSECURE_RANDOM_PATTERNS,
    *WEAK_CRYPTO_PATTERNS,
    *HARDCODED_SECRET_PATTERNS,
    *SENSITIVE_LOGGING_PATTERNS,
    *INSECURE_COOKIE_PATTERNS,
    *SSRF_PATTERNS,
    *CORS_PATTERNS,
) 