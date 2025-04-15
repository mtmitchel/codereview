"""
C#-specific security patterns for vulnerability detection.
"""

import re
from typing import List, Tuple, Pattern

# Compile regex patterns for improved performance
# Each pattern is a tuple of (regex_pattern, vulnerability_type, severity, message)

# SQL Injection Patterns
SQL_INJECTION_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'(?:ExecuteQuery|CommandText|SqlCommand).*?[\"\'].*?\+\s*(?:.*?)\s*(?:\+|$)'),
        "SQL Injection", 
        "critical",
        "Possible SQL injection vulnerability detected. Use parameterized queries with SqlParameter to mitigate SQL injection risks."
    ),
    (
        re.compile(r'(?:using\s*\()?\s*new\s+SqlCommand\s*\(\s*.*?\s*\+'),
        "SQL Injection", 
        "critical",
        "SQL query constructed with string concatenation. Use parameterized queries instead."
    ),
    (
        re.compile(r'string\s+sql\s*=.*?\+\s*.*?;'),
        "SQL Injection", 
        "high",
        "SQL query built using string concatenation. Use parameterized queries instead."
    ),
    (
        re.compile(r'\.ExecuteSqlRaw\(.*?\+'),
        "SQL Injection", 
        "critical",
        "Raw SQL execution with string concatenation in Entity Framework Core. Use parameterized queries instead."
    ),
    (
        re.compile(r'\.FromSqlRaw\(.*?\+'),
        "SQL Injection", 
        "critical",
        "Raw SQL execution with string concatenation in Entity Framework Core. Use parameterized queries instead."
    )
]

# Command Injection Patterns
COMMAND_INJECTION_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'Process\.Start\s*\(\s*.*?\s*\+'),
        "Command Injection", 
        "critical",
        "Command injection vulnerability detected. Avoid using user input in process execution."
    ),
    (
        re.compile(r'ProcessStartInfo.*?=.*?\+'),
        "Command Injection", 
        "critical",
        "Command arguments built with string concatenation. Validate inputs carefully."
    ),
    (
        re.compile(r'(?:System\.Diagnostics\.)?Process\.Start\s*\(\s*(?:\"|\')(?:cmd|cmd.exe|powershell|bash|sh)(?:\"|\')'),
        "Command Injection", 
        "high",
        "Use of shell commands. This can be dangerous if user input is involved."
    )
]

# XXE Vulnerability Patterns
XXE_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'new\s+XmlDocument\s*\(\s*\)\s*;.*?\.\s*Load\('),
        "XXE Vulnerability", 
        "critical",
        "Potential XXE vulnerability. Set XmlResolver to null or use secure parsing."
    ),
    (
        re.compile(r'XmlReaderSettings.*?XmlResolver\s*=\s*(?!null)'),
        "XXE Vulnerability", 
        "high",
        "XmlResolver should be set to null to prevent XXE attacks."
    ),
    (
        re.compile(r'XmlTextReader(?!\s*\(\s*.*?\s*\)\s*{\s*DtdProcessing\s*=\s*DtdProcessing\.Prohibit)'),
        "XXE Vulnerability", 
        "high",
        "XmlTextReader without disabling DTD processing might be vulnerable to XXE attacks."
    ),
    (
        re.compile(r'new\s+XmlSerializer\s*\('),
        "XXE Vulnerability", 
        "medium",
        "Using XmlSerializer without proper controls may pose XXE risks."
    )
]

# Deserialization Vulnerability Patterns
DESERIALIZATION_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'BinaryFormatter.*?(?:Deserialize|Serialize)'),
        "Insecure Deserialization", 
        "critical",
        "BinaryFormatter is insecure for deserializing untrusted data. Use a safer alternative like JSON.NET."
    ),
    (
        re.compile(r'SoapFormatter.*?(?:Deserialize|Serialize)'),
        "Insecure Deserialization", 
        "critical",
        "SoapFormatter is insecure for deserializing untrusted data. Use a safer alternative."
    ),
    (
        re.compile(r'NetDataContractSerializer.*?(?:Deserialize|Serialize)'),
        "Insecure Deserialization", 
        "critical",
        "NetDataContractSerializer can lead to remote code execution if not used carefully."
    ),
    (
        re.compile(r'TypeNameHandling(?!.*None)'),
        "Insecure Deserialization", 
        "high",
        "Setting TypeNameHandling to any value other than None can lead to deserialization vulnerabilities."
    ),
    (
        re.compile(r'JsonSerializerSettings.*?TypeNameHandling(?!.*None)'),
        "Insecure Deserialization", 
        "high",
        "JsonSerializerSettings with TypeNameHandling not set to None is vulnerable to attacks."
    )
]

# Path Traversal Vulnerability Patterns
PATH_TRAVERSAL_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'(?:File|Directory)\.(?:Open|Read|Write|Create|Delete)(?:Text)?.*?\+'),
        "Path Traversal", 
        "high",
        "Potential path traversal vulnerability. Sanitize file paths and validate user input."
    ),
    (
        re.compile(r'(?:Path\.Combine\(|Path\.Join\().*?(?:\+|Request)'),
        "Path Traversal", 
        "medium",
        "Building paths with user input can lead to path traversal. Ensure inputs are validated."
    ),
    (
        re.compile(r'(?:new FileStream|using.*?FileStream).*?\+'),
        "Path Traversal", 
        "high",
        "FileStream with concatenated paths might be vulnerable to path traversal attacks."
    )
]

# Weak Randomness Patterns
WEAK_RANDOMNESS_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'new\s+Random\s*\('),
        "Weak Randomness", 
        "medium",
        "System.Random produces predictable sequences. Use cryptographic random generators for security purposes."
    ),
    (
        re.compile(r'(?:Math|System\.Math)\.Random\s*\('),
        "Weak Randomness", 
        "medium",
        "Math.Random is not cryptographically secure. Use RNGCryptoServiceProvider instead."
    )
]

# Weak Cryptography Patterns
WEAK_CRYPTO_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'(?:SHA1|MD5|DES|TripleDES|RC2)(?:CryptoServiceProvider|\.\s*Create)'),
        "Weak Cryptography", 
        "high",
        "Using weak or outdated cryptographic algorithm. Use modern algorithms like SHA-256 or AES instead."
    ),
    (
        re.compile(r'new\s+(?:SHA1|MD5)(?:Managed|CryptoServiceProvider)'),
        "Weak Cryptography", 
        "high",
        "SHA1 and MD5 are cryptographically broken. Use SHA-256 or SHA-512 instead."
    ),
    (
        re.compile(r'CipherMode\.(?:ECB|OFB|CFB|CTS)'),
        "Weak Cryptography", 
        "high",
        "Using weak cipher mode. Use AES with GCM mode instead."
    ),
    (
        re.compile(r'PaddingMode\.(?:None|PKCS7|Zeros|ANSIX923)'),
        "Weak Cryptography", 
        "medium",
        "Some padding modes are vulnerable to padding oracle attacks."
    )
]

# Authentication and Identity Vulnerabilities
AUTH_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'RequireHttpsAttribute\s*=\s*false'),
        "Insecure Authentication", 
        "critical",
        "HTTPS is disabled for authentication endpoints. Always use HTTPS for authentication."
    ),
    (
        re.compile(r'CookieAuthenticationOptions.*?(?:Secure\s*=\s*false|HttpOnly\s*=\s*false)'),
        "Insecure Authentication", 
        "high",
        "Authentication cookies should have Secure and HttpOnly flags enabled."
    ),
    (
        re.compile(r'services\.AddDataProtection\(.*?\)(?!\s*\.ProtectKeysWithCertificate)'),
        "Weak Key Protection", 
        "medium",
        "Data protection keys should be protected with certificates in production environments."
    ),
    (
        re.compile(r'allowPasswordReset\s*:\s*false'),
        "Insecure Account Recovery",
        "medium",
        "Disabling password reset may lead to account lockout without recovery options."
    ),
    (
        re.compile(r'options\.Password\.RequiredLength\s*=\s*(?:[1-7]\b)'),
        "Weak Password Policy", 
        "medium",
        "Password policy allows short passwords. Minimum recommended length is 8 characters."
    )
]

# CSRF Vulnerability Patterns
CSRF_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'\[ValidateAntiForgeryToken\]\s*(?!\s*public\s+.*?Action)'),
        "CSRF Vulnerability", 
        "high",
        "Public action methods should be protected with ValidateAntiForgeryToken to prevent CSRF attacks."
    ),
    (
        re.compile(r'\.AddAntiforgery\(.*?\.SuppressXFrameOptionsHeader\s*=\s*true'),
        "CSRF Vulnerability", 
        "high",
        "Suppressing X-Frame-Options header increases CSRF risk. Keep this enabled for protection."
    ),
    (
        re.compile(r'options\.Cookie\.SameSite\s*=\s*SameSiteMode\.None'),
        "CSRF Vulnerability", 
        "high",
        "SameSite=None cookies are vulnerable to CSRF attacks unless the Secure flag is also set."
    ),
    (
        re.compile(r'\[HttpPost\](?!\s*\[ValidateAntiForgeryToken\])'),
        "CSRF Vulnerability", 
        "high",
        "POST endpoints should include ValidateAntiForgeryToken attribute to prevent CSRF attacks."
    )
]

# Insecure Cookie Settings
COOKIE_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'(?:HttpCookie|Cookie).*?Secure\s*=\s*false'),
        "Insecure Cookie", 
        "high",
        "Cookies should have the Secure flag enabled to ensure they are only sent over HTTPS."
    ),
    (
        re.compile(r'(?:HttpCookie|Cookie).*?HttpOnly\s*=\s*false'),
        "Insecure Cookie", 
        "high",
        "Cookies should have the HttpOnly flag enabled to prevent client-side script access."
    ),
    (
        re.compile(r'new\s+CookieOptions\s*\(\s*\).*?(?!Secure\s*=\s*true)'),
        "Insecure Cookie", 
        "medium",
        "Cookie options should set Secure=true to ensure cookies are only sent over HTTPS."
    ),
    (
        re.compile(r'SameSiteMode\.None(?!\s*.*?Secure\s*=\s*true)'),
        "Insecure Cookie", 
        "high",
        "Cookies with SameSite=None must use the Secure flag as well."
    )
]

# Direct Object References
IDOR_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'(?:Get|Find|Fetch|Load|Retrieve)(?:User|Account|Profile)\s*\(\s*(?:Request|id|userId)'),
        "Insecure Direct Object Reference", 
        "medium",
        "Potential IDOR vulnerability. Ensure proper authorization checks before loading objects by ID."
    ),
    (
        re.compile(r'\.(?:FirstOrDefault|Single)\s*\(\s*\w+\s*=>\s*\w+\.Id\s*==\s*(?:id|userId|Request)'),
        "Insecure Direct Object Reference", 
        "medium",
        "Fetching objects by ID from user input requires authorization checks to prevent IDOR vulnerabilities."
    )
]

# Sensitive Data Logging
LOGGING_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'Log(?:ger)?\.(?:Error|Info|Debug|Warning|Critical|Log)\s*\(\s*.*?(?:password|token|key|secret|credentials|private)'),
        "Sensitive Data Logging", 
        "high",
        "Sensitive data might be logged. Ensure passwords, tokens, and credentials are not included in logs."
    ),
    (
        re.compile(r'Console\.(?:WriteLine|Write)\s*\(\s*.*?(?:password|token|key|secret|credentials|private)'),
        "Sensitive Data Logging", 
        "high",
        "Sensitive data might be output to console. Avoid logging credentials or tokens."
    ),
    (
        re.compile(r'Log\.(?:Error|Info|Debug|Warning|Critical|Log)\s*\(\s*.*?Exception'),
        "Insecure Exception Logging", 
        "medium",
        "Logging exceptions may expose sensitive information. Consider sanitizing exception details."
    )
]

# Open Redirect Vulnerabilities
OPEN_REDIRECT_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'Redirect\s*\(\s*(?:.*?\+|Request\.Query|Request\.Form)'),
        "Open Redirect", 
        "high",
        "Potential open redirect vulnerability. Validate redirect URLs before processing them."
    ),
    (
        re.compile(r'Redirect(?:Result|ToRoute)?\s*\(\s*.*?(?:url|redirectUrl|returnUrl)(?:\s*\+|\[)'),
        "Open Redirect", 
        "high",
        "Unvalidated redirects can lead to phishing attacks. Validate all redirect URLs."
    )
]

# Debugging Code
DEBUG_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'#if\s+DEBUG'),
        "Debugging Code", 
        "low",
        "Debug conditional compilation directives should be reviewed before production deployment."
    ),
    (
        re.compile(r'[Dd]eveloperExceptionPage\s*\('),
        "Debugging Code", 
        "medium",
        "Developer exception page should not be enabled in production environments."
    ),
    (
        re.compile(r'UseDeveloperExceptionPage\s*\(\s*\)'),
        "Debugging Code", 
        "medium",
        "Developer exception page enabled. This should be configured only for development environment."
    ),
    (
        re.compile(r'Debugger\.(?:Break|Launch)'),
        "Debugging Code", 
        "low",
        "Debugger code should be removed before production deployment."
    )
]

# CORS Misconfiguration
CORS_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'builder\.WithOrigins\s*\(\s*"\*"'),
        "Insecure CORS Configuration", 
        "high",
        "Using wildcard origins in CORS policy allows any site to make cross-origin requests."
    ),
    (
        re.compile(r'AllowAnyOrigin\s*\(\s*\).*?AllowCredentials\s*\(\s*true'),
        "Insecure CORS Configuration", 
        "critical",
        "AllowAnyOrigin combined with AllowCredentials is insecure and violates CORS specification."
    ),
    (
        re.compile(r'app\.UseCors\s*\(\s*builder\s*=>\s*builder\.AllowAnyOrigin\s*\(\s*\)'),
        "Insecure CORS Configuration", 
        "high",
        "AllowAnyOrigin permits requests from any domain. Restrict to specific trusted origins."
    )
]

# Missing Input Validation
VALIDATION_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'Request\.Form\[\s*[\"\'].*?[\"\']\s*\](?!\s*\.(?:Validate|Sanitize|Contains))'),
        "Missing Input Validation", 
        "medium",
        "Form input is accessed without validation. Validate all user inputs to prevent injection attacks."
    ),
    (
        re.compile(r'Request\.QueryString\[\s*[\"\'].*?[\"\']\s*\](?!\s*\.(?:Validate|Sanitize|Contains))'),
        "Missing Input Validation", 
        "medium",
        "Query parameters are accessed without validation. Validate all user inputs."
    ),
    (
        re.compile(r'(?:int|long|decimal)\.Parse\s*\(\s*(?!model\.)'),
        "Unsafe Parsing", 
        "low",
        "Using Parse instead of TryParse can cause exceptions with invalid inputs."
    )
]

# Add new pattern categories for ASP.NET Core and modern .NET

# Server-Side Request Forgery Patterns
SSRF_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'new\s+HttpClient\(\).*?GetAsync\s*\(\s*.*?\+'),
        "Server-Side Request Forgery", 
        "high",
        "Potential SSRF vulnerability. Validate and sanitize URLs before making requests."
    ),
    (
        re.compile(r'using\s*\(\s*var\s+client\s*=\s*new\s+HttpClient\(\)\s*\).*?GetAsync\s*\(\s*.*?(?:url|uri|endpoint)(?:\s*\+|\[)'),
        "Server-Side Request Forgery", 
        "high",
        "HttpClient with user-supplied URLs can lead to SSRF attacks. Validate all URLs."
    ),
    (
        re.compile(r'WebRequest\.Create\s*\(\s*.*?\+'),
        "Server-Side Request Forgery", 
        "high",
        "WebRequest.Create with dynamic URLs can lead to SSRF vulnerabilities."
    )
]

# ASP.NET Core Specific Patterns
ASPNETCORE_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'services\.AddAuthentication\s*\(.*?\)(?!\s*\.AddJwtBearer)'),
        "Incomplete Authentication", 
        "medium",
        "Authentication is configured without specifying authentication scheme."
    ),
    (
        re.compile(r'JwtBearerOptions.*?(?:ValidateIssuer\s*=\s*false|ValidateAudience\s*=\s*false|ValidateLifetime\s*=\s*false)'),
        "Insecure JWT Configuration", 
        "high",
        "JWT Token validation is incomplete. Enable issuer, audience, and lifetime validation."
    ),
    (
        re.compile(r'builder\.Services\.AddRazorPages\(\)(?!\s*\.AddRazorPagesOptions\s*\(\s*options\s*=>\s*options\.Conventions\.AuthorizeFolder)'),
        "Missing Authorization", 
        "medium",
        "Consider applying authorization conventions to Razor Pages at registration."
    ),
    (
        re.compile(r'AddControllers\(\)(?!\s*\.AddAuthorization)'),
        "Missing Authorization", 
        "medium",
        "Controllers are registered without explicit authorization. Consider adding global authorization."
    )
]

# Blazor/WebAssembly Security Patterns
BLAZOR_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'@\(\s*.*?\s*\)\s*=>\s*{.*?}\s*(?:eval|Function)'),
        "Unsafe Blazor Execution", 
        "high",
        "Dynamic code execution in Blazor components can lead to code injection vulnerabilities."
    ),
    (
        re.compile(r'@\(\s*.*?\s*\)\s*=>\s*{.*?document\.write'),
        "Unsafe DOM Manipulation", 
        "high",
        "Direct DOM manipulation in Blazor can lead to XSS vulnerabilities. Use safe rendering instead."
    ),
    (
        re.compile(r'@\(\s*.*?\s*\)\s*=>\s*{.*?window\.location\s*='),
        "Client-Side Open Redirect", 
        "medium",
        "Dynamic redirects in Blazor should validate destination URLs."
    )
]

# Dependency Injection Security Patterns
DI_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'services\.AddSingleton<.*?>(?:\(.*?\){.*?}|\(.*?new\s+(?!HttpClient\())'),
        "Singleton Injection Risk", 
        "low",
        "Directly instantiating services in singleton registrations can lead to thread safety issues."
    ),
    (
        re.compile(r'services\.AddHttpClient\s*\(.*?\)(?!\s*\.ConfigurePrimaryHttpMessageHandler)'),
        "Insecure HttpClient", 
        "low",
        "HttpClient registered without configuring validation handlers. Consider adding certificate validation."
    ),
    (
        re.compile(r'services\.AddDbContext.*?optionsBuilder\.UseSqlServer\s*\(\s*.*?\+'),
        "Connection String Injection", 
        "high",
        "Dynamic connection strings in DbContext registration can lead to connection string injection attacks."
    )
]

# Insecure File Operations 
FILE_OPERATION_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'IFormFile.*?SaveAs\s*\(\s*.*?\+'),
        "Insecure File Upload", 
        "high",
        "File upload path is constructed using string concatenation. Validate and sanitize paths."
    ),
    (
        re.compile(r'PhysicalFileProvider\s*\(\s*.*?\+'),
        "Path Traversal", 
        "high",
        "Physical file providers with dynamic paths can lead to path traversal vulnerabilities."
    ),
    (
        re.compile(r'(?:\.UseStaticFiles|\.UseStaticWebAssets)\s*\(\s*\)(?!\s*\.Use(?:StaticFiles|StaticWebAssets)\s*\(\s*new\s+StaticFileOptions\s*{.*?ContentTypeProvider)'),
        "MIME Type Sniffing", 
        "medium",
        "Static files served without content type provider may be vulnerable to MIME sniffing attacks."
    )
]

# Health Check Exposure
HEALTH_CHECK_PATTERNS: List[Tuple[Pattern, str, str, str]] = [
    (
        re.compile(r'AddHealthChecks\s*\(\s*\).*?MapHealthChecks\s*\(\s*.*?\)(?!\s*\.RequireAuthorization)'),
        "Exposed Health Checks", 
        "low",
        "Health check endpoints are exposed without authorization, potentially leaking system information."
    ),
    (
        re.compile(r'MapHealthChecks\s*\(\s*.*?new\s+HealthCheckOptions\s*{.*?AllowCachingResponses\s*=\s*true'),
        "Cached Health Checks", 
        "low",
        "Health check responses should not be cached as they may contain sensitive system information."
    )
]

# Combine all patterns
CSHARP_PATTERNS = (
    SQL_INJECTION_PATTERNS +
    COMMAND_INJECTION_PATTERNS + 
    XXE_PATTERNS +
    DESERIALIZATION_PATTERNS +
    PATH_TRAVERSAL_PATTERNS +
    WEAK_RANDOMNESS_PATTERNS +
    WEAK_CRYPTO_PATTERNS +
    AUTH_PATTERNS +
    CSRF_PATTERNS +
    COOKIE_PATTERNS +
    IDOR_PATTERNS +
    LOGGING_PATTERNS +
    OPEN_REDIRECT_PATTERNS +
    DEBUG_PATTERNS +
    CORS_PATTERNS +
    VALIDATION_PATTERNS +
    # New pattern groups
    SSRF_PATTERNS +
    ASPNETCORE_PATTERNS +
    BLAZOR_PATTERNS +
    DI_PATTERNS +
    FILE_OPERATION_PATTERNS +
    HEALTH_CHECK_PATTERNS
)