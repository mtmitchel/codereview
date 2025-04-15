"""
Rust-specific security patterns for vulnerability detection.
"""

import re

# Unsafe code blocks
UNSAFE_CODE_PATTERNS = [
    (re.compile(r'unsafe\s+\{'), 
     "Use of Unsafe", "medium", "Using unsafe code block which bypasses Rust's safety guarantees. Ensure correctness."),
    (re.compile(r'unsafe\s+fn\s+\w+'), 
     "Use of Unsafe", "medium", "Defining an unsafe function that requires careful usage. Document safety requirements."),
    (re.compile(r'unsafe\s+trait\s+\w+'), 
     "Use of Unsafe", "medium", "Defining an unsafe trait. Ensure implementors respect all safety requirements."),
]

# Memory management vulnerabilities
MEMORY_MANAGEMENT_PATTERNS = [
    (re.compile(r'std::mem::transmute'), 
     "Transmute Use", "high", "Using transmute which is highly unsafe and can lead to undefined behavior."),
    (re.compile(r'std::ptr::(read|write|copy|copy_nonoverlapping)'), 
     "Raw Pointer Operations", "high", "Using raw pointer operations which can cause memory corruption if misused."),
    (re.compile(r'as\s+\*(?:mut|const)\s+\w+'), 
     "Raw Pointer Cast", "medium", "Converting to raw pointer which bypasses Rust's safety checks."),
    (re.compile(r'std::mem::forget'), 
     "Memory Leak", "medium", "Using std::mem::forget which intentionally leaks memory by preventing destructor execution."),
]

# Concurrency issues
CONCURRENCY_PATTERNS = [
    (re.compile(r'static\s+mut\s+\w+'), 
     "Mutable Static", "high", "Using mutable static variables can lead to data races in concurrent code."),
    (re.compile(r'std::sync::atomic::\w+::new\([^)]+\).load\([^)]*Ordering::Relaxed[^)]*\)'), 
     "Weak Memory Ordering", "medium", "Using Relaxed ordering for atomic operations may lead to unexpected behavior in concurrent code."),
    (re.compile(r'Arc::get_mut|Rc::get_mut'), 
     "Exclusive Reference Risk", "medium", "Getting exclusive reference to Arc/Rc content is only safe when there are no other references."),
]

# FFI vulnerabilities
FFI_PATTERNS = [
    (re.compile(r'extern\s+"C"\s+\{'), 
     "Foreign Function Interface", "medium", "Using FFI which can introduce memory safety issues. Ensure proper boundary checks."),
    (re.compile(r'#\[no_mangle\]'), 
     "No Mangle FFI Export", "low", "Exporting function for FFI usage. Ensure safety preconditions are documented."),
    (re.compile(r'CString::from_vec_unchecked|CString::from_raw|CStr::from_ptr'), 
     "Unsafe C String Handling", "high", "Using unsafe C string functions which can cause undefined behavior with invalid inputs."),
]

# Serialization vulnerabilities
SERIALIZATION_PATTERNS = [
    (re.compile(r'serde_json::from_str|serde_json::from_reader'), 
     "JSON Deserialization", "low", "JSON deserialization of untrusted input should be carefully validated."),
    (re.compile(r'serde::Deserialize'), 
     "Deserialization Implementation", "low", "Custom Deserialize implementation should handle malformed input safely."),
    (re.compile(r'bincode::deserialize'), 
     "Binary Deserialization", "medium", "Binary deserialization can be unsafe with untrusted input."),
]

# Debug and unwrap usage
UNWRAP_PATTERNS = [
    (re.compile(r'unwrap\(\)'), 
     "Unwrap Usage", "low", "Using unwrap() on Option/Result which can cause panics at runtime."),
    (re.compile(r'expect\([^\)]+\)'), 
     "Expect Usage", "low", "Using expect() on Option/Result which can cause panics at runtime."),
    (re.compile(r'unwrap_unchecked\(\)'), 
     "Unsafe Unwrap", "high", "Using unwrap_unchecked() which causes undefined behavior for None/Err values."),
]

# Panic usage
PANIC_PATTERNS = [
    (re.compile(r'panic!\s*\('), 
     "Use of Panic", "medium", "Using panic! which abruptly terminates the program. Handle errors gracefully."),
    (re.compile(r'unwrap\(\)'), 
     "Use of Panic", "low", "Using unwrap() which can lead to panic. Consider handle errors with match or if-let."),
    (re.compile(r'\.expect\('), 
     "Use of Panic", "low", "Using expect() which can lead to panic. Consider handling errors gracefully."),
]

# Command injection vulnerabilities
COMMAND_INJECTION_PATTERNS = [
    (re.compile(r'Command::new\s*\([^)]*\)\.arg\s*\([^)]*\+'), 
     "Command Injection", "high", "Possible command injection by concatenating user input in command arguments."),
    (re.compile(r'Command::new\s*\([^)]*\)\.args\s*\([^)]*\+'), 
     "Command Injection", "high", "Possible command injection by concatenating user input in command arguments."),
    (re.compile(r'Command::new\s*\([^)]*\+[^)]*\)'), 
     "Command Injection", "high", "Possible command injection by concatenating user input in command name."),
    (re.compile(r'(?:Command|cmd).*\.arg\s*\(\s*format!'), 
     "Command Injection", "high", "Possible command injection using format! with user input in command."),
    (re.compile(r'Command::new\("echo"\)\.arg\(format!'), 
     "Command Injection", "high", "Possible command injection using format! with user input."),
]

# SQL injection vulnerabilities
SQL_INJECTION_PATTERNS = [
    (re.compile(r'(?:query|execute|prepare)\s*\([\'"][^\'"]*[\'"] \+ '), 
     "SQL Injection", "critical", "Possible SQL injection through string concatenation. Use parameterized queries."),
    (re.compile(r'(?:query|execute|prepare)\s*\(format!\s*\([\'"][^\'"]*\{[^\}]*\}[^\'"]*[\'"]'),
     "SQL Injection", "critical", "Possible SQL injection using format! macro. Use parameterized queries instead."),
    (re.compile(r'(?:client|pool|conn)\.(?:query|execute)\s*\([\'"][^\'"]*[\'"] \+ '),
     "SQL Injection", "critical", "Possible SQL injection through string concatenation. Use parameterized queries."),
    (re.compile(r'(?:client|pool|conn)\.query\s*\(\s*format!'),
     "SQL Injection", "critical", "Possible SQL injection using format! macro. Use parameterized queries instead."),
    (re.compile(r'client\.query\(format!'),
     "SQL Injection", "critical", "Possible SQL injection using format! macro in client.query. Use parameterized queries instead."),
]

# Crypto security
CRYPTO_PATTERNS = [
    (re.compile(r'rand::thread_rng'), 
     "Non-Cryptographic RNG", "medium", "Using thread_rng() which is not suitable for cryptographic purposes."),
    (re.compile(r'md5::Md5|md5::compute'), 
     "Weak Hash Function", "medium", "Using MD5 which is cryptographically broken. Use SHA-256 or stronger."),
    (re.compile(r'sha1::Sha1|sha1::compute'), 
     "Weak Hash Function", "medium", "Using SHA1 which is cryptographically weak. Use SHA-256 or stronger."),
]

# Combine all patterns
RUST_PATTERNS = (
    UNSAFE_CODE_PATTERNS + 
    MEMORY_MANAGEMENT_PATTERNS + 
    CONCURRENCY_PATTERNS + 
    FFI_PATTERNS + 
    SERIALIZATION_PATTERNS + 
    UNWRAP_PATTERNS + 
    PANIC_PATTERNS +
    COMMAND_INJECTION_PATTERNS +
    SQL_INJECTION_PATTERNS +
    CRYPTO_PATTERNS
) 