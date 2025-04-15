"""
Security analyzer plugin for detecting security vulnerabilities in code.
"""

import ast
import logging
import re
import os
from functools import lru_cache
from typing import List, Dict, Any, Optional

from utils.analyzer_plugin import AnalyzerPlugin
from .vulnerability import SecurityVulnerability
from .patterns.python_patterns import PYTHON_PATTERNS
from .patterns.javascript_patterns import JAVASCRIPT_PATTERNS
from .patterns.rust_patterns import RUST_PATTERNS
from .patterns.php_patterns import PHP_PATTERNS
from .patterns.java_patterns import JAVA_PATTERNS
from .patterns.csharp_patterns import CSHARP_PATTERNS
from .ast_analysis import analyze_python_ast
from .parallel import analyze_files_parallel
from .patterns.go_patterns import GO_PATTERNS
from .patterns.ruby_patterns import RUBY_PATTERNS

logger = logging.getLogger("CodeReviewTool.SecurityAnalyzer")

class SecurityAnalyzerPlugin(AnalyzerPlugin):
    """
    Plugin for analyzing security vulnerabilities in code.
    Focuses on detecting common security issues based on patterns and AST analysis.
    """
    
    # Define severity levels and their weights
    SEVERITY_WEIGHTS = {
        "critical": 10,
        "high": 7,
        "medium": 4,
        "low": 1,
        "info": 0
    }
    
    # Cache for pattern compilation to improve performance
    _PATTERN_CACHE = {}
    
    @property
    def name(self) -> str:
        return "security_analyzer"

    @property
    def description(self) -> str:
        return "Analyzes code for common security vulnerabilities and weaknesses"

    @property
    def supported_languages(self) -> List[str]:
        return ["py", "js", "ts", "php", "java", "rb", "go", "cs", "rs"]  # Added Rust
    
    def __init__(self):
        # Initialize credential and comment patterns
        # Pre-compile common patterns for better performance
        self._compile_patterns()
        # Initialize context tracking for multiline analysis
        self._context = {}
    
    def _compile_patterns(self) -> None:
        """
        Pre-compile regex patterns for better performance.
        """
        # Hardcoded credentials patterns
        self._credential_patterns = [
            re.compile(r'(?i)(?:password|passwd|pwd|secret|token|api[-_]?key)\s*[:=]\s*[\'"][^\'"]+[\'"]'),
            re.compile(r'(?i)(?:username|user[-_]?name|user[-_]?id)\s*[:=]\s*[\'"][^\'"]+[\'"]'),
            re.compile(r'(?i)(?:https?:\/\/[^:\/\s]+:[^@\/\s]+@)'),  # URLs with embedded credentials
        ]
        
        # Insecure comment patterns
        self._comment_patterns = [
            re.compile(r'(?i)(?:fixme|todo|hack|workaround|temporary).*(?:security|auth|password|cred|token|secret)'),
            re.compile(r'(?i)disable\s*(?:security|authentication|auth|validation|verify|ssl)'),
            re.compile(r'(?i)bypass\s*(?:security|authentication|auth|validation|verify)'),
        ]
        
        # PHP, Java, Ruby, Go, C# and Rust patterns will be compiled by language-specific analyzers
    
    def _analyze_file_impl(self, file_path: str, file_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a file for security vulnerabilities.
        
        Args:
            file_path: Path to the file
            file_content: Optional content of the file (if already loaded)
            
        Returns:
            Dictionary with analysis results
        """
        vulnerabilities = []
        error = None
        
        try:
            # Get file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lstrip(".").lower()
            
            # Skip unsupported file types
            if ext not in self.supported_languages:
                return {
                    "vulnerabilities": [],
                    "risk_score": 0,
                    "message": f"File type '{ext}' not supported for security analysis"
                }
            
            # Read file content if not provided
            if file_content is None:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        file_content = f.read()
                except Exception as e:
                    error = f"Error reading file: {e}"
                    return {
                        "vulnerabilities": [],
                        "risk_score": 0,
                        "error": error
                    }
            
            # Split content into lines
            lines = file_content.split("\n")
            
            # Check for hardcoded credentials
            vulnerabilities.extend(self._check_hardcoded_credentials(file_path, lines))
            
            # Check for insecure comments
            vulnerabilities.extend(self._check_insecure_comments(file_path, lines))
            
            # Language-specific checks
            if ext == "py":
                vulnerabilities.extend(self._check_python_vulnerabilities(file_path, lines, file_content))
            elif ext in ["js", "ts", "jsx", "tsx"]:
                vulnerabilities.extend(self._check_javascript_vulnerabilities(file_path, lines))
            elif ext == "php":
                vulnerabilities.extend(self._check_php_vulnerabilities(file_path, lines))
            elif ext == "java":
                vulnerabilities.extend(self._check_java_vulnerabilities(file_path, lines))
            elif ext == "rb":
                vulnerabilities.extend(self._check_ruby_vulnerabilities(file_path, lines))
            elif ext == "go":
                vulnerabilities.extend(self._check_go_vulnerabilities(file_path, lines))
            elif ext == "cs":
                vulnerabilities.extend(self._check_csharp_vulnerabilities(file_path, lines))
            elif ext == "rs":
                vulnerabilities.extend(self._check_rust_vulnerabilities(file_path, lines))
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(vulnerabilities)
            
        except Exception as e:
            error = f"Error in security analysis: {e}"
            logger.error(f"Error analyzing {file_path}: {e}", exc_info=True)
            risk_score = 0
        
        return {
            "vulnerabilities": [v.to_dict() for v in vulnerabilities],
            "risk_score": risk_score,
            "error": error
        }
    
    def get_prompt_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from analysis for use in LLM prompts.
        
        Args:
            analysis_results: Results from analyze_file
            
        Returns:
            Dictionary with data for prompts
        """
        if not analysis_results:
            return {"security_analyzer_security_issues": "No security analysis results available."}
        
        vulnerabilities = analysis_results.get("vulnerabilities", [])
        
        if not vulnerabilities:
            return {"security_analyzer_security_issues": "No security issues detected."}
        
        # Format vulnerabilities for prompt
        formatted_vulnerabilities = []
        # Group by severity - start with critical
        for severity in ["critical", "high", "medium", "low"]:
            severity_vulns = [v for v in vulnerabilities if v.get("severity", "") == severity]
            if severity_vulns:
                formatted_vulnerabilities.append(f"\n{severity.upper()} Severity Issues:")
                for idx, vuln in enumerate(severity_vulns, 1):
                    line = vuln.get("line", "?")
                    vuln_type = vuln.get("vulnerability_type", "Unknown")
                    message = vuln.get("message", "No details provided")
                    
                    vulnerability_str = f"{idx}. Line {line}: {vuln_type} - {message}"
                    formatted_vulnerabilities.append(vulnerability_str)
        
        # Combine and add summary
        if formatted_vulnerabilities:
            summary = f"Security Analysis: Found {len(vulnerabilities)} potential security issues"
            # Add severity summary
            severity_counts = {}
            for v in vulnerabilities:
                severity = v.get("severity", "unknown")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            severity_summary = ", ".join(f"{count} {sev}" for sev, count in severity_counts.items())
            security_issues = f"{summary} ({severity_summary}):\n" + "\n".join(formatted_vulnerabilities)
        else:
            security_issues = "No security issues detected."
        
        return {"security_analyzer_security_issues": security_issues}
    
    def _calculate_risk_score(self, vulnerabilities: List[SecurityVulnerability]) -> int:
        """
        Calculate a risk score based on vulnerabilities.
        
        Args:
            vulnerabilities: List of vulnerabilities
            
        Returns:
            Risk score (0-100)
        """
        if not vulnerabilities:
            return 0
        
        # Base score on weighted severity
        total_score = 0
        for vulnerability in vulnerabilities:
            weight = self.SEVERITY_WEIGHTS.get(vulnerability.severity, 1)
            
            # Adjust for confidence
            confidence_factor = 1.0
            if vulnerability.confidence == "high":
                confidence_factor = 1.0
            elif vulnerability.confidence == "medium":
                confidence_factor = 0.8
            elif vulnerability.confidence == "low":
                confidence_factor = 0.5
            
            total_score += weight * confidence_factor
        
        # Normalize to 0-100 scale
        # Cap at 100 - a file with 10 critical issues will have a max score
        max_theoretical_score = 10 * self.SEVERITY_WEIGHTS["critical"]
        normalized_score = min(100, int((total_score / max_theoretical_score) * 100))
        
        return normalized_score
    
    def _check_hardcoded_credentials(self, file_path: str, lines: List[str]) -> List[SecurityVulnerability]:
        """
        Check for hardcoded credentials in code.
        
        Args:
            file_path: Path to the file
            lines: List of code lines
            
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        for i, line in enumerate(lines):
            # Skip comments or empty lines in credential detection
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//", "/*", "*", "'")):
                continue
                
            for pattern in self._credential_patterns:
                match = pattern.search(line)
                if match:
                    # Extract the credential identifier and the line
                    matched_text = match.group(0)
                    
                    # Determine if this is definitely a credential or possibly a false positive
                    has_credential_keyword = any(keyword in matched_text.lower() for keyword 
                             in ["password", "secret", "token", "key", "credential"])
                    
                    # Determine appropriate severity based on confidence
                    severity = "high" if has_credential_keyword else "medium"
                    confidence = "high" if has_credential_keyword else "medium"
                    
                    # Check for test/example context which reduces severity
                    is_test_context = (
                        "test" in file_path.lower() or 
                        any(test_word in line.lower() for test_word in ["test", "example", "mock", "dummy", "fake"])
                    )
                    
                    if is_test_context:
                        severity = "low"
                        confidence = "medium"
                    
                    # Create vulnerability
                    vuln = SecurityVulnerability(
                        file_path=file_path,
                        line=i+1,
                        vulnerability_type="Hardcoded Credentials",
                        severity=severity,
                        message=f"Possible hardcoded credential found: {matched_text}",
                        code_snippet=line.strip(),
                        confidence=confidence
                    )
                    vulnerabilities.append(vuln)
                    
                    # Once found, no need to check other patterns for this line
                    break
        
        return vulnerabilities
    
    def _check_insecure_comments(self, file_path: str, lines: List[str]) -> List[SecurityVulnerability]:
        """
        Check for comments indicating security issues.
        
        Args:
            file_path: Path to the file
            lines: List of code lines
            
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        # Look for indicators in comments
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Only process comment lines
            is_comment = False
            for comment_marker in ["#", "//", "/*", "*", "'"]:
                if stripped.startswith(comment_marker):
                    is_comment = True
                    break
            
            if not is_comment:
                # Also check for trailing comments
                for marker in ["#", "//"]:
                    if marker in stripped:
                        # Take only the comment part
                        comment_idx = stripped.find(marker)
                        stripped = stripped[comment_idx:]
                        is_comment = True
                        break
            
            if is_comment:
                for pattern in self._comment_patterns:
                    match = pattern.search(stripped)
                    if match:
                        message = f"Suspicious comment found: {stripped}"
                        vuln = SecurityVulnerability(
                            file_path=file_path,
                            line=i+1,
                            vulnerability_type="Security-related Comment",
                            severity="low",  # Security comments are low severity
                            message=message,
                            code_snippet=stripped,
                            confidence="medium"
                        )
                        vulnerabilities.append(vuln)
                        break
        
        return vulnerabilities
    
    def _check_python_vulnerabilities(self, file_path: str, lines: List[str], full_content: str) -> List[SecurityVulnerability]:
        """
        Check Python-specific vulnerabilities.
        
        Args:
            file_path: Path to the file
            lines: List of code lines
            full_content: Complete file content for AST parsing
            
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        # Use regex patterns first
        for i, line in enumerate(lines):
            for pattern, vuln_type, severity, message in PYTHON_PATTERNS:
                match = pattern.search(line)
                if match:
                    vuln = SecurityVulnerability(
                        file_path=file_path,
                        line=i+1,
                        vulnerability_type=vuln_type,
                        severity=severity,
                        message=message,
                        code_snippet=line.strip()
                    )
                    vulnerabilities.append(vuln)
        
        # Use AST analysis for deeper inspection
        try:
            tree = ast.parse(full_content, filename=file_path)
            ast_vulnerabilities = analyze_python_ast(file_path, tree, lines)
            vulnerabilities.extend(ast_vulnerabilities)
        except SyntaxError:
            # If parsing fails, log but continue with pattern-based results
            logger.warning(f"Failed to parse {file_path} for AST analysis")
        except Exception as e:
            logger.error(f"Error in AST analysis of {file_path}: {e}", exc_info=True)
        
        return vulnerabilities
    
    def _check_javascript_vulnerabilities(self, file_path: str, lines: List[str]) -> List[SecurityVulnerability]:
        """
        Check JavaScript-specific vulnerabilities.
        
        Args:
            file_path: Path to the file
            lines: List of code lines
            
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        for i, line in enumerate(lines):
            # Skip if the line has a nosec comment or security exception
            if any(marker in line for marker in ["// nosec", "/* nosec */", "security-exception"]):
                continue
                
            # Skip false positives for XSS that use sanitization
            if "innerHTML" in line and "DOMPurify.sanitize" in line:
                continue
                
            # Don't flag textContent as XSS (it's safe)
            if "textContent" in line:
                continue
                
            for pattern, vuln_type, severity, message in JAVASCRIPT_PATTERNS:
                match = pattern.search(line)
                if match:
                    # Determine confidence level based on file path
                    confidence = "medium"
                    if "test" in file_path.lower():
                        confidence = "low"
                        
                    vuln = SecurityVulnerability(
                        file_path=file_path,
                        line=i+1,
                        vulnerability_type=vuln_type,
                        severity=severity,
                        message=message,
                        code_snippet=line.strip(),
                        confidence=confidence
                    )
                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _check_php_vulnerabilities(self, file_path: str, lines: List[str]) -> List[SecurityVulnerability]:
        """
        Check PHP-specific vulnerabilities.
        
        Args:
            file_path: Path to the file
            lines: List of code lines
            
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        for i, line in enumerate(lines):
            for pattern, vuln_type, severity, message in PHP_PATTERNS:
                match = pattern.search(line)
                if match:
                    vuln = SecurityVulnerability(
                        file_path=file_path,
                        line=i+1,
                        vulnerability_type=vuln_type,
                        severity=severity,
                        message=message,
                        code_snippet=line.strip()
                    )
                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _check_java_vulnerabilities(self, file_path: str, lines: List[str]) -> List[SecurityVulnerability]:
        """
        Check Java-specific vulnerabilities.
        
        Args:
            file_path: Path to the file
            lines: List of code lines
            
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        # Build context for multi-line analysis
        context = {
            "in_test_class": False,
            "in_test_method": False,
            "has_validation": False,
            "has_sanitization": False,
            "imports": set(),
            "class_annotations": set(),
        }
        
        # First pass to gather context
        for line in lines:
            line = line.strip()
            # Look for imports that might indicate security libraries
            if line.startswith("import "):
                import_stmt = line[7:].strip().rstrip(";")
                context["imports"].add(import_stmt)
                
            # Detect test classes/methods
            if "@Test" in line or "public void test" in line:
                context["in_test_method"] = True
            if "class" in line and ("Test" in line or "Tests" in line or "MockMvc" in line):
                context["in_test_class"] = True
                
            # Look for validation annotations
            if any(anno in line for anno in ["@Valid", "@Validated", "@NotNull", "@Pattern", "@Size"]):
                context["has_validation"] = True
                
            # Look for security library usage
            if any(lib in line for lib in ["SecurityContext", "Authentication", "PermissionEvaluator", 
                                           "encode(", "matches(", "HtmlUtils.htmlEscape", 
                                           "StringEscapeUtils", "sanitize"]):
                context["has_sanitization"] = True
                
            # Collect class-level annotations
            if line.startswith("@") and "class" in lines[lines.index(line) + 1]:
                context["class_annotations"].add(line.strip())
        
        # Now check for vulnerabilities with context
        for i, line in enumerate(lines):
            # Skip if the line has a nosec comment or security exception
            if any(marker in line for marker in ["// nosec", "/* nosec */", "@SuppressWarnings", "security-exception"]):
                continue
                
            # Skip lines with expected exceptions for tests
            if "expected" in line and "Exception" in line:
                continue
                
            # Skip JPA/Hibernate createQuery (often false positive for SQL injection)
            if "createQuery" in line and any(orm in line for orm in ["em.", "entityManager.", "session."]):
                continue
                
            # Skip if using validated repository methods
            if ".findBy" in line and context["has_validation"]:
                continue
                
            # Skip false positives for prepared statements when using parameter binding
            if "prepareStatement" in line and "?" in line and ".set" in "".join(lines[i:min(i+5, len(lines))]):
                continue
                
            # Don't flag for patterns in test classes that are expected
            if (context["in_test_class"] or context["in_test_method"]) and any(
                    test_pattern in line for test_pattern in 
                    ["assertEquals", "assertThat", "mock", "when", "verify", "given", "willReturn"]):
                continue
                
            # Filter hardcoded secrets in test files
            if ("Hardcoded Secret" in str(JAVA_PATTERNS) and 
                (context["in_test_class"] or "test" in file_path.lower() or "mock" in file_path.lower())):
                continue
                
            for pattern, vuln_type, severity, message in JAVA_PATTERNS:
                match = pattern.search(line)
                if match:
                    # Base confidence calculation
                    confidence = "medium"
                    
                    # Adjust confidence based on context
                    if "test" in file_path.lower() or "tests" in file_path.lower() or context["in_test_class"]:
                        confidence = "low"
                    
                    # Higher confidence for certain vulnerability types with clear patterns
                    if vuln_type in ["SQL Injection", "Command Injection", "XML External Entity (XXE)"] and not context["has_sanitization"]:
                        if confidence != "low":
                            confidence = "high"
                    
                    # Look at surrounding context to reduce false positives
                    skip_vulnerability = False
                    
                    # Check for validation/sanitization in nearby lines
                    nearby_lines = lines[max(0, i-3):min(i+4, len(lines))]
                    if any("validate" in l.lower() or "sanitize" in l.lower() or "escape" in l.lower() 
                          or "StringUtils" in l or "prepareStatement" in l for l in nearby_lines):
                        confidence = "low"
                        
                    # Spring Security and other frameworks often handle many vulnerabilities automatically
                    if "org.springframework.security" in str(context["imports"]) and vuln_type in ["XSS", "CSRF"]:
                        confidence = "low"
                        
                    # Don't report if there's overwhelming evidence it's a false positive
                    if (confidence == "low" and 
                        (context["in_test_method"] or 
                         any(safe_pattern in line for safe_pattern in ["mock", "test", "assert", "example"]))):
                        skip_vulnerability = True
                        
                    if not skip_vulnerability:
                        vuln = SecurityVulnerability(
                            file_path=file_path,
                            line=i+1,
                            vulnerability_type=vuln_type,
                            severity=severity,
                            message=message,
                            code_snippet=line.strip(),
                            confidence=confidence
                        )
                        vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _check_ruby_vulnerabilities(self, file_path: str, lines: List[str]) -> List[SecurityVulnerability]:
        """
        Check Ruby-specific vulnerabilities.
        
        Args:
            file_path: Path to the file
            lines: List of code lines
            
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        # Skip security checks for test files but with reduced confidence
        is_test_file = any(pattern in file_path.lower() for pattern in ["test", "tests", "spec", "mock", "fake", "stub", "fixture"])
        test_confidence = "low" if is_test_file else "medium"
        
        # Check for Rails-specific mass assignment patterns
        is_rails_controller = any(line.strip().startswith("class") and "Controller" in line for line in lines)
        
        # Find method definitions for multi-line analysis
        method_start_indexes = []
        in_method = False
        current_method_start = 0
        
        for i, line in enumerate(lines):
            # Method definition in Ruby - can be identified by 'def' keyword
            if re.search(r'^\s*def\s+\w+', line):
                if in_method:
                    # Store the previous method's boundaries
                    method_start_indexes.append((current_method_start, i-1))
                
                # Start a new method
                in_method = True
                current_method_start = i
            
            # Skip if the line has a security comment or nosec marker
            if any(marker in line for marker in ["# nosec", "# rubocop:disable Security", "# noqa"]):
                continue
                
            # Check for line-level vulnerabilities
            for pattern, vuln_type, severity, message in RUBY_PATTERNS:
                match = pattern.search(line)
                if match:
                    # Check for false positives
                    if "SQL Injection" == vuln_type and ".where(" in line and ":" in line:
                        # Skip if using Rails' safe query methods with symbol params
                        continue
                    
                    # Skip permitted parameters for mass assignment
                    if "Mass Assignment" == vuln_type and ".permit" in line:
                        continue
                    
                    # Default confidence is medium, but set to low for test files
                    confidence = test_confidence
                    
                    # If it's a Rails controller and the vulnerability is Mass Assignment, increase confidence
                    if is_rails_controller and vuln_type == "Mass Assignment":
                        confidence = "high"
                    
                    vuln = SecurityVulnerability(
                        file_path=file_path,
                        line=i+1,
                        vulnerability_type=vuln_type,
                        severity=severity,
                        message=message,
                        code_snippet=line.strip(),
                        confidence=confidence
                    )
                    vulnerabilities.append(vuln)
        
        # Process any remaining method
        if in_method:
            method_start_indexes.append((current_method_start, len(lines)-1))
        
        # Check for Rails-specific controller patterns
        if is_rails_controller:
            for start, end in method_start_indexes:
                method_lines = lines[start:end+1]
                method_content = "\n".join(method_lines)
                
                # Check for missing input validation
                if any(attr in method_content for attr in ["def create", "def update", "def new"]):
                    # Look for params validation
                    has_validation = any(v in method_content for v in ["validates", "valid?", "presence", "if", "unless", "permit"])
                    if not has_validation:
                        vuln = SecurityVulnerability(
                            file_path=file_path,
                            line=start+1,
                            vulnerability_type="Missing Input Validation",
                            severity="medium",
                            message="Rails controller action may lack proper input validation. Consider using strong parameters or model validations.",
                            code_snippet=method_lines[0].strip(),
                            confidence=test_confidence
                        )
                        vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _check_go_vulnerabilities(self, file_path: str, lines: List[str]) -> List[SecurityVulnerability]:
        """
        Check Go-specific vulnerabilities.
        
        Args:
            file_path: Path to the file
            lines: List of code lines
            
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        for i, line in enumerate(lines):
            for pattern, vuln_type, severity, message in GO_PATTERNS:
                match = pattern.search(line)
                if match:
                    vuln = SecurityVulnerability(
                        file_path=file_path,
                        line=i+1,
                        vulnerability_type=vuln_type,
                        severity=severity,
                        message=message,
                        code_snippet=line.strip()
                    )
                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _check_csharp_vulnerabilities(self, file_path: str, lines: List[str]) -> List[SecurityVulnerability]:
        """
        Check C#-specific vulnerabilities.
        
        Args:
            file_path: Path to the file
            lines: List of code lines
            
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        # Skip security checks for test files but with reduced confidence
        is_test_file = any(pattern in file_path.lower() for pattern in ["test", "tests", "mock", "fake", "stub", "spec"])
        test_confidence = "low" if is_test_file else "medium"
        
        # Direct check for connection strings (for multi-line cases)
        joined_content = "\n".join(lines)
        connection_string_pattern = re.compile(r'(?:connectionString|ConnectionString).*?=\s*["\'].*?(?:User ID|uid|Password|pwd).*?["\']|["\'].*?(?:Server|Data Source).*?(?:User ID|uid).*?(?:Password|pwd).*?["\']', re.DOTALL | re.IGNORECASE)
        for conn_match in connection_string_pattern.finditer(joined_content):
            match_pos = conn_match.start()
            line_number = 0
            pos_in_content = 0
            for i, line in enumerate(lines):
                line_length = len(line) + 1  # +1 for newline
                if pos_in_content <= match_pos < pos_in_content + line_length:
                    line_number = i
                    break
                pos_in_content += line_length
                
            vuln = SecurityVulnerability(
                file_path=file_path,
                line=line_number + 1,
                vulnerability_type="Hardcoded Connection String",
                severity="critical",
                message="Connection string with credentials is hardcoded. Use secure configuration methods instead.",
                code_snippet=lines[line_number].strip(),
                confidence="high"
            )
            vulnerabilities.append(vuln)
        
        # Find method definitions for multi-line analysis
        method_regex = re.compile(r'(?:public|private|protected|internal)\s+(?:async\s+)?[A-Za-z0-9_<>]+\s+[A-Za-z0-9_]+\s*\([^)]*\)\s*(?:\s*where\s+[^{]+)?\s*\{', re.MULTILINE)
        for match in method_regex.finditer(joined_content):
            method_start = joined_content.count('\n', 0, match.start())
            
            # Find matching closing brace
            open_braces = 1
            method_end = method_start
            for i in range(method_start + 1, len(lines)):
                if i >= len(lines):
                    break
                open_braces += lines[i].count('{')
                open_braces -= lines[i].count('}')
                if open_braces == 0:
                    method_end = i
                    break
            
            # Extract method content
            method_lines = lines[method_start:method_end+1]
            
            # Skip empty methods
            if not method_lines:
                continue
                
            # Analyze method for multi-line patterns
            method_content = '\n'.join(method_lines)
            
            # Check for missing input validation in controller actions
            if "Controller" in file_path and any(attr in method_content for attr in ["[HttpGet]", "[HttpPost]", "[HttpPut]", "[HttpDelete]"]):
                # Look for parameters without validation
                param_regex = re.compile(r'\([^)]*\bstring\b[^)]*\)')
                for param_match in param_regex.finditer(method_content):
                    # Check if validation attributes are present
                    has_validation = any(v in method_content for v in ["[Required]", "[StringLength", "[RegularExpression", "[Range", "ModelState.IsValid", "FluentValidation"])
                    if not has_validation:
                        line_number = method_start + method_content[:param_match.start()].count('\n')
                        vuln = SecurityVulnerability(
                            file_path=file_path,
                            line=line_number + 1,
                            vulnerability_type="Missing Input Validation",
                            severity="medium",
                            message="Controller action parameters should have validation attributes or ModelState validation.",
                            code_snippet=method_lines[line_number - method_start].strip(),
                            confidence=test_confidence
                        )
                        vulnerabilities.append(vuln)
            
            # Check for insecure deserialization with type handling
            if "Deserialize" in method_content and "TypeNameHandling" in method_content:
                # Check if TypeNameHandling is set to a value other than None
                if re.search(r'TypeNameHandling\s*=\s*(?!TypeNameHandling\.None)', method_content):
                    line_number = method_start
                    type_name_pos = method_content.find("TypeNameHandling")
                    if type_name_pos >= 0:
                        line_number += method_content[:type_name_pos].count('\n')
                        
                    # Ensure line number is within method boundaries
                    if line_number - method_start < len(method_lines):
                        vuln = SecurityVulnerability(
                            file_path=file_path,
                            line=line_number + 1,
                            vulnerability_type="Insecure Deserialization",
                            severity="critical",
                            message="TypeNameHandling set to a value other than None allows attackers to execute arbitrary code during deserialization.",
                            code_snippet=method_lines[line_number - method_start].strip(),
                            confidence="high"
                        )
                        vulnerabilities.append(vuln)
        
        # Check line by line for single-line patterns
        for i, line in enumerate(lines):
            # Skip if the line has a nosec comment or security exception
            if any(marker in line for marker in ["// nosec", "/* nosec */", "security-exception", "#pragma warning disable SECURITY_WARNINGS"]):
                continue
                
            for pattern, vuln_type, severity, message in CSHARP_PATTERNS:
                match = pattern.search(line)
                if match:
                    # Default confidence is medium, but set to low for test files
                    confidence = test_confidence
                        
                    vuln = SecurityVulnerability(
                        file_path=file_path,
                        line=i+1,
                        vulnerability_type=vuln_type,
                        severity=severity,
                        message=message,
                        code_snippet=line.strip(),
                        confidence=confidence
                    )
                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _check_rust_vulnerabilities(self, file_path: str, lines: List[str]) -> List[SecurityVulnerability]:
        """
        Check Rust-specific vulnerabilities.
        
        Args:
            file_path: Path to the file
            lines: List of code lines
            
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        for i, line in enumerate(lines):
            for pattern, vuln_type, severity, message in RUST_PATTERNS:
                match = pattern.search(line)
                if match:
                    # Default confidence is medium, but set to low for test files
                    confidence = "medium"
                    # Check if the file is a test file
                    if "test" in file_path.lower() or "_test" in file_path.lower() or "tests" in file_path.lower():
                        confidence = "low"
                        
                    vuln = SecurityVulnerability(
                        file_path=file_path,
                        line=i+1,
                        vulnerability_type=vuln_type,
                        severity=severity,
                        message=message,
                        code_snippet=line.strip(),
                        confidence=confidence
                    )
                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def analyze_files_parallel(self, file_paths: List[str], max_workers: int = 8) -> Dict[str, Any]:
        """
        Analyze multiple files in parallel.
        
        Args:
            file_paths: List of file paths to analyze
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping file paths to analysis results
        """
        return analyze_files_parallel(self, file_paths, max_workers) 