"""
Abstract syntax tree-based security analysis for Python code.
"""

import ast
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from .vulnerability import SecurityVulnerability

logger = logging.getLogger("CodeReviewTool.SecurityAnalyzer.AstAnalysis")

def analyze_python_ast(file_path: str, tree: ast.AST, lines: List[str]) -> List[SecurityVulnerability]:
    """
    Analyze Python code using AST for security vulnerabilities.
    
    Args:
        file_path: Path to the file
        tree: AST tree of the Python code
        lines: Source code lines
        
    Returns:
        List of security vulnerabilities
    """
    vulnerabilities = []
    
    # Track potentially dangerous variables
    tainted_variables = set()
    
    # First pass: identify variables from user input or other tainted sources
    taint_tracker = TaintTracker()
    taint_tracker.visit(tree)
    tainted_variables = taint_tracker.tainted_variables
    
    # Second pass: track usage of tainted variables
    visitor = SecurityVisitor(file_path, lines, tainted_variables)
    visitor.visit(tree)
    vulnerabilities.extend(visitor.vulnerabilities)
    
    return vulnerabilities


class TaintTracker(ast.NodeVisitor):
    """Visit AST nodes to track tainted variables (user input, etc.)"""
    
    def __init__(self):
        self.tainted_variables: Set[str] = set()
        # Common input sources
        self.input_functions = {
            'input', 'raw_input',  # Command line input
            'request.get', 'request.post', 'request.args.get', 'request.form.get',  # Flask/Django
            'getattr', 'eval', 'exec',  # Dynamic attribute access and code execution
        }
        # Common network/file sources
        self.external_sources = {
            'open', 'urllib.request.urlopen', 'requests.get', 'requests.post',
            'socket.recv', 'subprocess.check_output', 'json.loads',
        }
    
    def visit_Call(self, node):
        # Check if the function call is to a known tainted source
        func_name = self._get_func_name(node.func)
        if func_name:
            # Check for input or external data sources
            if any(source in func_name for source in self.input_functions):
                # If the result is assigned to a variable, mark it as tainted
                self._mark_assignment_target(node)
            elif any(source in func_name for source in self.external_sources):
                # Mark variables assigned from external sources
                self._mark_assignment_target(node)
        
        # Visit all child nodes
        self.generic_visit(node)
    
    def _mark_assignment_target(self, node):
        """Mark the variable an expression is assigned to as tainted if we can find it."""
        # Navigate up to find assignment
        parent = getattr(node, 'parent', None)
        if parent and isinstance(parent, ast.Assign):
            for target in parent.targets:
                if isinstance(target, ast.Name):
                    self.tainted_variables.add(target.id)
    
    def _get_func_name(self, func_node) -> Optional[str]:
        """Extract function name from a Call node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            # For method calls like requests.get
            obj_name = self._get_func_name(func_node.value)
            return f"{obj_name}.{func_node.attr}" if obj_name else func_node.attr
        return None


class SecurityVisitor(ast.NodeVisitor):
    """AST visitor that checks for security issues."""
    
    def __init__(self, file_path: str, lines: List[str], tainted_variables: Set[str]):
        self.file_path = file_path
        self.lines = lines
        self.tainted_variables = tainted_variables
        self.vulnerabilities: List[SecurityVulnerability] = []
        self.current_line = 0
    
    def visit(self, node):
        """Override to track line numbers."""
        if hasattr(node, 'lineno'):
            self.current_line = node.lineno
        super().visit(node)
    
    def visit_Call(self, node):
        """Check function/method calls for security issues."""
        func_name = self._get_func_name(node.func)
        
        if func_name:
            # Check for SQL injection via execute with string formatting
            if any(db_call in func_name.lower() for db_call in ['execute', 'executemany', 'cursor.execute']):
                self._check_sql_injection(node)
            
            # Check for command injection with system/subprocess calls
            elif any(cmd in func_name.lower() for cmd in ['system', 'popen', 'subprocess.run', 'subprocess.call']):
                self._check_command_injection(node)
            
            # Check for unsafe deserialization
            elif any(deser in func_name.lower() for deser in ['pickle.load', 'pickle.loads', 'yaml.load']):
                self._check_deserialization(node, func_name)
            
            # Check for file access vulnerabilities
            elif any(file_op in func_name.lower() for file_op in ['open', 'read', 'write']):
                self._check_file_access(node)
            
            # Check for JWT security
            elif 'jwt.decode' in func_name.lower():
                self._check_jwt_security(node)
            
            # Check for weak cryptography
            elif 'md5' in func_name.lower() or 'sha1' in func_name.lower():
                self._add_vulnerability(
                    "Weak Hashing Algorithm",
                    "medium",
                    f"Using weak hashing algorithm ({func_name}). Consider using SHA-256 or stronger."
                )
            
            # Check for eval and exec
            elif func_name in ['eval', 'exec']:
                self._check_code_execution(node, func_name)
        
        # Continue visiting child nodes
        self.generic_visit(node)
    
    def _check_sql_injection(self, node):
        """Check for SQL injection vulnerabilities."""
        if not node.args:
            return
        
        # Look for string formatting in SQL queries
        sql_arg = node.args[0]
        
        # Check for f-strings (Python 3.6+)
        if isinstance(sql_arg, ast.JoinedStr):
            self._add_vulnerability(
                "SQL Injection",
                "critical",
                "Using f-string in SQL query. Use parameterized queries instead."
            )
            return
        
        # Check for string concatenation or formatting
        if isinstance(sql_arg, ast.BinOp) and isinstance(sql_arg.op, (ast.Add, ast.Mod)):
            self._add_vulnerability(
                "SQL Injection",
                "critical",
                "String concatenation or formatting in SQL query. Use parameterized queries instead."
            )
            return
        
        # Check for .format() method
        if isinstance(sql_arg, ast.Call) and hasattr(sql_arg.func, 'attr') and sql_arg.func.attr == 'format':
            self._add_vulnerability(
                "SQL Injection",
                "critical",
                "Using .format() in SQL query. Use parameterized queries instead."
            )
            return
    
    def _check_command_injection(self, node):
        """Check for command injection vulnerabilities."""
        # Check if shell=True is being used
        for keyword in node.keywords:
            if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                self._add_vulnerability(
                    "Command Injection Risk",
                    "high",
                    "Using shell=True can be dangerous. Avoid it when executing commands with variables."
                )
                break
        
        # Check if the command includes a variable or string concatenation
        if node.args:
            cmd_arg = node.args[0]
            # Check for string concatenation
            if isinstance(cmd_arg, ast.BinOp) and isinstance(cmd_arg.op, ast.Add):
                self._add_vulnerability(
                    "Command Injection",
                    "critical",
                    "String concatenation in command execution can lead to command injection."
                )
            # Check for f-strings
            elif isinstance(cmd_arg, ast.JoinedStr):
                self._add_vulnerability(
                    "Command Injection",
                    "critical",
                    "Using f-string in command execution can lead to command injection."
                )
            # Check for format method
            elif isinstance(cmd_arg, ast.Call) and hasattr(cmd_arg.func, 'attr') and cmd_arg.func.attr == 'format':
                self._add_vulnerability(
                    "Command Injection",
                    "critical",
                    "Using .format() in command execution can lead to command injection."
                )
    
    def _check_deserialization(self, node, func_name):
        """Check for unsafe deserialization vulnerabilities."""
        if 'pickle' in func_name.lower():
            self._add_vulnerability(
                "Insecure Deserialization",
                "high",
                "Pickle deserialization can lead to remote code execution. Never unpickle untrusted data."
            )
        elif 'yaml.load' in func_name.lower():
            # Check if safe_load is being used instead of load
            if 'safe_load' not in func_name.lower():
                self._add_vulnerability(
                    "Insecure Deserialization",
                    "high",
                    "yaml.load without SafeLoader can lead to remote code execution. Use yaml.safe_load() instead."
                )
    
    def _check_file_access(self, node):
        """Check for file access vulnerabilities."""
        # Check if file path includes a variable that might be tainted
        if node.args:
            file_arg = node.args[0]
            if isinstance(file_arg, ast.Name) and file_arg.id in self.tainted_variables:
                self._add_vulnerability(
                    "Path Traversal",
                    "high",
                    f"Using potentially tainted variable '{file_arg.id}' in file operation. Validate file paths."
                )
            elif isinstance(file_arg, ast.BinOp) and isinstance(file_arg.op, ast.Add):
                # Check for string concatenation with a tainted variable
                if self._contains_tainted_variable(file_arg):
                    self._add_vulnerability(
                        "Path Traversal",
                        "high",
                        "String concatenation with potentially tainted data in file operation. Validate file paths."
                    )
    
    def _check_jwt_security(self, node):
        """Check for JWT security issues."""
        # Check if verify=False is being used
        for keyword in node.keywords:
            if keyword.arg == 'verify' and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                self._add_vulnerability(
                    "Insecure JWT Verification",
                    "high",
                    "JWT verification is disabled (verify=False). Always verify JWT signatures."
                )
                break
        
        # Check if algorithms are specified (algorithm switching attacks)
        has_algorithms = False
        for keyword in node.keywords:
            if keyword.arg == 'algorithms' and isinstance(keyword.value, (ast.List, ast.Tuple)):
                has_algorithms = True
                break
        
        if not has_algorithms:
            self._add_vulnerability(
                "JWT Algorithm Vulnerability",
                "medium",
                "No algorithms specified for JWT verification. Specify algorithms to prevent algorithm switching attacks."
            )
    
    def _check_code_execution(self, node, func_name):
        """Check for dangerous code execution."""
        self._add_vulnerability(
            f"Use of {func_name}",
            "high",
            f"{func_name}() can execute arbitrary code and is dangerous with untrusted input.",
            confidence="high"
        )
        
        # Check if the argument contains a tainted variable
        if node.args and self._contains_tainted_variable(node.args[0]):
            self._add_vulnerability(
                f"Dangerous {func_name} with tainted data",
                "critical",
                f"Using {func_name}() with data from untrusted source. This can lead to code execution.",
                confidence="high"
            )
    
    def _contains_tainted_variable(self, node):
        """Check if an AST node contains any tainted variables."""
        if isinstance(node, ast.Name):
            return node.id in self.tainted_variables
        elif isinstance(node, ast.BinOp):
            return self._contains_tainted_variable(node.left) or self._contains_tainted_variable(node.right)
        elif isinstance(node, (ast.Call, ast.Attribute)):
            return any(self._contains_tainted_variable(arg) for arg in getattr(node, 'args', []))
        elif isinstance(node, ast.JoinedStr):
            return any(self._contains_tainted_variable(v) for v in node.values if isinstance(v, ast.FormattedValue))
        elif isinstance(node, ast.FormattedValue):
            return self._contains_tainted_variable(node.value)
        return False
    
    def _get_func_name(self, func_node) -> Optional[str]:
        """Extract function name from a Call node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            # For method calls like requests.get
            obj_name = self._get_func_name(func_node.value)
            return f"{obj_name}.{func_node.attr}" if obj_name else func_node.attr
        return None
    
    def _add_vulnerability(self, vuln_type: str, severity: str, message: str, confidence: str = "medium"):
        """Add a vulnerability to the list."""
        # Get the corresponding line of code
        line_idx = min(self.current_line - 1, len(self.lines) - 1)
        code_snippet = self.lines[line_idx].strip() if line_idx >= 0 else None
        
        vulnerability = SecurityVulnerability(
            file_path=self.file_path,
            line=self.current_line,
            vulnerability_type=vuln_type,
            severity=severity,
            message=message,
            code_snippet=code_snippet,
            confidence=confidence
        )
        self.vulnerabilities.append(vulnerability) 