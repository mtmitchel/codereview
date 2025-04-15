import ast
import logging
from typing import List, Dict, Any, Optional
from utils.analyzer_plugin import AnalyzerPlugin

logger = logging.getLogger("CodeReviewTool.SemanticAnalyzer")


class SemanticIssue:
    """
    Class to represent a semantic issue found in code.
    """
    def __init__(self, 
                 file_path: str, 
                 line: int,
                 end_line: Optional[int] = None,
                 issue_type: str = "",
                 severity: str = "medium", 
                 message: str = "",
                 code_snippet: Optional[str] = None):
        self.file_path = file_path
        self.line = line
        self.end_line = end_line or line
        self.issue_type = issue_type
        self.severity = severity
        self.message = message
        self.code_snippet = code_snippet
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the issue to a dictionary.
        
        Returns:
            Dictionary representation of the issue
        """
        return {
            "file_path": self.file_path,
            "line": self.line,
            "end_line": self.end_line,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "code_snippet": self.code_snippet
        }
    
    def __str__(self) -> str:
        loc = f"{self.file_path}:{self.line}"
        if self.end_line and self.end_line != self.line:
            loc += f"-{self.end_line}"
        return f"{loc} - {self.severity.upper()} {self.issue_type}: {self.message}"


class PythonSemanticAnalyzer:
    """Helper class for analyzing Python code semantics using AST."""
    
    # Threshold for function length (adjust as needed)
    MAX_FUNC_LINES = 50

    def __init__(self, file_path: str, code_content: str):
        self.file_path = file_path
        self.code_content = code_content
        self.lines = code_content.splitlines()
        self.issues = []
        self.ast_tree = None
        
        try:
            self.ast_tree = ast.parse(code_content)
        except SyntaxError as e:
            self.issues.append(SemanticIssue(
                file_path=file_path,
                line=e.lineno,
                issue_type="Syntax Error",
                severity="error",
                message=f"Python syntax error: {str(e)}",
                code_snippet=self.lines[e.lineno-1] if e.lineno-1 < len(self.lines) else None
            ))
        except Exception as parse_err: # Catch other potential parsing errors
            self.issues.append(SemanticIssue(
                file_path=file_path,
                line=1, # Default to line 1 if lineno not available
                issue_type="Parse Error",
                severity="error",
                message=f"Failed to parse Python code: {parse_err}"
            ))
            logging.getLogger("CodeReviewTool.SemanticAnalyzer").error(f"Failed to parse AST for {file_path}: {parse_err}")
    
    def analyze(self) -> List[SemanticIssue]:
        """Perform semantic analysis on the code."""
        if not self.ast_tree:
            return self.issues
            
        # Collect all variables
        self.variables = set()
        
        # Analyze code patterns
        self._check_unused_imports()
        self._check_unused_variables()
        self._check_complex_functions()
        self._check_nested_loops()
        self._check_large_classes()
        self._check_long_parameter_lists()
        self._check_magic_numbers()
        self._check_error_handling()
        self._check_long_functions()
        
        return self.issues
    
    def _check_unused_imports(self):
        """Check for unused imports."""
        imports = {}
        imported_names = set()  # names imported
        
        # Find all imports
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports[name.name] = name.asname or name.name
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    imported_name = name.asname or name.name
                    imports[f"{node.module}.{name.name}"] = imported_name
        
        # Find all used names
        used_names = set()
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # Identify unused imports
        for import_path, aliases in imports.items():
            for alias in aliases:
                if alias not in used_names:
                    # Get line number for the import
                    line_num = None
                    for node in ast.walk(self.ast_tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            for name in node.names:
                                if (isinstance(node, ast.Import) and name.name == import_path) or \
                                   (isinstance(node, ast.ImportFrom) and (name.asname or name.name) == alias):
                                    line_num = node.lineno
                                    break
                            if line_num:
                                break
                    
                    if line_num:
                        self.issues.append(SemanticIssue(
                            file_path=self.file_path,
                            line=line_num,
                            issue_type="Unused Import",
                            severity="info",
                            message=f"Import '{alias}' is not used in the code.",
                            code_snippet=self.lines[line_num-1] if line_num-1 < len(self.lines) else None
                        ))
    
    def _check_unused_variables(self):
        """Check for unused variables."""
        # Find all variable assignments
        assigned_vars = {}  # name -> [line numbers]
        used_vars = set()  # names used
        
        for node in ast.walk(self.ast_tree):
            # Check for assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in assigned_vars:
                            assigned_vars[target.id] = []
                        assigned_vars[target.id].append(node.lineno)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                if elt.id not in assigned_vars:
                                    assigned_vars[elt.id] = []
                                assigned_vars[elt.id].append(node.lineno)
            
            # Check for variable uses
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)
        
        # Special variables (parameters, etc.) that shouldn't trigger unused warnings
        special_vars = set(['self', 'cls'])
        
        # Identify unused variables
        for var_name, lines in assigned_vars.items():
            if var_name not in used_vars and var_name not in special_vars and not var_name.startswith('_'):
                for line_num in lines:
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=line_num,
                        issue_type="Unused Variable",
                        severity="info",
                        message=f"Variable '{var_name}' is assigned but never used.",
                        code_snippet=self.lines[line_num-1] if line_num-1 < len(self.lines) else None
                    ))
    
    def _check_complex_functions(self):
        """Check for overly complex functions (high cyclomatic complexity)."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = 1  # Base complexity
                
                # Count branches
                for subnode in ast.walk(node):
                    if isinstance(subnode, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                        complexity += 1
                    elif isinstance(subnode, ast.BoolOp) and isinstance(subnode.op, (ast.And, ast.Or)):
                        complexity += len(subnode.values) - 1
                
                if complexity > 10:
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        issue_type="Complex Function",
                        severity="medium",
                        message=f"Function '{node.name}' has high cyclomatic complexity ({complexity}). Consider refactoring.",
                        code_snippet=self.lines[node.lineno-1] if node.lineno-1 < len(self.lines) else None
                    ))
    
    def _check_nested_loops(self):
        """Check for deeply nested loops."""
        def find_nested_loops(node, depth=0):
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                depth += 1
                if depth >= 3:  # Flag triple-nested loops or deeper
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        issue_type="Deeply Nested Loops",
                        severity="medium",
                        message=f"Found deeply nested loop (depth {depth}). Consider refactoring for readability and performance.",
                        code_snippet=self.lines[node.lineno-1] if node.lineno-1 < len(self.lines) else None
                    ))
                
                # Check children for further nesting
                for child in ast.iter_child_nodes(node):
                    find_nested_loops(child, depth)
            else:
                # Continue checking children at current depth
                for child in ast.iter_child_nodes(node):
                    find_nested_loops(child, depth)
        
        # Start search from the root
        find_nested_loops(self.ast_tree)
    
    def _check_large_classes(self):
        """Check for overly large classes."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.ClassDef):
                method_count = sum(1 for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))
                
                start_line = node.lineno
                end_line = getattr(node, 'end_lineno', start_line)
                # Ensure end_line is properly fetched if available
                if hasattr(node, 'body') and node.body:
                     last_stmt = node.body[-1]
                     end_line = getattr(last_stmt, 'end_lineno', end_line)
                
                line_count = max(1, end_line - start_line + 1)
                
                # Adjust thresholds as needed
                MAX_METHODS = 15
                MAX_CLASS_LINES = 300
                
                if method_count > MAX_METHODS:
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=node.lineno,
                        issue_type="Large Class (Methods)",
                        severity="medium",
                        message=f"Class '{node.name}' has too many methods ({method_count} > {MAX_METHODS}). Consider splitting responsibilities.",
                        code_snippet=self.lines[node.lineno-1] if node.lineno-1 < len(self.lines) else None
                    ))
                if line_count > MAX_CLASS_LINES:
                     self.issues.append(SemanticIssue(
                         file_path=self.file_path,
                         line=node.lineno,
                         end_line=end_line,
                         issue_type="Large Class (Lines)",
                         severity="medium",
                         message=f"Class '{node.name}' is very long ({line_count} lines > {MAX_CLASS_LINES}). Consider refactoring.",
                         code_snippet=self.lines[node.lineno-1] if node.lineno-1 < len(self.lines) else None
                     ))
    
    def _check_long_functions(self):
        """Check for overly long functions or methods."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                # Try to get a reliable end line number
                end_line = getattr(node, 'end_lineno', None)
                if end_line is None and hasattr(node, 'body') and node.body:
                    # If end_lineno isn't directly available, get it from the last statement
                    last_stmt = node.body[-1]
                    end_line = getattr(last_stmt, 'end_lineno', start_line)
                
                # Fallback if end line still unknown (e.g., empty function)
                if end_line is None:
                     end_line = start_line 
                     
                line_count = end_line - start_line + 1
                
                if line_count > self.MAX_FUNC_LINES:
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=start_line,
                        end_line=end_line,
                        issue_type="Long Function/Method",
                        severity="low", # Assign severity (e.g., low)
                        message=f"Function/Method '{node.name}' is quite long ({line_count} lines > {self.MAX_FUNC_LINES}). Consider breaking it down.",
                        code_snippet=self.lines[start_line-1] if start_line-1 < len(self.lines) else None
                    ))
    
    def _check_long_parameter_lists(self):
        """Check for functions with too many parameters."""
        MAX_PARAMS = 5 # Define threshold
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                param_count = len(node.args.args)
                if param_count > MAX_PARAMS:
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=node.lineno,
                        issue_type="Long Parameter List",
                        severity="medium",
                        message=f"Function '{node.name}' has {param_count} parameters. Consider using a data class or dictionary.",
                        code_snippet=self.lines[node.lineno-1] if node.lineno-1 < len(self.lines) else None
                    ))
    
    def _check_magic_numbers(self):
        """Check for magic numbers."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Num) and not isinstance(node.parent, ast.Assign):
                # Skip common numbers like 0, 1, -1
                if node.n not in [0, 1, -1, 100, 10, 2, 24, 60, 3600]:
                    # Also skip if it's part of a named constant assignment
                    is_constant_def = False
                    if (isinstance(node.parent, ast.Assign) and 
                        len(node.parent.targets) == 1 and 
                        isinstance(node.parent.targets[0], ast.Name) and 
                        node.parent.targets[0].id.isupper()):
                        is_constant_def = True
                    
                    if not is_constant_def:
                        self.issues.append(SemanticIssue(
                            file_path=self.file_path,
                            line=node.lineno,
                            issue_type="Magic Number",
                            severity="info",
                            message=f"Found magic number {node.n}. Consider using a named constant.",
                            code_snippet=self.lines[node.lineno-1] if node.lineno-1 < len(self.lines) else None
                        ))
    
    def _check_error_handling(self):
        """Check for bare except clauses and other error handling issues."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None:  # Bare except
                        self.issues.append(SemanticIssue(
                            file_path=self.file_path,
                            line=handler.lineno,
                            issue_type="Bare Except",
                            severity="medium",
                            message="Using bare 'except:' without specifying exceptions. This can hide bugs and make debugging difficult.",
                            code_snippet=self.lines[handler.lineno-1] if handler.lineno-1 < len(self.lines) else None
                        ))
                    elif isinstance(handler.type, ast.Name) and handler.type.id == "Exception":
                        self.issues.append(SemanticIssue(
                            file_path=self.file_path,
                            line=handler.lineno,
                            issue_type="Broad Exception Handling",
                            severity="low",
                            message="Catching broad Exception class. Consider catching more specific exceptions.",
                            code_snippet=self.lines[handler.lineno-1] if handler.lineno-1 < len(self.lines) else None
                        ))


class JavaScriptSemanticAnalyzer:
    """Helper class for analyzing JavaScript code semantics.
    
    This is a simplified implementation as full JS parsing would require
    external libraries like Esprima or Babel.
    """
    
    def __init__(self, file_path: str, code_content: str):
        self.file_path = file_path
        self.code_content = code_content
        self.lines = code_content.splitlines()
        self.issues = []
    
    def analyze(self) -> List[SemanticIssue]:
        """Perform semantic analysis on the code."""
        # This would ideally use a proper JS parser
        # For demo purposes, we'll use simple regex checks
        
        self._check_console_logs()
        self._check_eval_usage()
        self._check_null_comparisons()
        self._check_jquery_usage()
        
        return self.issues
    
    def _check_console_logs(self):
        """Check for console.log statements that might be left in production code."""
        import re
        pattern = r'console\.(log|debug|info|warn|error)\('
        
        for i, line in enumerate(self.lines):
            matches = re.search(pattern, line)
            if matches:
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i+1,
                    issue_type="Console Statement",
                    severity="info",
                    message=f"Found console.{matches.group(1)} statement that may be left in production code.",
                    code_snippet=line.strip()
                ))
    
    def _check_eval_usage(self):
        """Check for eval() usage which is generally discouraged."""
        import re
        pattern = r'\beval\s*\('
        
        for i, line in enumerate(self.lines):
            if re.search(pattern, line):
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i+1,
                    issue_type="Eval Usage",
                    severity="high",
                    message="Found eval() usage which can lead to security vulnerabilities if not used carefully.",
                    code_snippet=line.strip()
                ))
    
    def _check_null_comparisons(self):
        """Check for loose null comparisons (== null) instead of strict (=== null)."""
        import re
        pattern = r'([^=!])={2}(\s*)(null|undefined)'
        
        for i, line in enumerate(self.lines):
            if re.search(pattern, line):
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i+1,
                    issue_type="Loose Comparison",
                    severity="medium",
                    message="Using loose equality (==) with null/undefined. Consider using strict equality (===).",
                    code_snippet=line.strip()
                ))
    
    def _check_jquery_usage(self):
        """Check for jQuery usage which might indicate legacy code."""
        import re
        pattern = r'(\$\(|\$\.|\bjQuery\b)'
        
        for i, line in enumerate(self.lines):
            if re.search(pattern, line):
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i+1,
                    issue_type="jQuery Usage",
                    severity="info",
                    message="Using jQuery. Consider using modern DOM APIs for new code.",
                    code_snippet=line.strip()
                ))


class TypeScriptSemanticAnalyzer(JavaScriptSemanticAnalyzer):
    """Helper class for analyzing TypeScript code semantics.
    
    Extends JavaScript analyzer with TypeScript-specific checks.
    This implementation uses regex for simplicity, but a proper TypeScript
    parser like typescript-estree would be better for production use.
    """
    
    def analyze(self) -> List[SemanticIssue]:
        """Perform semantic analysis on the TypeScript code."""
        # First run the JavaScript checks
        super().analyze()
        
        # Add TypeScript-specific checks
        self._check_any_type()
        self._check_non_null_assertion()
        self._check_type_assertions()
        self._check_unused_interfaces()
        self._check_enum_naming()
        
        return self.issues
    
    def _check_any_type(self):
        """Check for excessive use of 'any' type which defeats TypeScript's type checking."""
        import re
        pattern = r':\s*any\b'
        
        # Count occurrences of 'any' type
        any_count = 0
        any_lines = []
        
        for i, line in enumerate(self.lines):
            matches = re.findall(pattern, line)
            if matches:
                any_count += len(matches)
                any_lines.append(i+1)
        
        # Report if there are multiple uses of 'any'
        if any_count > 3:
            self.issues.append(SemanticIssue(
                file_path=self.file_path,
                line=any_lines[0] if any_lines else 1,
                issue_type="Excessive Any Type",
                severity="medium",
                message=f"Found {any_count} uses of 'any' type. This defeats TypeScript's type checking benefits.",
                code_snippet=None
            ))
        elif any_count > 0:
            for i in any_lines:
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i,
                    issue_type="Any Type",
                    severity="info",
                    message="Using 'any' type. Consider using a more specific type or 'unknown' if type is truly unknown.",
                    code_snippet=self.lines[i-1].strip()
                ))
    
    def _check_non_null_assertion(self):
        """Check for non-null assertion operator (!) which can lead to runtime errors."""
        import re
        pattern = r'\w+!\.|\w+!\['
        
        for i, line in enumerate(self.lines):
            matches = re.findall(pattern, line)
            if matches:
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i+1,
                    issue_type="Non-null Assertion",
                    severity="medium",
                    message="Using non-null assertion operator (!). Consider proper null checking instead.",
                    code_snippet=line.strip()
                ))
    
    def _check_type_assertions(self):
        """Check for type assertions (as Type) which can bypass type checking."""
        import re
        # Match both forms of type assertion: "as Type" and "<Type>"
        as_pattern = r'\bas\s+[A-Z]\w+'
        angle_pattern = r'<[A-Z]\w+>\s*\w+'
        
        for i, line in enumerate(self.lines):
            if re.search(as_pattern, line) or re.search(angle_pattern, line):
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i+1,
                    issue_type="Type Assertion",
                    severity="info",
                    message="Using type assertion. Consider redesigning to avoid the need for assertions.",
                    code_snippet=line.strip()
                ))
    
    def _check_unused_interfaces(self):
        """Check for potentially unused interfaces (very basic implementation)."""
        import re
        # Extract interface names
        interface_pattern = r'interface\s+(\w+)'
        interfaces = {}
        
        for i, line in enumerate(self.lines):
            matches = re.search(interface_pattern, line)
            if matches:
                interface_name = matches.group(1)
                interfaces[interface_name] = i+1
        
        # Check for usage of each interface
        for interface_name, line_num in interfaces.items():
            # Simple check: look for the interface name elsewhere in the code
            usage_count = 0
            for line in self.lines:
                if interface_name in line and not line.strip().startswith('interface'):
                    usage_count += 1
            
            if usage_count == 0:
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=line_num,
                    issue_type="Unused Interface",
                    severity="info",
                    message=f"Interface '{interface_name}' appears to be unused.",
                    code_snippet=self.lines[line_num-1].strip()
                ))
    
    def _check_enum_naming(self):
        """Check for enum naming conventions (should be PascalCase)."""
        import re
        enum_pattern = r'enum\s+(\w+)'
        
        for i, line in enumerate(self.lines):
            matches = re.search(enum_pattern, line)
            if matches:
                enum_name = matches.group(1)
                if not enum_name[0].isupper() or '_' in enum_name:
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=i+1,
                        issue_type="Enum Naming",
                        severity="info",
                        message=f"Enum '{enum_name}' does not follow PascalCase naming convention.",
                        code_snippet=line.strip()
                    ))


class GoSemanticAnalyzer:
    """Helper class for analyzing Go code semantics using regex-based pattern matching."""
    
    def __init__(self, file_path: str, code_content: str):
        self.file_path = file_path
        self.code_content = code_content
        self.lines = code_content.splitlines()
        self.issues = []
    
    def analyze(self) -> List[SemanticIssue]:
        """Perform semantic analysis on the Go code."""
        self._check_error_handling()
        self._check_naked_returns()
        self._check_unused_imports()
        self._check_empty_interfaces()
        self._check_large_structs()
        self._check_defer_in_loops()
        self._check_channel_usage()
        
        return self.issues
    
    def _check_error_handling(self):
        """Check for improper error handling patterns in Go code."""
        import re
        
        # Check for ignored errors (using _ for error variable)
        ignored_error_pattern = r'(?:[\w\[\]]+(?:\s*,\s*)+)?_\s*:=\s*[\w\d\.\(\)\[\]]+\.[\w\d]+\('
        
        for i, line in enumerate(self.lines):
            # Ignore comments
            if line.strip().startswith('//'):
                continue
                
            # Check for ignored errors
            if re.search(ignored_error_pattern, line):
                if not ('test' in self.file_path.lower() or 'mock' in self.file_path.lower()):
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=i+1,
                        issue_type="Ignored Error",
                        severity="medium",
                        message="Error value is being ignored. Consider handling the error properly.",
                        code_snippet=line.strip()
                    ))
    
    def _check_naked_returns(self):
        """Check for naked returns which can reduce code readability."""
        import re
        
        # Match function declarations with naked returns
        func_pattern = r'\bfunc\s+\w+[^{]*\)\s*\([^{]*\)\s*{'
        
        in_func = False
        naked_return_lines = []
        
        for i, line in enumerate(self.lines):
            line_stripped = line.strip()
            
            # Start of function with named return values
            if re.search(func_pattern, line) and ')' in line and '(' in line:
                # Check if it has named return values
                return_part = line.split(')')[-2]
                if re.search(r'\([\w\s,]+\s+[\w\[\]\*\.]+', return_part):
                    in_func = True
                    func_start_line = i+1
            
            # End of function
            if in_func and line_stripped == '}':
                in_func = False
                
                # Report naked returns if found
                if naked_return_lines:
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=func_start_line,
                        issue_type="Naked Return",
                        severity="info",
                        message="Function uses naked returns which can reduce code readability.",
                        code_snippet=f"Function at line {func_start_line} has naked returns at lines {', '.join(map(str, naked_return_lines))}"
                    ))
                    naked_return_lines = []
            
            # Check for naked returns inside function
            if in_func and line_stripped == 'return' and not line_stripped.startswith('//'):
                naked_return_lines.append(i+1)
    
    def _check_unused_imports(self):
        """Check for potentially unused imports."""
        import re
        
        # Extract all imports
        import_blocks = []
        in_import_block = False
        start_line = -1
        
        for i, line in enumerate(self.lines):
            line_stripped = line.strip()
            
            # Start of import block
            if line_stripped == 'import (' and not in_import_block:
                in_import_block = True
                start_line = i+1
                continue
            
            # End of import block
            if line_stripped == ')' and in_import_block:
                in_import_block = False
                import_blocks.append((start_line, i))
                continue
            
            # Single line import
            if re.match(r'import\s+"[\w\.\/]+"', line_stripped) and not in_import_block:
                # Extract the package name
                match = re.search(r'import\s+"([\w\.\/]+)"', line_stripped)
                if match:
                    package = match.group(1).split('/')[-1]
                    if not self._is_package_used(package):
                        self.issues.append(SemanticIssue(
                            file_path=self.file_path,
                            line=i+1,
                            issue_type="Unused Import",
                            severity="info",
                            message=f"Package '{package}' seems to be imported but not used.",
                            code_snippet=line_stripped
                        ))
        
        # Process multi-line import blocks
        for start, end in import_blocks:
            for i in range(start, end):
                line = self.lines[i].strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('//'):
                    continue
                
                # Extract package name
                if '"' in line:
                    package_path = re.search(r'"([^"]+)"', line)
                    if package_path:
                        full_path = package_path.group(1)
                        package = full_path.split('/')[-1]
                        
                        # Check if the package is aliased
                        alias = None
                        if re.match(r'[\w\d_]+\s+"', line):
                            alias = line.split('"')[0].strip()
                        
                        # Check if used
                        if not self._is_package_used(package) and not (alias and self._is_package_used(alias)):
                            self.issues.append(SemanticIssue(
                                file_path=self.file_path,
                                line=i+1,
                                issue_type="Unused Import",
                                severity="info",
                                message=f"Package '{package}' seems to be imported but not used.",
                                code_snippet=line
                            ))
    
    def _is_package_used(self, package: str) -> bool:
        """Check if a package is used in the code."""
        # This is a simple check - in a real implementation, this would be more sophisticated
        code_without_imports = '\n'.join(self.lines)
        return f"{package}." in code_without_imports or f"{package}.New" in code_without_imports
    
    def _check_empty_interfaces(self):
        """Check for empty interfaces which might indicate poor design."""
        import re
        
        empty_interface_pattern = r'type\s+\w+\s+interface\s*{\s*}'
        
        for i, line in enumerate(self.lines):
            if re.search(empty_interface_pattern, line):
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i+1,
                    issue_type="Empty Interface",
                    severity="low",
                    message="Empty interfaces should be avoided when possible. Consider defining more specific interfaces.",
                    code_snippet=line.strip()
                ))
    
    def _check_large_structs(self):
        """Check for large structs which might benefit from refactoring."""
        import re
        
        in_struct = False
        struct_start = -1
        field_count = 0
        
        for i, line in enumerate(self.lines):
            line_stripped = line.strip()
            
            # Start of struct
            if re.search(r'struct\s+\w+\s*{', line_stripped):
                in_struct = True
                struct_start = i+1
                field_count = 0
                continue
            
            # End of struct
            if in_struct and line_stripped == '}':
                in_struct = False
                if field_count > 15:  # Arbitrary threshold
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=struct_start,
                        issue_type="Large Struct",
                        severity="medium",
                        message=f"Struct has {field_count} fields. Consider breaking it down into smaller components.",
                        code_snippet=f"Struct at line {struct_start} has {field_count} fields"
                    ))
                continue
            
            # Count fields
            if in_struct and line_stripped and not line_stripped.startswith('//'):
                field_count += 1
    
    def _check_defer_in_loops(self):
        """Check for defer statements inside loops which can lead to resource exhaustion."""
        import re
        
        in_loop = False
        loop_start = -1
        defer_lines = []
        
        for i, line in enumerate(self.lines):
            line_stripped = line.strip()
            
            # Start of loop
            if re.search(r'^\s*for\s+', line):
                in_loop = True
                loop_start = i+1
                continue
            
            # End of loop
            if in_loop and line_stripped == '}':
                in_loop = False
                
                # Report defers in loop if found
                if defer_lines:
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=loop_start,
                        issue_type="Defer in Loop",
                        severity="high",
                        message="Using defer inside a loop can lead to resource exhaustion as deferred calls are only executed when the function returns.",
                        code_snippet=f"Loop at line {loop_start} has defer statements at lines {', '.join(map(str, defer_lines))}"
                    ))
                    defer_lines = []
                continue
            
            # Check for defer statements in loop
            if in_loop and re.search(r'^\s*defer\s+', line) and not line_stripped.startswith('//'):
                defer_lines.append(i+1)
    
    def _check_channel_usage(self):
        """Check for common channel misuse patterns."""
        import re
        
        for i, line in enumerate(self.lines):
            # Check for select with default but no other cases
            if 'select {' in line:
                # Find the end of the select block
                j = i + 1
                only_default = True
                has_default = False
                
                while j < len(self.lines) and '}' not in self.lines[j]:
                    case_line = self.lines[j].strip()
                    if case_line.startswith('case ') and not case_line.startswith('case <-'):
                        only_default = False
                    elif case_line.startswith('default:'):
                        has_default = True
                    j += 1
                
                if has_default and only_default:
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=i+1,
                        issue_type="Select With Only Default",
                        severity="medium",
                        message="Select statement has a default case but no channel operations. This might indicate a logical error.",
                        code_snippet=self.lines[i].strip()
                    ))


class RustSemanticAnalyzer:
    """Helper class for analyzing Rust code semantics using regex-based pattern matching."""
    
    def __init__(self, file_path: str, code_content: str):
        self.file_path = file_path
        self.code_content = code_content
        self.lines = code_content.splitlines()
        self.issues = []
    
    def analyze(self) -> List[SemanticIssue]:
        """Perform semantic analysis on the Rust code."""
        self._check_unwrap_usage()
        self._check_expect_usage()
        self._check_clone_usage()
        self._check_mutex_in_struct()
        self._check_large_enums()
        self._check_unsafe_blocks()
        self._check_mutable_borrowing()
        
        return self.issues
    
    def _check_unwrap_usage(self):
        """Check for unwrap() calls which can cause panics in production code."""
        import re
        
        unwrap_pattern = r'\.\s*unwrap\s*\(\)'
        
        for i, line in enumerate(self.lines):
            # Skip test modules and files
            if '#[test]' in line or 'mod test' in line or 'tests' in self.file_path.lower():
                continue
            
            # Skip comments
            if line.strip().startswith('//'):
                continue
            
            if re.search(unwrap_pattern, line):
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i+1,
                    issue_type="Unwrap Usage",
                    severity="medium",
                    message="Using unwrap() can cause panics. Consider proper error handling with match, if let, or ?.",
                    code_snippet=line.strip()
                ))
    
    def _check_expect_usage(self):
        """Check for expect() calls which can cause panics with custom messages."""
        import re
        
        expect_pattern = r'\.\s*expect\s*\(\s*["\'](.*?)["\']\s*\)'
        
        for i, line in enumerate(self.lines):
            # Skip test modules and files
            if '#[test]' in line or 'mod test' in line or 'tests' in self.file_path.lower():
                continue
            
            # Skip comments
            if line.strip().startswith('//'):
                continue
            
            if re.search(expect_pattern, line):
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i+1,
                    issue_type="Expect Usage",
                    severity="low",
                    message="Using expect() can cause panics. Consider proper error handling with match, if let, or ?.",
                    code_snippet=line.strip()
                ))
    
    def _check_clone_usage(self):
        """Check for potentially unnecessary clone() calls."""
        import re
        
        clone_pattern = r'\.\s*clone\s*\(\)'
        
        for i, line in enumerate(self.lines):
            # Skip comments
            if line.strip().startswith('//'):
                continue
            
            if re.search(clone_pattern, line):
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i+1,
                    issue_type="Clone Usage",
                    severity="info",
                    message="Potentially unnecessary clone(). Consider if a reference or borrowing would be more efficient.",
                    code_snippet=line.strip()
                ))
    
    def _check_mutex_in_struct(self):
        """Check for Mutex inside Struct fields that don't impl Copy."""
        import re
        
        in_struct = False
        struct_start_line = -1
        has_mutex = False
        mutex_line = -1
        struct_name = ""
        
        for i, line in enumerate(self.lines):
            line_stripped = line.strip()
            
            # Start of struct
            if re.search(r'struct\s+(\w+)\s*{', line_stripped):
                in_struct = True
                struct_start_line = i+1
                struct_name = re.search(r'struct\s+(\w+)', line_stripped).group(1)
                has_mutex = False
                continue
            
            # End of struct
            if in_struct and line_stripped == '}':
                in_struct = False
                if has_mutex:
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=mutex_line,
                        issue_type="Mutex in Struct",
                        severity="medium",
                        message=f"Struct '{struct_name}' contains a Mutex. Ensure that Clone/Copy traits are not derived for this struct to avoid deadlocks.",
                        code_snippet=f"Mutex field in struct at line {mutex_line}"
                    ))
                continue
            
            # Check for Mutex fields
            if in_struct and ('Mutex<' in line_stripped or 'std::sync::Mutex' in line_stripped):
                has_mutex = True
                mutex_line = i+1
    
    def _check_large_enums(self):
        """Check for large enums which might be better as structs with variants."""
        import re
        
        in_enum = False
        enum_start_line = -1
        variant_count = 0
        enum_name = ""
        
        for i, line in enumerate(self.lines):
            line_stripped = line.strip()
            
            # Start of enum
            if re.search(r'enum\s+(\w+)\s*{', line_stripped):
                in_enum = True
                enum_start_line = i+1
                enum_name = re.search(r'enum\s+(\w+)', line_stripped).group(1)
                variant_count = 0
                continue
            
            # End of enum
            if in_enum and line_stripped == '}':
                in_enum = False
                if variant_count > 10:  # Arbitrary threshold
                    self.issues.append(SemanticIssue(
                        file_path=self.file_path,
                        line=enum_start_line,
                        issue_type="Large Enum",
                        severity="medium",
                        message=f"Enum '{enum_name}' has {variant_count} variants. Consider using a struct with an enum field for variant type.",
                        code_snippet=f"Enum at line {enum_start_line} has {variant_count} variants"
                    ))
                continue
            
            # Count variants
            if in_enum and line_stripped and not line_stripped.startswith('//') and ',' in line_stripped:
                variant_count += 1
    
    def _check_unsafe_blocks(self):
        """Check for unsafe blocks which need careful review."""
        import re
        
        for i, line in enumerate(self.lines):
            if 'unsafe {' in line or 'unsafe fn' in line:
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=i+1,
                    issue_type="Unsafe Code",
                    severity="high",
                    message="Unsafe code detected. Ensure all invariants are properly maintained and the unsafe block is as small as possible.",
                    code_snippet=line.strip()
                ))
    
    def _check_mutable_borrowing(self):
        """Check for potential excessive mutable borrowing."""
        import re
        
        mutable_borrow_count = {}
        
        for i, line in enumerate(self.lines):
            # Skip comments
            if line.strip().startswith('//'):
                continue
            
            # Count mutable borrows
            mut_borrows = re.findall(r'&mut\s+(\w+)', line)
            for var in mut_borrows:
                if var not in mutable_borrow_count:
                    mutable_borrow_count[var] = []
                mutable_borrow_count[var].append(i+1)
        
        # Report excessive mutable borrowing
        for var, lines in mutable_borrow_count.items():
            if len(lines) > 5:  # Arbitrary threshold
                self.issues.append(SemanticIssue(
                    file_path=self.file_path,
                    line=lines[0],
                    issue_type="Excessive Mutable Borrowing",
                    severity="low",
                    message=f"Variable '{var}' is mutably borrowed {len(lines)} times. Consider refactoring to reduce mutable state.",
                    code_snippet=f"Mutable borrows at lines {', '.join(map(str, lines[:3]))}, ..."
                ))


class SemanticAnalyzerPlugin(AnalyzerPlugin):
    """
    Plugin for performing semantic code analysis.
    
    Analyzes code structure and patterns to identify potential issues
    that may not be caught by traditional linters.
    """
    
    @property
    def name(self) -> str:
        return "semantic_analyzer"
    
    @property
    def description(self) -> str:
        return "Analyzes code semantics to identify patterns and anti-patterns"
    
    @property
    def supported_languages(self) -> List[str]:
        return ["py", "js", "ts", "jsx", "tsx", "go", "rs"]
    
    def _analyze_file_impl(self, file_path: str, file_content: str = None) -> dict:
        """
        Dummy implementation to satisfy abstract base class requirements.
        Returns an empty result.
        """
        return {"issues": [], "metrics": {}, "file_path": file_path}
    
    def analyze_file(self, file_path: str, file_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a file for semantic issues.
        
        Args:
            file_path (str): Path to the file
            file_content (Optional[str]): File content if already loaded
            
        Returns:
            Dict[str, Any]: Analysis results with semantic issues found
        """
        results = {
            "issues": [],
            "metrics": {},
            "file_path": file_path
        }
        
        try:
            # Load file content if not provided
            if file_content is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            ext = file_path.split('.')[-1].lower()
            issues = []
            if ext == "py":
                analyzer = PythonSemanticAnalyzer(file_path, file_content)
                issues = analyzer.analyze()
            elif ext == "js":
                analyzer = JavaScriptSemanticAnalyzer(file_path, file_content)
                issues = analyzer.analyze()
            elif ext == "go":
                analyzer = GoSemanticAnalyzer(file_path, file_content)
                issues = analyzer.analyze()
            elif ext == "rs":
                analyzer = RustSemanticAnalyzer(file_path, file_content)
                issues = analyzer.analyze()
            # Store results
            results["issues"] = [issue.to_dict() for issue in issues]
            # Add metrics
            results["metrics"] = {
                "issue_count": len(issues),
                "issue_types": self._summarize_issue_types(issues),
                "severity_counts": self._summarize_severities(issues)
            }
            return results
        except Exception as e:
            logging.getLogger("CodeReviewTool.SemanticAnalyzer").error(f"Error analyzing file {file_path}: {e}")
            results["error"] = str(e)
            return results
    
    def analyze_files_parallel(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze a batch of files for semantic issues in parallel.
        Args:
            file_paths: List of file paths to analyze
        Returns:
            Dict mapping file paths to semantic analysis results
        """
        results = {}
        def analyze_one(fp):
            return (fp, self.analyze_file(fp))
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(analyze_one, fp): fp for fp in file_paths}
            for future in as_completed(futures):
                fp, res = future.result()
                results[fp] = res
        return results
    
    def get_prompt_data(self, analysis_results: Dict[str, Any], sensitivity: str = "medium") -> Dict[str, Any]:
        """
        Format semantic analysis results for inclusion in LLM prompts.
        
        Args:
            analysis_results (Dict[str, Any]): Analysis results from analyze_file
            sensitivity (str): Sensitivity level ('low', 'medium', 'high')
            
        Returns:
            Dict[str, Any]: Data to be included in LLM prompts
        """
        if "error" in analysis_results:
            return {"semantic_issues": f"Error during semantic analysis: {analysis_results['error']}"}
        
        issues = analysis_results.get("issues", [])
        if not issues:
            return {"semantic_issues": "No semantic issues detected."}
        
        # Filter issues based on sensitivity level
        filtered_issues = self._filter_by_sensitivity(issues, sensitivity)
        
        if not filtered_issues:
            return {"semantic_issues": f"No semantic issues detected at '{sensitivity}' sensitivity level."}
        
        # Format issues for prompt
        prompt_text = f"Semantic Analysis Results (sensitivity: {sensitivity}):\n"
        for issue in filtered_issues:
            prompt_text += f"- Line {issue['line']}: {issue['severity'].upper()} - {issue['issue_type']}: {issue['message']}\n"
        
        metrics = analysis_results.get("metrics", {})
        if metrics:
            prompt_text += f"\nSummary: {len(filtered_issues)} semantic issues detected.\n"
        
        return {"semantic_issues": prompt_text}
    
    def _filter_by_sensitivity(self, issues: List[Dict[str, Any]], sensitivity: str) -> List[Dict[str, Any]]:
        """
        Filter issues based on sensitivity level.
        
        Args:
            issues (List[Dict[str, Any]]): List of issue dictionaries
            sensitivity (str): Sensitivity level ('low', 'medium', 'high')
            
        Returns:
            List[Dict[str, Any]]: Filtered issues
        """
        # Define severity levels to include based on sensitivity
        if sensitivity == "low":
            # Only include high severity issues
            include_severities = ["high"]
        elif sensitivity == "medium":
            # Include medium and high severity issues
            include_severities = ["medium", "high"]
        else:  # high sensitivity
            # Include all issues
            include_severities = ["info", "low", "medium", "high"]
        
        return [issue for issue in issues if issue["severity"] in include_severities]
    
    def _summarize_issue_types(self, issues: List[SemanticIssue]) -> Dict[str, int]:
        """Count issues by type."""
        counts = {}
        for issue in issues:
            counts[issue.issue_type] = counts.get(issue.issue_type, 0) + 1
        return counts
    
    def _summarize_severities(self, issues: List[SemanticIssue]) -> Dict[str, int]:
        """Count issues by severity."""
        counts = {}
        for issue in issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts 