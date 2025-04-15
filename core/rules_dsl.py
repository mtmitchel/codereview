"""
Domain-Specific Language (DSL) for code review rules.

This module provides a simple DSL for defining code review rules
that can be used to customize the code review process.
"""

import re
import ast
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, Pattern, Set, Tuple
from pathlib import Path

logger = logging.getLogger("CodeReviewTool.RulesDSL")

class RuleContext:
    """Context object passed to rules when they are evaluated."""
    
    def __init__(self, 
                file_path: str, 
                content: str, 
                language: str = None,
                static_analysis: Dict[str, Any] = None,
                config: Dict[str, Any] = None):
        """
        Initialize the rule context.
        
        Args:
            file_path: Path to the file being analyzed
            content: Content of the file
            language: Language of the file (auto-detected if None)
            static_analysis: Static analysis results
            config: Configuration dictionary
        """
        self.file_path = file_path
        self.content = content
        self.lines = content.splitlines()
        self.line_count = len(self.lines)
        self._language = language
        self.static_analysis = static_analysis or {}
        self.config = config or {}
        self.ast = None
        
    @property
    def language(self) -> str:
        """Get the language of the file."""
        if self._language:
            return self._language
        
        # Auto-detect language based on file extension
        ext = Path(self.file_path).suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.rs': 'rust',
            '.md': 'markdown',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
        }
        
        self._language = language_map.get(ext, 'unknown')
        return self._language
    
    def get_line(self, line_number: int) -> str:
        """
        Get a specific line from the file.
        
        Args:
            line_number: Line number (1-based)
            
        Returns:
            The line content or empty string if out of bounds
        """
        if line_number < 1 or line_number > self.line_count:
            return ""
        
        return self.lines[line_number - 1]
    
    def get_ast(self) -> Optional[ast.AST]:
        """
        Get the AST for Python files.
        
        Returns:
            AST or None if not a Python file or parsing failed
        """
        if self.language != 'python' or self.ast is not None:
            return self.ast
        
        try:
            self.ast = ast.parse(self.content)
            return self.ast
        except Exception as e:
            logger.warning(f"Failed to parse Python AST for {self.file_path}: {e}")
            return None


class Rule(ABC):
    """Base class for code review rules."""
    
    def __init__(self, name: str, description: str, severity: str = "Medium"):
        """
        Initialize the rule.
        
        Args:
            name: Name of the rule
            description: Description of the rule
            severity: Severity level ("Critical", "High", "Medium", "Low", "Info")
        """
        self.name = name
        self.description = description
        self.severity = severity
        logger.debug(f"Initialized rule: {name}")
    
    @abstractmethod
    def evaluate(self, context: RuleContext) -> List[Dict[str, Any]]:
        """
        Evaluate the rule on a file.
        
        Args:
            context: Rule context
            
        Returns:
            List of findings, each with:
                - line_start: Starting line number
                - line_end: Ending line number
                - description: Description of the finding
                - severity: Severity level
                - recommendation: Recommendation for fixing the issue
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} ({self.severity})"


class PatternRule(Rule):
    """Rule that looks for regex patterns in code."""
    
    def __init__(self, 
                name: str, 
                description: str,
                pattern: str,
                language: Optional[str] = None,
                severity: str = "Medium",
                recommendation: str = "",
                ignore_case: bool = False,
                include_comments: bool = True):
        """
        Initialize the pattern rule.
        
        Args:
            name: Name of the rule
            description: Description of the rule
            pattern: Regex pattern to look for
            language: Language the rule applies to (None for all)
            severity: Severity level
            recommendation: Recommendation for fixing the issue
            ignore_case: Whether to ignore case in pattern matching
            include_comments: Whether to include comments in pattern matching
        """
        super().__init__(name, description, severity)
        self.pattern_str = pattern
        self.language = language
        self.recommendation = recommendation
        self.ignore_case = ignore_case
        self.include_comments = include_comments
        
        # Compile the pattern
        flags = re.IGNORECASE if ignore_case else 0
        self.pattern = re.compile(pattern, flags)
    
    def evaluate(self, context: RuleContext) -> List[Dict[str, Any]]:
        """
        Evaluate the rule on a file.
        
        Args:
            context: Rule context
            
        Returns:
            List of findings
        """
        findings = []
        
        # Check if rule applies to this language
        if self.language and context.language != self.language:
            return []
        
        # TODO: Add option to exclude comments based on language
        
        # Look for pattern in each line
        for i, line in enumerate(context.lines):
            if self.pattern.search(line):
                findings.append({
                    'line_start': i + 1,
                    'line_end': i + 1,
                    'description': self.description,
                    'severity': self.severity,
                    'recommendation': self.recommendation or f"Review potential issue with '{self.pattern_str}'"
                })
        
        return findings


class ASTRule(Rule):
    """Rule that analyzes Python AST."""
    
    def __init__(self, 
                name: str, 
                description: str,
                severity: str = "Medium",
                recommendation: str = ""):
        """
        Initialize the AST rule.
        
        Args:
            name: Name of the rule
            description: Description of the rule
            severity: Severity level
            recommendation: Recommendation for fixing the issue
        """
        super().__init__(name, description, severity)
        self.recommendation = recommendation
    
    def evaluate(self, context: RuleContext) -> List[Dict[str, Any]]:
        """
        Evaluate the rule on a file.
        
        Args:
            context: Rule context
            
        Returns:
            List of findings
        """
        findings = []
        
        # Only apply to Python files
        if context.language != 'python':
            return []
        
        # Get AST
        tree = context.get_ast()
        if not tree:
            return []
        
        # Evaluate AST
        self.visit_ast(tree, context, findings)
        
        return findings
    
    def visit_ast(self, node: ast.AST, context: RuleContext, findings: List[Dict[str, Any]]) -> None:
        """
        Visit AST nodes recursively.
        
        Args:
            node: AST node
            context: Rule context
            findings: List to add findings to
        """
        # Override this method in subclasses
        pass


class CodeComplexityRule(ASTRule):
    """Rule that checks for code complexity."""
    
    def __init__(self, 
                name: str = "Function Complexity",
                description: str = "Function is too complex (high cyclomatic complexity)",
                max_complexity: int = 10,
                severity: str = "Medium",
                recommendation: str = "Consider breaking this function into smaller, more focused functions."):
        """
        Initialize the code complexity rule.
        
        Args:
            name: Name of the rule
            description: Description of the rule
            max_complexity: Maximum allowed complexity
            severity: Severity level
            recommendation: Recommendation for fixing the issue
        """
        super().__init__(name, description, severity, recommendation)
        self.max_complexity = max_complexity
    
    def visit_ast(self, node: ast.AST, context: RuleContext, findings: List[Dict[str, Any]]) -> None:
        """
        Visit AST nodes to find complex functions.
        
        Args:
            node: AST node
            context: Rule context
            findings: List to add findings to
        """
        for node in ast.walk(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)
                
                if complexity > self.max_complexity:
                    findings.append({
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno),
                        'description': f"{self.description} (Complexity: {complexity}, Max: {self.max_complexity})",
                        'severity': self.severity,
                        'recommendation': self.recommendation
                    })
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """
        Calculate cyclomatic complexity of a function.
        
        Args:
            node: Function AST node
            
        Returns:
            Cyclomatic complexity
        """
        complexity = 1  # Base complexity
        
        # Count control flow statements
        for n in ast.walk(node):
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(n, ast.BoolOp) and isinstance(n.op, ast.And):
                complexity += len(n.values) - 1
            elif isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or):
                complexity += len(n.values) - 1
            elif isinstance(n, (ast.Break, ast.Continue)):
                complexity += 1
            elif isinstance(n, ast.Try):
                complexity += len(n.handlers)
        
        return complexity


class CompositeRule(Rule):
    """Rule that combines multiple rules."""
    
    def __init__(self, 
                name: str, 
                description: str,
                rules: List[Rule],
                severity: str = "Medium"):
        """
        Initialize the composite rule.
        
        Args:
            name: Name of the rule
            description: Description of the rule
            rules: List of rules to combine
            severity: Severity level
        """
        super().__init__(name, description, severity)
        self.rules = rules
    
    def evaluate(self, context: RuleContext) -> List[Dict[str, Any]]:
        """
        Evaluate all contained rules on a file.
        
        Args:
            context: Rule context
            
        Returns:
            Combined list of findings
        """
        findings = []
        
        for rule in self.rules:
            findings.extend(rule.evaluate(context))
        
        return findings


class RuleSet:
    """A set of code review rules."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the rule set.
        
        Args:
            name: Name of the rule set
            description: Description of the rule set
        """
        self.name = name
        self.description = description
        self.rules: List[Rule] = []
        logger.debug(f"Initialized rule set: {name}")
    
    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule to the set.
        
        Args:
            rule: Rule to add
        """
        self.rules.append(rule)
        logger.debug(f"Added rule {rule.name} to set {self.name}")
    
    def add_rules(self, rules: List[Rule]) -> None:
        """
        Add multiple rules to the set.
        
        Args:
            rules: Rules to add
        """
        for rule in rules:
            self.add_rule(rule)
    
    def evaluate(self, context: RuleContext) -> Dict[str, List[Dict[str, Any]]]:
        """
        Evaluate all rules on a file.
        
        Args:
            context: Rule context
            
        Returns:
            Dictionary mapping rule names to findings
        """
        results = {}
        
        for rule in self.rules:
            findings = rule.evaluate(context)
            if findings:
                results[rule.name] = findings
        
        return results
    
    def __len__(self) -> int:
        return len(self.rules)


class RuleRegistry:
    """Registry for code review rules and rule sets."""
    
    def __init__(self):
        """Initialize the rule registry."""
        self.rules: Dict[str, Rule] = {}
        self.rule_sets: Dict[str, RuleSet] = {}
        logger.debug("Initialized rule registry")
    
    def register_rule(self, rule: Rule) -> None:
        """
        Register a rule.
        
        Args:
            rule: Rule to register
        """
        self.rules[rule.name] = rule
        logger.debug(f"Registered rule: {rule.name}")
    
    def register_rule_set(self, rule_set: RuleSet) -> None:
        """
        Register a rule set.
        
        Args:
            rule_set: Rule set to register
        """
        self.rule_sets[rule_set.name] = rule_set
        logger.debug(f"Registered rule set: {rule_set.name}")
    
    def get_rule(self, name: str) -> Optional[Rule]:
        """
        Get a rule by name.
        
        Args:
            name: Name of the rule
            
        Returns:
            Rule or None if not found
        """
        return self.rules.get(name)
    
    def get_rule_set(self, name: str) -> Optional[RuleSet]:
        """
        Get a rule set by name.
        
        Args:
            name: Name of the rule set
            
        Returns:
            Rule set or None if not found
        """
        return self.rule_sets.get(name)
    
    def create_rule_from_dict(self, rule_dict: Dict[str, Any]) -> Optional[Rule]:
        """
        Create a rule from a dictionary.
        
        Args:
            rule_dict: Dictionary with rule configuration
            
        Returns:
            Rule or None if creation failed
            
        Rule dictionary format:
        {
            "type": "pattern|ast|complexity|composite",
            "name": "Rule Name",
            "description": "Rule Description",
            "severity": "Critical|High|Medium|Low|Info",
            // For pattern rules
            "pattern": "regex pattern",
            "language": "python|javascript|...",
            "ignore_case": true|false,
            "recommendation": "Fix recommendation",
            // For complexity rules
            "max_complexity": 10,
            // For composite rules
            "rules": ["Rule1", "Rule2", ...]
        }
        """
        try:
            rule_type = rule_dict.get("type", "")
            name = rule_dict.get("name", "")
            description = rule_dict.get("description", "")
            severity = rule_dict.get("severity", "Medium")
            
            if not name or not description:
                logger.warning("Rule must have a name and description")
                return None
            
            if rule_type == "pattern":
                pattern = rule_dict.get("pattern", "")
                if not pattern:
                    logger.warning("Pattern rule must have a pattern")
                    return None
                
                return PatternRule(
                    name=name,
                    description=description,
                    pattern=pattern,
                    language=rule_dict.get("language"),
                    severity=severity,
                    recommendation=rule_dict.get("recommendation", ""),
                    ignore_case=rule_dict.get("ignore_case", False),
                    include_comments=rule_dict.get("include_comments", True)
                )
                
            elif rule_type == "complexity":
                return CodeComplexityRule(
                    name=name,
                    description=description,
                    max_complexity=rule_dict.get("max_complexity", 10),
                    severity=severity,
                    recommendation=rule_dict.get("recommendation", "")
                )
                
            elif rule_type == "composite":
                rule_names = rule_dict.get("rules", [])
                rules = []
                
                for rule_name in rule_names:
                    rule = self.get_rule(rule_name)
                    if rule:
                        rules.append(rule)
                    else:
                        logger.warning(f"Rule not found: {rule_name}")
                
                if not rules:
                    logger.warning("Composite rule must have at least one valid rule")
                    return None
                
                return CompositeRule(
                    name=name,
                    description=description,
                    rules=rules,
                    severity=severity
                )
                
            else:
                logger.warning(f"Unknown rule type: {rule_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating rule from dict: {e}")
            return None
    
    def create_rule_set_from_dict(self, ruleset_dict: Dict[str, Any]) -> Optional[RuleSet]:
        """
        Create a rule set from a dictionary.
        
        Args:
            ruleset_dict: Dictionary with rule set configuration
            
        Returns:
            Rule set or None if creation failed
            
        Rule set dictionary format:
        {
            "name": "Rule Set Name",
            "description": "Rule Set Description",
            "rules": [
                // Rule dictionaries or rule names
                "ExistingRuleName",
                { rule dictionary },
                ...
            ]
        }
        """
        try:
            name = ruleset_dict.get("name", "")
            description = ruleset_dict.get("description", "")
            
            if not name:
                logger.warning("Rule set must have a name")
                return None
            
            rule_set = RuleSet(name, description)
            
            for rule_item in ruleset_dict.get("rules", []):
                if isinstance(rule_item, str):
                    # Rule name
                    rule = self.get_rule(rule_item)
                    if rule:
                        rule_set.add_rule(rule)
                    else:
                        logger.warning(f"Rule not found: {rule_item}")
                        
                elif isinstance(rule_item, dict):
                    # Rule dictionary
                    rule = self.create_rule_from_dict(rule_item)
                    if rule:
                        rule_set.add_rule(rule)
                        # Also register the rule
                        self.register_rule(rule)
                
            return rule_set
            
        except Exception as e:
            logger.error(f"Error creating rule set from dict: {e}")
            return None
    
    def load_builtin_rules(self) -> None:
        """Load built-in rules."""
        # Security rules
        self.register_rule(PatternRule(
            name="Hardcoded Secret",
            description="Potential hardcoded secret or API key",
            pattern=r"(password|secret|api.?key|token|credentials).*?['\"]([a-zA-Z0-9]{16,})['\"]",
            severity="Critical",
            recommendation="Never hardcode secrets. Use environment variables or a secure storage solution."
        ))
        
        self.register_rule(PatternRule(
            name="SQL Injection Risk",
            description="Potential SQL injection risk",
            pattern=r"(?:execute|query|select|insert|update|delete|drop).*?\+\s*.*?|.*?%s.*?\b(?:query|execute)",
            language="python",
            severity="Critical",
            recommendation="Use parameterized queries or an ORM instead of string concatenation."
        ))
        
        self.register_rule(PatternRule(
            name="Insecure Deserialization",
            description="Potential insecure deserialization",
            pattern=r"pickle\.loads|jsonpickle|yaml\.load\((?!.*Loader=yaml\.SafeLoader)",
            language="python",
            severity="High",
            recommendation="Use yaml.safe_load() instead of yaml.load() or specify the SafeLoader."
        ))
        
        # Code quality rules
        self.register_rule(PatternRule(
            name="TODO Comment",
            description="TODO comment found",
            pattern=r"#\s*TODO|\/\/\s*TODO",
            severity="Info",
            recommendation="Consider addressing TODO comments before code review or converting to tracked issues."
        ))
        
        self.register_rule(PatternRule(
            name="Print Statement",
            description="Print statement found in code",
            pattern=r"\bprint\s*\(",
            language="python",
            severity="Low",
            recommendation="Consider using a logger instead of print statements for better control in production."
        ))
        
        self.register_rule(CodeComplexityRule(
            name="Function Complexity",
            description="Function is too complex (high cyclomatic complexity)",
            max_complexity=10,
            severity="Medium",
            recommendation="Consider breaking this function into smaller, more focused functions."
        ))
        
        # Create a rule set
        security_ruleset = RuleSet(
            name="Security",
            description="Rules for identifying security issues"
        )
        security_ruleset.add_rules([
            self.get_rule("Hardcoded Secret"),
            self.get_rule("SQL Injection Risk"),
            self.get_rule("Insecure Deserialization")
        ])
        self.register_rule_set(security_ruleset)
        
        code_quality_ruleset = RuleSet(
            name="Code Quality",
            description="Rules for improving code quality"
        )
        code_quality_ruleset.add_rules([
            self.get_rule("TODO Comment"),
            self.get_rule("Print Statement"),
            self.get_rule("Function Complexity")
        ])
        self.register_rule_set(code_quality_ruleset)
        
        # Combined ruleset
        combined_ruleset = RuleSet(
            name="Default",
            description="Default rules for code review"
        )
        combined_ruleset.add_rules(list(self.rules.values()))
        self.register_rule_set(combined_ruleset)


# Global rule registry instance
_global_registry = RuleRegistry()

def get_rule_registry() -> RuleRegistry:
    """
    Get the global rule registry instance.
    
    Returns:
        Global RuleRegistry instance
    """
    return _global_registry


# Initialize with built-in rules
_global_registry.load_builtin_rules()