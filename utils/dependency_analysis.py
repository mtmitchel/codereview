import os
import re
import ast
import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class DependencyAnalyzer:
    """
    Analyzes dependencies between files in a codebase.
    Supports Python files with import statements and JavaScript files with import/require.
    """
    
    def __init__(self):
        """Initialize the dependency analyzer."""
        self.dependency_graph = {}  # file -> list of files it depends on
        self.reverse_dependency_graph = {}  # file -> list of files that depend on it
        self.module_to_file_map = {}  # module name -> file path
        self.scanned_files = set()
        self.analyzed = False
    
    def clear(self):
        """Clear all dependency information."""
        self.dependency_graph = {}
        self.reverse_dependency_graph = {}
        self.module_to_file_map = {}
        self.scanned_files = set()
        self.analyzed = False
    
    def analyze_directory(self, directory: str, exclude_patterns: List[str] = None) -> None:
        """
        Analyze all Python and JavaScript files in the directory to build dependency graphs.
        Optimized for speed by parallelizing dependency extraction.
        Args:
            directory: Root directory to analyze
            exclude_patterns: List of glob patterns to exclude
        """
        from pathlib import Path
        import fnmatch

        if exclude_patterns is None:
            exclude_patterns = []

        # First pass: build module to file map
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.py', '.js', '.jsx', '.ts', '.tsx')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, directory)
                    should_exclude = False
                    for pattern in exclude_patterns:
                        if fnmatch.fnmatch(relative_path, pattern):
                            should_exclude = True
                            break
                    if should_exclude:
                        continue
                    if file.endswith('.py'):
                        module_name = os.path.splitext(relative_path)[0].replace(os.path.sep, '.')
                        self.module_to_file_map[module_name] = file_path
                        package_path = os.path.dirname(relative_path)
                        if package_path:
                            package_name = package_path.replace(os.path.sep, '.')
                            self.module_to_file_map[package_name] = os.path.dirname(file_path)
                    elif file.endswith(('.js', '.jsx', '.ts', '.tsx')):
                        module_name = os.path.splitext(relative_path)[0]
                        self.module_to_file_map[module_name] = file_path
                        self.module_to_file_map[relative_path] = file_path

        # Second pass: analyze imports (parallelized)
        files_to_analyze = [fp for fp in self.module_to_file_map.values() if os.path.isfile(fp) and fp not in self.scanned_files]
        def analyze_file_task(fp):
            self._analyze_file_dependencies(fp, directory)
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(analyze_file_task, files_to_analyze))

        # Build reverse dependency graph
        self.reverse_dependency_graph = defaultdict(list)
        for file, deps in self.dependency_graph.items():
            for dep in deps:
                self.reverse_dependency_graph[dep].append(file)
        self.analyzed = True
        logger.info(f"Analyzed dependencies for {len(self.scanned_files)} files in {directory}")
    
    def _analyze_file_dependencies(self, file_path: str, base_dir: str) -> None:
        """
        Analyze dependencies for a single file.
        
        Args:
            file_path: Path to the file to analyze
            base_dir: Base directory for resolving relative imports
        """
        if file_path in self.scanned_files:
            return
        
        self.scanned_files.add(file_path)
        self.dependency_graph[file_path] = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if file_path.endswith('.py'):
                dependencies = self._extract_python_dependencies(file_path, content, base_dir)
            elif file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
                dependencies = self._extract_js_dependencies(file_path, content, base_dir)
            else:
                dependencies = []
            
            self.dependency_graph[file_path] = dependencies
            
        except Exception as e:
            logger.error(f"Error analyzing dependencies for {file_path}: {str(e)}")
    
    def _extract_python_dependencies(self, file_path: str, content: str, base_dir: str) -> List[str]:
        """
        Extract Python dependencies from import statements.
        
        Args:
            file_path: Path to the Python file
            content: File content
            base_dir: Base directory for resolving relative imports
            
        Returns:
            List of file paths that this file depends on
        """
        dependencies = []
        
        try:
            # Parse the file
            tree = ast.parse(content)
            
            # Extract the imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        module_name = name.name
                        dep_file = self._resolve_python_module(module_name, file_path, base_dir)
                        if dep_file and dep_file != file_path:
                            dependencies.append(dep_file)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Handle relative imports
                        if node.level > 0:
                            # Get the current package path
                            current_dir = os.path.dirname(file_path)
                            # Go up the directory tree for each level
                            for _ in range(node.level - 1):
                                current_dir = os.path.dirname(current_dir)
                            
                            # Full module name
                            if node.module:
                                package_path = os.path.join(current_dir, *node.module.split('.'))
                                module_name = node.module
                            else:
                                package_path = current_dir
                                relative_dir = os.path.relpath(current_dir, base_dir)
                                module_name = relative_dir.replace(os.path.sep, '.')
                            
                            # Add the directory itself
                            if os.path.isdir(package_path):
                                dependencies.append(package_path)
                            
                            # Check each imported name
                            for name in node.names:
                                possible_file = os.path.join(package_path, f"{name.name}.py")
                                if os.path.isfile(possible_file):
                                    dependencies.append(possible_file)
                        else:
                            # Normal import from
                            module_name = node.module
                            dep_file = self._resolve_python_module(module_name, file_path, base_dir)
                            if dep_file and dep_file != file_path:
                                dependencies.append(dep_file)
                                
                                # Also check for submodules if the imported name isn't '*'
                                if not any(n.name == '*' for n in node.names):
                                    for name in node.names:
                                        sub_module = f"{module_name}.{name.name}"
                                        sub_dep = self._resolve_python_module(sub_module, file_path, base_dir)
                                        if sub_dep and sub_dep != file_path and sub_dep != dep_file:
                                            dependencies.append(sub_dep)
        
        except SyntaxError:
            logger.warning(f"Syntax error in {file_path}, using regex fallback for imports")
            # Use regex as fallback
            import_pattern = re.compile(r'^(?:from\s+([\w.]+)\s+import|import\s+([\w.,\s]+))', re.MULTILINE)
            for match in import_pattern.finditer(content):
                from_import, direct_import = match.groups()
                if from_import:
                    dep_file = self._resolve_python_module(from_import, file_path, base_dir)
                    if dep_file and dep_file != file_path:
                        dependencies.append(dep_file)
                elif direct_import:
                    for module in re.split(r',\s*', direct_import):
                        module = module.strip()
                        dep_file = self._resolve_python_module(module, file_path, base_dir)
                        if dep_file and dep_file != file_path:
                            dependencies.append(dep_file)
        
        except Exception as e:
            logger.error(f"Error extracting Python dependencies from {file_path}: {str(e)}")
        
        return list(set(dependencies))  # Remove duplicates
    
    def _resolve_python_module(self, module_name: str, file_path: str, base_dir: str) -> Optional[str]:
        """
        Resolve a Python module name to a file path.
        
        Args:
            module_name: Name of the module to resolve
            file_path: Path to the file containing the import
            base_dir: Base directory for resolving relative imports
            
        Returns:
            File path for the module, or None if not found
        """
        # Skip standard library and third-party modules
        if module_name in ('os', 'sys', 're', 'json', 'time', 'datetime', 'pathlib',
                           'collections', 'typing', 'logging', 'math', 'random',
                           'sqlite3', 'io', 'tempfile', 'traceback', 'unittest'):
            return None
        
        # Check if the module is in our map
        if module_name in self.module_to_file_map:
            return self.module_to_file_map[module_name]
        
        # Try to find by searching for the file
        parts = module_name.split('.')
        for i in range(len(parts), 0, -1):
            # Try module as a file with .py extension
            partial_module = '.'.join(parts[:i])
            rel_path = os.path.join(*partial_module.split('.')) + '.py'
            abs_path = os.path.join(base_dir, rel_path)
            if os.path.isfile(abs_path):
                self.module_to_file_map[partial_module] = abs_path
                return abs_path
            
            # Try module as a directory with __init__.py
            init_path = os.path.join(base_dir, *partial_module.split('.'), '__init__.py')
            if os.path.isfile(init_path):
                self.module_to_file_map[partial_module] = os.path.dirname(init_path)
                return os.path.dirname(init_path)
        
        # Not found in this project
        return None
    
    def _extract_js_dependencies(self, file_path: str, content: str, base_dir: str) -> List[str]:
        """
        Extract JavaScript/TypeScript dependencies.
        
        Args:
            file_path: Path to the JS/TS file
            content: File content
            base_dir: Base directory for resolving relative imports
            
        Returns:
            List of file paths that this file depends on
        """
        dependencies = []
        
        try:
            # Match ES module imports: import X from 'path';
            import_pattern = re.compile(r'import\s+(?:[^;]+\s+from\s+)?[\'"]([^\'";]+)[\'"]')
            
            # Match require statements: const x = require('path');
            require_pattern = re.compile(r'require\s*\(\s*[\'"]([^\'")]+)[\'"]\s*\)')
            
            # Find all imports
            for pattern in [import_pattern, require_pattern]:
                for match in pattern.finditer(content):
                    path = match.group(1)
                    
                    # Skip node modules and absolute paths
                    if path.startswith('@') or path.startswith('http') or not (path.startswith('./') or path.startswith('../') or path.startswith('/')):
                        continue
                    
                    # Resolve the path
                    dep_file = self._resolve_js_module(path, file_path, base_dir)
                    if dep_file and dep_file != file_path:
                        dependencies.append(dep_file)
        
        except Exception as e:
            logger.error(f"Error extracting JS dependencies from {file_path}: {str(e)}")
        
        return list(set(dependencies))  # Remove duplicates
    
    def _resolve_js_module(self, import_path: str, file_path: str, base_dir: str) -> Optional[str]:
        """
        Resolve a JavaScript/TypeScript import path to a file path.
        
        Args:
            import_path: Path from import statement
            file_path: Path to the file containing the import
            base_dir: Base directory for resolving relative imports
            
        Returns:
            File path for the module, or None if not found
        """
        # Handle relative imports
        if import_path.startswith(('./', '../')):
            # Convert relative path to absolute path
            dir_path = os.path.dirname(file_path)
            abs_path = os.path.normpath(os.path.join(dir_path, import_path))
            
            # Check various extensions and patterns
            extensions = ['', '.js', '.jsx', '.ts', '.tsx']
            for ext in extensions:
                # Try direct file
                test_path = abs_path + ext
                if os.path.isfile(test_path):
                    return test_path
                
                # Try index files
                index_path = os.path.join(abs_path, 'index' + ext)
                if os.path.isfile(index_path):
                    return index_path
        
        # Handle absolute imports (within project)
        elif import_path.startswith('/'):
            # Treats it as absolute from the base directory
            abs_path = os.path.normpath(os.path.join(base_dir, import_path.lstrip('/')))
            
            # Check various extensions and patterns
            extensions = ['', '.js', '.jsx', '.ts', '.tsx']
            for ext in extensions:
                test_path = abs_path + ext
                if os.path.isfile(test_path):
                    return test_path
                
                # Try index files
                index_path = os.path.join(abs_path, 'index' + ext)
                if os.path.isfile(index_path):
                    return index_path
        
        # Use module map for known modules
        else:
            # Try exact match
            if import_path in self.module_to_file_map:
                return self.module_to_file_map[import_path]
            
            # Try with extensions
            for ext in ['.js', '.jsx', '.ts', '.tsx']:
                if import_path + ext in self.module_to_file_map:
                    return self.module_to_file_map[import_path + ext]
        
        return None
    
    def get_dependencies(self, file_path: str) -> List[str]:
        """
        Get the dependencies for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of file paths that this file depends on
        """
        if not self.analyzed:
            raise ValueError("Must call analyze_directory before getting dependencies")
        
        return self.dependency_graph.get(file_path, [])
    
    def get_dependents(self, file_path: str) -> List[str]:
        """
        Get files that depend on this file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of file paths that depend on this file
        """
        return self.reverse_dependency_graph.get(file_path, [])
    
    def get_full_dependency_tree(self, file_path: str) -> Dict[str, List[str]]:
        """
        Get a full dependency tree starting from the given file.
        
        Args:
            file_path: Root file path
            
        Returns:
            Dictionary mapping each file to its dependencies
        """
        result = {}
        visited = set()
        
        def traverse(path):
            if path in visited:
                return
            visited.add(path)
            
            deps = self.get_dependencies(path)
            result[path] = deps
            
            for dep in deps:
                traverse(dep)
        
        traverse(file_path)
        return result
    
    def get_dependency_cycles(self) -> List[List[str]]:
        """
        Detect dependency cycles in the codebase.
        
        Returns:
            List of cycles found, each represented as a list of file paths
        """
        if not self.analyzed:
            return []
        
        cycles = []
        visited = set()
        
        def find_cycles(node, path=None, start=None):
            if path is None:
                path = []
            
            if node in path:
                # Cycle detected
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited and node != start:
                return
            
            visited.add(node)
            path.append(node)
            
            for dependency in self.dependency_graph.get(node, []):
                find_cycles(dependency, path.copy(), start or node)
        
        for file in self.dependency_graph:
            if file not in visited:
                find_cycles(file)
        
        return cycles
    
    def get_file_complexity(self, file_path: str) -> Dict[str, int]:
        """
        Calculate dependency complexity metrics for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with complexity metrics
        """
        if not self.analyzed or file_path not in self.dependency_graph:
            return {
                "dependencies": 0,
                "dependents": 0,
                "complexity": 0
            }
        
        deps = len(self.get_dependencies(file_path))
        dependents = len(self.get_dependents(file_path))
        
        # Simple complexity score: product of dependencies and dependents
        complexity = deps * dependents
        
        return {
            "dependencies": deps,
            "dependents": dependents,
            "complexity": complexity
        }
    
    def get_most_complex_files(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most complex files based on dependency complexity.
        
        Args:
            limit: Maximum number of files to return
            
        Returns:
            List of (file_path, complexity) tuples
        """
        if not self.analyzed:
            return []
        
        complexities = []
        for file in self.dependency_graph:
            metrics = self.get_file_complexity(file)
            complexities.append((file, metrics["complexity"]))
        
        # Sort by complexity, highest first
        complexities.sort(key=lambda x: x[1], reverse=True)
        
        return complexities[:limit]

    def analyze_file_dependencies(self, file_path: str) -> str:
        """
        Analyze dependencies for a single file and return a formatted string.
        Used by the EnhancedCodeAnalyzer for code review.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            String representation of the dependencies
        """
        # Make sure dependencies have been analyzed
        if not self.analyzed:
            base_dir = os.path.dirname(file_path)
            self.analyze_directory(base_dir)
        
        # Get dependencies for this file
        deps = self.get_dependencies(file_path)
        if not deps:
            return "No dependencies detected."
        
        # Format the dependency information
        formatted_deps = []
        for dep in deps:
            # Convert absolute paths to relative/module names where possible
            rel_name = os.path.basename(dep)
            formatted_deps.append(rel_name)
        
        if not formatted_deps:
            return "No dependencies detected."
            
        # Return formatted string
        return f"Dependencies: {', '.join(formatted_deps)}"


def analyze_dependencies(file_paths: List[str]) -> Dict[str, Any]:
    """Analyze dependencies for a list of files."""
    analyzer = DependencyAnalyzer.get_analyzer_for_files(file_paths)
    analyzer.analyze_files(file_paths)
    
    # Create a summary of the analysis
    summary = {
        'dependency_count': {file: len(deps) for file, deps in analyzer.dependencies.items()},
        'reverse_dependency_count': {file: len(deps) for file, deps in analyzer.reverse_dependencies.items()},
        'cycles': analyzer.find_cycles(),
        'dependency_graph': {file: list(deps) for file, deps in analyzer.dependencies.items()},
        'reverse_dependency_graph': {file: list(deps) for file, deps in analyzer.reverse_dependencies.items()}
    }
    
    # Identify key files (most depended upon)
    if analyzer.reverse_dependencies:
        most_depended = max(analyzer.reverse_dependencies.items(), key=lambda x: len(x[1]))[0]
        summary['most_depended_file'] = {
            'file': most_depended,
            'dependent_count': len(analyzer.reverse_dependencies[most_depended])
        }
    
    return summary


def analyze_project_dependencies(directory: str, exclude_patterns: List[str] = None) -> Dict[str, Any]:
    """Analyze dependencies for an entire project."""
    exclude_patterns = exclude_patterns or []
    file_paths = []
    
    # Collect all files
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, directory)
            
            # Skip files matching exclude patterns
            if any(re.match(pattern, rel_path) for pattern in exclude_patterns):
                continue
            
            # Only include Python, JavaScript, and TypeScript files
            if file_path.endswith(('.py', '.js', '.jsx', '.ts', '.tsx')):
                file_paths.append(file_path)
    
    return analyze_dependencies(file_paths)


def get_file_impact(file_path: str, file_paths: List[str]) -> Dict[str, Any]:
    """Calculate the impact of changes to a specific file."""
    analyzer = DependencyAnalyzer.get_analyzer_for_files(file_paths)
    analyzer.analyze_files(file_paths)
    
    # Get all files that would be impacted by changes to this file
    direct_impact = analyzer.get_reverse_dependencies(file_path)
    all_impact = analyzer.get_all_reverse_dependencies(file_path)
    
    return {
        'file': file_path,
        'direct_impact': list(direct_impact),
        'direct_impact_count': len(direct_impact),
        'transitive_impact': list(all_impact),
        'transitive_impact_count': len(all_impact),
        'impact_percentage': round(len(all_impact) / len(file_paths) * 100, 2) if file_paths else 0
    } 