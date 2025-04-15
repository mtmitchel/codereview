import os
import time
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional, Callable, Generator, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from utils.analyzer_plugin import PluginManager
from core.constants import (
    MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, DEFAULT_CHUNK_SIZE, 
    DEFAULT_BINARY_CHUNK_SIZE, BINARY_FILE_EXTENSIONS
)

logger = logging.getLogger("CodeReviewTool.ParallelAnalyzer")

class ParallelAnalyzer:
    """
    Class for analyzing code in parallel using multiple threads.
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 plugin_manager: Optional[PluginManager] = None,
                 status_callback: Optional[Callable[[str], None]] = None,
                 progress_callback: Optional[Callable[[int, int], None]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the parallel analyzer.
        
        Args:
            max_workers: Maximum number of parallel workers
            plugin_manager: Plugin manager for analyzer plugins
            status_callback: Callback for status updates
            progress_callback: Callback for progress updates
            config: Configuration dictionary
        """
        self.max_workers = max(1, min(max_workers, os.cpu_count() or 4))
        self.plugin_manager = plugin_manager or PluginManager()
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.stop_requested = False
        self.processed_files = 0
        self.total_files = 0
        self.file_hashes = {}  # Store file hashes for incremental analysis
        self.file_cache_path = os.path.join(os.path.expanduser("~"), ".code_reviewer_file_hashes.json")
        self._load_file_hashes()
        self.executor = None  # Initialize later
        
        # Chunking configuration
        self.config = config or {}
        self.adaptive_chunking = self.config.get('enable_adaptive_chunking', True)
        self.min_chunk_size = self.config.get('min_chunk_size', MIN_CHUNK_SIZE)
        self.max_chunk_size = self.config.get('max_chunk_size', MAX_CHUNK_SIZE)
        self.default_chunk_size = self.config.get('chunk_size', DEFAULT_CHUNK_SIZE)
        self.binary_chunk_size = self.config.get('binary_chunk_size', DEFAULT_BINARY_CHUNK_SIZE)
        
        # Model management
        self.model_manager = None
        try:
            from api.model_manager import ModelManager
            self.model_manager = ModelManager(config)
        except ImportError:
            logger.warning("ModelManager not available, using default model selection")
    
    def _load_file_hashes(self) -> None:
        """Load file hashes from cache."""
        try:
            import json
            if os.path.exists(self.file_cache_path):
                with open(self.file_cache_path, 'r', encoding='utf-8') as f:
                    self.file_hashes = json.load(f)
        except Exception as e:
            logger.error(f"Error loading file hashes: {str(e)}")
            self.file_hashes = {}
    
    def _save_file_hashes(self) -> None:
        """Save file hashes to cache."""
        try:
            import json
            with open(self.file_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.file_hashes, f)
        except Exception as e:
            logger.error(f"Error saving file hashes: {str(e)}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        Calculate a hash for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hash string for the file
        """
        try:
            import hashlib
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error calculating file hash for {file_path}: {str(e)}")
            return ""
    
    def _file_has_changed(self, file_path: str) -> bool:
        """
        Check if a file has changed since last analysis.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file has changed, False otherwise
        """
        file_hash = self._get_file_hash(file_path)
        if not file_hash:
            return True  # If we can't get a hash, assume it changed
            
        previous_hash = self.file_hashes.get(file_path)
        if not previous_hash or previous_hash != file_hash:
            self.file_hashes[file_path] = file_hash
            return True
            
        return False
    
    def update_status(self, message: str) -> None:
        """Update status using the callback if available."""
        if self.status_callback:
            self.status_callback(message)
            
    def update_progress(self, current: int, total: int) -> None:
        """Update progress using the callback if available."""
        if self.progress_callback:
            self.progress_callback(current, total)
            
    def request_stop(self) -> None:
        """Request to stop processing."""
        self.stop_requested = True
        logger.info("Stop requested for parallel analyzer")
    
    def get_file_list(self, 
                      directory_path: str, 
                      file_types: Optional[List[str]] = None,
                      exclude_patterns: Optional[List[str]] = None,
                      incremental: bool = True) -> List[str]:
        """
        Get list of files to process.
        
        Args:
            directory_path: Path to the directory
            file_types: File extensions to include
            exclude_patterns: Patterns to exclude
            incremental: Whether to use incremental analysis (only analyze changed files)
            
        Returns:
            List of file paths to process
        """
        from utils.file_utils import is_excluded
        
        if file_types is None:
            file_types = []
        if exclude_patterns is None:
            exclude_patterns = []
            
        file_list = []
        directory = Path(directory_path)
        
        for file_path in directory.rglob('*'):
            if self.stop_requested:
                break
                
            if file_path.is_file():
                path_str = str(file_path)
                
                # Skip excluded files
                if is_excluded(path_str, exclude_patterns, directory_path):
                    continue
                    
                # Check file extensions
                ext = file_path.suffix.lower()
                if not file_types or ext in file_types or ext[1:] in file_types:
                    # Check if file has changed (if incremental)
                    if not incremental or self._file_has_changed(path_str):
                        try:
                            # Get file size for binary detection
                            file_size = os.path.getsize(path_str)
                            
                            # Skip very large binary files (likely not code)
                            if file_size > 100 * 1024 * 1024 and ext in BINARY_FILE_EXTENSIONS:  # 100MB
                                logger.info(f"Skipping large binary file: {path_str} ({file_size / (1024*1024):.2f} MB)")
                                continue
                            
                            # Check if file is readable for text files
                            if ext not in BINARY_FILE_EXTENSIONS:
                                with open(path_str, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                if not content:  # Skip empty files
                                    continue
                            
                            file_list.append(path_str)
                        except Exception as e:
                            self.update_status(f"Warning: Could not read file {path_str}: {e}")
        
        # Save file hashes for next time
        self._save_file_hashes()
        
        return file_list
    
    def get_dependent_files(self, 
                            file_list: List[str], 
                            changed_files: List[str],
                            dependency_analyzer) -> List[str]:
        """
        Get list of files that depend on changed files.
        
        Args:
            file_list: List of all files
            changed_files: List of changed files
            dependency_analyzer: Dependency analyzer to use
            
        Returns:
            List of dependent files that need to be analyzed
        """
        if not dependency_analyzer:
            return []
            
        # Get dependencies for all files
        dependency_analyzer.analyze_files(file_list)
        
        # Get all files that depend on changed files
        dependent_files = set()
        for changed_file in changed_files:
            dependencies = dependency_analyzer.get_reverse_dependencies(changed_file)
            dependent_files.update(dependencies)
        
        # Only include files from our file list
        file_set = set(file_list)
        return [f for f in dependent_files if f in file_set]
    
    def analyze_files(self, 
                      file_list: List[str], 
                      incremental: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Analyze files in parallel.
        
        Args:
            file_list: List of files to analyze
            incremental: Whether to use incremental analysis
            
        Returns:
            Analysis results for each file
        """
        if not file_list:
            return {}
            
        self.stop_requested = False
        self.processed_files = 0
        self.total_files = len(file_list)
        
        results = {}
        start_time = time.time()
        
        self.update_status(f"Starting parallel analysis of {self.total_files} files with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all analysis tasks
            future_to_file = {
                executor.submit(self._analyze_file, file_path): file_path
                for file_path in file_list
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                if self.stop_requested:
                    for f in future_to_file:
                        f.cancel()
                    break
                
                file_path = future_to_file[future]
                try:
                    file_results = future.result()
                    results[file_path] = file_results
                except Exception as e:
                    logger.error(f"Error analyzing {file_path}: {str(e)}")
                    results[file_path] = {"error": str(e)}
                
                self.processed_files += 1
                self.update_progress(self.processed_files, self.total_files)
        
        elapsed_time = time.time() - start_time
        self.update_status(f"Completed analysis of {self.processed_files} files in {elapsed_time:.2f} seconds")
        
        return results
    
    def _determine_chunk_size(self, file_path: str, file_size: int) -> int:
        """
        Determine optimal chunk size based on file characteristics.
        
        Args:
            file_path: Path to the file
            file_size: Size of the file in bytes
            
        Returns:
            Optimal chunk size in bytes
        """
        if not self.adaptive_chunking:
            return self.default_chunk_size
            
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())
        
        # Binary files need smaller chunks for efficient processing
        if ext in BINARY_FILE_EXTENSIONS:
            # Smaller chunks for very large binary files
            if file_size > 50 * 1024 * 1024:  # > 50MB
                return max(self.min_chunk_size, self.binary_chunk_size // 2)
            elif file_size > 10 * 1024 * 1024:  # > 10MB
                return max(self.min_chunk_size, self.binary_chunk_size)
            else:
                return max(self.min_chunk_size, self.binary_chunk_size * 2)
                
        # For text-based files, use larger chunks
        # Scale based on file size
        if file_size > 5 * 1024 * 1024:  # > 5MB
            # For very large text files, use smaller chunks to avoid timeouts
            return max(self.min_chunk_size, min(self.default_chunk_size, 2048))
        elif file_size > 1 * 1024 * 1024:  # > 1MB
            return max(self.min_chunk_size, min(self.default_chunk_size, 4096))
        else:
            # For small text files, can use larger chunks
            return min(self.max_chunk_size, max(self.min_chunk_size, self.default_chunk_size))
    
    def _is_binary_file(self, file_path: str) -> bool:
        """
        Check if a file is likely binary based on extension and content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is likely binary, False otherwise
        """
        # Check extension first
        _, ext = os.path.splitext(file_path.lower())
        if ext in BINARY_FILE_EXTENSIONS:
            return True
            
        # Check content if extension doesn't indicate binary
        try:
            with open(file_path, 'rb') as f:
                # Read first 1024 bytes
                chunk = f.read(1024)
                # Check for null bytes which usually indicate binary content
                if b'\x00' in chunk:
                    return True
                    
            # No binary indicators found
            return False
        except Exception:
            # If can't read file, assume it's not binary to be safe
            return False
    
    def _select_model_for_file(self, file_path: str, file_size: int) -> str:
        """
        Select an appropriate model for processing the file.
        
        Args:
            file_path: Path to the file
            file_size: Size of the file in bytes
            
        Returns:
            Model name to use
        """
        if not self.model_manager:
            # If model manager is not available, return None to use default
            return None
            
        # Use model manager to select appropriate model
        return self.model_manager.get_model_for_file(file_path, file_size, 0)
    
    def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single file using all available plugins.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Analysis results from all plugins
        """
        try:
            # Get file size and determine if binary
            file_size = os.path.getsize(file_path)
            is_binary = self._is_binary_file(file_path)
            
            # Get language from file extension
            ext = os.path.splitext(file_path)[1][1:].lower()
            
            # Custom model selection based on file characteristics
            model = self._select_model_for_file(file_path, file_size)
            
            # For non-binary files, read content
            file_content = None
            if not is_binary:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
            
            # Get analysis results from all plugins
            plugin_results = self.plugin_manager.analyze_file(file_path, file_content)
            
            # Determine optimal chunk size for this file
            chunk_size = self._determine_chunk_size(file_path, file_size)
            
            # Combine results
            results = {
                "plugin_results": plugin_results,
                "file_path": file_path,
                "language": ext,
                "timestamp": time.time(),
                "file_size": file_size,
                "is_binary": is_binary,
                "chunk_size": chunk_size,
                "model": model
            }
            
            return results
        except Exception as e:
            logger.error(f"Error in _analyze_file for {file_path}: {str(e)}")
            raise
    
    def run_analysis(self, 
                    directory_path: str, 
                    file_types: Optional[List[str]] = None,
                    exclude_patterns: Optional[List[str]] = None,
                    incremental: bool = True,
                    dependency_analyzer = None) -> Dict[str, Dict[str, Any]]:
        """
        Run analysis on a directory.
        
        Args:
            directory_path: Path to the directory
            file_types: File extensions to include
            exclude_patterns: Patterns to exclude
            incremental: Whether to use incremental analysis
            dependency_analyzer: Optional dependency analyzer for finding related files
            
        Returns:
            Analysis results for each file
        """
        # Get list of files to analyze
        self.update_status(f"Finding files in {directory_path}")
        file_list = self.get_file_list(directory_path, file_types, exclude_patterns, incremental)
        
        if not file_list:
            self.update_status("No files found to analyze")
            return {}
        
        # If incremental, add dependent files
        if incremental and dependency_analyzer and len(file_list) < self.total_files:
            self.update_status("Finding dependent files...")
            dependent_files = self.get_dependent_files(
                self.get_file_list(directory_path, file_types, exclude_patterns, False),  # All files
                file_list,  # Changed files
                dependency_analyzer
            )
            
            if dependent_files:
                self.update_status(f"Found {len(dependent_files)} dependent files to analyze")
                file_list.extend(dependent_files)
        
        self.update_status(f"Found {len(file_list)} files to analyze")
        
        # Analyze files
        results = self.analyze_files(file_list, incremental)
        
        return results

    def split_into_chunks(self, content: str, file_path: str, chunk_size: int = 0, overlap: int = 200) -> List[str]:
        """
        Split content into chunks with overlap.
        
        Args:
            content: Content to split
            file_path: Path to the file (for determining chunk size)
            chunk_size: Chunk size in characters (0 to use adaptive)
            overlap: Overlap between chunks
            
        Returns:
            List of content chunks
        """
        if not content:
            return []
            
        # If chunk size is not specified, determine it adaptively
        if chunk_size <= 0:
            file_size = len(content.encode('utf-8'))
            chunk_size = self._determine_chunk_size(file_path, file_size)
            
        chunks = []
        start = 0
        content_len = len(content)
        
        while start < content_len:
            # Calculate end position for this chunk
            end = min(start + chunk_size, content_len)
            
            # If not at the end of content, try to find a good break point
            if end < content_len:
                # Try to find a newline to break at
                newline_pos = content.rfind('\n', start, end)
                if newline_pos > start + chunk_size // 2:  # Only use if we're not cutting the chunk too short
                    end = newline_pos + 1  # Include the newline
            
            # Extract the chunk
            chunks.append(content[start:end])
            
            # Move to next chunk position, accounting for overlap
            start = max(0, end - overlap)
            
            # Avoid getting stuck in an infinite loop
            if start >= end:
                break
        
        return chunks
    
    def determine_optimal_params(self, file_path: str, content: str = None) -> Tuple[int, str]:
        """
        Determine optimal parameters for processing a file.
        
        Args:
            file_path: Path to the file
            content: Optional file content
            
        Returns:
            Tuple of (chunk_size, model_name)
        """
        try:
            # Get file size
            if content:
                file_size = len(content.encode('utf-8'))
            else:
                file_size = os.path.getsize(file_path)
                
            # Determine optimal chunk size
            chunk_size = self._determine_chunk_size(file_path, file_size)
            
            # Select model
            model = self._select_model_for_file(file_path, file_size)
            
            return (chunk_size, model)
        except Exception as e:
            logger.error(f"Error determining optimal parameters for {file_path}: {e}")
            return (self.default_chunk_size, None)
        
    def process_chunks(self,
                      chunks: List[str],
                      file_path: str,
                      dependencies: str = "",
                      flake8_output: str = "",
                      semgrep_findings: str = "",
                      plugin_data: Dict[str, Any] = None,
                      build_prompt_func: Callable = None,
                      call_llm_func: Callable = None,
                      model_override: str = None) -> List[Dict[str, Any]]:
        """
        Process code chunks in parallel using thread pool.
        
        Args:
            chunks: List of code chunks to process
            file_path: Path to the file
            dependencies: Dependencies string
            flake8_output: Flake8 analysis output
            semgrep_findings: Semgrep analysis output
            plugin_data: Plugin data
            build_prompt_func: Function to build prompt for LLM
            call_llm_func: Function to call LLM API
            model_override: Optional model to use (overrides automatic selection)
            
        Returns:
            List of results for each chunk
        """
        if not chunks:
            return []
            
        if plugin_data is None:
            plugin_data = {}
            
        self.stop_requested = False
        total_chunks = len(chunks)
        processed_chunks = 0
        
        # Get file size for model selection
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else sum(len(c.encode('utf-8')) for c in chunks)
        
        # Select model based on file characteristics
        model = model_override
        if not model and self.model_manager:
            model = self.model_manager.get_model_for_file(file_path, file_size, max(len(c) for c in chunks))
            logger.info(f"Selected model {model} for processing {file_path}")
        
        logger.info(f"Processing {total_chunks} chunks for {file_path} in parallel")
        
        results = []
        
        # Define the worker function for processing a chunk
        def process_chunk(chunk_index: int) -> Dict[str, Any]:
            if self.stop_requested:
                return {"error": "Processing stopped", "chunk_index": chunk_index}
                
            chunk = chunks[chunk_index]
            
            try:
                # Build the prompt if a function is provided
                if build_prompt_func:
                    # Combine flake8 and semgrep into context
                    context = flake8_output
                    if semgrep_findings and semgrep_findings != "N/A":
                        context += "\n\n" + semgrep_findings
                        
                    prompt = build_prompt_func(
                        file_path=file_path,
                        code_chunk=chunk,
                        dependencies=dependencies,
                        flake8_output=context,
                        plugin_data=plugin_data
                    )
                else:
                    # Simple default prompt if no function provided
                    prompt = f"Review the following code chunk {chunk_index+1}/{total_chunks} from {file_path}:\n\n{chunk}"
                
                # Call LLM with built prompt
                if call_llm_func:
                    context = f"chunk {chunk_index+1}/{total_chunks} of {file_path}"
                    
                    # If we have a model from intelligent selection, use it
                    if model:
                        # Temporarily override the model in the context
                        orig_model = None
                        if hasattr(call_llm_func, '__self__') and hasattr(call_llm_func.__self__, 'config'):
                            client = call_llm_func.__self__
                            if 'model' in client.config:
                                orig_model = client.config['model']
                                client.config['model'] = model
                                
                        response = call_llm_func(prompt, context)
                        
                        # Restore original model if we changed it
                        if orig_model is not None and hasattr(call_llm_func, '__self__'):
                            call_llm_func.__self__.config['model'] = orig_model
                    else:
                        response = call_llm_func(prompt, context)
                    
                    # Process response
                    result = {
                        "code": chunk,
                        "review": response,
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "error": False
                    }
                    
                    # Extract parsed issues if the response appears to be JSON-formatted
                    if response.strip().startswith("{") and response.strip().endswith("}"):
                        try:
                            import json
                            parsed = json.loads(response)
                            if isinstance(parsed, dict) and "issues" in parsed:
                                result["parsed_issues"] = parsed["issues"]
                        except Exception as json_err:
                            logger.debug(f"Failed to parse LLM response as JSON: {json_err}")
                            
                    return result
                else:
                    return {"error": "No LLM function provided", "chunk_index": chunk_index}
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_index+1}/{total_chunks} of {file_path}: {e}")
                return {
                    "error": str(e),
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks
                }
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processing tasks
            future_to_chunk = {
                executor.submit(process_chunk, chunk_idx): chunk_idx
                for chunk_idx in range(total_chunks)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                if self.stop_requested:
                    for f in future_to_chunk:
                        f.cancel()
                    break
                
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                    results.append(chunk_result)
                except Exception as e:
                    logger.error(f"Failed to get result for chunk {chunk_idx+1}/{total_chunks}: {e}")
                    results.append({
                        "error": f"Failed to process: {str(e)}",
                        "chunk_index": chunk_idx,
                        "total_chunks": total_chunks
                    })
                
                processed_chunks += 1
                if self.progress_callback:
                    self.progress_callback(processed_chunks, total_chunks)
        
        # Sort results by chunk index
        results.sort(key=lambda x: x.get("chunk_index", 0))
        
        return results
