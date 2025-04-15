"""
Parallel processing module for security analysis of multiple files.
"""

import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("CodeReviewTool.SecurityAnalyzer.Parallel")

def analyze_files_parallel(analyzer, file_paths: List[str], max_workers: int = 8) -> Dict[str, Any]:
    """
    Analyze multiple files in parallel using ThreadPoolExecutor.
    
    Args:
        analyzer: The SecurityAnalyzerPlugin instance
        file_paths: List of file paths to analyze
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary mapping file paths to analysis results
    """
    results = {}
    
    # Safety check
    if not file_paths:
        return results
    
    # Limit the number of threads based on file count
    actual_workers = min(max_workers, len(file_paths))
    if actual_workers <= 0:
        actual_workers = 1
    
    logger.info(f"Starting parallel security analysis of {len(file_paths)} files with {actual_workers} workers")
    
    try:
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Create a future for each file
            future_to_file = {
                executor.submit(analyzer._analyze_file_impl, file_path): file_path 
                for file_path in file_paths
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results[file_path] = result
                except Exception as e:
                    logger.error(f"Error analyzing {file_path}: {e}")
                    results[file_path] = {
                        "error": str(e),
                        "vulnerabilities": [],
                        "risk_score": 0
                    }
                    
    except Exception as e:
        logger.error(f"Error in parallel analysis execution: {e}")
    
    logger.info(f"Completed parallel security analysis of {len(results)} files")
    
    return results 