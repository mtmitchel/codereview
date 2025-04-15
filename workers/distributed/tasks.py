"""
Celery tasks for distributed processing in Code Review Tool.

These tasks handle the distributed execution of code review operations,
allowing the system to scale for large codebases by distributing work across
multiple processes or machines.
"""

import logging
import uuid
from typing import Dict, Any
from celery import Celery, Task
# from core.config import CodeReviewerConfig
from workers.analyzer import CodeReviewController

logger = logging.getLogger("CodeReviewTool.DistributedTasks")

class CodeReviewTask(Task):
    """Base task class with error handling for code review tasks."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(f"Task {task_id} failed: {exc}", exc_info=True)
        super().on_failure(exc, task_id, args, kwargs, einfo)

@shared_task(bind=True, base=CodeReviewTask, max_retries=2)
def process_file_task(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single file in a distributed worker.
    
    Args:
        file_path: Path to the file to process
        config: Configuration dictionary
        
    Returns:
        Dictionary with file analysis results
    """
    try:
        from workers.file_processor import FileProcessor
        
        # Create temporary file processor for this task
        file_processor = FileProcessor(config=config)
        
        # Process the file with memory-efficient streaming
        result = file_processor.process_file_streaming(file_path)
        
        return {
            "file_path": file_path,
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
        # Retry for certain errors, but not for all
        if isinstance(e, (OSError, IOError)):
            # Retry system-related errors
            raise self.retry(exc=e, countdown=5, max_retries=2)
        
        # For other errors, return error status
        return {
            "file_path": file_path,
            "status": "error",
            "error": str(e)
        }

@shared_task(bind=True, base=CodeReviewTask, max_retries=1)
def analyze_chunk_task(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a code chunk with LLM in a distributed worker.
    
    Args:
        chunk_data: Dictionary with chunk information and context
            {
                "chunk": str,
                "file_path": str, 
                "dependencies": str,
                "flake8_output": str,
                "plugin_data": dict,
                "config": dict
            }
            
    Returns:
        Dictionary with chunk analysis results
    """
    try:
        # Import here to avoid circular imports
        from workers.analyzer.llm_client import create_llm_client
        
        # Extract data from chunk_data
        chunk = chunk_data.get("chunk", "")
        file_path = chunk_data.get("file_path", "")
        dependencies = chunk_data.get("dependencies", "")
        flake8_output = chunk_data.get("flake8_output", "")
        plugin_data = chunk_data.get("plugin_data", {})
        config = chunk_data.get("config", {})
        build_prompt_func = chunk_data.get("build_prompt_func_name")
        
        # Create LLM client for this task
        llm_client = create_llm_client(config)
        
        # Build prompt (use enhanced code analyzer's build_prompt method)
        # We use a named function to avoid serialization issues
        if build_prompt_func == "build_enhanced_prompt":
            from workers.analyzer import CodeReviewController
            # Create temporary analyzer just for building the prompt
            temp_analyzer = CodeReviewController(config)
            prompt = temp_analyzer.build_enhanced_prompt(
                file_path=file_path,
                code_chunk=chunk,
                dependencies=dependencies,
                flake8_output=flake8_output,
                plugin_data=plugin_data
            )
        else:
            # Default basic prompt if function name is not recognized
            prompt = f"Review the following code from {file_path}:\n\n{chunk}"
        
        # Call the LLM API
        result = llm_client.call_api(prompt, context=f"distributed analysis of {file_path}")
        
        # Return the result
        return {
            "status": "completed",
            "result": result,
            "file_path": file_path
        }
        
    except APIError as e:
        logger.error(f"API error analyzing chunk for {chunk_data.get('file_path', 'unknown')}: {e}")
        # Retry API errors with exponential backoff
        raise self.retry(exc=e, countdown=30, max_retries=1)
        
    except Exception as e:
        logger.error(f"Error analyzing chunk: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "file_path": chunk_data.get("file_path", "unknown")
        }

@shared_task(bind=True, base=CodeReviewTask)
def run_code_review_task(self, task_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a complete code review as a distributed task.
    
    Args:
        task_id: Unique identifier for this review task
        config: Configuration dictionary
    
    Returns:
        Dictionary with review results summary
    """
    try:
        from workers.analyzer import CodeReviewController
        from core.metrics import MetricsCollector
        
        # Create metrics collector for this task
        metrics = MetricsCollector()
        metrics.start_review(task_id)
        
        # Initialize the code review controller with the provided config
        analyzer = CodeReviewController(
            config=config,
            metrics_collector=metrics,
            use_distributed=True  # Enable distributed mode
        )
        
        # Run the code review
        results = analyzer.run_code_review()
        
        # Finalize metrics
        metrics.complete_review(
            file_count=results.get("files_analyzed", 0),
            issue_count=results.get("total_issues", 0)
        )
        
        # Add metrics to results
        results["performance_metrics"] = metrics.get_performance_metrics()
        
        return results
        
    except Exception as e:
        logger.error(f"Error running distributed code review: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "task_id": task_id
        }