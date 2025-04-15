"""
Distributed processing module for Code Review Tool.

This module provides distributed task processing capabilities using Celery and Redis,
allowing the tool to scale for very large codebases by distributing work across
multiple processes or machines.
"""

from .celery_app import app
from .tasks import process_file_task, analyze_chunk_task, run_code_review_task

__all__ = [
    'app',
    'process_file_task',
    'analyze_chunk_task',
    'run_code_review_task'
]