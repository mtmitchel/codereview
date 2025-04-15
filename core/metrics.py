"""
Metrics collection system for Code Review Tool.

This module provides utilities for collecting, aggregating, and reporting
various metrics about code reviews and tool performance.
"""

import logging
import time
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from collections import defaultdict

from core.events import get_event_bus, EventEmitter
from core.events import EVENT_ANALYSIS_COMPLETED, EVENT_ANALYSIS_ERROR, EVENT_FILE_PROCESSED

logger = logging.getLogger("CodeReviewTool.Metrics")

class MetricsCollector(EventEmitter):
    """
    Collects and aggregates metrics about code reviews and tool performance.
    """
    
    def __init__(self, metrics_dir: Optional[str] = None):
        """
        Initialize the metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics data (None for no persistence)
        """
        super().__init__()
        self.metrics_dir = metrics_dir
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        
        self.lock = threading.RLock()
        
        # Performance metrics
        self.performance_metrics = {
            'analysis_time_total': 0.0,  # Total time spent on analysis
            'analysis_time_by_file': {},  # Analysis time by file
            'llm_api_calls': 0,  # Number of LLM API calls
            'llm_api_time_total': 0.0,  # Total time spent on LLM API calls
            'llm_token_usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
            },
            'processing_stages': {}  # Time spent in each processing stage
        }
        
        # Usage metrics
        self.usage_metrics = {
            'files_analyzed': 0,  # Number of files analyzed
            'files_by_language': {},  # Files analyzed by language
            'lines_analyzed': 0,  # Total lines of code analyzed
            'issues_found': 0,  # Total issues found
            'issues_by_severity': {
                'Critical': 0,
                'High': 0,
                'Medium': 0,
                'Low': 0,
                'Info': 0,
            },
            'issues_by_category': {},  # Issues by category
            'analysis_counts': 0,  # Number of analyses run
            'analysis_counts_by_date': {},  # Analyses by date
        }
        
        # Last update timestamp
        self.last_update = time.time()
        
        # Set up event subscriptions
        self._setup_event_listeners()
        
        logger.debug("Initialized metrics collector")
    
    def _setup_event_listeners(self) -> None:
        """Set up event listeners for metrics collection."""
        event_bus = get_event_bus()
        
        # Subscribe to relevant events
        event_bus.subscribe(EVENT_ANALYSIS_COMPLETED, self._on_analysis_completed)
        event_bus.subscribe(EVENT_ANALYSIS_ERROR, self._on_analysis_error)
        event_bus.subscribe(EVENT_FILE_PROCESSED, self._on_file_processed)
        
        logger.debug("Set up metrics event listeners")
    
    def _on_analysis_completed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle analysis completed event.
        
        Args:
            event_data: Event data
        """
        with self.lock:
            # Update analysis counts
            self.usage_metrics['analysis_counts'] += 1
            
            # Update analysis counts by date
            today = datetime.now().strftime('%Y-%m-%d')
            self.usage_metrics['analysis_counts_by_date'][today] = \
                self.usage_metrics['analysis_counts_by_date'].get(today, 0) + 1
            
            # Update analysis time
            if 'analysis_time' in event_data:
                self.performance_metrics['analysis_time_total'] += event_data['analysis_time']
            
            # Update token usage
            if 'token_usage' in event_data:
                token_usage = event_data['token_usage']
                self.performance_metrics['llm_token_usage']['prompt_tokens'] += token_usage.get('prompt_tokens', 0)
                self.performance_metrics['llm_token_usage']['completion_tokens'] += token_usage.get('completion_tokens', 0)
                self.performance_metrics['llm_token_usage']['total_tokens'] += token_usage.get('total_tokens', 0)
    
    def _on_analysis_error(self, event_data: Dict[str, Any]) -> None:
        """
        Handle analysis error event.
        
        Args:
            event_data: Event data
        """
        # Could track error types or frequencies here
        pass
    
    def _on_file_processed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle file processed event.
        
        Args:
            event_data: Event data
        """
        with self.lock:
            self.usage_metrics['files_analyzed'] += 1
            
            # Update files by language
            language = event_data.get('language', 'unknown')
            self.usage_metrics['files_by_language'][language] = \
                self.usage_metrics['files_by_language'].get(language, 0) + 1
            
            # Update lines analyzed
            if 'lines' in event_data:
                self.usage_metrics['lines_analyzed'] += event_data['lines']
            
            # Update issues
            if 'issues' in event_data:
                issues = event_data['issues']
                self.usage_metrics['issues_found'] += len(issues)
                
                # Update issues by severity
                for issue in issues:
                    severity = issue.get('severity', 'Medium')
                    if severity in self.usage_metrics['issues_by_severity']:
                        self.usage_metrics['issues_by_severity'][severity] += 1
                    
                    # Update issues by category
                    category = issue.get('category', 'Uncategorized')
                    self.usage_metrics['issues_by_category'][category] = \
                        self.usage_metrics['issues_by_category'].get(category, 0) + 1
            
            # Update analysis time by file
            if 'file_path' in event_data and 'analysis_time' in event_data:
                file_path = event_data['file_path']
                analysis_time = event_data['analysis_time']
                self.performance_metrics['analysis_time_by_file'][file_path] = analysis_time
    
    def update_llm_metrics(self, 
                          api_call_time: float, 
                          prompt_tokens: int = 0, 
                          completion_tokens: int = 0) -> None:
        """
        Update LLM API call metrics.
        
        Args:
            api_call_time: Time spent on the API call
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
        """
        with self.lock:
            self.performance_metrics['llm_api_calls'] += 1
            self.performance_metrics['llm_api_time_total'] += api_call_time
            self.performance_metrics['llm_token_usage']['prompt_tokens'] += prompt_tokens
            self.performance_metrics['llm_token_usage']['completion_tokens'] += completion_tokens
            self.performance_metrics['llm_token_usage']['total_tokens'] += prompt_tokens + completion_tokens
    
    def update_stage_metrics(self, stage_name: str, execution_time: float) -> None:
        """
        Update processing stage metrics.
        
        Args:
            stage_name: Name of the processing stage
            execution_time: Time spent in the stage
        """
        with self.lock:
            if stage_name not in self.performance_metrics['processing_stages']:
                self.performance_metrics['processing_stages'][stage_name] = {
                    'total_time': 0.0,
                    'call_count': 0,
                    'avg_time': 0.0
                }
            
            self.performance_metrics['processing_stages'][stage_name]['total_time'] += execution_time
            self.performance_metrics['processing_stages'][stage_name]['call_count'] += 1
            
            # Update average time
            call_count = self.performance_metrics['processing_stages'][stage_name]['call_count']
            total_time = self.performance_metrics['processing_stages'][stage_name]['total_time']
            self.performance_metrics['processing_stages'][stage_name]['avg_time'] = total_time / call_count
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            metrics = self.performance_metrics.copy()
            
            # Add derived metrics
            if metrics['llm_api_calls'] > 0:
                metrics['avg_llm_api_time'] = metrics['llm_api_time_total'] / metrics['llm_api_calls']
            
            if self.usage_metrics['files_analyzed'] > 0:
                metrics['avg_analysis_time_per_file'] = metrics['analysis_time_total'] / self.usage_metrics['files_analyzed']
            
            return metrics
    
    def get_usage_metrics(self) -> Dict[str, Any]:
        """
        Get usage metrics.
        
        Returns:
            Dictionary with usage metrics
        """
        with self.lock:
            metrics = self.usage_metrics.copy()
            
            # Add derived metrics
            if metrics['files_analyzed'] > 0:
                metrics['avg_issues_per_file'] = metrics['issues_found'] / metrics['files_analyzed']
                metrics['avg_lines_per_file'] = metrics['lines_analyzed'] / metrics['files_analyzed']
            
            return metrics
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            'performance': self.get_performance_metrics(),
            'usage': self.get_usage_metrics(),
            'timestamp': time.time()
        }
    
    def save_metrics(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Save metrics to a file.
        
        Args:
            filename: Optional filename (uses timestamp if not provided)
            
        Returns:
            Path to the saved file or None if saving failed
        """
        if not self.metrics_dir:
            logger.warning("No metrics directory configured. Metrics not saved.")
            return None
        
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"metrics_{timestamp}.json"
            
            filepath = os.path.join(self.metrics_dir, filename)
            
            # Get all metrics
            metrics = self.get_all_metrics()
            
            # Add metadata
            metrics['metadata'] = {
                'timestamp': time.time(),
                'date': datetime.now().isoformat()
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Saved metrics to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return None
    
    def reset_metrics(self) -> None:
        """Reset all metrics to initial values."""
        with self.lock:
            # Save current metrics before resetting
            if self.metrics_dir:
                self.save_metrics()
            
            # Reset performance metrics
            self.performance_metrics = {
                'analysis_time_total': 0.0,
                'analysis_time_by_file': {},
                'llm_api_calls': 0,
                'llm_api_time_total': 0.0,
                'llm_token_usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                },
                'processing_stages': {}
            }
            
            # Reset usage metrics (but keep analysis_counts_by_date)
            analysis_counts_by_date = self.usage_metrics['analysis_counts_by_date'].copy()
            analysis_counts = self.usage_metrics['analysis_counts']
            
            self.usage_metrics = {
                'files_analyzed': 0,
                'files_by_language': {},
                'lines_analyzed': 0,
                'issues_found': 0,
                'issues_by_severity': {
                    'Critical': 0,
                    'High': 0,
                    'Medium': 0,
                    'Low': 0,
                    'Info': 0,
                },
                'issues_by_category': {},
                'analysis_counts': analysis_counts,
                'analysis_counts_by_date': analysis_counts_by_date,
            }
            
            logger.info("Reset metrics")
    
    def get_statistics_report(self) -> str:
        """
        Generate a human-readable statistics report.
        
        Returns:
            Markdown formatted statistics report
        """
        metrics = self.get_all_metrics()
        performance = metrics['performance']
        usage = metrics['usage']
        
        report = "# Code Review Tool Statistics Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Analysis summary
        report += "## Analysis Summary\n\n"
        report += f"- Total analyses run: {usage['analysis_counts']}\n"
        report += f"- Files analyzed: {usage['files_analyzed']}\n"
        report += f"- Lines of code analyzed: {usage['lines_analyzed']}\n"
        report += f"- Issues found: {usage['issues_found']}\n"
        
        if usage['files_analyzed'] > 0:
            report += f"- Average issues per file: {usage['issues_found'] / usage['files_analyzed']:.2f}\n"
        
        # Performance metrics
        report += "\n## Performance Metrics\n\n"
        report += f"- Total analysis time: {performance['analysis_time_total']:.2f} seconds\n"
        
        if usage['files_analyzed'] > 0:
            report += f"- Average time per file: {performance['analysis_time_total'] / usage['files_analyzed']:.2f} seconds\n"
        
        report += f"- LLM API calls: {performance['llm_api_calls']}\n"
        report += f"- Total LLM API time: {performance['llm_api_time_total']:.2f} seconds\n"
        
        if performance['llm_api_calls'] > 0:
            report += f"- Average LLM API time: {performance['llm_api_time_total'] / performance['llm_api_calls']:.2f} seconds\n"
        
        # Token usage
        token_usage = performance['llm_token_usage']
        report += "\n## Token Usage\n\n"
        report += f"- Prompt tokens: {token_usage['prompt_tokens']}\n"
        report += f"- Completion tokens: {token_usage['completion_tokens']}\n"
        report += f"- Total tokens: {token_usage['total_tokens']}\n"
        
        # Issues by severity
        report += "\n## Issues by Severity\n\n"
        for severity, count in usage['issues_by_severity'].items():
            report += f"- {severity}: {count}\n"
        
        # Issues by category
        if usage['issues_by_category']:
            report += "\n## Issues by Category\n\n"
            for category, count in sorted(usage['issues_by_category'].items(), key=lambda x: x[1], reverse=True):
                report += f"- {category}: {count}\n"
        
        # Files by language
        if usage['files_by_language']:
            report += "\n## Files by Language\n\n"
            for language, count in sorted(usage['files_by_language'].items(), key=lambda x: x[1], reverse=True):
                report += f"- {language}: {count}\n"
        
        # Processing stages
        if performance['processing_stages']:
            report += "\n## Processing Stage Performance\n\n"
            report += "| Stage | Total Time (s) | Calls | Avg Time (s) |\n"
            report += "|-------|---------------|-------|-------------|\n"
            
            for stage, data in sorted(performance['processing_stages'].items(), key=lambda x: x[1]['total_time'], reverse=True):
                report += f"| {stage} | {data['total_time']:.2f} | {data['call_count']} | {data['avg_time']:.2f} |\n"
        
        return report


# Global metrics collector instance
_global_metrics_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.
    
    Returns:
        Global MetricsCollector instance
    """
    return _global_metrics_collector


def configure_metrics_collector(metrics_dir: Optional[str] = None) -> None:
    """
    Configure the global metrics collector.
    
    Args:
        metrics_dir: Directory to store metrics data
    """
    global _global_metrics_collector
    _global_metrics_collector = MetricsCollector(metrics_dir)
    logger.info(f"Configured metrics collector with directory: {metrics_dir}")