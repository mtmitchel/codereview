"""
Pipeline module for Code Review Tool.

This module provides a flexible pipeline pattern implementation
for creating modular and reusable workflows.
"""

import logging
import uuid
from typing import Dict, List, Any, Callable, Optional, Generic, TypeVar, cast, Union

from core.events import EventEmitter, get_event_bus
from core.exceptions import PipelineExecutionError

logger = logging.getLogger("CodeReviewTool.Pipeline")

# Type for context and result
T = TypeVar('T')
U = TypeVar('U')

# Type for middleware functions
PipelineMiddleware = Callable[[Dict[str, Any], Callable], Dict[str, Any]]
PipelineStep = Callable[[Dict[str, Any]], Dict[str, Any]]


class Pipeline(EventEmitter, Generic[T, U]):
    """
    A flexible pipeline that processes data through a series of steps.
    Implements the pipeline pattern to create modular, reusable workflows.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize a pipeline.
        
        Args:
            name: Optional name for the pipeline
        """
        super().__init__()
        self.name = name or f"Pipeline-{uuid.uuid4().hex[:8]}"
        self.steps: List[PipelineStep] = []
        self.middleware: List[PipelineMiddleware] = []
        self.error_handlers: List[Callable[[Exception, Dict[str, Any]], None]] = []
        
        # Set event source to pipeline name
        self.set_event_source(self.name)
        
        logger.debug(f"Initialized pipeline: {self.name}")
    
    def add_step(self, step: PipelineStep) -> 'Pipeline[T, U]':
        """
        Add a processing step to the pipeline.
        
        Args:
            step: Function that takes a context dict and returns a modified context
            
        Returns:
            The pipeline instance for method chaining
        """
        self.steps.append(step)
        logger.debug(f"Added step to pipeline {self.name}: {step.__name__ if hasattr(step, '__name__') else 'anonymous'}")
        return self
    
    def add_steps(self, steps: List[PipelineStep]) -> 'Pipeline[T, U]':
        """
        Add multiple processing steps to the pipeline.
        
        Args:
            steps: List of step functions
            
        Returns:
            The pipeline instance for method chaining
        """
        self.steps.extend(steps)
        logger.debug(f"Added {len(steps)} steps to pipeline {self.name}")
        return self
    
    def use_middleware(self, middleware: PipelineMiddleware) -> 'Pipeline[T, U]':
        """
        Add middleware to the pipeline.
        
        Middleware wraps the entire pipeline execution and can perform
        cross-cutting concerns like logging, timing, etc.
        
        Args:
            middleware: Middleware function
            
        Returns:
            The pipeline instance for method chaining
        """
        self.middleware.append(middleware)
        logger.debug(f"Added middleware to pipeline {self.name}: {middleware.__name__ if hasattr(middleware, '__name__') else 'anonymous'}")
        return self
    
    def on_error(self, handler: Callable[[Exception, Dict[str, Any]], None]) -> 'Pipeline[T, U]':
        """
        Add an error handler to the pipeline.
        
        Args:
            handler: Function that takes an exception and context
            
        Returns:
            The pipeline instance for method chaining
        """
        self.error_handlers.append(handler)
        logger.debug(f"Added error handler to pipeline {self.name}")
        return self
    
    def _wrapped_pipeline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline steps on the context.
        
        Args:
            context: Input context dictionary
            
        Returns:
            Modified context after all steps
        """
        current_context = context.copy()
        
        # Emit pipeline started event
        self.emit_event("pipeline.started", {
            "pipeline": self.name,
            "context": {k: v for k, v in current_context.items() if isinstance(v, (str, int, float, bool, type(None)))}
        })
        
        try:
            # Execute all steps in sequence
            for i, step in enumerate(self.steps):
                step_name = step.__name__ if hasattr(step, '__name__') else f"step_{i}"
                
                # Add step tracking to context
                current_context["_current_step"] = step_name
                current_context["_step_index"] = i
                current_context["_total_steps"] = len(self.steps)
                
                # Emit step started event
                self.emit_event("pipeline.step.started", {
                    "pipeline": self.name,
                    "step": step_name,
                    "step_index": i,
                    "total_steps": len(self.steps)
                })
                
                try:
                    # Execute the step
                    logger.debug(f"Executing pipeline step {i+1}/{len(self.steps)}: {step_name}")
                    current_context = step(current_context)
                    
                    # Emit step completed event
                    self.emit_event("pipeline.step.completed", {
                        "pipeline": self.name,
                        "step": step_name,
                        "step_index": i,
                        "total_steps": len(self.steps)
                    })
                    
                except Exception as e:
                    # Emit step error event
                    self.emit_event("pipeline.step.error", {
                        "pipeline": self.name,
                        "step": step_name,
                        "step_index": i,
                        "error": str(e)
                    })
                    raise PipelineExecutionError(f"Error in pipeline step {step_name}: {e}") from e
        
        except Exception as e:
            # Handle the error
            for handler in self.error_handlers:
                try:
                    handler(e, current_context)
                except Exception as handler_error:
                    logger.error(f"Error in pipeline error handler: {handler_error}")
            
            # Emit pipeline error event
            self.emit_event("pipeline.error", {
                "pipeline": self.name,
                "error": str(e),
                "step": current_context.get("_current_step")
            })
            
            # Re-raise the error
            raise
        
        # Emit pipeline completed event
        self.emit_event("pipeline.completed", {
            "pipeline": self.name
        })
        
        return current_context
    
    def execute(self, context: T) -> U:
        """
        Execute the pipeline with the given context.
        
        Args:
            context: Input data
            
        Returns:
            Processed result
            
        Raises:
            PipelineExecutionError: If an error occurs during execution
        """
        # Create initial context dictionary
        initial_context = {"input": context, "result": None}
        
        # Apply middleware (in reverse order for proper nesting)
        pipeline_executor = self._wrapped_pipeline
        for middleware in reversed(self.middleware):
            # Each middleware wraps the previous pipeline executor
            pipeline_executor = lambda ctx, m=middleware, p=pipeline_executor: m(ctx, p)
        
        try:
            # Execute the pipeline with all middleware applied
            final_context = pipeline_executor(initial_context)
            return cast(U, final_context.get("result", final_context))
        except Exception as e:
            if not isinstance(e, PipelineExecutionError):
                # Wrap the exception if it's not already a pipeline error
                raise PipelineExecutionError(f"Pipeline {self.name} execution failed: {e}") from e
            raise
    
    def compose(self, other_pipeline: 'Pipeline') -> 'Pipeline[T, U]':
        """
        Compose this pipeline with another one.
        
        Args:
            other_pipeline: Another pipeline to compose with
            
        Returns:
            A new pipeline that combines both pipelines
        """
        composed_pipeline = Pipeline(f"{self.name}+{other_pipeline.name}")
        composed_pipeline.steps = self.steps + other_pipeline.steps
        composed_pipeline.middleware = self.middleware + other_pipeline.middleware
        composed_pipeline.error_handlers = self.error_handlers + other_pipeline.error_handlers
        
        logger.debug(f"Composed pipeline {self.name} with {other_pipeline.name}")
        return composed_pipeline
    
    def __call__(self, context: T) -> U:
        """
        Make the pipeline callable directly.
        
        Args:
            context: Input data
            
        Returns:
            Processed result
        """
        return self.execute(context)


def create_step(name: str = None) -> Callable[[Callable[[Dict[str, Any]], Dict[str, Any]]], PipelineStep]:
    """
    Decorator to create a named pipeline step.
    
    Args:
        name: Name for the step (defaults to function name)
    
    Returns:
        Decorator function
    
    Example:
        @create_step("process_files")
        def process_files_step(context):
            # Process files
            return context
    """
    def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> PipelineStep:
        func.__name__ = name or func.__name__
        return func
    
    return decorator


# Common middleware functions

def logging_middleware(context: Dict[str, Any], next_middleware: Callable) -> Dict[str, Any]:
    """
    Middleware that logs pipeline execution.
    
    Args:
        context: Pipeline context
        next_middleware: Next middleware in the chain
        
    Returns:
        Updated context after pipeline execution
    """
    logger.info(f"Starting pipeline execution with context: {context.get('_current_step', 'initial')}")
    
    try:
        result = next_middleware(context)
        logger.info(f"Pipeline execution completed successfully")
        return result
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise
    

def timing_middleware(context: Dict[str, Any], next_middleware: Callable) -> Dict[str, Any]:
    """
    Middleware that times pipeline execution.
    
    Args:
        context: Pipeline context
        next_middleware: Next middleware in the chain
        
    Returns:
        Updated context after pipeline execution with timing info
    """
    import time
    
    start_time = time.time()
    
    try:
        result = next_middleware(context)
        
        # Add timing information to result
        execution_time = time.time() - start_time
        result["_execution_time"] = execution_time
        
        logger.debug(f"Pipeline execution took {execution_time:.2f} seconds")
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        logger.debug(f"Pipeline execution failed after {execution_time:.2f} seconds: {e}")
        raise


def checkpoint_middleware(context: Dict[str, Any], next_middleware: Callable) -> Dict[str, Any]:
    """
    Middleware that creates checkpoints during pipeline execution.
    
    Args:
        context: Pipeline context
        next_middleware: Next middleware in the chain
        
    Returns:
        Updated context after pipeline execution
    """
    from core.events import get_event_bus, EVENT_CHECKPOINT_CREATED
    
    # Create initial checkpoint
    checkpoint_id = str(uuid.uuid4())
    context["_checkpoints"] = context.get("_checkpoints", {})
    context["_checkpoints"]["initial"] = {
        "id": checkpoint_id,
        "timestamp": time.time(),
        "context": context.copy()
    }
    
    # Emit checkpoint created event
    get_event_bus().publish(EVENT_CHECKPOINT_CREATED, {
        "checkpoint_id": checkpoint_id,
        "checkpoint_type": "initial",
        "pipeline_context": {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool, type(None)))}
    })
    
    try:
        # Execute the pipeline
        result = next_middleware(context)
        
        # Create final checkpoint
        checkpoint_id = str(uuid.uuid4())
        result["_checkpoints"] = result.get("_checkpoints", {})
        result["_checkpoints"]["final"] = {
            "id": checkpoint_id,
            "timestamp": time.time(),
            "context": result.copy()
        }
        
        # Emit final checkpoint created event
        get_event_bus().publish(EVENT_CHECKPOINT_CREATED, {
            "checkpoint_id": checkpoint_id,
            "checkpoint_type": "final",
            "pipeline_context": {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, type(None)))}
        })
        
        return result
    except Exception as e:
        # Create error checkpoint
        checkpoint_id = str(uuid.uuid4())
        context["_checkpoints"] = context.get("_checkpoints", {})
        context["_checkpoints"]["error"] = {
            "id": checkpoint_id,
            "timestamp": time.time(),
            "context": context.copy(),
            "error": str(e)
        }
        
        # Emit error checkpoint created event
        get_event_bus().publish(EVENT_CHECKPOINT_CREATED, {
            "checkpoint_id": checkpoint_id,
            "checkpoint_type": "error",
            "pipeline_context": {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool, type(None)))},
            "error": str(e)
        })
        
        raise


# Pipeline factory functions

def create_pipeline(name: str = None) -> Pipeline:
    """
    Create a new pipeline.
    
    Args:
        name: Optional name for the pipeline
        
    Returns:
        A new Pipeline instance
    """
    return Pipeline(name)


def create_logging_pipeline(name: str = None) -> Pipeline:
    """
    Create a new pipeline with logging middleware.
    
    Args:
        name: Optional name for the pipeline
        
    Returns:
        A new Pipeline instance with logging middleware
    """
    pipeline = Pipeline(name)
    pipeline.use_middleware(logging_middleware)
    return pipeline


def create_timed_pipeline(name: str = None) -> Pipeline:
    """
    Create a new pipeline with timing middleware.
    
    Args:
        name: Optional name for the pipeline
        
    Returns:
        A new Pipeline instance with timing middleware
    """
    pipeline = Pipeline(name)
    pipeline.use_middleware(timing_middleware)
    return pipeline


def create_checkpoint_pipeline(name: str = None) -> Pipeline:
    """
    Create a new pipeline with checkpoint middleware.
    
    Args:
        name: Optional name for the pipeline
        
    Returns:
        A new Pipeline instance with checkpoint middleware
    """
    pipeline = Pipeline(name)
    pipeline.use_middleware(checkpoint_middleware)
    return pipeline
