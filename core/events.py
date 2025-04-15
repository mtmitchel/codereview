"""
Event system implementation for Code Review Tool.

This module provides a centralized event system for inter-component
communication, implementing the observer pattern.
"""

import logging
import threading
import time
import queue
import uuid
from typing import Dict, List, Any, Callable, Optional, Set, TypeVar, Generic

logger = logging.getLogger("CodeReviewTool.Events")

# Type for event handlers
EventHandler = Callable[[Dict[str, Any]], None]


class Event:
    """Represents an event with metadata and payload."""
    
    def __init__(self, event_type: str, data: Dict[str, Any], source: str = None):
        """
        Initialize an event.
        
        Args:
            event_type: Type of the event
            data: Event data/payload
            source: Source of the event
        """
        self.event_type = event_type
        self.data = data
        self.source = source
        self.id = str(uuid.uuid4())
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        return f"Event[{self.event_type}] from {self.source} at {self.timestamp}"


class EventBus:
    """
    Central event bus for publishing and subscribing to events.
    Implements the observer pattern for loose coupling between components.
    """
    
    def __init__(self, async_dispatch: bool = True):
        """
        Initialize the event bus.
        
        Args:
            async_dispatch: Whether to dispatch events asynchronously
        """
        self.subscribers: Dict[str, List[EventHandler]] = {}
        self.global_subscribers: List[EventHandler] = []
        self.lock = threading.RLock()
        
        # For async dispatch
        self.async_dispatch = async_dispatch
        self.event_queue = queue.Queue()
        self.dispatch_thread = None
        self.running = False
        
        if async_dispatch:
            self._start_dispatch_thread()
            
        logger.debug("Initialized event bus")
    
    def _start_dispatch_thread(self) -> None:
        """Start the async dispatch thread."""
        self.running = True
        self.dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="EventDispatchThread",
            daemon=True
        )
        self.dispatch_thread.start()
        logger.debug("Started event dispatch thread")
    
    def _dispatch_loop(self) -> None:
        """Event dispatch loop for async dispatching."""
        while self.running:
            try:
                # Wait for events with timeout to allow clean shutdown
                try:
                    event, handlers = self.event_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process the event
                self._dispatch_to_handlers(event, handlers)
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in event dispatch loop: {e}")
                # Brief sleep to avoid tight error loop
                time.sleep(0.1)
    
    def _dispatch_to_handlers(self, event: Event, handlers: List[EventHandler]) -> None:
        """
        Dispatch an event to the given handlers.
        
        Args:
            event: Event to dispatch
            handlers: List of handlers to notify
        """
        for handler in handlers:
            try:
                # Create event data dictionary with metadata
                event_data = {
                    'type': event.event_type,
                    'id': event.id,
                    'timestamp': event.timestamp,
                    'source': event.source,
                    **event.data
                }
                
                # Call the handler
                handler(event_data)
                
            except Exception as e:
                logger.error(f"Error calling event handler for {event.event_type}: {e}")
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Handler function to call when event occurs
        """
        with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            
            if handler not in self.subscribers[event_type]:
                self.subscribers[event_type].append(handler)
                logger.debug(f"Added subscriber for event: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Unsubscribe from a specific event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler function to remove
        """
        with self.lock:
            if event_type in self.subscribers and handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
                logger.debug(f"Removed subscriber for event: {event_type}")
    
    def subscribe_to_all(self, handler: EventHandler) -> None:
        """
        Subscribe to all events.
        
        Args:
            handler: Handler function to call for all events
        """
        with self.lock:
            if handler not in self.global_subscribers:
                self.global_subscribers.append(handler)
                logger.debug("Added global event subscriber")
    
    def unsubscribe_from_all(self, handler: EventHandler) -> None:
        """
        Unsubscribe from all events.
        
        Args:
            handler: Handler function to remove
        """
        with self.lock:
            if handler in self.global_subscribers:
                self.global_subscribers.remove(handler)
                logger.debug("Removed global event subscriber")
    
    def publish(self, event_type: str, data: Dict[str, Any], source: str = None) -> str:
        """
        Publish an event.
        
        Args:
            event_type: Type of event to publish
            data: Event data
            source: Source of the event
            
        Returns:
            Event ID
        """
        event = Event(event_type, data, source)
        
        with self.lock:
            # Get specific handlers for this event type
            specific_handlers = self.subscribers.get(event_type, []).copy()
            
            # Add global handlers
            all_handlers = specific_handlers + self.global_subscribers.copy()
            
            if not all_handlers:
                # No handlers for this event
                logger.debug(f"No handlers for event: {event_type}")
                return event.id
            
            if self.async_dispatch:
                # Queue the event for async dispatch
                self.event_queue.put((event, all_handlers))
            else:
                # Dispatch the event immediately
                self._dispatch_to_handlers(event, all_handlers)
        
        logger.debug(f"Published event: {event_type} (id: {event.id})")
        return event.id
    
    def shutdown(self) -> None:
        """Shutdown the event bus."""
        if self.async_dispatch and self.running:
            self.running = False
            
            # Wait for dispatch thread to exit
            if self.dispatch_thread and self.dispatch_thread.is_alive():
                self.dispatch_thread.join(timeout=2.0)
            
            logger.debug("Event bus shutdown")


class EventEmitter:
    """
    Mixin class that provides event publishing capabilities to a class.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the event emitter.
        
        Args:
            event_bus: Event bus to use
        """
        self._event_bus = event_bus or _global_event_bus
        self._event_source = self.__class__.__name__
    
    def emit_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Emit an event.
        
        Args:
            event_type: Type of event to emit
            data: Event data
            
        Returns:
            Event ID
        """
        return self._event_bus.publish(event_type, data, source=self._event_source)
    
    def set_event_source(self, source: str) -> None:
        """
        Set the event source name.
        
        Args:
            source: Source name
        """
        self._event_source = source


# Global event bus instance
_global_event_bus = EventBus()

def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.
    
    Returns:
        Global EventBus instance
    """
    return _global_event_bus


# Standard event types
EVENT_ANALYSIS_STARTED = "analysis.started"
EVENT_ANALYSIS_COMPLETED = "analysis.completed"
EVENT_ANALYSIS_ERROR = "analysis.error"
EVENT_ANALYSIS_PROGRESS = "analysis.progress"
EVENT_FILE_PROCESSED = "file.processed"
EVENT_CONFIG_CHANGED = "config.changed"
EVENT_RESOURCE_WARNING = "resource.warning"
EVENT_UI_ACTION = "ui.action"
EVENT_PLUGIN_LOADED = "plugin.loaded"
EVENT_CHECKPOINT_CREATED = "checkpoint.created"
EVENT_ERROR_OCCURRED = "error.occurred"