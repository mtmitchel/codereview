"""
Command Pattern implementation for Code Review Tool.

This module implements the Command Pattern to enable undo/redo functionality,
action history tracking, and complex operation sequencing.
"""

import logging
import time
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, TypeVar, Generic, cast
from enum import Enum
import pickle
import threading
from utils.error_utils import ErrorHandler

from core.events import EventEmitter, get_event_bus

logger = logging.getLogger("CodeReviewTool.Commands")

# Type aliases
T = TypeVar('T')
CommandResult = Dict[str, Any]
CommandParams = Dict[str, Any]


class CommandStatus(Enum):
    """Enum for command execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNDONE = "undone"
    REDONE = "redone"


class Command(ABC, EventEmitter):
    """
    Abstract base class for all commands.
    
    The Command pattern encapsulates a request as an object, allowing:
    - Parameterization of clients with different requests
    - Queuing of requests
    - Logging of requests
    - Undoable operations
    """
    
    def __init__(self, params: Optional[CommandParams] = None):
        """
        Initialize a command.
        
        Args:
            params: Command parameters
        """
        super().__init__()
        self.params = params or {}
        self.id = f"{self.__class__.__name__}_{int(time.time() * 1000)}"
        self.status = CommandStatus.PENDING
        self.result = None
        self.error = None
        self.execution_time = 0
        self.created_at = time.time()
        self.executed_at = None
        self.undone_at = None
        self.redone_at = None
        
        logger.debug(f"Created command {self.id}")
    
    @abstractmethod
    async def execute(self) -> CommandResult:
        """
        Execute the command.
        
        Returns:
            Dictionary with command result
        """
        pass
    
    @abstractmethod
    async def undo(self) -> CommandResult:
        """
        Undo the command.
        
        Returns:
            Dictionary with undo result
        """
        pass
    
    async def redo(self) -> CommandResult:
        """
        Redo the command.
        
        By default, this calls execute again, but can be overridden
        if a command needs special handling for redo operations.
        
        Returns:
            Dictionary with redo result
        """
        self.status = CommandStatus.PENDING
        return await self.execute()
    
    def can_undo(self) -> bool:
        """
        Check if the command can be undone.
        
        Returns:
            True if the command can be undone, False otherwise
        """
        return self.status == CommandStatus.COMPLETED
    
    def can_redo(self) -> bool:
        """
        Check if the command can be redone.
        
        Returns:
            True if the command can be redone, False otherwise
        """
        return self.status == CommandStatus.UNDONE
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the command to a dictionary.
        
        Returns:
            Dictionary representation of the command
        """
        return {
            "id": self.id,
            "class": self.__class__.__name__,
            "params": self.params,
            "status": self.status.value,
            "created_at": self.created_at,
            "executed_at": self.executed_at,
            "undone_at": self.undone_at,
            "redone_at": self.redone_at,
            "execution_time": self.execution_time,
            "result": self.result,
            "error": self.error
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Command':
        """
        Deserialize a command from a dictionary.
        
        Args:
            data: Dictionary representation of the command
            
        Returns:
            Command instance
        """
        # This is implemented by subclasses
        raise NotImplementedError("Deserialization should be implemented by subclasses")
    
    def __str__(self) -> str:
        """Get string representation of the command."""
        return f"{self.__class__.__name__}(id={self.id}, status={self.status.value})"
    
    def __repr__(self) -> str:
        """Get detailed string representation of the command."""
        return f"{self.__class__.__name__}(id={self.id}, status={self.status.value}, params={self.params})"


class CommandRegistry:
    """
    Registry for all available command classes.
    
    This enables command deserialization and dynamic command creation.
    """
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, command_class: type) -> type:
        """
        Register a command class.
        
        Can be used as a decorator:
        
        @CommandRegistry.register
        class MyCommand(Command):
            ...
        
        Args:
            command_class: Command class to register
            
        Returns:
            The registered command class (for decorator usage)
        """
        class_name = command_class.__name__
        cls._registry[class_name] = command_class
        logger.debug(f"Registered command class {class_name}")
        return command_class
    
    @classmethod
    def get_command_class(cls, class_name: str) -> Optional[type]:
        """
        Get a command class by name.
        
        Args:
            class_name: Name of the command class
            
        Returns:
            Command class or None if not found
        """
        return cls._registry.get(class_name)
    
    @classmethod
    def create_command(cls, class_name: str, params: Optional[CommandParams] = None) -> Optional[Command]:
        """
        Create a command instance by class name.
        
        Args:
            class_name: Name of the command class
            params: Command parameters
            
        Returns:
            Command instance or None if class not found
        """
        command_class = cls.get_command_class(class_name)
        if command_class:
            return command_class(params)
        return None
    
    @classmethod
    def deserialize_command(cls, data: Dict[str, Any]) -> Optional[Command]:
        """
        Deserialize a command from a dictionary.
        
        Args:
            data: Dictionary representation of the command
            
        Returns:
            Command instance or None if class not found
        """
        class_name = data.get("class")
        if not class_name:
            return None
        
        command_class = cls.get_command_class(class_name)
        if not command_class:
            return None
        
        try:
            return command_class.deserialize(data)
        except Exception as e:
            logger.error(f"Error deserializing command: {e}")
            return None
    
    @classmethod
    def list_commands(cls) -> List[str]:
        """
        Get a list of all registered command class names.
        
        Returns:
            List of command class names
        """
        return list(cls._registry.keys())


class CommandManager(EventEmitter):
    """
    Manages command execution, history, undo/redo, and serialization.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, max_history: int = 100):
        """
        Create or return the CommandManager singleton instance.
        
        Args:
            max_history: Maximum number of commands to keep in history
            
        Returns:
            CommandManager instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CommandManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, max_history: int = 100):
        """
        Initialize the command manager.
        
        Args:
            max_history: Maximum number of commands to keep in history
        """
        # Only initialize once
        if not hasattr(self, 'initialized'):
            super().__init__()
            self.max_history = max_history
            self.history: List[Command] = []
            self.current_index = -1
            self.event_bus = get_event_bus()
            self.initialized = True
            logger.debug(f"Initialized CommandManager with max_history={max_history}")
    
    async def execute(self, command: Command) -> CommandResult:
        """
        Execute a command and add it to history.
        
        Args:
            command: Command to execute
            
        Returns:
            Command result
        """
        # Truncate history if we're not at the end (undo was used)
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        
        # Add command to history
        self.history.append(command)
        self.current_index = len(self.history) - 1
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.current_index = len(self.history) - 1
        
        # Execute command
        start_time = time.time()
        command.status = CommandStatus.RUNNING
        command.executed_at = time.time()
        
        self.emit_event("command.executing", {
            "command_id": command.id,
            "command_class": command.__class__.__name__,
            "params": command.params
        })
        
        try:
            result = await command.execute()
            command.result = result
            command.status = CommandStatus.COMPLETED
            command.execution_time = time.time() - start_time
            
            self.emit_event("command.executed", {
                "command_id": command.id,
                "command_class": command.__class__.__name__,
                "execution_time": command.execution_time,
                "success": True,
                "result": result
            })
            
            return result
            
        except Exception as e:
            ErrorHandler.handle_error(e, context=f"Executing command {command.id}")
            
            command.error = str(e)
            command.status = CommandStatus.FAILED
            command.execution_time = time.time() - start_time
            
            self.emit_event("command.executed", {
                "command_id": command.id,
                "command_class": command.__class__.__name__,
                "execution_time": command.execution_time,
                "success": False,
                "error": str(e)
            })
            
            # Re-raise the exception
            raise
    
    async def undo(self) -> Optional[CommandResult]:
        """
        Undo the last command.
        
        Returns:
            Undo result or None if nothing to undo
        """
        if self.current_index < 0 or self.current_index >= len(self.history):
            logger.warning("Nothing to undo")
            return None
        
        command = self.history[self.current_index]
        
        if not command.can_undo():
            logger.warning(f"Command {command.id} cannot be undone (status: {command.status.value})")
            return None
        
        self.emit_event("command.undoing", {
            "command_id": command.id,
            "command_class": command.__class__.__name__
        })
        
        try:
            result = await command.undo()
            command.status = CommandStatus.UNDONE
            command.undone_at = time.time()
            
            self.emit_event("command.undone", {
                "command_id": command.id,
                "command_class": command.__class__.__name__,
                "success": True,
                "result": result
            })
            
            # Move current index back
            self.current_index -= 1
            
            return result
            
        except Exception as e:
            ErrorHandler.handle_error(e, context=f"Undoing command {command.id}")
            
            self.emit_event("command.undone", {
                "command_id": command.id,
                "command_class": command.__class__.__name__,
                "success": False,
                "error": str(e)
            })
            
            # Re-raise the exception
            raise
    
    async def redo(self) -> Optional[CommandResult]:
        """
        Redo the next command.
        
        Returns:
            Redo result or None if nothing to redo
        """
        if self.current_index + 1 >= len(self.history):
            logger.warning("Nothing to redo")
            return None
        
        # Move to the next command
        self.current_index += 1
        command = self.history[self.current_index]
        
        if not command.can_redo():
            logger.warning(f"Command {command.id} cannot be redone (status: {command.status.value})")
            self.current_index -= 1  # Move back
            return None
        
        self.emit_event("command.redoing", {
            "command_id": command.id,
            "command_class": command.__class__.__name__
        })
        
        try:
            result = await command.redo()
            command.status = CommandStatus.COMPLETED
            command.redone_at = time.time()
            
            self.emit_event("command.redone", {
                "command_id": command.id,
                "command_class": command.__class__.__name__,
                "success": True,
                "result": result
            })
            
            return result
            
        except Exception as e:
            ErrorHandler.handle_error(e, context=f"Redoing command {command.id}")
            
            self.emit_event("command.redone", {
                "command_id": command.id,
                "command_class": command.__class__.__name__,
                "success": False,
                "error": str(e)
            })
            
            # Re-raise the exception
            raise
    
    def can_undo(self) -> bool:
        """
        Check if undo is available.
        
        Returns:
            True if undo is available, False otherwise
        """
        if self.current_index < 0 or self.current_index >= len(self.history):
            return False
        
        command = self.history[self.current_index]
        return command.can_undo()
    
    def can_redo(self) -> bool:
        """
        Check if redo is available.
        
        Returns:
            True if redo is available, False otherwise
        """
        if self.current_index + 1 >= len(self.history):
            return False
        
        command = self.history[self.current_index + 1]
        return command.can_redo()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the command history.
        
        Returns:
            List of serialized commands
        """
        return [cmd.serialize() for cmd in self.history]
    
    def clear_history(self) -> None:
        """Clear the command history."""
        self.history = []
        self.current_index = -1
        logger.debug("Cleared command history")
    
    def serialize_history(self) -> Dict[str, Any]:
        """
        Serialize the command history to a dictionary.
        
        Returns:
            Dictionary representation of the command history
        """
        return {
            "current_index": self.current_index,
            "commands": [cmd.serialize() for cmd in self.history]
        }
    
    def deserialize_history(self, data: Dict[str, Any]) -> bool:
        """
        Deserialize the command history from a dictionary.
        
        Args:
            data: Dictionary representation of the command history
            
        Returns:
            True if successful, False otherwise
        """
        try:
            commands = []
            for cmd_data in data.get("commands", []):
                cmd = CommandRegistry.deserialize_command(cmd_data)
                if cmd:
                    commands.append(cmd)
            
            self.history = commands
            self.current_index = data.get("current_index", -1)
            
            # Ensure current_index is valid
            if self.current_index >= len(self.history):
                self.current_index = len(self.history) - 1
            
            logger.debug(f"Deserialized {len(self.history)} commands with current_index={self.current_index}")
            return True
            
        except Exception as e:
            ErrorHandler.handle_error(e, context="Deserializing command history")
            return False
    
    def save_history(self, file_path: str) -> bool:
        """
        Save the command history to a file.
        
        Args:
            file_path: Path to the history file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = self.serialize_history()
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Saved command history to {file_path}")
            return True
            
        except Exception as e:
            ErrorHandler.handle_error(e, context=f"Saving command history to {file_path}")
            return False
    
    def load_history(self, file_path: str) -> bool:
        """
        Load the command history from a file.
        
        Args:
            file_path: Path to the history file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"History file not found: {file_path}")
                return False
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            return self.deserialize_history(data)
            
        except Exception as e:
            ErrorHandler.handle_error(e, context=f"Loading command history from {file_path}")
            return False


# Example concrete commands

@CommandRegistry.register
class EditFileCommand(Command):
    """
    Command for editing a file.
    
    This demonstrates a command that can be undone/redone.
    """
    
    def __init__(self, params: Optional[CommandParams] = None):
        """
        Initialize the edit file command.
        
        Args:
            params: Command parameters with file_path, new_content, and original_content (optional)
        """
        super().__init__(params)
        
        # Required parameters
        self.file_path = self.params.get("file_path")
        self.new_content = self.params.get("new_content")
        
        # Optional parameters
        self.original_content = self.params.get("original_content")
        
        if not self.file_path or self.new_content is None:
            raise ValueError("file_path and new_content are required parameters")
    
    async def execute(self) -> CommandResult:
        """
        Execute the edit file command.
        
        Returns:
            Dictionary with command result
        """
        import aiofiles
        import os
        
        # If original content was not provided, read it from the file
        def read_original():
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return ""
        self.original_content = await ErrorHandler.safe_execute(read_original, context=f"Reading original content: {self.file_path}", error_value="")
        
        # Save the new content to the file
        async def save_content():
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(self.new_content)
            return {
                "file_path": self.file_path,
                "success": True,
                "message": f"File edited successfully: {self.file_path}"
            }
        return await ErrorHandler.safe_execute(save_content, context=f"Editing file: {self.file_path}")
    
    async def undo(self) -> CommandResult:
        """
        Undo the edit file command by restoring the original content.
        
        Returns:
            Dictionary with undo result
        """
        import aiofiles
        
        try:
            if self.original_content is None:
                return {
                    "file_path": self.file_path,
                    "success": False,
                    "message": "Cannot undo: original content not available"
                }
            
            # Restore the original content
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(self.original_content)
            
            return {
                "file_path": self.file_path,
                "success": True,
                "message": f"File restored to original state: {self.file_path}"
            }
            
        except Exception as e:
            ErrorHandler.handle_error(e, context=f"Undoing file edit: {self.file_path}")
            raise
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'EditFileCommand':
        """
        Deserialize a command from a dictionary.
        
        Args:
            data: Dictionary representation of the command
            
        Returns:
            EditFileCommand instance
        """
        cmd = cls(data.get("params", {}))
        cmd.id = data.get("id", cmd.id)
        cmd.status = CommandStatus(data.get("status", CommandStatus.PENDING.value))
        cmd.created_at = data.get("created_at", cmd.created_at)
        cmd.executed_at = data.get("executed_at")
        cmd.undone_at = data.get("undone_at")
        cmd.redone_at = data.get("redone_at")
        cmd.execution_time = data.get("execution_time", 0)
        cmd.result = data.get("result")
        cmd.error = data.get("error")
        
        return cmd


@CommandRegistry.register
class ReviewCodeCommand(Command):
    """
    Command for reviewing code with an LLM.
    
    This demonstrates a more complex command that may not be fully undoable.
    """
    
    def __init__(self, params: Optional[CommandParams] = None):
        """
        Initialize the review code command.
        
        Args:
            params: Command parameters with file_paths and review_type
        """
        super().__init__(params)
        
        # Required parameters
        self.file_paths = self.params.get("file_paths", [])
        self.review_type = self.params.get("review_type", "general")
        
        # Optional parameters
        self.llm_provider = self.params.get("llm_provider", "default")
        self.additional_context = self.params.get("additional_context", "")
        
        if not self.file_paths:
            raise ValueError("file_paths is a required parameter")
    
    async def execute(self) -> CommandResult:
        """
        Execute the review code command.
        
        Returns:
            Dictionary with command result
        """
        # This would normally call the code review logic
        # For demonstration purposes, we'll just return a mock result
        
        files_str = ", ".join(self.file_paths)
        
        return {
            "file_paths": self.file_paths,
            "review_type": self.review_type,
            "review_results": f"Mock review results for {files_str}",
            "success": True,
            "message": f"Code review completed for {len(self.file_paths)} files"
        }
    
    async def undo(self) -> CommandResult:
        """
        Undo the review code command.
        
        Since reviewing code doesn't change any state, this is more of a logical undo
        that might just clear results from the UI.
        
        Returns:
            Dictionary with undo result
        """
        return {
            "success": True,
            "message": "Code review results cleared",
            "note": "This is a logical undo since code review doesn't change state"
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'ReviewCodeCommand':
        """
        Deserialize a command from a dictionary.
        
        Args:
            data: Dictionary representation of the command
            
        Returns:
            ReviewCodeCommand instance
        """
        cmd = cls(data.get("params", {}))
        cmd.id = data.get("id", cmd.id)
        cmd.status = CommandStatus(data.get("status", CommandStatus.PENDING.value))
        cmd.created_at = data.get("created_at", cmd.created_at)
        cmd.executed_at = data.get("executed_at")
        cmd.undone_at = data.get("undone_at")
        cmd.redone_at = data.get("redone_at")
        cmd.execution_time = data.get("execution_time", 0)
        cmd.result = data.get("result")
        cmd.error = data.get("error")
        
        return cmd


@CommandRegistry.register
class MacroCommand(Command):
    """
    A command that executes multiple commands as a single transaction.
    
    This implements the Composite pattern to treat a group of commands
    as a single command.
    """
    
    def __init__(self, params: Optional[CommandParams] = None):
        """
        Initialize the macro command.
        
        Args:
            params: Command parameters with commands list
        """
        super().__init__(params)
        
        # Required parameters
        commands_data = self.params.get("commands", [])
        
        # Create command instances
        self.commands: List[Command] = []
        for cmd_data in commands_data:
            class_name = cmd_data.get("class")
            cmd_params = cmd_data.get("params", {})
            
            if class_name:
                cmd = CommandRegistry.create_command(class_name, cmd_params)
                if cmd:
                    self.commands.append(cmd)
        
        if not self.commands:
            raise ValueError("At least one valid command is required")
    
    async def execute(self) -> CommandResult:
        """
        Execute all commands in the macro.
        
        If any command fails, the already executed commands will be undone.
        
        Returns:
            Dictionary with command result
        """
        results = []
        executed_commands = []
        
        try:
            for cmd in self.commands:
                result = await cmd.execute()
                results.append(result)
                executed_commands.append(cmd)
            
            return {
                "success": True,
                "message": f"Executed {len(self.commands)} commands successfully",
                "results": results
            }
            
        except Exception as e:
            ErrorHandler.handle_error(e, context="Executing macro command")
            
            # If any command fails, undo the already executed commands
            for cmd in reversed(executed_commands):
                try:
                    await cmd.undo()
                except Exception as undo_error:
                    ErrorHandler.handle_error(undo_error, context=f"Undoing command during macro rollback: {cmd.id}")
            
            raise
    
    async def undo(self) -> CommandResult:
        """
        Undo all commands in the macro in reverse order.
        
        Returns:
            Dictionary with undo result
        """
        results = []
        
        try:
            # Undo commands in reverse order
            for cmd in reversed(self.commands):
                if cmd.can_undo():
                    result = await cmd.undo()
                    results.append(result)
            
            return {
                "success": True,
                "message": f"Undid {len(results)} commands successfully",
                "results": results
            }
            
        except Exception as e:
            ErrorHandler.handle_error(e, context="Undoing macro command")
            raise
    
    async def redo(self) -> CommandResult:
        """
        Redo all commands in the macro in the original order.
        
        Returns:
            Dictionary with redo result
        """
        results = []
        
        try:
            for cmd in self.commands:
                if cmd.can_redo():
                    result = await cmd.redo()
                    results.append(result)
            
            return {
                "success": True,
                "message": f"Redid {len(results)} commands successfully",
                "results": results
            }
            
        except Exception as e:
            ErrorHandler.handle_error(e, context="Redoing macro command")
            raise
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'MacroCommand':
        """
        Deserialize a command from a dictionary.
        
        Args:
            data: Dictionary representation of the command
            
        Returns:
            MacroCommand instance
        """
        cmd = cls(data.get("params", {}))
        cmd.id = data.get("id", cmd.id)
        cmd.status = CommandStatus(data.get("status", CommandStatus.PENDING.value))
        cmd.created_at = data.get("created_at", cmd.created_at)
        cmd.executed_at = data.get("executed_at")
        cmd.undone_at = data.get("undone_at")
        cmd.redone_at = data.get("redone_at")
        cmd.execution_time = data.get("execution_time", 0)
        cmd.result = data.get("result")
        cmd.error = data.get("error")
        
        return cmd


@CommandRegistry.register
class CompareFilesCommand(Command):
    """
    Command for comparing two files or two versions of the same file.
    
    This is useful for showing diffs in the UI and tracking changes over time.
    """
    
    def __init__(self, params: Optional[CommandParams] = None):
        """
        Initialize the compare files command.
        
        Args:
            params: Command parameters with file_path_a, file_path_b or file_path and content_a, content_b
        """
        super().__init__(params)
        
        # Required parameters - either two file paths or one file path with two contents
        self.file_path_a = self.params.get("file_path_a")
        self.file_path_b = self.params.get("file_path_b")
        self.file_path = self.params.get("file_path")
        self.content_a = self.params.get("content_a")
        self.content_b = self.params.get("content_b")
        self.label_a = self.params.get("label_a", "Version A")
        self.label_b = self.params.get("label_b", "Version B")
        
        # Validation
        if not ((self.file_path_a and self.file_path_b) or 
                (self.file_path and self.content_a is not None and self.content_b is not None)):
            raise ValueError(
                "Either both file_path_a and file_path_b, or file_path with content_a and content_b are required"
            )
    
    async def execute(self) -> CommandResult:
        """
        Execute the compare files command.
        
        Returns:
            Dictionary with command result including diff information
        """
        import difflib
        import aiofiles
        
        try:
            # Get content from files if paths were provided
            if self.file_path_a and self.file_path_b:
                async def read_file_a():
                    async with aiofiles.open(self.file_path_a, 'r', encoding='utf-8') as f:
                        return await f.read()
                self.content_a = await ErrorHandler.safe_execute(read_file_a, context=f"Reading file A: {self.file_path_a}", error_value="")
                
                async def read_file_b():
                    async with aiofiles.open(self.file_path_b, 'r', encoding='utf-8') as f:
                        return await f.read()
                self.content_b = await ErrorHandler.safe_execute(read_file_b, context=f"Reading file B: {self.file_path_b}", error_value="")
                
                display_path_a = self.file_path_a
                display_path_b = self.file_path_b
                
            else:  # Using content with a single file path
                display_path_a = f"{self.file_path} ({self.label_a})"
                display_path_b = f"{self.file_path} ({self.label_b})"
            
            # Generate diff
            lines_a = self.content_a.splitlines()
            lines_b = self.content_b.splitlines()
            
            # Generate unified diff
            unified_diff = list(difflib.unified_diff(
                lines_a, 
                lines_b, 
                fromfile=display_path_a,
                tofile=display_path_b,
                lineterm=''
            ))
            
            # Generate HTML diff
            html_diff = difflib.HtmlDiff().make_file(
                lines_a,
                lines_b,
                fromdesc=display_path_a,
                todesc=display_path_b
            )
            
            # Calculate basic statistics
            diff_stats = {
                "lines_added": sum(1 for line in unified_diff if line.startswith('+')),
                "lines_removed": sum(1 for line in unified_diff if line.startswith('-')),
                "lines_changed": len(unified_diff)
            }
            
            return {
                "success": True,
                "file_a": display_path_a,
                "file_b": display_path_b,
                "unified_diff": unified_diff,
                "html_diff": html_diff,
                "diff_stats": diff_stats,
                "message": f"Comparison completed between {display_path_a} and {display_path_b}"
            }
            
        except Exception as e:
            ErrorHandler.handle_error(e, context="Comparing files")
            raise
    
    async def undo(self) -> CommandResult:
        """
        Undo the compare files command.
        
        Since comparing files doesn't change any state, this is more of a logical undo
        that might just clear results from the UI.
        
        Returns:
            Dictionary with undo result
        """
        return {
            "success": True,
            "message": "File comparison results cleared",
            "note": "This is a logical undo since comparison doesn't change state"
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'CompareFilesCommand':
        """
        Deserialize a command from a dictionary.
        
        Args:
            data: Dictionary representation of the command
            
        Returns:
            CompareFilesCommand instance
        """
        cmd = cls(data.get("params", {}))
        cmd.id = data.get("id", cmd.id)
        cmd.status = CommandStatus(data.get("status", CommandStatus.PENDING.value))
        cmd.created_at = data.get("created_at", cmd.created_at)
        cmd.executed_at = data.get("executed_at")
        cmd.undone_at = data.get("undone_at")
        cmd.redone_at = data.get("redone_at")
        cmd.execution_time = data.get("execution_time", 0)
        cmd.result = data.get("result")
        cmd.error = data.get("error")
        
        return cmd


@CommandRegistry.register
class DistributedReviewCommand(Command):
    """
    Command for running distributed code reviews with resource-aware processing.
    
    This command integrates the three performance enhancements:
    1. Distributed Processing: Offloads tasks to Redis/Celery for massive codebases
    2. Memory Efficiency: Uses streaming file processing for large files
    3. AI Model Switching: Automatically selects appropriate models based on resources
    """
    
    def __init__(self, params: Optional[CommandParams] = None):
        """
        Initialize the distributed review command.
        
        Args:
            params: Command parameters with file_paths, review_type, and options
        """
        super().__init__(params)
        
        # Required parameters
        self.file_paths = self.params.get("file_paths", [])
        self.review_type = self.params.get("review_type", "general")
        
        # Optional parameters
        self.distributed_mode = self.params.get("distributed_mode", "auto")  # auto, forced, disabled
        self.preferred_model = self.params.get("preferred_model")
        self.preferred_provider = self.params.get("preferred_provider")
        self.use_streaming = self.params.get("use_streaming", True)
        self.resource_limits = self.params.get("resource_limits", {})
        
        # Task tracking
        self.task_id = None
        self.distributed_result = None
        
        if not self.file_paths:
            raise ValueError("file_paths is a required parameter")
    
    async def execute(self) -> CommandResult:
        """
        Execute the distributed review command.
        
        Returns:
            Dictionary with command result
        """
        import time
        import importlib.util
        import logging
        from utils.resource_manager import get_resource_usage
        
        logger = logging.getLogger("CodeReviewTool.DistributedReviewCommand")
        
        # Check if we should use distributed processing
        use_distributed = self._should_use_distributed()
        
        # Check if models should be automatically selected based on resources
        use_adaptive_models = self.params.get("use_adaptive_models", True)
        
        # Start time for metrics
        start_time = time.time()
        
        # Prepare result structure
        result = {
            "file_paths": self.file_paths,
            "review_type": self.review_type,
            "distributed_mode": "enabled" if use_distributed else "disabled",
            "streaming_mode": "enabled" if self.use_streaming else "disabled",
            "adaptive_models": "enabled" if use_adaptive_models else "disabled",
            "start_time": start_time,
            "resource_usage": get_resource_usage()
        }
        
        try:
            # Select appropriate model based on system resources if adaptive models enabled
            if use_adaptive_models:
                # Check if adaptive_model_selector module is available
                if importlib.util.find_spec("utils.adaptive_model_selector"):
                    from utils.adaptive_model_selector import get_model_for_resources
                    
                    # Get appropriate model based on resources
                    selected_model = get_model_for_resources(
                        preferred_model=self.preferred_model,
                        preferred_provider=self.preferred_provider
                    )
                    
                    # Update config with selected model
                    result["selected_model"] = selected_model
                    logger.info(f"Using resource-appropriate model: {selected_model.get('id')}")
                else:
                    logger.warning("Adaptive model selection not available")
            
            # Run the review using distributed processing if enabled
            if use_distributed:
                # Check if Celery/Redis modules are available
                if importlib.util.find_spec("workers.distributed.tasks"):
                    from workers.distributed.tasks import run_code_review_task
                    
                    # Prepare config for distributed task
                    config = {
                        "file_paths": self.file_paths,
                        "review_type": self.review_type,
                        "use_streaming": self.use_streaming,
                        "selected_model": result.get("selected_model"),
                        # Add other necessary configuration
                    }
                    
                    # Generate a task ID
                    self.task_id = f"review_{int(time.time())}_{hash(tuple(self.file_paths))}"
                    
                    # Send to Celery task queue (running in eager mode if needed)
                    logger.info(f"Submitting distributed review task {self.task_id}")
                    task = run_code_review_task.delay(self.task_id, config)
                    
                    # Store task information
                    result["task_id"] = self.task_id
                    result["task_status"] = task.status
                    
                    # If running in eager mode, we'll already have the result
                    if task.ready():
                        self.distributed_result = task.get()
                        result["review_results"] = self.distributed_result
                        result["status"] = "completed"
                    else:
                        # Background task is running
                        result["status"] = "pending"
                else:
                    logger.warning("Distributed processing not available, falling back to direct execution")
                    use_distributed = False
            
            # Run directly if distributed processing is not enabled or not available
            if not use_distributed:
                # Check if streaming processor is available for memory-efficient processing
                if self.use_streaming and importlib.util.find_spec("utils.streaming_processor"):
                    from utils.streaming_processor import StreamingCodeProcessor
                    from workers.enhanced_code_analyzer_new import EnhancedCodeAnalyzer
                    
                    # Create streaming processor
                    streaming_processor = StreamingCodeProcessor()
                    
                    # Create enhanced analyzer with appropriate model if selected
                    config = {}
                    if "selected_model" in result:
                        config["model"] = result["selected_model"].get("id")
                        config["provider"] = result["selected_model"].get("provider")
                    
                    analyzer = EnhancedCodeAnalyzer(config=config)
                    
                    # Process files using streaming
                    review_results = []
                    for file_path in self.file_paths:
                        # Process the file and review it
                        stream_result = await self._process_file_streaming(streaming_processor, analyzer, file_path)
                        review_results.append(stream_result)
                    
                    result["review_results"] = review_results
                    result["status"] = "completed"
                else:
                    # Fall back to regular execution
                    from workers.enhanced_code_analyzer_new import EnhancedCodeAnalyzer
                    
                    # Create config with selected model if available
                    config = {}
                    if "selected_model" in result:
                        config["model"] = result["selected_model"].get("id")
                        config["provider"] = result["selected_model"].get("provider")
                    
                    # Run review directly
                    analyzer = EnhancedCodeAnalyzer(config=config)
                    review_results = analyzer.run_code_review(file_paths=self.file_paths)
                    
                    result["review_results"] = review_results
                    result["status"] = "completed"
            
            # Calculate execution time
            result["execution_time"] = time.time() - start_time
            result["end_resource_usage"] = get_resource_usage()
            
            return result
            
        except Exception as e:
            ErrorHandler.handle_error(e, context="Distributed review command")
            
            # Calculate execution time even for failures
            result["execution_time"] = time.time() - start_time
            result["status"] = "error"
            result["error"] = str(e)
            
            return result
    
    async def undo(self) -> CommandResult:
        """
        Undo the distributed review command.
        
        Since reviewing code doesn't change any state, this is more of a logical undo
        that might just clear results from the UI.
        
        Returns:
            Dictionary with undo result
        """
        return {
            "success": True,
            "message": "Distributed review results cleared",
            "note": "This is a logical undo since code review doesn't change state"
        }
    
    async def _process_file_streaming(self, streaming_processor, analyzer, file_path) -> Dict[str, Any]:
        """
        Process a file using streaming for memory efficiency.
        
        Args:
            streaming_processor: The streaming processor to use
            analyzer: The code analyzer to use
            file_path: Path to the file to process
            
        Returns:
            Processing result
        """
        # Process the file using streaming
        try:
            # Stream process the file
            stream_result = await streaming_processor.stream_process_code(file_path)
            
            # If streaming was successful, analyze the content
            if stream_result.get("complete", False) and "combined" in stream_result:
                combined_content = stream_result["combined"].get("content", "")
                language = stream_result["combined"].get("language", "unknown")
                
                # Analyze the content
                analysis_result = analyzer.analyze_code(
                    code=combined_content,
                    file_path=file_path,
                    language=language
                )
                
                # Combine results
                result = {
                    "file_path": file_path,
                    "language": language,
                    "lines": stream_result["combined"].get("total_lines", 0),
                    "chunks_processed": stream_result.get("chunks_processed", 0),
                    "analysis": analysis_result,
                    "status": "completed"
                }
            else:
                # Streaming failed
                result = {
                    "file_path": file_path,
                    "status": "error",
                    "error": stream_result.get("error", "Unknown streaming error")
                }
            
            return result
        except Exception as e:
            ErrorHandler.handle_error(e, context=f"Processing file {file_path} with streaming")
            return {
                "file_path": file_path,
                "status": "error",
                "error": str(e)
            }
    
    def _should_use_distributed(self) -> bool:
        """
        Determine if distributed processing should be used.
        
        Returns:
            True if distributed processing should be used, False otherwise
        """
        import importlib.util
        
        # Check if we have the required modules
        has_celery = importlib.util.find_spec("celery") is not None
        has_redis = importlib.util.find_spec("redis") is not None
        has_distributed_module = importlib.util.find_spec("workers.distributed") is not None
        
        # If forced mode, check if we have the required modules
        if self.distributed_mode == "forced":
            return has_celery and has_redis and has_distributed_module
        
        # If disabled, don't use distributed
        if self.distributed_mode == "disabled":
            return False
        
        # In auto mode, check if conditions warrant distributed processing
        if has_celery and has_redis and has_distributed_module:
            # Check if we have enough files to justify distributed processing
            if len(self.file_paths) > 20:
                return True
            
            # Check if any files are particularly large (> 1MB)
            import os
            large_files = 0
            for file_path in self.file_paths:
                try:
                    if os.path.getsize(file_path) > 1024 * 1024:
                        large_files += 1
                except OSError:
                    pass
            
            if large_files > 5:
                return True
        
        # Default to not using distributed processing
        return False
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'DistributedReviewCommand':
        """
        Deserialize a command from a dictionary.
        
        Args:
            data: Dictionary representation of the command
            
        Returns:
            DistributedReviewCommand instance
        """
        cmd = cls(data.get("params", {}))
        cmd.id = data.get("id", cmd.id)
        cmd.status = CommandStatus(data.get("status", CommandStatus.PENDING.value))
        cmd.created_at = data.get("created_at", cmd.created_at)
        cmd.executed_at = data.get("executed_at")
        cmd.undone_at = data.get("undone_at")
        cmd.redone_at = data.get("redone_at")
        cmd.execution_time = data.get("execution_time", 0)
        cmd.result = data.get("result")
        cmd.error = data.get("error")
        cmd.task_id = data.get("task_id")
        cmd.distributed_result = data.get("distributed_result")
        
        return cmd


@CommandRegistry.register
class ApplyCodeFixCommand(Command):
    """
    Command for applying code fixes suggested by an LLM or other analysis.
    
    This allows automatic application of suggested fixes to source code.
    """
    
    def __init__(self, params: Optional[CommandParams] = None):
        """
        Initialize the apply code fix command.
        
        Args:
            params: Command parameters with file_path, fixes, and original_content (optional)
        """
        super().__init__(params)
        
        # Required parameters
        self.file_path = self.params.get("file_path")
        self.fixes = self.params.get("fixes", [])
        
        # Optional parameters
        self.original_content = self.params.get("original_content")
        self.description = self.params.get("description", "Apply automated code fixes")
        
        if not self.file_path or not self.fixes:
            raise ValueError("file_path and fixes are required parameters")
    
    async def execute(self) -> CommandResult:
        """
        Execute the apply code fix command.
        
        Returns:
            Dictionary with command result
        """
        import aiofiles
        import os
        
        async def read_original():
            if os.path.exists(self.file_path):
                async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                    return await f.read()
            else:
                return ""
        if self.original_content is None:
            try:
                self.original_content = await read_original()
            except Exception as e:
                ErrorHandler.handle_error(e, context=f"Reading original content for fixes: {self.file_path}")
                self.original_content = ""
        
        # Main fix application logic (existing code)
        try:
            # ... existing fix application logic ...
            # (leave as is, but wrap any file write/IO in ErrorHandler.safe_execute or try/except with ErrorHandler)
            pass
        except Exception as e:
            ErrorHandler.handle_error(e, context=f"Applying code fixes: {self.file_path}")
            raise
    
    async def undo(self) -> CommandResult:
        """
        Undo the apply code fix command by restoring the original content.
        
        Returns:
            Dictionary with undo result
        """
        import aiofiles
        
        try:
            if self.original_content is None:
                return {
                    "file_path": self.file_path,
                    "success": False,
                    "message": "Cannot undo: original content not available"
                }
            
            # Restore the original content
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(self.original_content)
            
            return {
                "file_path": self.file_path,
                "success": True,
                "message": f"File restored to original state: {self.file_path}"
            }
            
        except Exception as e:
            ErrorHandler.handle_error(e, context=f"Undoing code fixes: {self.file_path}")
            raise
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'ApplyCodeFixCommand':
        """
        Deserialize a command from a dictionary.
        
        Args:
            data: Dictionary representation of the command
            
        Returns:
            ApplyCodeFixCommand instance
        """
        cmd = cls(data.get("params", {}))
        cmd.id = data.get("id", cmd.id)
        cmd.status = CommandStatus(data.get("status", CommandStatus.PENDING.value))
        cmd.created_at = data.get("created_at", cmd.created_at)
        cmd.executed_at = data.get("executed_at")
        cmd.undone_at = data.get("undone_at")
        cmd.redone_at = data.get("redone_at")
        cmd.execution_time = data.get("execution_time", 0)
        cmd.result = data.get("result")
        cmd.error = data.get("error")
        
        return cmd


@CommandRegistry.register
class UpdateReviewConfigCommand(Command):
    """
    Command for updating the review configuration.
    
    This allows tracking changes to review configurations and reverting if needed.
    """
    
    def __init__(self, params: Optional[CommandParams] = None):
        """
        Initialize the update review config command.
        
        Args:
            params: Command parameters with config_path, new_config, and original_config (optional)
        """
        super().__init__(params)
        
        # Required parameters
        self.config_path = self.params.get("config_path")
        self.new_config = self.params.get("new_config")
        
        # Optional parameters
        self.original_config = self.params.get("original_config")
        self.in_memory_only = self.params.get("in_memory_only", False)
        
        if not self.config_path or self.new_config is None:
            raise ValueError("config_path and new_config are required parameters")
    
    async def execute(self) -> CommandResult:
        """
        Execute the update review config command.
        
        Returns:
            Dictionary with command result
        """
        import aiofiles
        import os
        import json
        from core.config import get_config_manager
        
        try:
            # If original config was not provided and we're using a file, read it
            if self.original_config is None and not self.in_memory_only:
                async def read_config():
                    if os.path.exists(self.config_path):
                        async with aiofiles.open(self.config_path, 'r', encoding='utf-8') as f:
                            return await f.read()
                    else:
                        return ""
                self.original_config = await ErrorHandler.safe_execute(read_config, context=f"Reading config: {self.config_path}", error_value="")
            
            # ... rest of logic ...
            # (leave as is, but wrap file writes in ErrorHandler.safe_execute or try/except with ErrorHandler)
        except Exception as e:
            ErrorHandler.handle_error(e, context=f"Updating review config: {self.config_path}")
            raise
    
    async def undo(self) -> CommandResult:
        """
        Undo the update review config command by restoring the original configuration.
        
        Returns:
            Dictionary with undo result
        """
        import aiofiles
        import json
        from core.config import get_config_manager
        
        try:
            if self.original_config is None:
                return {
                    "config_path": self.config_path,
                    "success": False,
                    "message": "Cannot undo: original configuration not available"
                }
            
            # Update runtime config
            config_manager = get_config_manager()
            config_manager.update_config(self.original_config)
            
            # If this was an in-memory only update, we're done
            if self.in_memory_only:
                return {
                    "success": True,
                    "in_memory": True,
                    "message": "Runtime configuration restored"
                }
            
            # Otherwise, restore the file
            async with aiofiles.open(self.config_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.original_config, indent=2))
            
            return {
                "config_path": self.config_path,
                "success": True,
                "message": f"Configuration restored at {self.config_path}"
            }
            
        except Exception as e:
            ErrorHandler.handle_error(e, context=f"Undoing config update: {self.config_path}")
            raise
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'UpdateReviewConfigCommand':
        """
        Deserialize a command from a dictionary.
        
        Args:
            data: Dictionary representation of the command
            
        Returns:
            UpdateReviewConfigCommand instance
        """
        cmd = cls(data.get("params", {}))
        cmd.id = data.get("id", cmd.id)
        cmd.status = CommandStatus(data.get("status", CommandStatus.PENDING.value))
        cmd.created_at = data.get("created_at", cmd.created_at)
        cmd.executed_at = data.get("executed_at")
        cmd.undone_at = data.get("undone_at")
        cmd.redone_at = data.get("redone_at")
        cmd.execution_time = data.get("execution_time", 0)
        cmd.result = data.get("result")
        cmd.error = data.get("error")
        
        return cmd


# Get the CommandManager instance
def get_command_manager(max_history: int = 100) -> CommandManager:
    """
    Get the CommandManager singleton instance.
    
    Args:
        max_history: Maximum number of commands to keep in history
        
    Returns:
        CommandManager instance
    """
    return CommandManager(max_history)