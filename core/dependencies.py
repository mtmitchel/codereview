"""
Dependencies module for Code Review Tool.

This module provides a lightweight dependency injection container
to manage service dependencies throughout the application.
"""

import logging
import importlib
import inspect
import threading
from typing import Dict, Any, Type, TypeVar, Optional, cast, List, Callable, Set, Union, Generic

logger = logging.getLogger("CodeReviewTool.Dependencies")

T = TypeVar('T')
U = TypeVar('U')
Interface = Type[T]  # Type alias for better readability


class Lifecycle:
    """Enumeration of service lifecycle types."""
    TRANSIENT = "transient"  # New instance each time
    SINGLETON = "singleton"  # Single instance for entire app
    SCOPED = "scoped"        # Single instance per scope


class ServiceRegistration(Generic[T]):
    """Represents a registered service in the container."""
    
    def __init__(self, 
                service_type: Type[T], 
                implementation: Union[Type[T], Callable[..., T], T], 
                lifecycle: str = Lifecycle.TRANSIENT,
                factory_args: Optional[List[Any]] = None,
                factory_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize a service registration.
        
        Args:
            service_type: The interface or type to register
            implementation: The implementation type, factory, or instance
            lifecycle: Service lifecycle (transient, singleton, scoped)
            factory_args: Optional positional args for the factory
            factory_kwargs: Optional keyword args for the factory
        """
        self.service_type = service_type
        self.implementation = implementation
        self.lifecycle = lifecycle
        self.factory_args = factory_args or []
        self.factory_kwargs = factory_kwargs or {}
        self.instance = None  # For singleton and scoped instances


class ServiceScope:
    """
    Represents a scope for scoped service instances.
    """
    
    def __init__(self, container: 'ServiceContainer'):
        """
        Initialize a service scope.
        
        Args:
            container: The parent container
        """
        self.container = container
        self.instances: Dict[Type, Any] = {}
    
    def resolve(self, service_type: Type[T]) -> Optional[T]:
        """
        Resolve a service within this scope.
        
        Args:
            service_type: The type of service to resolve
            
        Returns:
            The resolved service instance
        """
        # For scoped services, check if we already have an instance
        if service_type in self.instances:
            return cast(T, self.instances[service_type])
        
        # Resolve from container
        instance = self.container._resolve_impl(service_type, scope=self)
        
        # If this is a scoped service, store the instance
        registration = self.container._get_registration(service_type)
        if registration and registration.lifecycle == Lifecycle.SCOPED:
            self.instances[service_type] = instance
        
        return instance
    
    def __enter__(self) -> 'ServiceScope':
        """Enter the scope context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the scope context."""
        self.dispose()
    
    def dispose(self) -> None:
        """Dispose of the scope and its instances."""
        # Call dispose method on any instances that have it
        for instance in self.instances.values():
            if hasattr(instance, 'dispose') and callable(instance.dispose):
                try:
                    instance.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing instance {instance}: {e}")
        
        self.instances.clear()


class ServiceContainer:
    """
    A lightweight dependency injection container that manages service registrations
    and resolutions throughout the application.
    """
    
    def __init__(self):
        """Initialize an empty service container."""
        self._registrations: Dict[Type, ServiceRegistration] = {}
        self._plugin_registry: Dict[str, List[Type]] = {}
        self._lock = threading.RLock()
        self._initializing: Set[Type] = set()  # For circular dependency detection
        logger.debug("Service container initialized")
    
    def register(self, service_type: Type[T], implementation: Union[Type[U], Callable[..., U], U], 
                lifecycle: str = Lifecycle.TRANSIENT, **kwargs) -> None:
        """
        Register a service with the container.
        
        Args:
            service_type: The service type or interface
            implementation: The implementation type, factory function, or instance
            lifecycle: Service lifecycle (transient, singleton, scoped)
            **kwargs: Optional arguments to pass to factory
            
        Example:
            container.register(LLMClient, OllamaClient, Lifecycle.SINGLETON)
            container.register(ConfigManager, lambda: ConfigManager(config_path="config.json"))
            container.register(Logger, existing_logger_instance)
        """
        with self._lock:
            registration = ServiceRegistration(
                service_type=service_type,
                implementation=implementation,
                lifecycle=lifecycle,
                factory_kwargs=kwargs if kwargs else None
            )
            
            self._registrations[service_type] = registration
            
            # If implementation is already an instance and lifecycle is singleton, store it
            if not callable(implementation) or (isinstance(implementation, type) and lifecycle != Lifecycle.TRANSIENT):
                if lifecycle == Lifecycle.SINGLETON and not isinstance(implementation, type):
                    registration.instance = implementation
            
            logger.debug(f"Registered {lifecycle} implementation for {service_type.__name__}")
    
    def register_singleton(self, service_type: Type[T], implementation: Union[Type[U], Callable[..., U], U], **kwargs) -> None:
        """
        Register a singleton service with the container.
        
        Args:
            service_type: The service type or interface
            implementation: The implementation type, factory function, or instance
            **kwargs: Optional arguments to pass to factory
        """
        self.register(service_type, implementation, Lifecycle.SINGLETON, **kwargs)
    
    def register_scoped(self, service_type: Type[T], implementation: Union[Type[U], Callable[..., U]], **kwargs) -> None:
        """
        Register a scoped service with the container.
        
        Args:
            service_type: The service type or interface
            implementation: The implementation type or factory function
            **kwargs: Optional arguments to pass to factory
        """
        self.register(service_type, implementation, Lifecycle.SCOPED, **kwargs)
    
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """
        Register an existing instance as a singleton.
        
        Args:
            service_type: The service type or interface
            instance: The existing instance
        """
        self.register(service_type, instance, Lifecycle.SINGLETON)
    
    def register_factory(self, service_type: Type[T], factory: Callable[..., T], 
                       lifecycle: str = Lifecycle.TRANSIENT, **kwargs) -> None:
        """
        Register a factory function to create instances of a service.
        
        Args:
            service_type: The service type or interface
            factory: A callable that creates instances of the service
            lifecycle: Service lifecycle (transient, singleton, scoped)
            **kwargs: Optional arguments to pass to factory
        """
        self.register(service_type, factory, lifecycle, **kwargs)
    
    def _get_registration(self, service_type: Type[T]) -> Optional[ServiceRegistration]:
        """
        Get the registration for a service type.
        
        Args:
            service_type: The service type to look up
            
        Returns:
            The service registration or None if not found
        """
        return self._registrations.get(service_type)
    
    def _resolve_impl(self, service_type: Type[T], scope: Optional[ServiceScope] = None) -> Optional[T]:
        """
        Internal implementation of resolve that supports scopes.
        
        Args:
            service_type: The service type to resolve
            scope: Optional current scope
            
        Returns:
            The resolved instance or None
        """
        # Check if the service is registered
        registration = self._get_registration(service_type)
        if not registration:
            logger.warning(f"No implementation registered for {service_type.__name__}")
            return None
        
        # Check for circular dependencies
        if service_type in self._initializing:
            raise Exception(f"Circular dependency detected while resolving {service_type.__name__}")
        
        # Handle different lifecycles
        if registration.lifecycle == Lifecycle.SINGLETON:
            # Return existing instance if available
            if registration.instance is not None:
                return cast(T, registration.instance)
            
            # Initialize the singleton
            self._initializing.add(service_type)
            try:
                implementation = registration.implementation
                
                # Create the instance
                if isinstance(implementation, type):
                    # It's a class, instantiate it
                    instance = implementation(*registration.factory_args, **registration.factory_kwargs)
                elif callable(implementation):
                    # It's a factory function
                    instance = implementation(*registration.factory_args, **registration.factory_kwargs)
                else:
                    # It's already an instance
                    instance = implementation
                
                # Store the instance
                registration.instance = instance
                logger.debug(f"Initialized singleton for {service_type.__name__}")
                return cast(T, instance)
            finally:
                self._initializing.remove(service_type)
                
        elif registration.lifecycle == Lifecycle.SCOPED:
            # Scoped services require a scope
            if scope is None:
                logger.warning(f"Attempted to resolve scoped service {service_type.__name__} without a scope")
                # Fall back to transient behavior
                return self._create_instance(registration)
            
            # Return existing instance from scope if available
            if service_type in scope.instances:
                return cast(T, scope.instances[service_type])
            
            # Create new instance for this scope
            instance = self._create_instance(registration)
            scope.instances[service_type] = instance
            return instance
            
        else:  # TRANSIENT
            # Always create a new instance
            return self._create_instance(registration)
    
    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """
        Create a new instance from a registration.
        
        Args:
            registration: The service registration
            
        Returns:
            The created instance
        """
        self._initializing.add(registration.service_type)
        try:
            implementation = registration.implementation
            
            if isinstance(implementation, type):
                # It's a class, instantiate it
                return implementation(*registration.factory_args, **registration.factory_kwargs)
            elif callable(implementation):
                # It's a factory function
                return implementation(*registration.factory_args, **registration.factory_kwargs)
            else:
                # It's already an instance
                return implementation
                
        finally:
            self._initializing.remove(registration.service_type)
    
    def resolve(self, service_type: Type[T]) -> Optional[T]:
        """
        Resolve a service implementation.
        
        Args:
            service_type: The service type to resolve
            
        Returns:
            The resolved instance or None if not registered
        """
        with self._lock:
            return self._resolve_impl(service_type)
    
    def create_scope(self) -> ServiceScope:
        """
        Create a new service scope.
        
        Returns:
            A new ServiceScope instance
            
        Example:
            with container.create_scope() as scope:
                service = scope.resolve(ScopedService)
        """
        return ServiceScope(self)
    
    def has_implementation(self, service_type: Type[T]) -> bool:
        """
        Check if an implementation is registered for the service type.
        
        Args:
            service_type: The service type to check
            
        Returns:
            True if an implementation is registered, False otherwise
        """
        return service_type in self._registrations
    
    def register_plugin(self, plugin_type: str, plugin_class: Type) -> None:
        """
        Register a plugin implementation for a specific plugin type.
        
        Args:
            plugin_type: The category of plugin (e.g., 'analyzer', 'formatter')
            plugin_class: The plugin class to register
        """
        with self._lock:
            if plugin_type not in self._plugin_registry:
                self._plugin_registry[plugin_type] = []
            
            if plugin_class not in self._plugin_registry[plugin_type]:
                self._plugin_registry[plugin_type].append(plugin_class)
                logger.debug(f"Registered plugin {plugin_class.__name__} for type {plugin_type}")
    
    def get_plugins(self, plugin_type: str) -> List[Type]:
        """
        Get all registered plugin classes for a specific type.
        
        Args:
            plugin_type: The category of plugin to retrieve
            
        Returns:
            List of plugin classes registered for that type
        """
        return self._plugin_registry.get(plugin_type, []).copy()
    
    def instantiate_plugins(self, plugin_type: str, *args, **kwargs) -> List[Any]:
        """
        Instantiate all plugins of a specific type with the given arguments.
        
        Args:
            plugin_type: The category of plugin to instantiate
            *args, **kwargs: Arguments to pass to plugin constructors
            
        Returns:
            List of instantiated plugin objects
        """
        plugin_instances = []
        for plugin_class in self.get_plugins(plugin_type):
            try:
                plugin_instance = plugin_class(*args, **kwargs)
                plugin_instances.append(plugin_instance)
                logger.debug(f"Instantiated plugin {plugin_class.__name__}")
            except Exception as e:
                logger.error(f"Error instantiating plugin {plugin_class.__name__}: {e}")
        
        return plugin_instances
    
    def discover_plugins(self, module_paths: List[str], base_class: Type, plugin_type: str) -> None:
        """
        Discover and register plugins from specified modules that inherit from a base class.
        
        Args:
            module_paths: List of module paths to search for plugins
            base_class: The base class that plugins should inherit from
            plugin_type: The category to register discovered plugins under
        """
        for module_path in module_paths:
            try:
                module = importlib.import_module(module_path)
                for name, obj in inspect.getmembers(module):
                    # Check if it's a class that inherits from base_class but is not base_class itself
                    if (inspect.isclass(obj) and issubclass(obj, base_class) and 
                            obj != base_class and not inspect.isabstract(obj)):
                        self.register_plugin(plugin_type, obj)
                        logger.info(f"Discovered plugin {obj.__name__} in {module_path}")
            except ImportError as e:
                logger.warning(f"Could not import module {module_path} for plugin discovery: {e}")
    
    def clear(self) -> None:
        """Clear all registrations and instances."""
        with self._lock:
            # Dispose any singleton instances with a dispose method
            for registration in self._registrations.values():
                if registration.instance and hasattr(registration.instance, 'dispose'):
                    try:
                        registration.instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing instance {registration.instance}: {e}")
            
            self._registrations.clear()
            self._plugin_registry.clear()
            logger.debug("Service container cleared")
    
    def build_provider(self, config_action: Optional[Callable[['ServiceContainerBuilder'], None]] = None) -> 'ServiceProvider':
        """
        Build a service provider with a fluent configuration API.
        
        Args:
            config_action: Optional configuration action
            
        Returns:
            A new ServiceProvider
            
        Example:
            provider = container.build_provider(lambda builder:
                builder.add_singleton(ConfigManager)
                      .add_transient(LLMClient, OllamaClient)
            )
        """
        builder = ServiceContainerBuilder(self)
        if config_action:
            config_action(builder)
        return ServiceProvider(self)


class ServiceContainerBuilder:
    """
    Fluent API for configuring a service container.
    """
    
    def __init__(self, container: ServiceContainer):
        """
        Initialize the builder.
        
        Args:
            container: The service container to configure
        """
        self.container = container
    
    def add_singleton(self, service_type: Type[T], implementation: Optional[Union[Type[U], Callable[..., U], U]] = None, **kwargs) -> 'ServiceContainerBuilder':
        """
        Add a singleton service.
        
        Args:
            service_type: The service type
            implementation: The implementation (defaults to service_type if None)
            **kwargs: Optional arguments for the implementation
            
        Returns:
            The builder for method chaining
        """
        impl = implementation or service_type
        self.container.register_singleton(service_type, impl, **kwargs)
        return self
    
    def add_scoped(self, service_type: Type[T], implementation: Optional[Union[Type[U], Callable[..., U]]] = None, **kwargs) -> 'ServiceContainerBuilder':
        """
        Add a scoped service.
        
        Args:
            service_type: The service type
            implementation: The implementation (defaults to service_type if None)
            **kwargs: Optional arguments for the implementation
            
        Returns:
            The builder for method chaining
        """
        impl = implementation or service_type
        self.container.register_scoped(service_type, impl, **kwargs)
        return self
    
    def add_transient(self, service_type: Type[T], implementation: Optional[Union[Type[U], Callable[..., U]]] = None, **kwargs) -> 'ServiceContainerBuilder':
        """
        Add a transient service.
        
        Args:
            service_type: The service type
            implementation: The implementation (defaults to service_type if None)
            **kwargs: Optional arguments for the implementation
            
        Returns:
            The builder for method chaining
        """
        impl = implementation or service_type
        self.container.register(service_type, impl, Lifecycle.TRANSIENT, **kwargs)
        return self
    
    def add_instance(self, service_type: Type[T], instance: T) -> 'ServiceContainerBuilder':
        """
        Add an existing instance.
        
        Args:
            service_type: The service type
            instance: The instance to register
            
        Returns:
            The builder for method chaining
        """
        self.container.register_instance(service_type, instance)
        return self
    
    def discover_plugins(self, module_paths: List[str], base_class: Type, plugin_type: str) -> 'ServiceContainerBuilder':
        """
        Discover plugins from modules.
        
        Args:
            module_paths: Module paths to search
            base_class: Base class of plugins
            plugin_type: Type of plugins
            
        Returns:
            The builder for method chaining
        """
        self.container.discover_plugins(module_paths, base_class, plugin_type)
        return self


class ServiceProvider:
    """
    Service provider that exposes a simplified API for resolving services.
    """
    
    def __init__(self, container: ServiceContainer):
        """
        Initialize the service provider.
        
        Args:
            container: The underlying service container
        """
        self.container = container
    
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """
        Get a service by type.
        
        Args:
            service_type: The service type to resolve
            
        Returns:
            The resolved service or None if not registered
        """
        return self.container.resolve(service_type)
    
    def create_scope(self) -> ServiceScope:
        """
        Create a new service scope.
        
        Returns:
            A new ServiceScope instance
        """
        return self.container.create_scope()


# Create a global container instance
container = ServiceContainer()


def get_container() -> ServiceContainer:
    """
    Get the global service container instance.
    
    Returns:
        Global ServiceContainer instance
    """
    return container


def get_service(service_type: Type[T]) -> Optional[T]:
    """
    Resolve a service from the global container.
    
    Args:
        service_type: The service type to resolve
        
    Returns:
        The resolved service or None if not registered
    """
    return container.resolve(service_type)


def configure_services(config_action: Callable[[ServiceContainerBuilder], None]) -> ServiceProvider:
    """
    Configure services using a fluent API.
    
    Args:
        config_action: Configuration action
        
    Returns:
        A new ServiceProvider
        
    Example:
        provider = configure_services(lambda services:
            services.add_singleton(ConfigManager)
                   .add_transient(LLMClient, OllamaClient)
        )
    """
    return container.build_provider(config_action)