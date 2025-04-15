"""
LRU Cache Implementation for Code Review Tool.

This module provides an enhanced Least Recently Used (LRU) cache implementation
with size limits and memory monitoring to prevent memory issues with large codebases.
"""

import logging
import threading
import time
import sys
import json
import hashlib
from collections import OrderedDict
from typing import Dict, Any, Optional, Callable, Tuple, List, Union, Generic, TypeVar

from utils.resource_manager import MemoryMonitor

logger = logging.getLogger("CodeReviewTool.LRUCache")

# Create a memory monitor for cache resource management
_memory_monitor = MemoryMonitor()

K = TypeVar('K')
V = TypeVar('V')

class LRUCache(Generic[K, V]):
    """
    Enhanced LRU (Least Recently Used) cache with size and memory limits.
    
    This cache automatically evicts the least recently used items when:
    1. The maximum number of items is reached
    2. The maximum memory usage is reached
    
    It also provides metrics and monitoring capabilities.
    """
    
    def __init__(self, 
                 max_items: int = 1000, 
                 max_memory_mb: Optional[int] = None, 
                 ttl: Optional[int] = None,
                 size_estimator: Optional[Callable[[V], int]] = None):
        """
        Initialize the LRU cache.
        
        Args:
            max_items: Maximum number of items to store
            max_memory_mb: Maximum memory usage in MB (optional)
            ttl: Default time-to-live for items in seconds (optional)
            size_estimator: Function to estimate item size in bytes (optional)
        """
        self.max_items = max(1, max_items)  # At least 1 item
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        self.default_ttl = ttl
        self.size_estimator = size_estimator or self._default_size_estimator
        
        # Use OrderedDict for O(1) operations
        self._cache: OrderedDict[K, Tuple[V, Optional[float], int]] = OrderedDict()
        # key -> (value, expiry_time, size_bytes)
        
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.memory_eviction_count = 0
        self.last_prune_time = time.time()
        self.prune_interval = 60  # Seconds between auto-pruning expired items
        self.estimated_memory_usage = 0  # Estimated bytes used
        
        logger.info(f"Initialized LRU cache with max_items={max_items}, "
                  f"max_memory_mb={max_memory_mb}, default_ttl={ttl}")
    
    def get(self, key: K, increment_miss: bool = True) -> Optional[V]:
        """
        Get a value from the cache.
        
        This operation marks the item as recently used.
        
        Args:
            key: Cache key
            increment_miss: Whether to increment the miss counter if the key is not found
            
        Returns:
            Cached value or None if not found or expired
        """
        with self.lock:
            if key not in self._cache:
                if increment_miss:
                    self.miss_count += 1
                return None
            
            value, expiry_time, size_bytes = self._cache[key]
            
            # Check if item has expired
            if expiry_time is not None and time.time() > expiry_time:
                # Remove expired item
                self._remove_item(key)
                if increment_miss:
                    self.miss_count += 1
                return None
            
            # Mark as recently used by removing and re-adding
            self._cache.move_to_end(key)
            
            self.hit_count += 1
            
            # Check if we need to auto-prune
            current_time = time.time()
            if current_time - self.last_prune_time > self.prune_interval:
                self._prune_expired()
                self.last_prune_time = current_time
            
            return value
    
    def set(self, key: K, value: V, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional, overrides default)
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Estimate value size
            try:
                size_bytes = self.size_estimator(value)
            except Exception as e:
                logger.warning(f"Error estimating size: {e}")
                size_bytes = sys.getsizeof(value)
            
            # Calculate expiry time if TTL is provided
            expiry_time = None
            if ttl is not None or self.default_ttl is not None:
                expiry_time = time.time() + (ttl if ttl is not None else self.default_ttl)
            
            # If key already exists, update estimated memory usage
            if key in self._cache:
                _, _, old_size = self._cache[key]
                self.estimated_memory_usage -= old_size
            
            # Check if we need to make room 
            # First check memory limit
            if self.max_memory_bytes is not None:
                # If this single item exceeds our memory limit, we can't cache it
                if size_bytes > self.max_memory_bytes:
                    logger.warning(f"Item too large to cache: {size_bytes} bytes > {self.max_memory_bytes} bytes")
                    return False
                
                # Make room for the new item
                while (self.estimated_memory_usage + size_bytes > self.max_memory_bytes 
                       and len(self._cache) > 0):
                    self._evict_lru_item(reason="memory")
            
            # Then check item count limit
            while len(self._cache) >= self.max_items and len(self._cache) > 0:
                self._evict_lru_item(reason="count")
            
            # Add the new item
            self._cache[key] = (value, expiry_time, size_bytes)
            self.estimated_memory_usage += size_bytes
            
            # Move to end (mark as most recently used)
            self._cache.move_to_end(key)
            
            return True
    
    def delete(self, key: K) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        with self.lock:
            if key in self._cache:
                self._remove_item(key)
                return True
            return False
    
    def _remove_item(self, key: K) -> None:
        """
        Remove an item from the cache and update memory usage.
        
        Args:
            key: Cache key
        """
        _, _, size_bytes = self._cache[key]
        del self._cache[key]
        self.estimated_memory_usage -= size_bytes
    
    def _evict_lru_item(self, reason: str = "count") -> None:
        """
        Evict the least recently used item.
        
        Args:
            reason: Reason for eviction (for tracking)
        """
        if not self._cache:
            return
        
        # Get the oldest item (first in OrderedDict)
        lru_key = next(iter(self._cache))
        
        # Remove it
        self._remove_item(lru_key)
        
        # Update stats
        self.eviction_count += 1
        if reason == "memory":
            self.memory_eviction_count += 1
        
        logger.debug(f"Evicted LRU cache entry for key: {lru_key} (reason: {reason})")
    
    def _prune_expired(self) -> int:
        """
        Remove all expired items from the cache.
        
        Returns:
            Number of items pruned
        """
        if not self._cache:
            return 0
        
        current_time = time.time()
        expired_keys = []
        
        # Find all expired keys
        for key, (_, expiry_time, _) in self._cache.items():
            if expiry_time is not None and current_time > expiry_time:
                expired_keys.append(key)
        
        # Remove them
        for key in expired_keys:
            self._remove_item(key)
        
        pruned_count = len(expired_keys)
        if pruned_count > 0:
            logger.debug(f"Pruned {pruned_count} expired items from cache")
        
        return pruned_count
    
    def clear(self) -> int:
        """
        Clear all items from the cache.
        
        Returns:
            Number of items cleared
        """
        with self.lock:
            count = len(self._cache)
            self._cache.clear()
            self.estimated_memory_usage = 0
            # Reset counters
            self.hit_count = 0
            self.miss_count = 0
            self.eviction_count = 0
            self.memory_eviction_count = 0
            return count
    
    def _default_size_estimator(self, value: V) -> int:
        """
        Default function to estimate the memory size of a value.
        
        This is a simple estimate and may not be accurate for all types.
        
        Args:
            value: Value to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        try:
            # For objects that can be JSON serialized, use the string length
            # as a rough approximation of memory usage
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                json_str = json.dumps(value)
                return len(json_str.encode('utf-8'))
            
            # For more complex objects, use sys.getsizeof()
            return sys.getsizeof(value)
            
        except (TypeError, OverflowError):
            # Fallback for objects that can't be JSON serialized
            return sys.getsizeof(value)
    
    def contains(self, key: K) -> bool:
        """
        Check if a key exists in the cache.
        
        Unlike get(), this does not update usage statistics.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key exists and is not expired, False otherwise
        """
        with self.lock:
            if key not in self._cache:
                return False
                
            _, expiry_time, _ = self._cache[key]
            
            # Check if item has expired
            if expiry_time is not None and time.time() > expiry_time:
                return False
                
            return True
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            hit_ratio = self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
            memory_percent = (self.estimated_memory_usage / self.max_memory_bytes) * 100 if self.max_memory_bytes else 0
            
            # Get actual memory usage of the process for comparison
            memory_info = _memory_monitor.get_memory_usage()
            
            return {
                'items': len(self._cache),
                'max_items': self.max_items,
                'items_percent': (len(self._cache) / self.max_items) * 100 if self.max_items else 0,
                'hits': self.hit_count,
                'misses': self.miss_count,
                'hit_ratio': hit_ratio,
                'evictions': self.eviction_count,
                'memory_evictions': self.memory_eviction_count,
                'estimated_memory_bytes': self.estimated_memory_usage,
                'max_memory_bytes': self.max_memory_bytes,
                'memory_percent': memory_percent,
                'system_memory_percent': memory_info.get('percent', None),
                'available_system_memory_gb': memory_info.get('available_gb', None),
                'memory_warning': memory_info.get('warning', False),
                'memory_critical': memory_info.get('critical', False)
            }
    
    def keys(self) -> List[K]:
        """
        Get a list of all keys in the cache.
        
        Returns:
            List of keys
        """
        with self.lock:
            return list(self._cache.keys())
    
    def resize(self, max_items: int, max_memory_mb: Optional[int] = None) -> None:
        """
        Resize the cache limits.
        
        Args:
            max_items: New maximum number of items
            max_memory_mb: New maximum memory usage in MB (optional)
        """
        with self.lock:
            self.max_items = max(1, max_items)
            if max_memory_mb is not None:
                self.max_memory_bytes = max_memory_mb * 1024 * 1024
            
            # Evict items if we're over the new limits
            while len(self._cache) > self.max_items:
                self._evict_lru_item(reason="resize")
            
            if self.max_memory_bytes is not None:
                while self.estimated_memory_usage > self.max_memory_bytes and len(self._cache) > 0:
                    self._evict_lru_item(reason="resize")
            
            logger.info(f"Resized cache to max_items={max_items}, max_memory_mb={max_memory_mb}")
    
    def __len__(self) -> int:
        """Get the number of items in the cache."""
        with self.lock:
            return len(self._cache)
    
    def __contains__(self, key: K) -> bool:
        """Check if a key exists in the cache (may be expired)."""
        with self.lock:
            return key in self._cache


class LRUCacheManager:
    """
    Manager for multiple LRU caches with different purposes.
    
    This allows the application to have separate caches for different
    types of data with their own size limits and eviction policies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LRU cache manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.caches: Dict[str, LRUCache] = {}
        self.lock = threading.RLock()
        
        # Limits from configuration
        self.default_max_items = self.config.get('default_max_items', 1000)
        self.default_max_memory_mb = self.config.get('default_max_memory_mb', 200)
        self.default_ttl = self.config.get('default_ttl', None)
        
        # Initialize with predefined caches from configuration
        predefined = self.config.get('caches', {})
        for name, cache_config in predefined.items():
            max_items = cache_config.get('max_items', self.default_max_items)
            max_memory_mb = cache_config.get('max_memory_mb', self.default_max_memory_mb)
            ttl = cache_config.get('ttl', self.default_ttl)
            
            self.caches[name] = LRUCache(
                max_items=max_items,
                max_memory_mb=max_memory_mb,
                ttl=ttl
            )
        
        # Always create these standard caches if not already defined
        standard_caches = {
            'llm_responses': {
                'max_items': 200,
                'max_memory_mb': 100,
                'ttl': 3600 * 24  # 24 hours
            },
            'dependency_graphs': {
                'max_items': 50,
                'max_memory_mb': 50,
                'ttl': 3600 * 24  # 24 hours
            },
            'code_chunks': {
                'max_items': 500,
                'max_memory_mb': 50,
                'ttl': 3600 * 2  # 2 hours
            }
        }
        
        for name, cache_config in standard_caches.items():
            if name not in self.caches:
                max_items = cache_config.get('max_items', self.default_max_items)
                max_memory_mb = cache_config.get('max_memory_mb', self.default_max_memory_mb)
                ttl = cache_config.get('ttl', self.default_ttl)
                
                self.caches[name] = LRUCache(
                    max_items=max_items,
                    max_memory_mb=max_memory_mb,
                    ttl=ttl
                )
        
        logger.info(f"Initialized LRU cache manager with {len(self.caches)} caches")
    
    def get_cache(self, name: str) -> LRUCache:
        """
        Get or create a cache by name.
        
        Args:
            name: Cache name
            
        Returns:
            LRUCache instance
        """
        with self.lock:
            if name not in self.caches:
                # Create a new cache with default settings
                self.caches[name] = LRUCache(
                    max_items=self.default_max_items,
                    max_memory_mb=self.default_max_memory_mb,
                    ttl=self.default_ttl
                )
                logger.debug(f"Created new cache: {name}")
            
            return self.caches[name]
    
    def get(self, name: str, key: Any) -> Optional[Any]:
        """
        Get a value from a named cache.
        
        Args:
            name: Cache name
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        cache = self.get_cache(name)
        return cache.get(key)
    
    def set(self, name: str, key: Any, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in a named cache.
        
        Args:
            name: Cache name
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        cache = self.get_cache(name)
        return cache.set(key, value, ttl)
    
    def delete(self, name: str, key: Any) -> bool:
        """
        Delete a value from a named cache.
        
        Args:
            name: Cache name
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if name in self.caches:
            cache = self.caches[name]
            return cache.delete(key)
        return False
    
    def clear(self, name: Optional[str] = None) -> int:
        """
        Clear a named cache or all caches.
        
        Args:
            name: Cache name or None to clear all
            
        Returns:
            Number of items cleared
        """
        with self.lock:
            if name is not None:
                if name in self.caches:
                    return self.caches[name].clear()
                return 0
            
            # Clear all caches
            total = 0
            for cache_name, cache in self.caches.items():
                count = cache.clear()
                total += count
                logger.debug(f"Cleared {count} items from cache: {cache_name}")
            
            logger.info(f"Cleared {total} items from all caches")
            return total
    
    def create_cache(self, name: str, max_items: int, max_memory_mb: Optional[int] = None, 
                     ttl: Optional[int] = None) -> LRUCache:
        """
        Create a new cache with specific settings.
        
        Args:
            name: Cache name
            max_items: Maximum number of items
            max_memory_mb: Maximum memory usage in MB
            ttl: Default time-to-live in seconds
            
        Returns:
            New LRUCache instance
        """
        with self.lock:
            cache = LRUCache(
                max_items=max_items,
                max_memory_mb=max_memory_mb,
                ttl=ttl
            )
            self.caches[name] = cache
            logger.debug(f"Created new cache: {name} (max_items={max_items}, "
                        f"max_memory_mb={max_memory_mb}, ttl={ttl})")
            return cache
    
    def resize_cache(self, name: str, max_items: int, max_memory_mb: Optional[int] = None) -> bool:
        """
        Resize an existing cache.
        
        Args:
            name: Cache name
            max_items: New maximum number of items
            max_memory_mb: New maximum memory usage in MB
            
        Returns:
            True if successful, False if cache not found
        """
        with self.lock:
            if name in self.caches:
                self.caches[name].resize(max_items, max_memory_mb)
                return True
            return False
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.
        
        Returns:
            Dictionary mapping cache names to their statistics
        """
        with self.lock:
            stats = {}
            for name, cache in self.caches.items():
                stats[name] = cache.stats()
            return stats
    
    def get_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific cache.
        
        Args:
            name: Cache name
            
        Returns:
            Statistics dictionary or None if cache not found
        """
        with self.lock:
            if name in self.caches:
                return self.caches[name].stats()
            return None
    
    def get_memory_usage(self) -> int:
        """
        Get total estimated memory usage of all caches.
        
        Returns:
            Total memory usage in bytes
        """
        with self.lock:
            total = 0
            for cache in self.caches.values():
                total += cache.estimated_memory_usage
            return total
    
    def contains(self, name: str, key: Any) -> bool:
        """
        Check if a key exists in a named cache.
        
        Args:
            name: Cache name
            key: Cache key
            
        Returns:
            True if the key exists, False otherwise
        """
        if name in self.caches:
            return self.caches[name].contains(key)
        return False
    
    def get_hash(self, data: Any) -> str:
        """
        Generate a hash for cache keys.
        
        Args:
            data: Data to hash
            
        Returns:
            String hash
        """
        if isinstance(data, str):
            return hashlib.sha256(data.encode('utf-8')).hexdigest()
        
        if isinstance(data, (int, float, bool)):
            return hashlib.sha256(str(data).encode('utf-8')).hexdigest()
        
        try:
            # Try to serialize to JSON for complex objects
            json_str = json.dumps(data, sort_keys=True)
            return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        except:
            # Fallback for non-JSON serializable objects
            return hashlib.sha256(str(data).encode('utf-8')).hexdigest()


# Single global instance
_cache_manager = LRUCacheManager()

def get_cache_manager() -> LRUCacheManager:
    """
    Get the global LRU cache manager instance.
    
    Returns:
        Global LRUCacheManager instance
    """
    return _cache_manager
