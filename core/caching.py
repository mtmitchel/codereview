"""
Caching system for Code Review Tool.

This module provides flexible caching strategies for review results and
intermediate data to improve performance.
"""

import os
import json
import hashlib
import base64
import time
import logging
import shutil
import threading
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Set

from core.events import EventEmitter, get_event_bus

logger = logging.getLogger("CodeReviewTool.Cache")

# Default cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".code_reviewer_cache")

# For custom class serialization/deserialization
class JsonSerializer:
    """Helper class for JSON serialization of complex objects."""
    
    @staticmethod
    def serialize(obj):
        """Convert Python objects into JSON serializable objects."""
        if isinstance(obj, (set, frozenset)):
            return {"__type__": "set", "items": list(obj)}
        if isinstance(obj, bytes):
            return {"__type__": "bytes", "value": base64.b64encode(obj).decode('ascii')}
        # Add more custom type handlers as needed
        return obj
    
    @staticmethod
    def deserialize(obj):
        """Convert JSON serialized objects back to Python objects."""
        if isinstance(obj, dict) and "__type__" in obj:
            if obj["__type__"] == "set":
                return set(obj["items"])
            if obj["__type__"] == "bytes":
                return base64.b64decode(obj["value"])
        return obj
    
    @classmethod
    def dumps(cls, obj):
        """Serialize object to JSON string."""
        return json.dumps(obj, default=cls.serialize)
    
    @classmethod
    def loads(cls, json_str):
        """Deserialize JSON string to object."""
        return json.loads(json_str, object_hook=cls.deserialize)


class CacheStrategy(ABC):
    """Abstract base class for different cache strategies."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """
        Clear all cached values.
        
        Returns:
            Number of entries cleared
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        pass


class FileSystemCache(CacheStrategy):
    """File system-based cache strategy using JSON files."""
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        """
        Initialize the file system cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        logger.debug(f"Initialized file system cache in {cache_dir}")
    
    def _get_cache_file_path(self, key: str) -> str:
        """
        Get the cache file path for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Cache file path
        """
        # Use the key as the filename, escaping any problematic characters
        safe_key = key.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{safe_key}.json")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self.lock:
            cache_file = self._get_cache_file_path(key)
            
            if not os.path.exists(cache_file):
                self.miss_count += 1
                return None
            
            try:
                with open(cache_file, 'r') as f:
                    # Format: [value, expiry_time or null]
                    cached_data = JsonSerializer.loads(f.read())
                    
                    # Check if data has expired
                    if isinstance(cached_data, list) and len(cached_data) >= 2:
                        value, expiry_time = cached_data
                        
                        if expiry_time is not None and time.time() > expiry_time:
                            # Data has expired, delete it
                            logger.debug(f"Cache entry expired for key: {key}")
                            os.remove(cache_file)
                            self.miss_count += 1
                            return None
                        
                        self.hit_count += 1
                        return value
                    else:
                        # Legacy format or invalid data
                        self.hit_count += 1
                        return cached_data
                        
            except Exception as e:
                logger.warning(f"Error reading from cache: {e}")
                self.miss_count += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            cache_file = self._get_cache_file_path(key)
            
            try:
                # Calculate expiry time if TTL is provided
                expiry_time = None
                if ttl is not None:
                    expiry_time = time.time() + ttl
                
                # Save with expiry time
                with open(cache_file, 'w') as f:
                    json_data = JsonSerializer.dumps([value, expiry_time])
                    f.write(json_data)
                
                logger.debug(f"Cached data for key: {key}")
                return True
                
            except Exception as e:
                logger.warning(f"Error writing to cache: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            cache_file = self._get_cache_file_path(key)
            
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    logger.debug(f"Deleted cache entry for key: {key}")
                    return True
                except Exception as e:
                    logger.warning(f"Error deleting cache entry: {e}")
                    return False
            
            return False
    
    def clear(self) -> int:
        """
        Clear all cached values.
        
        Returns:
            Number of entries cleared
        """
        with self.lock:
            count = 0
            
            if not os.path.exists(self.cache_dir):
                return 0
            
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, file_name)
                    try:
                        os.remove(file_path)
                        count += 1
                    except Exception as e:
                        logger.warning(f"Error removing cache file {file_path}: {e}")
            
            # Reset stats
            self.hit_count = 0
            self.miss_count = 0
            
            logger.info(f"Cleared {count} cache entries")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_access = self.hit_count + self.miss_count
            hit_ratio = 0
            if total_access > 0:
                hit_ratio = self.hit_count / total_access
            
            # Count and calculate size of cache entries
            entry_count = 0
            total_size = 0
            
            try:
                for file_name in os.listdir(self.cache_dir):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(self.cache_dir, file_name)
                        entry_count += 1
                        total_size += os.path.getsize(file_path)
            except Exception as e:
                logger.warning(f"Error calculating cache stats: {e}")
            
            return {
                'hits': self.hit_count,
                'misses': self.miss_count,
                'hit_ratio': hit_ratio,
                'entry_count': entry_count,
                'size_bytes': total_size,
                'type': 'file_system'
            }


class SQLiteCache(CacheStrategy):
    """SQLite-based cache strategy for larger datasets."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the SQLite cache.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or os.path.join(CACHE_DIR, "cache.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
        # Initialize the database
        self._init_db()
        logger.debug(f"Initialized SQLite cache at {self.db_path}")
    
    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create the cache table if it doesn't exist
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    expiry INTEGER,
                    created INTEGER,
                    access_count INTEGER DEFAULT 0
                )
                ''')
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"Error initializing SQLite cache: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get the cached value
                cursor.execute(
                    'SELECT value, expiry, access_count FROM cache WHERE key = ?',
                    (key,)
                )
                result = cursor.fetchone()
                
                if result:
                    value_blob, expiry_time, access_count = result
                    
                    # Check if entry has expired
                    if expiry_time is not None and time.time() > expiry_time:
                        # Delete expired entry
                        cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
                        conn.commit()
                        conn.close()
                        
                        self.miss_count += 1
                        logger.debug(f"Cache entry expired for key: {key}")
                        return None
                    
                    # Update access count
                    cursor.execute(
                        'UPDATE cache SET access_count = ? WHERE key = ?',
                        (access_count + 1, key)
                    )
                    conn.commit()
                    
                    # Deserialize the value
                    value = JsonSerializer.loads(value_blob)
                    
                    conn.close()
                    self.hit_count += 1
                    return value
                
                conn.close()
                self.miss_count += 1
                return None
                
            except Exception as e:
                logger.warning(f"Error reading from SQLite cache: {e}")
                self.miss_count += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Serialize the value
                value_blob = JsonSerializer.dumps(value)
                
                # Calculate expiry time if TTL is provided
                expiry_time = None
                if ttl is not None:
                    expiry_time = int(time.time() + ttl)
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Insert or replace the cache entry
                created_time = int(time.time())
                cursor.execute(
                    '''
                    INSERT OR REPLACE INTO cache (key, value, expiry, created, access_count)
                    VALUES (?, ?, ?, ?, 0)
                    ''',
                    (key, value_blob, expiry_time, created_time)
                )
                
                conn.commit()
                conn.close()
                
                logger.debug(f"Cached data for key: {key}")
                return True
                
            except Exception as e:
                logger.warning(f"Error writing to SQLite cache: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Delete the cache entry
                cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
                deleted = cursor.rowcount > 0
                
                conn.commit()
                conn.close()
                
                if deleted:
                    logger.debug(f"Deleted cache entry for key: {key}")
                
                return deleted
                
            except Exception as e:
                logger.warning(f"Error deleting from SQLite cache: {e}")
                return False
    
    def clear(self) -> int:
        """
        Clear all cached values.
        
        Returns:
            Number of entries cleared
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Count entries before deletion
                cursor.execute('SELECT COUNT(*) FROM cache')
                count = cursor.fetchone()[0]
                
                # Delete all entries
                cursor.execute('DELETE FROM cache')
                
                conn.commit()
                conn.close()
                
                # Reset stats
                self.hit_count = 0
                self.miss_count = 0
                
                logger.info(f"Cleared {count} cache entries")
                return count
                
            except Exception as e:
                logger.warning(f"Error clearing SQLite cache: {e}")
                return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            stats = {
                'hits': self.hit_count,
                'misses': self.miss_count,
                'hit_ratio': 0,
                'entry_count': 0,
                'size_bytes': 0,
                'type': 'sqlite'
            }
            
            total_access = self.hit_count + self.miss_count
            if total_access > 0:
                stats['hit_ratio'] = self.hit_count / total_access
            
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Count entries
                cursor.execute('SELECT COUNT(*) FROM cache')
                stats['entry_count'] = cursor.fetchone()[0]
                
                # Calculate approximate size (SQLite doesn't have a direct way to do this)
                cursor.execute('SELECT SUM(length(value)) FROM cache')
                result = cursor.fetchone()
                if result[0] is not None:
                    stats['size_bytes'] = result[0]
                
                conn.close()
                
            except Exception as e:
                logger.warning(f"Error calculating SQLite cache stats: {e}")
            
            return stats


class InMemoryCache(CacheStrategy):
    """In-memory cache strategy for faster access."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the in-memory cache.
        
        Args:
            max_size: Maximum number of entries to store
        """
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, Optional[float]]] = {}  # (value, expiry_time)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        self.access_times: Dict[str, float] = {}  # Key -> last access time
        logger.debug(f"Initialized in-memory cache with max size {max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            value, expiry_time = self.cache[key]
            
            # Check if entry has expired
            if expiry_time is not None and time.time() > expiry_time:
                # Remove expired entry
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                
                self.miss_count += 1
                logger.debug(f"Cache entry expired for key: {key}")
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            
            self.hit_count += 1
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Check if we need to evict an entry
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_entry()
            
            # Calculate expiry time if TTL is provided
            expiry_time = None
            if ttl is not None:
                expiry_time = time.time() + ttl
            
            # Store the value
            self.cache[key] = (value, expiry_time)
            self.access_times[key] = time.time()
            
            logger.debug(f"Cached data for key: {key}")
            return True
    
    def _evict_entry(self) -> None:
        """Evict the least recently used entry."""
        if not self.access_times:
            return
        
        # Find the least recently used key
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        # Remove it
        if lru_key in self.cache:
            del self.cache[lru_key]
        if lru_key in self.access_times:
            del self.access_times[lru_key]
        
        logger.debug(f"Evicted LRU cache entry for key: {lru_key}")
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                
                logger.debug(f"Deleted cache entry for key: {key}")
                return True
            
            return False
    
    def clear(self) -> int:
        """
        Clear all cached values.
        
        Returns:
            Number of entries cleared
        """
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            self.access_times.clear()
            
            # Reset stats
            self.hit_count = 0
            self.miss_count = 0
            
            logger.info(f"Cleared {count} cache entries")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_access = self.hit_count + self.miss_count
            hit_ratio = 0
            if total_access > 0:
                hit_ratio = self.hit_count / total_access
            
            return {
                'hits': self.hit_count,
                'misses': self.miss_count,
                'hit_ratio': hit_ratio,
                'entry_count': len(self.cache),
                'max_size': self.max_size,
                'size_bytes': None,  # Not easily calculable in memory
                'type': 'in_memory'
            }


class CacheManager(EventEmitter):
    """
    Manager class for handling different cache strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cache manager.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.default_strategy_name = self.config.get('cache_strategy', 'file_system')
        self.cache_dir = self.config.get('cache_dir', CACHE_DIR)
        self.lock = threading.RLock()
        
        # Create cache strategies
        self.strategies: Dict[str, CacheStrategy] = {
            'file_system': FileSystemCache(self.cache_dir),
            'sqlite': SQLiteCache(os.path.join(self.cache_dir, "cache.db")),
            'in_memory': InMemoryCache(self.config.get('memory_cache_size', 1000))
        }
        
        logger.info(f"Initialized cache manager with default strategy: {self.default_strategy_name}")
    
    def get_strategy(self, strategy_name: Optional[str] = None) -> CacheStrategy:
        """
        Get a cache strategy by name.
        
        Args:
            strategy_name: Name of the strategy or None for default
            
        Returns:
            Cache strategy
        """
        with self.lock:
            name = strategy_name or self.default_strategy_name
            if name not in self.strategies:
                logger.warning(f"Strategy '{name}' not found, using file_system")
                name = 'file_system'
            
            return self.strategies[name]
    
    def get(self, key: str, strategy_name: Optional[str] = None) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            strategy_name: Name of the strategy to use or None for default
            
        Returns:
            Cached value or None if not found
        """
        strategy = self.get_strategy(strategy_name)
        return strategy.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, strategy_name: Optional[str] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            strategy_name: Name of the strategy to use or None for default
            
        Returns:
            True if successful, False otherwise
        """
        strategy = self.get_strategy(strategy_name)
        result = strategy.set(key, value, ttl)
        if result:
            self.emit_event("cache.set", {
                'key': key,
                'strategy': strategy_name or self.default_strategy_name
            })
        return result
    
    def delete(self, key: str, strategy_name: Optional[str] = None) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            strategy_name: Name of the strategy to use or None for default
            
        Returns:
            True if successful, False otherwise
        """
        strategy = self.get_strategy(strategy_name)
        result = strategy.delete(key)
        if result:
            self.emit_event("cache.delete", {
                'key': key,
                'strategy': strategy_name or self.default_strategy_name
            })
        return result
    
    def clear(self, strategy_name: Optional[str] = None) -> int:
        """
        Clear all values from a cache strategy.
        
        Args:
            strategy_name: Name of the strategy to clear or None for default
            
        Returns:
            Number of entries cleared
        """
        if strategy_name is None:
            # Clear all strategies
            total = 0
            for name, strategy in self.strategies.items():
                count = strategy.clear()
                total += count
                self.emit_event("cache.clear", {
                    'strategy': name,
                    'count': count
                })
            return total
        else:
            # Clear specific strategy
            strategy = self.get_strategy(strategy_name)
            count = strategy.clear()
            self.emit_event("cache.clear", {
                'strategy': strategy_name,
                'count': count
            })
            return count
    
    def get_stats(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            strategy_name: Name of the strategy to get stats for or None for all
            
        Returns:
            Dictionary with cache statistics
        """
        if strategy_name is None:
            # Get stats for all strategies
            stats = {}
            for name, strategy in self.strategies.items():
                stats[name] = strategy.get_stats()
            return stats
        else:
            # Get stats for specific strategy
            strategy = self.get_strategy(strategy_name)
            return strategy.get_stats()
    
    def get_file_hash(self, file_path: str, file_content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Calculate a unique hash for a file based on its path, content, and metadata.
        
        Args:
            file_path: Path to file
            file_content: String content of file
            metadata: Optional dict of additional metadata
            
        Returns:
            String hash representing the file
        """
        # Create a composite key from path and content
        composite = f"{file_path}:{file_content}"
        
        # If metadata is provided, include it in the hash calculation
        # Sort keys to ensure consistent hashing regardless of dict order
        if metadata:
            # Convert metadata to sorted JSON to ensure consistent serialization
            metadata_str = json.dumps(metadata, sort_keys=True)
            composite = f"{composite}:{metadata_str}"
        
        # Use SHA-256 instead of MD5 for better security
        return hashlib.sha256(composite.encode('utf-8')).hexdigest()
    
    def get_cached_review(self, file_path: str, file_content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached review results if available and valid.
        
        Args:
            file_path: Path to the file
            file_content: Content of the file
            metadata: Additional metadata that affects the review
            
        Returns:
            Cached review data if found, None otherwise
        """
        # Generate file hash for cache key
        file_hash = self.get_file_hash(file_path, file_content, metadata)
        
        # Try to get from cache
        return self.get(file_hash)
    
    def cache_review(self, file_path: str, file_content: str, review_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None, ttl: Optional[int] = None) -> bool:
        """
        Save review results to cache.
        
        Args:
            file_path: Path to the file
            file_content: Content of the file
            review_data: Review data to cache
            metadata: Additional metadata that affects the review
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if caching was successful, False otherwise
        """
        # Generate file hash for cache key
        file_hash = self.get_file_hash(file_path, file_content, metadata)
        
        # Save to cache
        return self.set(file_hash, review_data, ttl)


# Global cache manager instance
_cache_manager = CacheManager()

def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        Global CacheManager instance
    """
    return _cache_manager


# Legacy functions for backward compatibility
def get_file_hash(file_path, file_content, metadata=None):
    """
    Calculate a unique hash for a file based on its path, content, and metadata.
    
    Args:
        file_path: Path to file
        file_content: String content of file
        metadata: Optional dict of additional metadata
        
    Returns:
        String hash representing the file
    """
    # Create a composite key from path and content
    composite = f"{file_path}:{file_content}"
    
    # If metadata is provided, include it in the hash calculation
    # Sort keys to ensure consistent hashing regardless of dict order
    if metadata:
        # Convert metadata to sorted JSON to ensure consistent serialization
        metadata_str = json.dumps(metadata, sort_keys=True)
        composite = f"{composite}:{metadata_str}"
    
    # Use SHA-256 instead of MD5 for better security
    return hashlib.sha256(composite.encode('utf-8')).hexdigest()

def get_cached_review(file_path, file_content, metadata=None, cache_dir=CACHE_DIR):
    """
    Get a cached review for a file.
    
    Args:
        file_path: Path to file
        file_content: String content of file
        metadata: Optional dict of additional metadata
        cache_dir: Optional override for cache directory
        
    Returns:
        Cached review data or None if not found
    """
    file_hash = get_file_hash(file_path, file_content, metadata)
    cache_file = os.path.join(cache_dir, f"{file_hash}.json")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r') as f:
            cached_data = JsonSerializer.loads(f.read())
            if isinstance(cached_data, list) and len(cached_data) >= 2:
                review_data, expiry_time = cached_data
                
                # Check if data has expired
                if expiry_time is not None and time.time() > expiry_time:
                    # Data has expired
                    os.remove(cache_file)
                    return None
                
                return review_data
            else:
                # Legacy format
                return cached_data
    except Exception as e:
        logger.warning(f"Error reading cached review: {e}")
        return None

def cache_review(file_path, file_content, review_data, metadata=None, cache_dir=CACHE_DIR):
    """
    Cache a review for a file.
    
    Args:
        file_path: Path to file
        file_content: String content of file
        review_data: Review data to cache
        metadata: Optional dict of additional metadata
        cache_dir: Optional override for cache directory
        
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_hash = get_file_hash(file_path, file_content, metadata)
    cache_file = os.path.join(cache_dir, f"{file_hash}.json")
    
    try:
        with open(cache_file, 'w') as f:
            # Store review data with no expiry (None)
            f.write(JsonSerializer.dumps([review_data, None]))
        return True
    except Exception as e:
        logger.warning(f"Error caching review: {e}")
        return False

def clear_cache(cache_dir=CACHE_DIR):
    """
    Clear all cached reviews.
    
    Args:
        cache_dir: Optional override for cache directory
        
    Returns:
        Number of cache entries cleared
    """
    count = 0
    
    if not os.path.exists(cache_dir):
        return count
    
    for file_name in os.listdir(cache_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(cache_dir, file_name)
            try:
                os.remove(file_path)
                count += 1
            except Exception as e:
                logger.warning(f"Error removing cache file {file_path}: {e}")
    
    return count