"""DEPRECATED: This cache module is deprecated in favor of the state recovery system.

Please use rh_agents.core.state_recovery and rh_agents.state_backends instead.

Old cache system (hash-based):
- Used hash of inputs to cache results
- Required cache_backend parameter
- Limited to result caching only

New state recovery system (address-based):
- Uses execution address for event matching
- Supports full state save/restore
- Enables user-in-the-loop workflows
- Supports selective replay via resume_from_address
- Artifacts stored separately with content-addressable IDs

Migration guide:
1. Replace cache_backend with state_backend and artifact_backend
2. Use ExecutionState.save_checkpoint() instead of cache.set()
3. Use ExecutionState.load_from_state_id() instead of cache.get()
4. Events automatically replay with cached results

See docs/STATE_RECOVERY_SPEC.md for details.
"""

import warnings

warnings.warn(
    "rh_agents.core.cache is deprecated. Use rh_agents.core.state_recovery instead.",
    DeprecationWarning,
    stacklevel=2
)

# Original docstring preserved below:
"""
Cache abstraction for execution result caching and recovery.

This module contains the abstract base class and core utilities.
Concrete implementations are in rh_agents.cache_backends module.
"""
from __future__ import annotations
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar('T')


class CachedResult(BaseModel, Generic[T]):
    """Represents a cached execution result with metadata."""
    result: T
    cached_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    input_hash: str
    cache_key: str
    actor_name: str
    actor_version: str
    expires_at: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if the cached result has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > datetime.fromisoformat(self.expires_at)


class CacheBackend(ABC):
    """
    Abstract base class for cache storage backends.
    
    Concrete implementations should be placed in rh_agents.cache_backends module.
    """
    
    @abstractmethod
    def get(self, cache_key: str) -> Optional[CachedResult]:
        """Retrieve a cached result by key."""
        pass
    
    @abstractmethod
    def set(self, cache_key: str, cached_result: CachedResult, ttl: Optional[int] = None):
        """Store a cached result with optional TTL in seconds."""
        pass
    
    @abstractmethod
    def invalidate(self, cache_key: str) -> bool:
        """Invalidate a specific cache entry. Returns True if entry existed."""
        pass
    
    @abstractmethod
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all cache entries matching a pattern. Returns count of invalidated entries."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all cached entries."""
        pass
    
    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics (hits, misses, size, etc.)."""
        pass


def compute_cache_key(address: str, input_data: Any, actor_name: str, actor_version: str) -> tuple[str, str]:
    """
    Compute a cache key and input hash for an execution.
    
    Args:
        address: Execution address (context path)
        input_data: Input data to hash
        actor_name: Name of the actor
        actor_version: Version of the actor
    
    Returns:
        Tuple of (cache_key, input_hash)
    """
    # Serialize input data
    if isinstance(input_data, BaseModel):
        try:
            # Try to serialize with exclude_unset to skip defaults
            input_str = input_data.model_dump_json(exclude_unset=False)
            # Sort the JSON string by re-parsing and dumping with sorted keys
            input_dict = json.loads(input_str)
            input_str = json.dumps(input_dict, sort_keys=True)
        except Exception as e:
            # If serialization fails, try with custom serialization that excludes non-serializable fields
            try:
                # Use model_dump with mode='json' to get JSON-compatible dict
                input_dict = input_data.model_dump(mode='json')
                # Remove any remaining non-serializable items (like functions, complex objects)
                input_dict = _make_json_serializable(input_dict)
                input_str = json.dumps(input_dict, sort_keys=True)
            except Exception:
                # Last resort: use string representation
                input_str = str(input_data)
    elif isinstance(input_data, (dict, list)):
        input_dict = _make_json_serializable(input_data)
        input_str = json.dumps(input_dict, sort_keys=True, default=str)
    else:
        input_str = str(input_data)
    
    # Compute input hash
    input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]
    
    # Compute cache key from all components
    cache_content = f"{address}::{actor_name}::{actor_version}::{input_hash}"
    cache_key = hashlib.sha256(cache_content.encode()).hexdigest()
    
    return cache_key, input_hash


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert an object to a JSON-serializable form.
    Removes non-serializable items like functions, complex objects, etc.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            try:
                # Try to serialize the value
                serialized = _make_json_serializable(value)
                # Test if it's actually serializable
                json.dumps(serialized)
                result[key] = serialized
            except (TypeError, ValueError):
                # Skip non-serializable values
                continue
        return result
    elif isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            try:
                serialized = _make_json_serializable(item)
                json.dumps(serialized)
                result.append(serialized)
            except (TypeError, ValueError):
                continue
        return result
    elif isinstance(obj, BaseModel):
        try:
            return _make_json_serializable(obj.model_dump(mode='json'))
        except Exception:
            return str(obj)
    else:
        # For other types, try to convert to string
        try:
            return str(obj)
        except Exception:
            return None
