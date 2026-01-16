"""DEPRECATED: This cache_backends module is deprecated.

Please use rh_agents.state_backends instead.

The new state recovery system provides superior functionality:
- Full execution state persistence (not just results)
- Smart replay with event skipping
- Selective replay via resume_from_address
- Artifact storage with content-addressable IDs
- State diffing and comparison
- Metadata tagging and querying

See docs/STATE_RECOVERY_SPEC.md for migration guide.
"""

import warnings

warnings.warn(
    "rh_agents.cache_backends is deprecated. Use rh_agents.state_backends instead.",
    DeprecationWarning,
    stacklevel=2
)

# Original implementation preserved for backward compatibility:
"""
Cache backend implementations for execution result caching.
"""
import json
from pathlib import Path
from typing import Any, Optional
from rh_agents.core.cache import CacheBackend, CachedResult


class InMemoryCacheBackend(CacheBackend):
    """In-memory cache backend for development and testing."""
    
    def __init__(self):
        self._cache: dict[str, CachedResult] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, cache_key: str) -> Optional[CachedResult]:
        cached = self._cache.get(cache_key)
        if cached is None:
            self._misses += 1
            return None
        
        if cached.is_expired():
            del self._cache[cache_key]
            self._misses += 1
            return None
        
        self._hits += 1
        return cached
    
    def set(self, cache_key: str, cached_result: CachedResult, ttl: Optional[int] = None):
        if ttl is not None:
            from datetime import datetime, timedelta
            cached_result.expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()
        self._cache[cache_key] = cached_result
    
    def invalidate(self, cache_key: str) -> bool:
        if cache_key in self._cache:
            del self._cache[cache_key]
            return True
        return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Simple pattern matching with * wildcard."""
        import fnmatch
        matching_keys = [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]
        for key in matching_keys:
            del self._cache[key]
        return len(matching_keys)
    
    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> dict[str, Any]:
        hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        return {
            "backend": "in_memory",
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate
        }


class FileCacheBackend(CacheBackend):
    """File-based cache backend for persistent storage."""
    
    def __init__(self, cache_dir: Path | str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        self._index_file = self.cache_dir / "cache_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r') as f:
                    data = json.load(f)
                    self._hits = data.get('hits', 0)
                    self._misses = data.get('misses', 0)
            except Exception:
                self._hits = 0
                self._misses = 0
    
    def _save_index(self):
        """Save cache index to disk."""
        try:
            from datetime import datetime
            with open(self._index_file, 'w') as f:
                json.dump({
                    'hits': self._hits,
                    'misses': self._misses,
                    'last_updated': datetime.now().isoformat()
                }, f)
        except Exception:
            pass
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, cache_key: str) -> Optional[CachedResult]:
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            self._misses += 1
            self._save_index()
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                
                # Reconstruct ExecutionResult from dict if needed
                if isinstance(data.get('result'), dict) and not isinstance(data['result'], type):
                    from rh_agents.core.events import ExecutionResult
                    exec_result_data = data['result']
                    
                    # Also reconstruct the nested result object (e.g., Doctrine) if it's a dict
                    if isinstance(exec_result_data.get('result'), dict):
                        # Try to reconstruct using Pydantic model validation
                        # This will work for any BaseModel subclass
                        try:
                            from pydantic import BaseModel, ValidationError
                            # Leave as dict - the actor's handler will need to handle this
                            # or we parse it later when we know the type
                            pass
                        except Exception:
                            pass
                    
                    data['result'] = ExecutionResult(**exec_result_data)
                
                cached = CachedResult(**data)
            
            if cached.is_expired():
                cache_path.unlink()
                self._misses += 1
                self._save_index()
                return None
            
            self._hits += 1
            self._save_index()
            return cached
            
        except Exception as e:
            self._misses += 1
            self._save_index()
            return None
    
    def set(self, cache_key: str, cached_result: CachedResult, ttl: Optional[int] = None):
        if ttl is not None:
            from datetime import datetime, timedelta
            cached_result.expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()
        
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(cached_result.model_dump(), f, indent=2)
        except Exception as e:
            # Silently fail on write errors
            pass
    
    def invalidate(self, cache_key: str) -> bool:
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all cache files matching pattern."""
        import fnmatch
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name == "cache_index.json":
                continue
            cache_key = cache_file.stem
            if fnmatch.fnmatch(cache_key, pattern):
                cache_file.unlink()
                count += 1
        return count
    
    def clear(self):
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name != "cache_index.json":
                cache_file.unlink()
        self._hits = 0
        self._misses = 0
        self._save_index()
    
    def get_stats(self) -> dict[str, Any]:
        hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files if f.name != "cache_index.json")
        
        return {
            "backend": "file",
            "cache_dir": str(self.cache_dir),
            "size": len(cache_files) - 1,  # Exclude index file
            "total_bytes": total_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate
        }
