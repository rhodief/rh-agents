# Caching Quick Reference Card

## 🚀 Quick Start

```python
from rh_agents.cache_backends import FileCacheBackend
from rh_agents.core.execution import ExecutionState

# Setup cache
cache = FileCacheBackend(".cache")
state = ExecutionState(cache_backend=cache)

# That's it! LLM calls now cached automatically
```

## 🎯 Default Behavior

| Actor Type | Cached? | TTL    | Reason                    |
|------------|---------|--------|---------------------------|
| **LLM**    | ✅ YES  | 1 hour | Expensive API calls       |
| **Tool**   | ❌ NO   | N/A    | May have side effects     |
| **Agent**  | ❌ NO   | N/A    | Orchestration logic       |

## ⚙️ Configuration

### Enable caching for a Tool
```python
class MyTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            cacheable=True,      # Enable
            cache_ttl=1800,      # 30 min
            version="1.0.0"      # For invalidation
        )
```

### Customize LLM caching
```python
llm = OpenAILLM(
    cacheable=True,    # Default
    cache_ttl=7200,    # 2 hours
    version="1.1.0"    # Bump to invalidate
)
```

### Disable caching
```python
llm = OpenAILLM(cacheable=False)
```

## 📦 Cache Backends

### In-Memory (Fast, Non-Persistent)
```python
from rh_agents.cache_backends import InMemoryCacheBackend
cache = InMemoryCacheBackend()
```

### File-Based (Persistent)
```python
from rh_agents.cache_backends import FileCacheBackend
cache = FileCacheBackend(".cache/executions")
```

## 🔍 Check if Cached

```python
# In event handler
if event.from_cache:
    print("✨ Result from cache!")

# Check status
if event.execution_status == ExecutionStatus.RECOVERED:
    print("♻️ Recovered from cache")
```

## 📊 Statistics

```python
# Get cache stats
stats = cache_backend.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Size: {stats['size']} entries")
```

## 🗑️ Cache Management

### Clear specific entry
```python
cache.invalidate(cache_key)
```

### Pattern-based invalidation
```python
# Invalidate all LLM calls
cache.invalidate_pattern("*::OpenAI-LLM::*")

# Invalidate all v1.0.0 results
cache.invalidate_pattern("*::1.0.0::*")

# Invalidate all entries for specific actor
cache.invalidate_pattern("*::MyActor::*")
```

### Clear all
```python
cache.clear()
```

## 🔄 Version Management

```python
# Bump version to invalidate old cache
llm = OpenAILLM(version="1.0.0")  # Old
# ... make changes ...
llm = OpenAILLM(version="1.1.0")  # New - won't use old cache
```

## 📈 Performance

| Operation              | In-Memory | File-Based | LLM API  |
|------------------------|-----------|------------|----------|
| Cache hit              | < 1ms     | ~5-10ms    | -        |
| Cache miss (overhead)  | < 2ms     | ~15ms      | -        |
| LLM call (typical)     | -         | -          | 1-5s     |
| **Speedup**            | **1000x** | **200x**   | -        |

## 🎨 Visual Indicators

In event logs:
```
♻ 🧠 OpenAI-LLM [RECOVERED] (0μs)
  ├─ 💾 [FROM CACHE] {...}
  ├─ ✨ Recovered from cache (saved at 2025-12-22T10:30:45)
  └────────────────────────────────────────
```

## ⚠️ Best Practices

### ✅ DO
- Cache LLM calls (default)
- Use file cache in production
- Set appropriate TTLs
- Bump versions when changing behavior
- Monitor hit rates

### ❌ DON'T
- Cache operations with side effects
- Cache time-sensitive operations
- Cache authentication checks
- Use unbounded TTLs for dynamic data

## 🐛 Troubleshooting

### Cache not working?
1. ✓ Check `cacheable=True` on actor
2. ✓ Verify `cache_backend` set on ExecutionState
3. ✓ Check TTL hasn't expired
4. ✓ Verify input data is identical

### Wrong results?
```python
# Bump version to force re-execution
actor.version = "1.1.0"
```

### Cache too large?
```python
# Use shorter TTLs
llm = OpenAILLM(cache_ttl=300)  # 5 minutes

# Or clear periodically
cache.clear()
```

## 📚 See Also

- [CACHING.md](CACHING.md) - Detailed documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details
- [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) - Visual diagrams
- [examples/cached_index.py](../examples/cached_index.py) - Working example
- [tests/test_caching.py](../tests/test_caching.py) - Test suite

---

**TIP**: Run your script twice to see caching in action! First run caches, second run uses cache. 🚀
