# Execution Caching and Recovery System

This document describes the caching and recovery system for execution results in the RH Agents framework.

## Overview

The caching system allows expensive operations (especially LLM calls) to be cached and reused, improving performance and reducing costs. It uses a content-addressed approach where cache keys are computed from:

- Execution address (context path)
- Input data (hashed)
- Actor name
- Actor version

This ensures that cached results are only reused when all relevant factors match.

## Architecture

### Core Components

1. **CacheBackend (Abstract)** - Base interface for cache storage
2. **InMemoryCacheBackend** - Fast in-memory cache for development/testing
3. **FileCacheBackend** - Persistent file-based cache for production
4. **CachedResult** - Wrapper for cached execution results with metadata
5. **ExecutionEvent** - Enhanced to check cache before execution

### Cache Key Computation

```python
cache_key = SHA256(address :: actor_name :: actor_version :: input_hash)
```

Where:
- `address`: Full execution path (e.g., "OmniAgent::DoctrineReceverAgent::OpenAI-LLM::llm_call")
- `actor_name`: Name of the actor being executed
- `actor_version`: Version string for cache invalidation
- `input_hash`: SHA256 hash of input data (first 16 chars)

## Usage

### Basic Setup

```python
from rh_agents.cache_backends import FileCacheBackend
from rh_agents.core.execution import ExecutionState

# Create cache backend
cache_backend = FileCacheBackend(cache_dir=".cache")

# Create execution state with caching
execution_state = ExecutionState(cache_backend=cache_backend)
```

### Enabling Caching for Actors

By default:
- **LLM calls**: `cacheable=True`, `cache_ttl=3600` (1 hour)
- **Tool calls**: `cacheable=False` (may have side effects)
- **Agent calls**: `cacheable=False` (orchestration logic)

#### Enable caching for a Tool:

```python
class MyTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="A cacheable tool",
            input_model=MyInput,
            handler=my_handler,
            cacheable=True,      # Enable caching
            cache_ttl=1800,      # 30 minutes
            version="1.0.0"      # For cache invalidation
        )
```

#### Customize LLM caching:

```python
llm = OpenAILLM(
    cacheable=True,
    cache_ttl=7200,  # 2 hours
    version="1.1.0"  # Bump version to invalidate old cache
)
```

### Cache Backends

#### In-Memory Cache

Fast but non-persistent. Good for testing.

```python
from rh_agents.cache_backends import InMemoryCacheBackend

cache = InMemoryCacheBackend()
execution_state = ExecutionState(cache_backend=cache)
```

#### File-Based Cache

Persistent across runs. Good for production.

```python
from rh_agents.cache_backends import FileCacheBackend
from pathlib import Path

cache = FileCacheBackend(cache_dir=Path(".cache/executions"))
execution_state = ExecutionState(cache_backend=cache)
```

### Cache Management

#### Get Statistics

```python
stats = cache_backend.get_stats()
print(f"Cached entries: {stats['size']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

#### Invalidate Specific Entry

```python
cache_backend.invalidate(cache_key)
```

#### Invalidate by Pattern

```python
# Invalidate all entries for a specific actor
cache_backend.invalidate_pattern("*::OpenAI-LLM::*")

# Invalidate all entries for version 1.0.0
cache_backend.invalidate_pattern("*::1.0.0::*")
```

#### Clear All Cache

```python
cache_backend.clear()
```

## Execution Flow with Caching

1. **Event Creation**: `ExecutionEvent` is created with an actor
2. **Cache Key Computation**: Address, input, actor name, and version are hashed
3. **Cache Lookup**: Check if result exists in cache
4. **Cache Hit**: 
   - Result is returned immediately
   - `from_cache=True` flag is set
   - Status is `RECOVERED` instead of `COMPLETED`
   - Event is logged with `[CACHED]` prefix in detail
5. **Cache Miss**:
   - Normal execution proceeds
   - Result is computed
   - If `cacheable=True`, result is stored in cache

## Event Tracking

Cached executions are tracked differently in events:

```python
# Check if execution was cached
if event.from_cache:
    print(f"✨ {event.actor.name} recovered from cache!")
    print(f"   Saved execution time: {expected_time}s")

# Status will be RECOVERED instead of COMPLETED
assert event.execution_status == ExecutionStatus.RECOVERED
```

## Version Management

Use the `version` field to invalidate cache when actor behavior changes:

```python
# Version 1.0.0 - original implementation
llm = OpenAILLM(version="1.0.0")

# ... some time later, you change the system prompt ...

# Version 1.1.0 - new system prompt
llm = OpenAILLM(version="1.1.0")  # Old cached results won't be used
```

## Best Practices

### ✅ DO Cache:

- LLM calls with deterministic inputs
- Expensive API calls that return static data
- Database queries for immutable data
- File reads for unchanging files

### ❌ DON'T Cache:

- Operations with side effects (writes, sends, creates)
- Time-sensitive operations (`datetime.now()`)
- Random number generation
- User authentication checks
- Operations that depend on external state

### Cache TTL Guidelines:

| Operation Type | Suggested TTL |
|----------------|---------------|
| LLM calls | 1-2 hours |
| Static API data | 24 hours |
| Dynamic API data | 5-15 minutes |
| File reads | Until file changes |
| Database queries | Depends on update frequency |

## Cache Storage Structure

### File Cache Layout

```
.cache/
├── cache_index.json          # Statistics and metadata
├── a1b2c3d4e5f6...json      # Cached result 1
├── f6e5d4c3b2a1...json      # Cached result 2
└── ...
```

### Cached Result Format

```json
{
  "result": {
    "result": { ... },
    "execution_time": 2.456,
    "ok": true,
    "erro_message": null
  },
  "cached_at": "2025-12-22T10:30:45.123456",
  "input_hash": "a1b2c3d4e5f6...",
  "cache_key": "f6e5d4c3b2a1...",
  "actor_name": "OpenAI-LLM",
  "actor_version": "1.0.0",
  "expires_at": "2025-12-22T11:30:45.123456"
}
```

## Performance Considerations

### Memory Usage

- **In-Memory Cache**: Grows unbounded unless manually cleared
- **File Cache**: Uses disk space but doesn't consume RAM

### Cache Key Computation

Cache key computation is fast (< 1ms) using SHA256 hashing.

### Cache Lookup

- **In-Memory**: O(1) dictionary lookup
- **File**: O(1) filesystem lookup + JSON parsing

### Cache Write

- **In-Memory**: O(1) dictionary write
- **File**: O(1) JSON serialization + file write

## Monitoring and Debugging

### Enable Cache Logging

```python
# The EventPrinter automatically shows cached executions
printer = EventPrinter(show_timestamp=True, show_address=True)
bus.subscribe(printer)
```

Output:
```
[2025-12-22 10:30:45] 🔄 RECOVERED OpenAI-LLM
  Address: OmniAgent::DoctrineReceverAgent::OpenAI-LLM::llm_call
  Detail: [CACHED] {...}
  Message: Recovered from cache (saved at 2025-12-22T10:25:30)
```

### Cache Statistics

```python
stats = cache_backend.get_stats()
print(f"""
Cache Performance:
  Backend: {stats['backend']}
  Total entries: {stats['size']}
  Cache hits: {stats['hits']}
  Cache misses: {stats['misses']}
  Hit rate: {stats['hit_rate']:.2%}
""")
```

## Examples

See [examples/cached_index.py](../examples/cached_index.py) for a complete working example.

## Future Enhancements

Potential improvements for the caching system:

1. **Distributed Cache**: Redis/Memcached backend for multi-process scenarios
2. **Cache Warming**: Pre-populate cache with common queries
3. **Smart Invalidation**: Auto-invalidate based on time-series patterns
4. **Compression**: Compress large cached results
5. **Cache Analytics**: Dashboard for cache performance
6. **Partial Matching**: Fuzzy cache matching for similar inputs
7. **Cost Tracking**: Track $ saved via caching

## Troubleshooting

### Cache Not Working

1. Check `cacheable=True` on the actor
2. Verify `cache_backend` is set on `ExecutionState`
3. Check cache TTL hasn't expired
4. Verify input data is identical (including whitespace)

### Cache Growing Too Large

```python
# Clear old entries periodically
cache_backend.clear()

# Or use shorter TTLs
llm = OpenAILLM(cache_ttl=300)  # 5 minutes
```

### Wrong Results from Cache

Bump the actor version to invalidate old cache:

```python
llm = OpenAILLM(version="1.1.0")  # Forces re-execution
```
