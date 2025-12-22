# Artifact System Implementation

## Overview
The artifact system provides a specialized caching mechanism for actors that produce important state artifacts (like Doctrine objects) that should be stored in the ExecutionState rather than in the regular cache backend.

## Key Components

### 1. ExecutionStore Enhancement
**File:** [rh_agents/core/execution.py](../rh_agents/core/execution.py)

Added artifact storage capabilities to `ExecutionStore`:
- `artifacts: dict[str, Any]` - Dictionary to store artifacts
- `get_artifact(key: str)` - Retrieve an artifact by key
- `set_artifact(key: str, value: Any)` - Store an artifact by key
- `has_artifact(key: str)` - Check if an artifact exists

### 2. Actor Configuration
**File:** [rh_agents/core/actors.py](../rh_agents/core/actors.py)

Added `is_artifact` parameter to all actor types:
- `BaseActor.is_artifact: bool` - Flag to mark actors that produce artifacts
- Default is `False` for all actors
- Agents, Tools, and LLMs can be marked as artifact producers

### 3. Cache Logic Enhancement
**File:** [rh_agents/core/events.py](../rh_agents/core/events.py)

#### Cache Retrieval (`__try_retrieve_from_cache`)
1. For artifact actors (`is_artifact=True`):
   - First checks `ExecutionState.storage.artifacts`
   - Does **not** require `cache_backend` to be set
   - Returns artifact directly from storage if found

2. For regular actors:
   - Checks the `cache_backend` as before
   - Requires `cache_backend` to be configured

#### Cache Storage (`__store_result_in_cache`)
1. For artifact actors:
   - Stores result in `ExecutionState.storage.artifacts`
   - Does **not** use the `cache_backend`
   - Persists for the lifetime of the ExecutionState

2. For regular actors:
   - Stores in `cache_backend` as before
   - Subject to TTL and cache eviction policies

## Usage

### Marking an Actor as Artifact Producer

```python
class DoctrineReceverAgent(Agent):
    def __init__(self, llm: LLM, tools: list[Tool] = None):
        # ... handler implementation ...
        
        super().__init__(
            name="DoctrineReceverAgent",
            description=INTENT_PARSER_PROMPT,
            input_model=Message,
            output_model=Doctrine,
            handler=handler,
            event_type=EventType.AGENT_CALL,
            llm=llm,
            tools=ToolSet(tools) if tools else ToolSet(),
            is_artifact=True,      # Mark as artifact producer
            cacheable=True         # Enable caching (required)
        )
```

### How It Works

1. **First Execution:**
   ```
   User Input → DoctrineReceverAgent → LLM → Doctrine
                                                    ↓
                                    ExecutionState.storage.artifacts[key] = Doctrine
   ```

2. **Subsequent Executions with Same Input:**
   ```
   User Input → DoctrineReceverAgent → Check artifacts storage
                                                    ↓
                                    Return cached Doctrine (no LLM call)
   ```

## Benefits

1. **No Cache Backend Required:** Artifacts work without configuring a cache backend
2. **Session Persistence:** Artifacts persist throughout the ExecutionState lifetime
3. **Fast Retrieval:** Direct memory access without serialization/deserialization
4. **State Consistency:** Multiple steps in same execution can access the same Doctrine
5. **Reduced LLM Calls:** Doctrine creation is expensive; caching saves API costs

## Example: Doctrine as Artifact

The `DoctrineReceverAgent` is marked as an artifact producer because:
- **Expensive to create:** Requires LLM call with function calling
- **Shared state:** Multiple execution steps may need the Doctrine
- **Session-scoped:** Valid for the entire execution session
- **Critical data:** Represents the execution plan and constraints

## Cache Key Generation

Artifacts use the same cache key computation as regular cache:
```
cache_key = hash(address + actor_name + actor_version + input_hash)
```

Where:
- `address`: Execution context path (e.g., `OmniAgent::DoctrineReceverAgent::agent_call`)
- `actor_name`: Name of the actor
- `actor_version`: Version for cache invalidation
- `input_hash`: Hash of the input data

## Testing

See [test_artifacts.py](../test_artifacts.py) for comprehensive tests:
- Artifact storage verification
- Artifact retrieval testing
- Doctrine agent configuration validation

## Implementation Notes

1. **Cacheable Requirement:** An actor must have `cacheable=True` for artifacts to work
2. **No Cache Backend Needed:** Artifacts bypass the cache backend entirely
3. **Memory-Based:** Artifacts are stored in memory within ExecutionState
4. **No TTL:** Artifacts don't expire; they live with the ExecutionState
5. **Thread-Safe:** ExecutionState is typically per-execution, so no concurrency issues

## Future Enhancements

Potential improvements:
1. Artifact serialization for persistence across sessions
2. Artifact versioning and migration
3. Artifact size limits and eviction policies
4. Cross-execution artifact sharing
5. Artifact dependency tracking
