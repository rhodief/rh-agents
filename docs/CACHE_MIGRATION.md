# Migrating from Cache System to State Recovery

**Status:** Cache system deprecated as of Phase 3  
**Replacement:** State Recovery System (rh_agents.core.state_recovery)

---

## Why Migrate?

The old hash-based cache system has been replaced with a more powerful **state recovery system** that provides:

### Old Cache System (Deprecated)
- ❌ Hash-based result caching only
- ❌ No execution state persistence
- ❌ Limited to single-session recovery
- ❌ No support for user-in-the-loop workflows
- ❌ Cannot resume from specific points
- ❌ No execution history tracking

### New State Recovery System
- ✅ Full execution state persistence (history, storage, artifacts)
- ✅ Smart replay with automatic event skipping
- ✅ Resume from specific address (selective replay)
- ✅ Cross-session state restoration
- ✅ User-in-the-loop workflows (pause and resume)
- ✅ Error recovery (restore and retry)
- ✅ Complete audit trail with metadata
- ✅ State diffing and comparison
- ✅ Validation mode for determinism testing

---

## Migration Guide

### Before: Old Cache System

```python
from rh_agents.cache_backends import FileCacheBackend
from rh_agents.core.execution import ExecutionState

# Old approach - cache backend for result caching
cache_backend = FileCacheBackend(cache_dir=".cache")
state = ExecutionState(cache_backend=cache_backend)

# Execute events - results cached by input hash
result1 = await event1(input_data, None, state)
result2 = await event2(result1, None, state)

# Cache hits on subsequent runs with same inputs
result1_cached = await event1(input_data, None, state)  # From cache
```

### After: State Recovery System

```python
from rh_agents.state_backends import FileSystemStateBackend, FileSystemArtifactBackend
from rh_agents.core.execution import ExecutionState
from rh_agents.core.state_recovery import StateStatus, StateMetadata

# New approach - state recovery for full execution persistence
state_backend = FileSystemStateBackend(".state_store")
artifact_backend = FileSystemArtifactBackend(".state_store/artifacts")

state = ExecutionState(
    state_backend=state_backend,
    artifact_backend=artifact_backend
)

# Execute events
result1 = await event1(input_data, None, state)
result2 = await event2(result1, None, state)

# Save checkpoint manually
state.save_checkpoint(
    status=StateStatus.PAUSED,
    metadata=StateMetadata(tags=["checkpoint1"])
)

# Later: restore and replay
restored_state = ExecutionState.load_from_state_id(
    state_id=state.state_id,
    state_backend=state_backend,
    artifact_backend=artifact_backend
)

# Events automatically skip if already executed
result1_replayed = await event1(input_data, None, restored_state)  # Skipped, instant
result2_replayed = await event2(result1, None, restored_state)     # Skipped, instant
result3_new = await event3(result2, None, restored_state)          # Executes fresh
```

---

## Key Differences

| Feature | Old Cache | State Recovery |
|---------|-----------|----------------|
| **Matching** | Input hash | Execution address |
| **Persistence** | Results only | Full state (history, storage, artifacts) |
| **Recovery** | Same inputs required | Any point in execution |
| **Lifecycle** | Single session | Cross-session |
| **Metadata** | None | Tags, lineage, custom fields |
| **Selective Replay** | No | Yes (resume_from_address) |
| **Validation** | No | Yes (compare results) |

---

## Migration Steps

### 1. Update Imports

**Before:**
```python
from rh_agents.cache_backends import FileCacheBackend, InMemoryCacheBackend
```

**After:**
```python
from rh_agents.state_backends import FileSystemStateBackend, FileSystemArtifactBackend
from rh_agents.core.state_recovery import StateStatus, StateMetadata, ReplayMode
```

### 2. Replace Backend Initialization

**Before:**
```python
cache_backend = FileCacheBackend(cache_dir=".cache")
state = ExecutionState(cache_backend=cache_backend)
```

**After:**
```python
state_backend = FileSystemStateBackend(".state_store")
artifact_backend = FileSystemArtifactBackend(".state_store/artifacts")
state = ExecutionState(
    state_backend=state_backend,
    artifact_backend=artifact_backend
)
```

### 3. Replace Manual Caching with Checkpoints

**Before:**
```python
# Cache automatically stores on execution
result = await event(input_data, None, state)
# Result cached by input hash
```

**After:**
```python
# Execute normally
result = await event(input_data, None, state)

# Explicitly save checkpoint when needed
state.save_checkpoint(
    status=StateStatus.PAUSED,
    metadata=StateMetadata(tags=["after-event1"])
)
```

### 4. Replace Cache Retrieval with State Restoration

**Before:**
```python
# Re-run with same inputs to hit cache
result_cached = await event(input_data, None, state)
```

**After:**
```python
# Restore from checkpoint
restored_state = ExecutionState.load_from_state_id(
    state_id=state.state_id,
    state_backend=state_backend,
    artifact_backend=artifact_backend
)

# Replay events - automatic skipping
result_replayed = await event(input_data, None, restored_state)
```

### 5. Remove Actor Caching Configuration

**Before:**
```python
actor = Tool(
    name="my_tool",
    handler=handler_func,
    cacheable=True,  # ← Remove this
    cache_ttl=3600,  # ← Remove this
    # ...
)
```

**After:**
```python
actor = Tool(
    name="my_tool",
    handler=handler_func,
    # Caching removed - use state recovery instead
    # ...
)
```

---

## Advanced Features

### User-in-the-Loop Workflows

```python
# Step 1: Execute until user input needed
state = ExecutionState(state_backend=state_backend, artifact_backend=artifact_backend)

result1 = await validate_data_event(user_input, None, state)
result2 = await analyze_data_event(result1, None, state)

# Save and wait for user decision
state.save_checkpoint(
    status=StateStatus.AWAITING,
    metadata=StateMetadata(
        tags=["awaiting-user-approval"],
        custom_data={"user_id": "123"}
    )
)

# ... user makes decision in UI ...

# Step 2: Resume with user's decision
restored_state = ExecutionState.load_from_state_id(
    state_id=state.state_id,
    state_backend=state_backend,
    artifact_backend=artifact_backend
)

# Previous events skipped instantly
await validate_data_event(user_input, None, restored_state)
await analyze_data_event(result1, None, restored_state)

# New work with user's decision
result3 = await process_decision_event(user_decision, None, restored_state)
```

### Error Recovery

```python
# Pipeline fails at step 3
try:
    result1 = await event1(data, None, state)
    result2 = await event2(result1, None, state)
    result3 = await event3(result2, None, state)  # FAILS
except Exception as e:
    state.save_checkpoint(
        status=StateStatus.FAILED,
        metadata=StateMetadata(tags=["failed-at-event3"])
    )

# Later: retry from checkpoint
restored_state = ExecutionState.load_from_state_id(
    state_id=state.state_id,
    state_backend=state_backend,
    artifact_backend=artifact_backend
)

# Steps 1-2 replay instantly (skipped)
await event1(data, None, restored_state)
await event2(result1, None, restored_state)

# Step 3 retries fresh
result3 = await event3(result2, None, restored_state)
```

### Selective Replay (Debug Mode)

```python
# User reports issue with step 3 output
# Developer wants to re-run step 3 with different parameters

restored_state = ExecutionState.load_from_state_id(
    state_id=reported_state_id,
    state_backend=state_backend,
    artifact_backend=artifact_backend,
    resume_from_address="process_data::tool_call"  # Step 3 address
)

# Steps 1-2: skipped (cached)
await event1(data, None, restored_state)
await event2(result1, None, restored_state)

# Step 3: re-executed with modified parameters
event3_debug = ExecutionEvent(actor=ProcessDataActor(debug_mode=True))
result3_debug = await event3_debug(result2, None, restored_state)

# All subsequent steps re-execute with new data
await event4(result3_debug, None, restored_state)
```

---

## FAQ

### Q: Do I lose all my cached data?

**A:** The cache system is separate from state recovery. Old cached results will remain in `.cache` directory but won't be used. State recovery creates new snapshots in `.state_store` directory.

### Q: Can I use both systems during transition?

**A:** No. The cache_backend parameter has been removed from ExecutionState. You must migrate to state recovery.

### Q: How do I query saved states?

**A:**
```python
# List states with filters
backend = FileSystemStateBackend(".state_store")
states = backend.list_states(
    tags=["production"],
    status=StateStatus.PAUSED,
    limit=10
)

for snapshot in states:
    print(f"State {snapshot.state_id}: {snapshot.metadata.description}")
```

### Q: How do I delete old states?

**A:**
```python
backend = FileSystemStateBackend(".state_store")
backend.delete_state(state_id)
```

### Q: What about artifacts?

**A:** Artifacts are now stored separately in the artifact backend with content-addressable IDs. They're automatically referenced in snapshots and restored on load.

### Q: How does replay matching work?

**A:** 
- **Old cache**: Matched by hash of inputs → same inputs = cache hit
- **New state recovery**: Matched by execution address → same event in pipeline = skip

The address-based approach is more reliable for stateless pipelines where you want to resume from a specific point regardless of input changes.

---

## Breaking Changes

1. **Removed `cache_backend` parameter** from ExecutionState.__init__()
2. **Removed `cacheable`, `cache_ttl` fields** from actor configuration (still present but ignored)
3. **Removed cache methods** from ExecutionEvent:
   - `__try_retrieve_from_cache()`
   - `__store_result_in_cache()`
4. **Deprecated modules** (still importable but show warnings):
   - `rh_agents.core.cache`
   - `rh_agents.cache_backends`

---

## Resources

- **State Recovery Spec**: [docs/STATE_RECOVERY_SPEC.md](./STATE_RECOVERY_SPEC.md)
- **Implementation Plan**: [docs/IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)
- **Phase 2 Summary**: [docs/PHASE2_SUMMARY.md](./PHASE2_SUMMARY.md)
- **Progress Log**: [docs/PROGRESS.md](./PROGRESS.md)

---

## Need Help?

The state recovery system is more powerful but also more explicit. If you need assistance migrating:

1. Review the examples in this guide
2. Check [docs/PHASE2_SUMMARY.md](./PHASE2_SUMMARY.md) for detailed usage patterns
3. Run the test suite for working examples:
   - `python test_phase1_state_recovery.py`
   - `python test_phase2_replay.py`
