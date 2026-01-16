# State Recovery System Specification

## Executive Summary

This specification outlines the conversion of the current hash-based cache system into a comprehensive state recovery system. The goal is to enable full serialization and restoration of `ExecutionState`, allowing pipelines to resume from specific checkpoints by replaying events intelligently—returning cached results for already-executed events and executing only new or incomplete events.

---

## Current System Analysis

### ExecutionState Serializability Assessment

**Current State:**
- `ExecutionState` is a Pydantic model, which provides inherent serialization capability
- Contains serializable components:
  - `storage: ExecutionStore` - Dictionary-based storage (serializable)
  - `history: HistorySet` - List of events (serializable)
  - `execution_stack: list[str]` - Context stack (serializable)
  
**Non-Serializable Components:**
- `event_bus: EventBus` - Contains:
  - `subscribers: list[Callable]` - Function references (NOT serializable)
  - `queue: asyncio.Queue` - Runtime async structure (NOT serializable)
- `_cache_backend: Optional[Any]` - External cache backend instance (NOT serializable)

**Verdict:** 
✅ **Partially Serializable** - Core execution data (history, storage, stack) can be serialized, but runtime components (event bus subscribers, async queue, cache backend) cannot and must be reconstructed on deserialization.

### Current Cache System Analysis

**How it Works:**
1. Cache key computed from: `hash(address + actor_name + actor_version + input_hash)`
2. Input hash computed from serialized input data
3. Two-tier storage for artifacts:
   - In-memory: `ExecutionStore.artifacts` (fast access within session)
   - Persistent: `CacheBackend` (cross-session persistence)
4. Regular actors use only `CacheBackend`

**Limitations for State Recovery:**
- Hash-based keys don't preserve execution state identity
- No way to identify "this specific execution run"
- No mechanism to replay events in order
- Cache is result-oriented, not state-oriented
- No tracking of execution progress/checkpoints

---

## Proposed State Recovery System

### Core Concept

Transform from **result caching** to **state recovery** by:
1. Adding **State ID** (UUID) to uniquely identify an execution session
2. Making `ExecutionState` fully serializable and restorable
3. Implementing **smart event replay** that checks execution status before re-execution
4. Adding **selective event publishing** to avoid duplicate bus notifications
5. Storing complete state snapshots for checkpoint/restore functionality

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    State Recovery System                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. State Snapshot                                          │
│     ┌──────────────────────────────────────────┐           │
│     │ StateID (UUID) → Serialized State        │           │
│     │  - History (all events)                  │           │
│     │  - Storage (data + artifacts)            │           │
│     │  - Execution Stack                       │           │
│     │  - Metadata (timestamp, version)         │           │
│     └──────────────────────────────────────────┘           │
│                                                              │
│  2. Event Replay Engine                                     │
│     ┌──────────────────────────────────────────┐           │
│     │ For each event in pipeline:              │           │
│     │   - Check if in restored history         │           │
│     │   - If exists & COMPLETED → use result   │           │
│     │   - If not exists → execute              │           │
│     │   - Track: republish_events flag         │           │
│     └──────────────────────────────────────────┘           │
│                                                              │
│  3. State Backend                                           │
│     ┌──────────────────────────────────────────┐           │
│     │ StateID → StateSnapshot                  │           │
│     │ - Store: save(state_id, snapshot)        │           │
│     │ - Retrieve: load(state_id)               │           │
│     │ - List: list_states(filters)             │           │
│     └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Design

### 1. State Identity and Metadata

**New Model: `StateSnapshot`**

```python
class StateSnapshot(BaseModel):
    state_id: str  # UUID4
    created_at: str  # ISO datetime
    updated_at: str  # ISO datetime
    execution_state: dict  # Serialized ExecutionState
    metadata: dict[str, Any]  # User-defined tags, labels
    version: str  # Schema version for migration
    status: StateStatus  # RUNNING, PAUSED, COMPLETED, FAILED
```

**New Enum: `StateStatus`**
```python
class StateStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
```

### 2. ExecutionState Modifications

**Add State ID Tracking:**
```python
class ExecutionState(BaseModel):
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    storage: ExecutionStore = Field(default_factory=ExecutionStore)
    history: HistorySet = Field(default_factory=HistorySet)
    execution_stack: list[str] = Field(default_factory=list)
    
    # Runtime components (excluded from serialization)
    event_bus: EventBus = Field(default_factory=EventBus, exclude=True)
    _cache_backend: Optional[Any] = Field(default=None, exclude=True)
    _state_backend: Optional[StateBackend] = Field(default=None, exclude=True)
    
    # Recovery control
    replay_mode: bool = Field(default=False)
    republish_events: bool = Field(default=True)
```

**Serialization Methods:**
```python
def to_snapshot(self) -> StateSnapshot:
    """Create a snapshot of current state"""
    
def from_snapshot(snapshot: StateSnapshot, 
                  event_bus: EventBus,
                  cache_backend: Optional[Any],
                  state_backend: Optional[StateBackend]) -> ExecutionState:
    """Restore state from snapshot, reconstructing runtime components"""
```

### 3. ExecutionEvent Modifications

**Add Replay Control Flag:**
```python
class ExecutionEvent(BaseModel, Generic[OutputT]):
    # ... existing fields ...
    
    # New field for controlling event publishing during replay
    skip_republish: bool = Field(
        default=False, 
        description="If True, skip publishing to event bus (used during state replay)"
    )
```

**Modified Event Publishing Logic:**
```python
async def add_event(self, event: Any, status: ExecutionStatus):
    """Add event to history and conditionally publish to event bus"""
    event.address = self.get_current_address(event.actor.event_type)
    event.execution_status = status
    self.history.add(event)
    
    # Only publish if not in skip mode
    if not event.skip_republish:
        await self.event_bus.publish(event)
```

### 4. Event Replay Logic

**Smart Replay in `ExecutionEvent.__call__`:**

```python
async def __call__(self, input_data, extra_context, execution_state: ExecutionState):
    # If in replay mode, check history first
    if execution_state.replay_mode:
        existing_event = execution_state.history.get_event_by_address_and_status(
            self.actor.name, 
            ExecutionStatus.COMPLETED
        )
        if existing_event is not None:
            # Event already completed - return cached result
            self.skip_republish = not execution_state.republish_events
            return existing_event.result
    
    # Not in replay OR event not found - execute normally
    # ... existing execution logic ...
```

### 5. State Backend Interface

**Abstract Class:**
```python
class StateBackend(ABC):
    @abstractmethod
    def save_state(self, snapshot: StateSnapshot) -> bool:
        """Persist state snapshot"""
        
    @abstractmethod
    def load_state(self, state_id: str) -> Optional[StateSnapshot]:
        """Load state snapshot by ID"""
        
    @abstractmethod
    def list_states(self, filters: dict[str, Any] = None) -> list[StateSnapshot]:
        """List available states with optional filtering"""
        
    @abstractmethod
    def delete_state(self, state_id: str) -> bool:
        """Delete a state snapshot"""
        
    @abstractmethod
    def update_state(self, snapshot: StateSnapshot) -> bool:
        """Update existing state snapshot"""
```

### 6. Recovery Workflow

**Saving State (Checkpoint):**
```python
# During execution, periodically save state
execution_state.save_checkpoint()  # Internally: state_backend.save_state(to_snapshot())
```

**Restoring State:**
```python
# Load state by ID
restored_state = ExecutionState.from_state_id(
    state_id="abc-123-def",
    state_backend=my_state_backend,
    event_bus=new_event_bus,
    cache_backend=my_cache_backend,
    replay_mode=True,
    republish_events=False  # Don't re-notify subscribers of old events
)

# Continue execution from checkpoint
result = await pipeline.run(input_data, execution_state=restored_state)
```

---

## Implementation Questions & Design Decisions

### Q1: State Backend Storage Location

**Issue:** Where should state snapshots be persisted?

**Options:**

**A) File System (JSON/Pickle)**
- **Pros:** Simple, no dependencies, human-readable (JSON), easy debugging
- **Cons:** No concurrent access control, no querying, file I/O overhead
- **Use Case:** Local development, single-process applications

**B) Database (SQLite/PostgreSQL)**
- **Pros:** ACID guarantees, query support, handles concurrency, metadata indexing
- **Cons:** Requires DB setup, serialization of complex types (artifacts)
- **Use Case:** Production systems, multi-process/distributed setups

**C) Key-Value Store (Redis/DynamoDB)**
- **Pros:** Fast access, distributed, TTL support, simple interface
- **Cons:** Size limits, additional infrastructure, serialization required
- **Use Case:** High-throughput distributed systems

**D) Reuse Existing CacheBackend**
- **Pros:** Single storage system, leverage existing infrastructure
- **Cons:** Mixing concerns (cache vs state), cache eviction could lose states
- **Use Case:** Simplify architecture if cache backend already robust

**Your Decision:** Create a abstraction for StateBackend and the actual store and retrive will be a implementation. You can create a file Adapter in order to have a basic implementation for MVP. About  the CacheBackend, remove it completly. With that recovery system, we no longer need it. 

---

### Q2: Event History Matching Strategy

**Issue:** When replaying, how do we match an event in the pipeline to an event in the restored history?

**Options:**

**A) By Address Only (Current)**
- **Pros:** Simple, already implemented, matches execution path
- **Cons:** Doesn't handle retries, can't distinguish multiple executions at same address
- **Implementation:** `history[address]` lookup

**B) By Address + Input Hash**
- **Pros:** Ensures same input = same cached result, deterministic
- **Cons:** Different input at same address won't match (is this desired?), compute overhead
- **Implementation:** `history[(address, input_hash)]` lookup

**C) By Address + Execution Order**
- **Pros:** Handles multiple executions at same address (loops), preserves order
- **Cons:** More complex tracking, address might not be stable across runs
- **Implementation:** `history[(address, execution_index)]` lookup

**D) By Actor Name + Input Hash (Ignoring Address)**
- **Pros:** Portable across execution paths, reusable results
- **Cons:** Loses context, could return wrong result if same actor+input used differently
- **Implementation:** `history[(actor_name, input_hash)]` lookup

**Your Decision:** This is important. You can use just "A", because this system state recovery is just to make the whole pipeline stateless. For ex: if something is processing and we need a user confirmation or something (user in the loop), it should be possible to send to user the message and when they answer, the pipeline should restore its state and use that new information to resume the processing. The same for error, we it was casted "replay", "retry", they just recover the last state and execute from that. If the user request to change same previous state, I'm gonna get the "address" of that state and passe it to the ExecutionState as "address_to_replace" to tell the system that from that point, everything will be run again regardless it could be run previously. 

---

### Q3: Handling Non-Deterministic Actors (LLMs, Random, Time-based)

**Issue:** Some actors produce different outputs each time (LLMs, random number generation, timestamps). How should replay handle these?

**Options:**

**A) Always Return Cached Result (Deterministic Replay)**
- **Pros:** True state recovery, reproducible results, debugging-friendly
- **Cons:** LLM responses frozen, no adaptation to new context
- **Implementation:** Ignore non-determinism, always use history

**B) Re-Execute Non-Deterministic Actors**
- **Pros:** Fresh LLM responses, adapts to current conditions
- **Cons:** Breaks true state recovery, results diverge from snapshot
- **Implementation:** Mark actors as `deterministic: bool`, skip cache if False

**C) User Choice Per Replay**
- **Pros:** Flexibility, different use cases (debug vs adapt)
- **Cons:** More API complexity, user must understand implications
- **Implementation:** `replay_mode: ReplayMode` enum (STRICT, ADAPTIVE, HYBRID)

**D) Versioned Actors + Cache Invalidation**
- **Pros:** Controlled re-execution when logic changes, keeps determinism otherwise
- **Cons:** Requires version management, doesn't solve inherent randomness
- **Implementation:** Increment `actor.version` when behavior changes

**Your Decision:** 
Checkout Q3 answer. This is not a state recovery strictly, but a stateless system. If store a state I'd like to recover them just to resume the execution. I can trust in everything passed in the recovery state, unless that flag to replay from a specfific address. 

---

### Q4: Event Republishing Strategy

**Issue:** When replaying, should events be republished to the event bus?

**Options:**

**A) Never Republish During Replay**
- **Pros:** Clean separation, no duplicate notifications, faster
- **Cons:** Observers miss historical context, can't reconstruct state from events
- **Implementation:** `skip_republish = True` during replay

**B) Always Republish All Events**
- **Pros:** Complete event stream, observers get full history, event sourcing compatible
- **Cons:** Duplicate processing, potential side effects, slower
- **Implementation:** `skip_republish = False` always

**C) Republish Only New Events**
- **Pros:** Observers see continuation, no duplicates, efficient
- **Cons:** Observers might need initialization with historical context
- **Implementation:** Track "last replayed event index", only publish beyond that

**D) Configurable Per Observer (Event Filtering)**
- **Pros:** Maximum flexibility, each observer decides
- **Cons:** Complex, requires observer metadata, harder to reason about
- **Implementation:** Observers register with `receive_replayed: bool` flag

**Your Decision:** Option C. (Republish Only New Events) as default, with global flag `republish_all: bool` for debugging. This preserves event stream continuity while avoiding duplicates. Mark replayed events with `event.is_replayed = True` so observers can distinguish if needed.

---

### Q5: Artifact Handling During State Recovery

**Issue:** Large artifacts (embeddings, models, datasets) in `ExecutionStore.artifacts` - how to serialize/restore?

**Options:**

**A) Inline Serialization (Pickle/JSON)**
- **Pros:** Simple, all data in one place, no external dependencies
- **Cons:** Massive snapshot size, slow serialization, memory issues
- **Implementation:** `pickle.dumps(artifacts)`

**B) Reference Storage (Store Artifacts Separately)**
- **Pros:** Small snapshots, efficient, can share artifacts across states
- **Cons:** Two-phase save/load, artifact lifecycle management needed
- **Implementation:** Artifacts stored by hash, snapshot contains references

**C) Artifact Registry (Centralized Store with IDs)**
- **Pros:** Deduplication, version control, clean separation
- **Cons:** Additional infrastructure, complexity, reference integrity
- **Implementation:** Artifact storage backend, snapshot has artifact IDs

**D) Lazy Loading (Don't Store, Recompute if Needed)**
- **Pros:** No storage overhead, always fresh data
- **Cons:** Slow recovery, requires re-execution, defeats purpose
- **Implementation:** Mark artifacts as `cacheable=False` during serialization

**Your Decision:** Option B: Store artifacts separately keyed by `artifact_hash`, snapshot contains `{key: artifact_ref}` mapping. On restore, load artifacts from reference store. This keeps snapshots lightweight while preserving artifacts. Implement garbage collection for orphaned artifacts. So I can store the artfact in a way its more convinient for their type. The artfact loading and save should be abstraction and the real implementation will be done as adapter for each case. 

---

### Q6: State Snapshot Frequency and Triggers

**Issue:** When should the system automatically save state snapshots?

**Options:**

**A) Manual Only**
- **Pros:** Full control, predictable, no overhead
- **Cons:** Users must remember, easy to forget, no auto-recovery
- **Implementation:** `execution_state.save_checkpoint()` called explicitly

**B) After Every Event**
- **Pros:** Maximum granularity, no data loss
- **Cons:** Massive overhead, slow execution, storage explosion
- **Implementation:** Hook in `add_event()`

**C) Periodic Time-Based (Every N Seconds)**
- **Pros:** Bounded overhead, predictable intervals
- **Cons:** Time-based doesn't align with logic, might save mid-operation
- **Implementation:** Background task with timer

**D) Event-Based Triggers (After Specific Events)**
- **Pros:** Logical checkpoints, aligns with execution flow, configurable
- **Cons:** Requires event selection logic, might miss important points
- **Implementation:** `save_after_events: list[str]` configuration

**E) Hybrid (Manual + Auto-Triggers)**
- **Pros:** Best of both worlds, safety net + control
- **Cons:** Most complex, needs tuning
- **Implementation:** Combine A + D

**Your Decision:** Option E (Hybrid) - Default to **A (Manual)** for explicit control, add opt-in **D (Event-Based)** with smart defaults:
- After LLM calls (expensive to recompute)
- After tool executions (external side effects)
- After artifact generation
- User can override: `auto_checkpoint_events: list[EventType]`

---

### Q7: State Version and Schema Evolution

**Issue:** As system evolves, state snapshot schema might change. How to handle compatibility?

**Options:**

**A) No Versioning (Break Compatibility)**
- **Pros:** Simplest, no migration code
- **Cons:** Old states unloadable after updates, data loss
- **Implementation:** Ignore the problem

**B) Schema Version + Migration Functions**
- **Pros:** Forward compatibility, graceful evolution, no data loss
- **Cons:** Migration code maintenance, testing burden, complexity
- **Implementation:** `StateSnapshot.version`, migrate v1→v2→v3...

**C) Best-Effort Loading (Try to Adapt)**
- **Pros:** No explicit versioning, flexible
- **Cons:** Unpredictable failures, silent data corruption, hard to debug
- **Implementation:** Try/except with field defaults

**D) Immutable States + Converter Tool**
- **Pros:** Clear separation, testable, no runtime impact
- **Cons:** Manual conversion step, duplicate storage during transition
- **Implementation:** CLI tool to convert old states to new format

**Your Decision:** Option B (Schema Version + Migrations)- Add `version: str` to `StateSnapshot`, implement migration functions for each version bump. Fail loudly on incompatible versions with clear error message. This is the most robust approach for production systems.

---

### Q8: Integration with Existing Cache System

**Issue:** Current system has `CacheBackend` for result caching. How should state recovery interact with it?

**Options:**

**A) Completely Replace Cache with State Recovery**
- **Pros:** Single system, no duplication, consistent
- **Cons:** Overkill for simple result caching, performance overhead
- **Implementation:** Remove `CacheBackend`, use `StateBackend` for everything

**B) Keep Both Separate (State = Long-term, Cache = Short-term)**
- **Pros:** Optimized for each use case, clear separation of concerns
- **Cons:** Two systems to maintain, potential confusion
- **Implementation:** Use cache for intermediate results, state for checkpoints

**C) State Uses Cache as Storage Layer**
- **Pros:** Reuse infrastructure, unified backend
- **Cons:** Mixed responsibilities, cache eviction could lose states
- **Implementation:** `StateBackend` wraps `CacheBackend` with no-evict flag

**D) Cache Uses State for Persistence**
- **Pros:** State is source of truth, cache is derived
- **Cons:** Cache rebuilding overhead, tight coupling
- **Implementation:** Cache misses trigger state load

**Your Decision:** Option A. Replace it completed, we no longer need it. 

---

## Additional Suggestions

### S1: State Metadata and Tagging

**Issue:** How to organize and identify states for later retrieval?

**Suggestion:** Add rich metadata to `StateSnapshot`:

```python
class StateMetadata(BaseModel):
    tags: list[str] = Field(default_factory=list)  # ["production", "user_123"]
    description: str = ""  # Human-readable description
    pipeline_name: str = ""  # Which pipeline created this
    parent_state_id: Optional[str] = None  # For state lineage
    custom: dict[str, Any] = Field(default_factory=dict)  # User-defined metadata
```

**Implementation:** Add to `StateSnapshot.metadata: StateMetadata`

**Benefits:**
- Query states by tag: `state_backend.list_states(tags=["production"])`
- Track state lineage (which state spawned this one)
- Debugging: attach error context, input parameters
- Audit trail: user, timestamp, reason

**Cons:** Metadata management overhead, indexing needed for performance

**Your Decision:** that's greate. Do it

---

### S2: Partial State Replay (Resume from Specific Event)

**Issue:** Full replay might be unnecessary; user might want to resume from specific point.

**Suggestion:** Add resume-from capability:

```python
restored_state = ExecutionState.from_state_id(
    state_id="abc-123",
    resume_from_address="agent::tool_call::llm",  # Skip everything before this
    replay_mode=ReplayMode.ADAPTIVE
)
```

**Implementation:** 
- Find event index matching `resume_from_address`
- Skip all events before that point
- Start replaying from that event onward

**Benefits:**
- Faster recovery (skip completed sections)
- Selective re-execution (e.g., only redo LLM call with new prompt)
- Debugging (jump to failure point)

**Cons:** 
- Complex to ensure consistency (skipped events might have side effects)
- Address matching ambiguity (what if address appears multiple times?)
- User must understand execution graph

- Find event index matching `resume_from_address`
**Your Decision:** Yep. This is what I'd like with "address_to_replace" I mentioned ealier. But I like most you name "resume_from_address". Do it. 

---

### S3: State Diffing and Comparison

**Issue:** Users might want to understand what changed between state versions.

**Suggestion:** Add state comparison utility:

```python
diff = StateSnapshot.diff(state1, state2)
# Returns:
# {
#   "new_events": [...],
#   "changed_storage": {"key": (old_val, new_val)},
#   "new_artifacts": [...]
# }
```

**Implementation:** Compare histories, storage dicts, artifact keys

**Benefits:**
- Debugging (what changed between checkpoints?)
- Audit trail (track execution evolution)
- Testing (assert expected state changes)

**Cons:** Computational overhead, memory for storing multiple states

**Your Decision:** That's awesome! Do it. 

---

### S4: Event Replay Validation Mode

**Issue:** How to ensure replay produces same results as original execution?

**Suggestion:** Add validation mode:

```python
restored_state = ExecutionState.from_state_id(
    state_id="abc-123",
    validation_mode=True  # Re-execute AND compare to cached result
)
```

**Implementation:**
- Re-execute each event even if cached
- Compare new result with cached result
- Log/raise if mismatch detected
- Useful for testing determinism

**Benefits:**
- Detect non-determinism bugs
- Verify actor version compatibility
- Test harness for replay logic

**Cons:** Doubles execution time, requires strict equality checking (hard for floats, complex objects)

**Your Decision:** This is great. Do it

---

## Migration Path

### Phase 1: Foundation (Minimal Viable State Recovery)
1. Add `state_id: str` to `ExecutionState`
2. Implement `StateSnapshot` model
3. Create `FileSystemStateBackend` (simple JSON storage)
4. Add `ExecutionState.to_snapshot()` and `from_snapshot()`
5. Manual checkpoint: `execution_state.save_checkpoint()`

### Phase 2: Smart Replay
1. Add `replay_mode: bool` to `ExecutionState`
2. Modify `ExecutionEvent.__call__` to check history during replay
3. Implement address-based matching
4. Add `skip_republish: bool` flag to `ExecutionEvent`
5. Implement "republish only new events" logic

### Phase 3: Advanced Features
1. Add `StateBackend` abstraction + DB implementation
2. Implement artifact reference storage
3. Add event-based auto-checkpointing
4. Implement schema versioning and migrations

### Phase 4: Production Hardening
1. Add state metadata and tagging
2. Implement partial replay
3. Add validation mode
4. Performance optimization (async save, compression)
5. Monitoring and observability

---

## Open Questions Summary

Please provide your decisions on the following:

1. **[Q1]** State Backend Storage: _____________
2. **[Q2]** Event Matching Strategy: _____________
3. **[Q3]** Non-Deterministic Actors: _____________
4. **[Q4]** Event Republishing: _____________
5. **[Q5]** Artifact Handling: _____________
6. **[Q6]** Snapshot Frequency: _____________
7. **[Q7]** Schema Evolution: _____________
8. **[Q8]** Cache Integration: _____________
9. **[S1]** State Metadata: _____________
10. **[S2]** Partial Replay: _____________
11. **[S3]** State Diffing: _____________
12. **[S4]** Validation Mode: _____________

---

## Expected Outcomes

After implementation:

✅ **State Persistence:** Full serialization of `ExecutionState` to durable storage  
✅ **State Restoration:** Load execution state by `state_id` and resume  
✅ **Smart Replay:** Automatically skip already-executed events  
✅ **Selective Publishing:** Control whether replayed events trigger bus notifications  
✅ **Artifact Recovery:** Restore large artifacts without re-computation  
✅ **Failure Recovery:** Resume pipelines after crashes or interruptions  
✅ **Debugging:** Step through execution history, compare states  
✅ **Testing:** Validate determinism and replay correctness  

---

## Next Steps

1. **Review this specification** and answer the decision questions above
2. **Prioritize features** - which are must-have vs nice-to-have?
3. **Approve Phase 1 scope** - confirm minimal viable feature set
4. **Implementation** - begin coding once design is validated

