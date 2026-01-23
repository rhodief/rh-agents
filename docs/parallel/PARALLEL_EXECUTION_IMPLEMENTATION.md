# Parallel Execution Implementation Specification

**Version:** 1.0  
**Date:** January 23, 2026  
**Status:** Implementation Ready

---

## 📋 Overview

This specification defines the phased implementation of parallel execution for the RH-Agents framework. The system will enable controlled parallel execution of independent events with proper concurrency management, progress tracking, and organized event emission.

### Design Decisions Summary

Based on requirements review, the following design decisions have been confirmed:

1. **API Design**: Context Manager API (Option C)
2. **Event Display**: Hybrid Mode - Configurable (Option D)
3. **Concurrency Control**: Per-Group Semaphores (Option B)
4. **Result Gathering**: Multiple Strategies (Option D)
5. **Error Handling**: Configurable Strategy with Fail Slow default (Option C)

### Additional Features to Include

- ✅ Timeout support for parallel groups
- ✅ Retry logic for failed parallel events
- ✅ Circuit breaker pattern for external calls
- ✅ Parallel execution metrics/observability

---

## 🎯 Core Requirements

### Functional Requirements

1. **Parallel Execution Context Manager**
   - Clean, Pythonic API using async context managers
   - Support for adding multiple events to parallel group
   - Automatic resource cleanup on exit

2. **Concurrency Control**
   - Per-group semaphore-based limiting
   - Configurable max_workers per parallel group
   - Prevent resource exhaustion

3. **Result Collection**
   - List mode (order preserved)
   - Streaming mode (as completed)
   - Dictionary mode (named results)

4. **Progress Visualization**
   - Real-time mode (interleaved output)
   - Progress bar mode (grouped with progress)
   - Configurable display strategy

5. **Error Handling**
   - Fail slow (collect all results) - default
   - Fail fast (cancel on first error) - optional
   - Proper error aggregation and reporting

6. **State Recovery Compatibility**
   - Parallel groups must be replayable
   - Deterministic group IDs
   - Preserved event ordering in history

---

## 🏗️ Architecture

### Component Structure

```
rh_agents/
├── core/
│   ├── parallel.py              # NEW - Core parallel execution logic
│   ├── events.py                # MODIFY - Add parallel metadata
│   └── execution.py             # MODIFY - Add parallel context manager
├── bus_handlers.py              # MODIFY - Add ParallelEventPrinter
└── models.py                    # MODIFY - Add parallel models if needed
```

### Key Components

#### 1. ParallelEventGroup (Data Model)

Represents a group of events executing in parallel.

```python
class ParallelEventGroup(BaseModel):
    """Metadata for a group of events executing in parallel."""
    
    group_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = Field(default=None)
    total: int = Field(default=0)
    completed: int = Field(default=0)
    failed: int = Field(default=0)
    status: GroupStatus = Field(default=GroupStatus.PENDING)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_strategy: ErrorStrategy = Field(default=ErrorStrategy.FAIL_SLOW)
    max_workers: int = Field(default=5)
    timeout: Optional[float] = None
```

#### 2. ParallelExecutionManager (Core Logic)

Orchestrates parallel execution with concurrency control.

```python
class ParallelExecutionManager:
    """Manages parallel execution of events with concurrency control."""
    
    def __init__(
        self,
        execution_state: ExecutionState,
        max_workers: int = 5,
        error_strategy: ErrorStrategy = ErrorStrategy.FAIL_SLOW,
        timeout: Optional[float] = None
    ):
        self.execution_state = execution_state
        self.semaphore = asyncio.Semaphore(max_workers)
        self.error_strategy = error_strategy
        self.timeout = timeout
        self.group = ParallelEventGroup(max_workers=max_workers)
        self.tasks: list[asyncio.Task] = []
        self.results: list[ExecutionResult] = []
    
    async def __aenter__(self) -> Self:
        """Enter context manager."""
        self.group.start_time = time()
        self.group.status = GroupStatus.RUNNING
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - auto-gather if not done."""
        if self.tasks and not self.results:
            await self.gather()
    
    def add(self, coro: Awaitable, name: Optional[str] = None):
        """Add a coroutine to the parallel group."""
        pass
    
    async def gather(self) -> list[ExecutionResult]:
        """Execute all tasks and gather results (list mode)."""
        pass
    
    async def gather_dict(self) -> dict[str, ExecutionResult]:
        """Execute all tasks and return named results."""
        pass
    
    async def stream(self) -> AsyncGenerator[ExecutionResult, None]:
        """Execute tasks and yield results as they complete."""
        pass
    
    async def _execute_with_semaphore(self, coro: Awaitable, index: int):
        """Execute single coroutine with semaphore limiting."""
        pass
```

#### 3. ExecutionEvent Extensions

Add parallel execution metadata to existing ExecutionEvent model.

```python
class ExecutionEvent(BaseModel, Generic[OutputT]):
    # ... existing fields ...
    
    # Parallel execution fields
    group_id: Optional[str] = Field(default=None, description="Parallel group ID if part of parallel execution")
    parallel_index: Optional[int] = Field(default=None, description="Index within parallel group")
    is_parallel: bool = Field(default=False, description="True if event is part of parallel group")
```

#### 4. ExecutionState Extensions

Add parallel execution support to ExecutionState.

```python
class ExecutionState(BaseModel):
    # ... existing fields ...
    
    def parallel(
        self,
        max_workers: int = 5,
        error_strategy: ErrorStrategy = ErrorStrategy.FAIL_SLOW,
        timeout: Optional[float] = None,
        name: Optional[str] = None
    ) -> ParallelExecutionManager:
        """Create a parallel execution context."""
        return ParallelExecutionManager(
            execution_state=self,
            max_workers=max_workers,
            error_strategy=error_strategy,
            timeout=timeout
        )
```

#### 5. ParallelEventPrinter

Enhanced printer with parallel execution visualization.

```python
class ParallelEventPrinter(EventPrinter):
    """Event printer with parallel execution awareness."""
    
    def __init__(
        self,
        parallel_mode: ParallelDisplayMode = ParallelDisplayMode.PROGRESS,
        show_progress: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.parallel_mode = parallel_mode
        self.show_progress = show_progress
        self.parallel_groups: dict[str, ParallelGroupTracker] = {}
    
    def print_event(self, event: ExecutionEvent):
        """Print event with parallel group awareness."""
        if event.is_parallel:
            self._handle_parallel_event(event)
        else:
            super().print_event(event)
    
    def _handle_parallel_event(self, event: ExecutionEvent):
        """Handle printing for parallel events."""
        pass
    
    def _render_progress_bar(self, completed: int, total: int, width: int = 10) -> str:
        """Render a progress bar."""
        pass
```

---

## 📝 Implementation Phases

### Phase 1: Core Data Models and Enums (2-3 hours)

**Goal**: Define all data structures, enums, and type definitions needed for parallel execution.

**Tasks**:
1. Create `rh_agents/core/parallel.py` with:
   - `GroupStatus` enum (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
   - `ErrorStrategy` enum (FAIL_SLOW, FAIL_FAST)
   - `ParallelDisplayMode` enum (REALTIME, PROGRESS)
   - `ParallelEventGroup` model
   - `ParallelGroupTracker` model (for printer state)

2. Add parallel fields to `ExecutionEvent` in `rh_agents/core/events.py`:
   - `group_id: Optional[str]`
   - `parallel_index: Optional[int]`
   - `is_parallel: bool`

**Validation**:
- All models can be instantiated
- Models can serialize/deserialize correctly
- No import errors
- Run: `python -c "from rh_agents.core.parallel import *; print('✓ Models loaded')"`

**Files Modified**:
- `rh_agents/core/parallel.py` (NEW)
- `rh_agents/core/events.py` (MODIFY)

---

### Phase 2: Basic ParallelExecutionManager (4-6 hours)

**Goal**: Implement core parallel execution logic with semaphore-based concurrency control.

**Tasks**:
1. Implement `ParallelExecutionManager` class in `rh_agents/core/parallel.py`:
   - `__init__` with semaphore creation
   - `__aenter__` and `__aexit__` for context manager
   - `add()` method to queue coroutines
   - `_execute_with_semaphore()` helper for controlled execution
   - `gather()` method for list-based result collection

2. Handle event metadata:
   - Assign `group_id` to all events in group
   - Assign `parallel_index` to each event
   - Set `is_parallel = True`

3. Implement basic error handling:
   - Fail slow: collect all results, wrap errors in ExecutionResult
   - Proper exception propagation

**Validation**:
- Can create ParallelExecutionManager instance
- Can add multiple coroutines
- Semaphore correctly limits concurrency
- Results returned in correct order
- Errors handled without crashing

**Test Script**:
```python
async def test_basic_parallel():
    state = ExecutionState()
    
    async def dummy_task(n):
        await asyncio.sleep(0.1)
        return n * 2
    
    async with state.parallel(max_workers=3) as p:
        p.add(dummy_task(1))
        p.add(dummy_task(2))
        p.add(dummy_task(3))
        results = await p.gather()
    
    assert len(results) == 3
    print("✓ Basic parallel execution works")
```

**Files Modified**:
- `rh_agents/core/parallel.py` (UPDATE)

---

### Phase 3: ExecutionState Integration (2-3 hours)

**Goal**: Integrate parallel execution into ExecutionState.

**Tasks**:
1. Add `parallel()` method to `ExecutionState` in `rh_agents/core/execution.py`:
   - Returns configured ParallelExecutionManager
   - Passes self as execution_state reference
   
2. Ensure parallel events are properly recorded in history:
   - Each event added to history with unique address
   - Group metadata preserved
   
3. Test state recovery compatibility:
   - Parallel events can be replayed
   - Group structure preserved

**Validation**:
- `execution_state.parallel()` returns ParallelExecutionManager
- Parallel events appear in history
- Events have correct addresses

**Test Script**:
```python
async def test_state_integration():
    state = ExecutionState()
    
    async def dummy_actor(input_data, context, state):
        await asyncio.sleep(0.05)
        return f"Processed {input_data}"
    
    # Create execution events
    event1 = ExecutionEvent(actor=DummyActor(), ...)
    event2 = ExecutionEvent(actor=DummyActor(), ...)
    
    async with state.parallel(max_workers=2) as p:
        p.add(event1(input1, {}, state))
        p.add(event2(input2, {}, state))
        await p.gather()
    
    # Check history
    assert len(state.history.events) >= 2
    print("✓ ExecutionState integration works")
```

**Files Modified**:
- `rh_agents/core/execution.py` (MODIFY)
- `rh_agents/core/parallel.py` (UPDATE)

---

### Phase 4: Advanced Result Collection Modes (3-4 hours)

**Goal**: Implement streaming and dictionary result collection modes.

**Tasks**:
1. Implement `stream()` method in ParallelExecutionManager:
   - Yield results as tasks complete
   - Use `asyncio.as_completed()`
   - Include event metadata with each result

2. Implement `gather_dict()` method:
   - Accept named events: `add(coro, name="task1")`
   - Return dict mapping names to results
   - Preserve error handling

3. Add timeout support:
   - Per-group timeout via `asyncio.wait_for()`
   - Proper timeout exception handling
   - Partial results on timeout (if fail_slow)

**Validation**:
- Streaming yields results in completion order
- Dictionary mode returns correct name mappings
- Timeout cancels remaining tasks properly

**Test Script**:
```python
async def test_advanced_modes():
    state = ExecutionState()
    
    # Test streaming
    async with state.parallel(max_workers=3) as p:
        for i in range(5):
            p.add(dummy_task(i))
        
        results = []
        async for result in p.stream():
            results.append(result)
            print(f"Got result: {result}")
        
        assert len(results) == 5
    
    # Test dictionary mode
    async with state.parallel(max_workers=2) as p:
        p.add(dummy_task(1), name="first")
        p.add(dummy_task(2), name="second")
        results = await p.gather_dict()
        
        assert "first" in results
        assert "second" in results
    
    print("✓ Advanced result modes work")
```

**Files Modified**:
- `rh_agents/core/parallel.py` (UPDATE)

---

### Phase 5: Error Handling Strategies (2-3 hours)

**Goal**: Implement configurable error handling (fail fast, fail slow).

**Tasks**:
1. Implement fail fast strategy:
   - Cancel all pending tasks on first error
   - Propagate error immediately
   - Cleanup resources properly

2. Implement fail slow strategy (already partial):
   - Collect all results including errors
   - Wrap errors in ExecutionResult
   - Report all errors at end

3. Add retry logic:
   - `max_retries` parameter for ParallelExecutionManager
   - Exponential backoff between retries
   - Per-event retry tracking

4. Circuit breaker (basic):
   - Track failure rate
   - Temporary failure mode after threshold
   - Auto-recovery after cooldown

**Validation**:
- Fail fast cancels remaining tasks
- Fail slow collects all results
- Retry logic works correctly
- Circuit breaker triggers and recovers

**Test Script**:
```python
async def test_error_handling():
    state = ExecutionState()
    
    async def failing_task():
        await asyncio.sleep(0.1)
        raise ValueError("Intentional error")
    
    async def success_task():
        await asyncio.sleep(0.2)
        return "Success"
    
    # Test fail slow (default)
    async with state.parallel(max_workers=2) as p:
        p.add(failing_task())
        p.add(success_task())
        results = await p.gather()
    
    assert len(results) == 2
    assert not results[0].ok
    assert results[1].ok
    
    # Test fail fast
    try:
        async with state.parallel(max_workers=2, error_strategy=ErrorStrategy.FAIL_FAST) as p:
            p.add(failing_task())
            p.add(success_task())
            await p.gather()
        assert False, "Should have raised"
    except ValueError:
        print("✓ Fail fast works")
    
    print("✓ Error handling works")
```

**Files Modified**:
- `rh_agents/core/parallel.py` (UPDATE)

---

### Phase 6: Basic Event Printer Support (3-4 hours)

**Goal**: Add parallel event detection and basic visualization to EventPrinter.

**Tasks**:
1. Implement `ParallelEventPrinter` class in `rh_agents/bus_handlers.py`:
   - Extend `EventPrinter`
   - Add `parallel_groups` tracking dict
   - Implement real-time mode (interleaved output)

2. Detect parallel events:
   - Check `event.is_parallel` flag
   - Track group state by `group_id`
   
3. Real-time mode implementation:
   - Print events as they arrive
   - Indent parallel events consistently
   - Show group membership clearly

4. Add group lifecycle events:
   - GROUP_STARTED event when first parallel task starts
   - GROUP_COMPLETED event when all tasks complete
   - GROUP_FAILED event if any task fails (and fail fast)

**Validation**:
- ParallelEventPrinter can be instantiated
- Detects parallel events correctly
- Real-time mode displays events clearly
- Group lifecycle visible

**Test Script**:
```python
async def test_printer_basic():
    printer = ParallelEventPrinter(parallel_mode=ParallelDisplayMode.REALTIME)
    bus = EventBus()
    bus.subscribe(printer.print_event)
    
    state = ExecutionState(event_bus=bus)
    
    async with state.parallel(max_workers=3) as p:
        for i in range(5):
            p.add(dummy_event(i))
        await p.gather()
    
    # Visual inspection of output
    print("✓ Basic printer works")
```

**Files Modified**:
- `rh_agents/bus_handlers.py` (MODIFY)
- `rh_agents/core/parallel.py` (UPDATE - add group events)

---

### Phase 7: Progress Bar Visualization (4-5 hours)

**Goal**: Implement progress bar mode with visual group tracking.

**Tasks**:
1. Implement progress bar rendering:
   - `_render_progress_bar()` helper method
   - Percentage calculation
   - Visual bar with completion indicators
   - Timing information

2. Progress mode implementation:
   - Group header with progress bar
   - Update bar as events complete
   - Final summary on group completion

3. Terminal handling:
   - Use ANSI escape codes for updating lines
   - Handle terminal width
   - Graceful fallback for non-TTY

4. Statistics aggregation:
   - Track completed/failed per group
   - Calculate total execution time
   - Show group-level metrics

**Validation**:
- Progress bar renders correctly
- Updates in real-time
- Final state shows 100%
- Statistics accurate

**Test Script**:
```python
async def test_progress_bar():
    printer = ParallelEventPrinter(parallel_mode=ParallelDisplayMode.PROGRESS)
    bus = EventBus()
    bus.subscribe(printer.print_event)
    
    state = ExecutionState(event_bus=bus)
    
    async with state.parallel(max_workers=3, name="Processing Docs") as p:
        for i in range(10):
            p.add(dummy_event(i))
        await p.gather()
    
    # Visual inspection - should see progress bar
    print("✓ Progress bar works")
```

**Files Modified**:
- `rh_agents/bus_handlers.py` (UPDATE)

---

### Phase 8: Integration Testing & Examples (3-4 hours)

**Goal**: Create comprehensive tests and usage examples.

**Tasks**:
1. Create test file `tests/test_parallel_execution.py`:
   - Test all result collection modes
   - Test error handling strategies
   - Test concurrency limiting
   - Test timeout behavior
   - Test state recovery compatibility

2. Create examples:
   - `examples/parallel_basic.py` - Simple parallel execution
   - `examples/parallel_streaming.py` - Streaming results
   - `examples/parallel_error_handling.py` - Error strategies
   - `examples/parallel_with_printer.py` - Visualization demo

3. Add documentation:
   - Docstrings for all public APIs
   - Type hints everywhere
   - Usage examples in docstrings

**Validation**:
- All tests pass
- Examples run without errors
- Coverage > 90%

**Files Created**:
- `tests/test_parallel_execution.py` (NEW)
- `examples/parallel_basic.py` (NEW)
- `examples/parallel_streaming.py` (NEW)
- `examples/parallel_error_handling.py` (NEW)
- `examples/parallel_with_printer.py` (NEW)

---

### Phase 9: Advanced Features & Polish (4-5 hours)

**Goal**: Add advanced features and polish the implementation.

**Tasks**:
1. Metrics and observability:
   - Track parallel group execution metrics
   - Expose metrics through ExecutionState
   - Add metric export functionality

2. Performance optimization:
   - Profile critical paths
   - Optimize event metadata copying
   - Reduce memory allocations

3. Edge case handling:
   - Empty parallel groups
   - Single event groups
   - Nested parallel contexts (prevention/handling)
   - Cancellation during execution

4. Documentation polish:
   - Update main README with parallel execution section
   - Create migration guide for existing code
   - Add best practices document

**Validation**:
- Metrics collected correctly
- Performance acceptable (overhead < 5%)
- Edge cases handled gracefully
- Documentation complete

**Files Modified**:
- All relevant files (polish)
- `README.md` (UPDATE)
- `docs/parallel/PARALLEL_BEST_PRACTICES.md` (NEW)

---

## 🧪 Testing Strategy

### Unit Tests (Per Phase)

Each phase should include focused unit tests:

1. **Models**: Serialization, validation, defaults
2. **Manager**: Concurrency control, task execution, result collection
3. **Error Handling**: All strategies, edge cases
4. **Printer**: Event detection, formatting, progress rendering

### Integration Tests (Phase 8)

Full workflow tests:

1. End-to-end parallel execution with real actors
2. State recovery with parallel events
3. Cache interactions during parallel execution
4. Event bus under concurrent load

### Performance Tests

Benchmarks to track:

1. **Overhead**: Time overhead for parallel vs sequential
2. **Scalability**: Performance with 2, 5, 10, 20, 50 workers
3. **Memory**: Memory usage with large parallel groups
4. **Throughput**: Events processed per second

### Manual Testing

Visual verification:

1. Progress bar appearance and updates
2. Real-time mode output clarity
3. Error message formatting
4. Group lifecycle visualization

---

## 📊 Success Criteria

### Functional

- ✅ All result collection modes work correctly
- ✅ Concurrency properly limited by semaphore
- ✅ Error handling strategies function as specified
- ✅ Progress visualization displays accurately
- ✅ State recovery preserves parallel execution

### Performance

- ✅ 3-5x speedup for 5 independent CPU-bound events
- ✅ Overhead < 5% for single event "parallel" groups
- ✅ Memory growth O(n) with event count

### Quality

- ✅ Test coverage > 90%
- ✅ All public APIs documented
- ✅ Type hints on all functions
- ✅ No mypy errors in strict mode

### Usability

- ✅ Less than 5 lines to enable parallelism
- ✅ Clear error messages
- ✅ Helpful warnings for common mistakes

---

## 🔧 Implementation Guidelines

### Code Style

- Follow existing codebase conventions
- Use Pydantic models for data structures
- Prefer async/await over callbacks
- Type hints required for all public APIs

### Error Messages

Make errors helpful:

```python
# Good
raise ValueError(
    f"Parallel group '{group.name}' timed out after {timeout}s. "
    f"Completed {group.completed}/{group.total} events. "
    f"Consider increasing timeout or max_workers."
)

# Bad
raise ValueError("Timeout")
```

### Logging

Add strategic logging:

```python
logger.debug(f"Starting parallel group {group.group_id} with {len(tasks)} tasks")
logger.info(f"Parallel group {group.name} completed: {completed}/{total} succeeded")
logger.warning(f"Parallel group {group.name} had {failed} failures")
```

### Documentation

Every public API needs:

1. Docstring with description
2. Args documentation
3. Returns documentation
4. Raises documentation
5. Example usage

Example:

```python
async def gather(self) -> list[ExecutionResult]:
    """
    Execute all tasks in parallel and return results in order.
    
    Results are returned in the same order as tasks were added,
    regardless of completion order. This method respects the
    configured error_strategy.
    
    Returns:
        List of ExecutionResult objects, one per task, in order.
        Failed tasks have ok=False and erro_message set.
    
    Raises:
        asyncio.TimeoutError: If group timeout exceeded
        Exception: If error_strategy is FAIL_FAST and any task fails
    
    Example:
        ```python
        async with state.parallel(max_workers=5) as p:
            p.add(process_doc(doc1))
            p.add(process_doc(doc2))
            results = await p.gather()
        
        for result in results:
            if result.ok:
                print(f"Success: {result.result}")
            else:
                print(f"Failed: {result.erro_message}")
        ```
    """
    pass
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Existing RH-Agents framework
- Development environment set up

### Phase-by-Phase Workflow

For each phase:

1. **Read phase specification** carefully
2. **Create/modify files** as specified
3. **Implement functionality** incrementally
4. **Write tests** as you go
5. **Run validation** script provided
6. **Commit** with clear message
7. **Move to next phase** only when current phase validates

### Validation Commands

After each phase:

```bash
# Run tests
python -m pytest tests/test_parallel_execution.py -v

# Type check
mypy rh_agents/core/parallel.py

# Run example
python examples/parallel_basic.py

# Check formatting
black --check rh_agents/core/parallel.py
```

---

## 📈 Progress Tracking

### Implementation Checklist

**Phase 1: Core Data Models** ☐
- [ ] Enums created
- [ ] ParallelEventGroup model
- [ ] ExecutionEvent extended
- [ ] Models validated

**Phase 2: Basic Manager** ☐
- [ ] ParallelExecutionManager class
- [ ] Context manager protocol
- [ ] add() method
- [ ] gather() method
- [ ] Semaphore control
- [ ] Basic tests pass

**Phase 3: State Integration** ☐
- [ ] ExecutionState.parallel() method
- [ ] History integration
- [ ] State recovery compatible
- [ ] Integration tests pass

**Phase 4: Advanced Modes** ☐
- [ ] stream() method
- [ ] gather_dict() method
- [ ] Timeout support
- [ ] Tests pass

**Phase 5: Error Handling** ☐
- [ ] Fail fast strategy
- [ ] Fail slow strategy
- [ ] Retry logic
- [ ] Circuit breaker
- [ ] Error tests pass

**Phase 6: Basic Printer** ☐
- [ ] ParallelEventPrinter class
- [ ] Real-time mode
- [ ] Group detection
- [ ] Lifecycle events
- [ ] Visual validation

**Phase 7: Progress Bar** ☐
- [ ] Progress bar rendering
- [ ] Progress mode
- [ ] Terminal handling
- [ ] Statistics
- [ ] Visual validation

**Phase 8: Testing & Examples** ☐
- [ ] Comprehensive tests
- [ ] Usage examples
- [ ] Documentation
- [ ] Coverage > 90%

**Phase 9: Polish** ☐
- [ ] Metrics
- [ ] Performance optimization
- [ ] Edge cases
- [ ] Documentation complete

---

## 🎯 Key Design Principles

### 1. Backward Compatibility

Existing code must continue to work:

```python
# This still works without any changes
event = ExecutionEvent(actor=MyAgent())
result = await event(input_data, context, state)
```

### 2. Progressive Enhancement

Features are opt-in:

```python
# Sequential (existing behavior)
result1 = await event1(input1, context, state)
result2 = await event2(input2, context, state)

# Parallel (new feature, opt-in)
async with state.parallel() as p:
    p.add(event1(input1, context, state))
    p.add(event2(input2, context, state))
    results = await p.gather()
```

### 3. Fail Safe Defaults

Default behavior should be safe:

- Default `max_workers=5` prevents resource exhaustion
- Default `error_strategy=FAIL_SLOW` preserves partial results
- No timeout by default (let tasks complete naturally)
- Real-time printer mode as fallback

### 4. Clear Intent

Code should be self-documenting:

```python
# Intent is crystal clear
async with state.parallel(max_workers=10, name="Document Processing") as p:
    for doc in documents:
        p.add(process_document(doc))
    results = await p.gather()
```

### 5. Observable Behavior

Users should see what's happening:

- Progress bars show completion status
- Events emit for group lifecycle
- Statistics available after execution
- Errors are descriptive

---

## 🔍 Implementation Notes

### Semaphore Usage

```python
# Semaphore limits concurrent execution
async with self.semaphore:
    # Only max_workers tasks execute this block simultaneously
    result = await coro
```

### Event Metadata Flow

```python
# Manager assigns metadata before execution
event.group_id = self.group.group_id
event.parallel_index = index
event.is_parallel = True

# ExecutionEvent reads metadata during execution
if event.is_parallel:
    # Special handling for parallel events
    pass
```

### Progress Bar Update Pattern

```python
# Update on event completion
def _handle_parallel_event(self, event: ExecutionEvent):
    if event.execution_status == ExecutionStatus.STARTED:
        # Increment started count
        pass
    elif event.execution_status == ExecutionStatus.COMPLETED:
        # Increment completed, update progress bar
        self._update_progress_bar(event.group_id)
```

### Result Collection Pattern

```python
# List mode - preserves order
results = await asyncio.gather(*tasks, return_exceptions=True)

# Stream mode - completion order
for coro in asyncio.as_completed(tasks):
    result = await coro
    yield result

# Dict mode - named results
results = {}
for name, task in named_tasks.items():
    results[name] = await task
```

---

## ✅ Definition of Done

A phase is complete when:

1. ✅ All tasks in phase specification implemented
2. ✅ Tests written and passing
3. ✅ Validation script runs successfully
4. ✅ Code reviewed (self-review at minimum)
5. ✅ Documentation updated
6. ✅ No regression in existing tests
7. ✅ Committed with clear message

Overall feature is complete when:

1. ✅ All 9 phases done
2. ✅ All success criteria met
3. ✅ Examples working and documented
4. ✅ Integration tests passing
5. ✅ Performance benchmarks acceptable
6. ✅ README updated with usage guide

---

## 📚 API Reference Summary

### User-Facing API

```python
# Create parallel context
async with execution_state.parallel(
    max_workers=5,
    error_strategy=ErrorStrategy.FAIL_SLOW,
    timeout=30.0,
    name="My Parallel Group"
) as parallel:
    
    # Add tasks
    parallel.add(coro1)
    parallel.add(coro2, name="task2")
    
    # Gather results (various modes)
    results = await parallel.gather()              # List mode
    results = await parallel.gather_dict()         # Dict mode
    async for result in parallel.stream():         # Stream mode
        process(result)

# Configure printer
printer = ParallelEventPrinter(
    parallel_mode=ParallelDisplayMode.PROGRESS,  # or REALTIME
    show_progress=True
)
```

### Internal API

```python
# ParallelExecutionManager (internal)
manager = ParallelExecutionManager(state, max_workers=5)
await manager._execute_with_semaphore(coro, index)
manager._emit_group_event(GroupStatus.COMPLETED)

# ParallelEventPrinter (internal)
printer._handle_parallel_event(event)
printer._update_progress_bar(group_id)
printer._render_progress_bar(completed, total)
```

---

## 🎓 Development Tips

### For LLM Implementation

1. **Work Phase by Phase**: Complete one phase fully before moving to next
2. **Run Validation Early**: Test after each small change
3. **Copy Patterns**: Use existing code patterns from the framework
4. **Read Context**: Check existing implementations before creating new ones
5. **Keep It Simple**: Don't over-engineer, follow the spec
6. **Test Incrementally**: Write tests as you write code
7. **Use Type Hints**: Let the type checker catch errors early
8. **Check Examples**: Look at existing examples for code style

### Common Pitfalls to Avoid

1. ❌ Don't create semaphore per task - use one per group
2. ❌ Don't forget to handle exceptions in async code
3. ❌ Don't block the event loop with sync operations
4. ❌ Don't modify shared state without synchronization
5. ❌ Don't ignore timeout errors
6. ❌ Don't forget to clean up resources in __aexit__
7. ❌ Don't assume task completion order
8. ❌ Don't leak tasks (always await or cancel)

---

**Document Status:** ✅ Ready for Implementation  
**Estimated Total Time:** 28-38 hours  
**Implementation Style:** Phased, incremental, test-driven  
**Risk Level:** Low (isolated changes, backward compatible)
