# Parallel Execution - Implementation Checklist

**Status:** Ready to implement  
**Specification:** [PARALLEL_EXECUTION_IMPLEMENTATION.md](./PARALLEL_EXECUTION_IMPLEMENTATION.md)

---

## 📦 Prerequisites - All Available ✅

### Existing Components
- ✅ `ExecutionState` - core/execution.py (line 139+)
- ✅ `ExecutionEvent` - core/events.py (line 14+)
- ✅ `EventBus` - core/execution.py (line 103+)
- ✅ `EventPrinter` - bus_handlers.py (line 14+)
- ✅ `ExecutionStatus` enum - core/types.py
- ✅ `EventType` enum - core/types.py
- ✅ `ExecutionResult` - core/events.py
- ✅ `HistorySet` - core/execution.py
- ✅ `BaseActor` - core/actors.py

### Framework Capabilities
- ✅ Async/await throughout
- ✅ Pydantic models for serialization
- ✅ Event bus publish/subscribe
- ✅ State recovery & replay
- ✅ History tracking

---

## 🎯 Implementation Phases - Step by Step

### Phase 1: Core Data Models (START HERE) ⏳
**Time:** 2-3 hours  
**Goal:** Define all data structures

#### Tasks:
1. Create `rh_agents/core/parallel.py`:
   ```python
   # Enums
   - GroupStatus (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
   - ErrorStrategy (FAIL_SLOW, FAIL_FAST)
   - ParallelDisplayMode (REALTIME, PROGRESS)
   
   # Models
   - ParallelEventGroup
   - ParallelGroupTracker
   ```

2. Modify `rh_agents/core/events.py`:
   ```python
   # Add to ExecutionEvent class:
   group_id: Optional[str] = None
   parallel_index: Optional[int] = None
   is_parallel: bool = False
   ```

#### Validation:
```bash
python -c "from rh_agents.core.parallel import *; print('✓ Phase 1 complete')"
```

---

### Phase 2: Basic ParallelExecutionManager ⏳
**Time:** 4-6 hours  
**Goal:** Core parallel execution logic

#### Tasks:
1. In `rh_agents/core/parallel.py`:
   ```python
   class ParallelExecutionManager:
       - __init__(execution_state, max_workers, error_strategy, timeout)
       - __aenter__ / __aexit__
       - add(coro, name=None)
       - _execute_with_semaphore(coro, index)
       - gather() -> list[ExecutionResult]
   ```

2. Implement:
   - Semaphore-based concurrency control
   - Event metadata assignment (group_id, parallel_index, is_parallel)
   - Basic error handling (fail slow)

#### Validation:
```python
# test_phase2.py
async def test():
    state = ExecutionState()
    async def task(n):
        await asyncio.sleep(0.1)
        return n * 2
    
    async with ParallelExecutionManager(state, max_workers=3) as p:
        p.add(task(1))
        p.add(task(2))
        p.add(task(3))
        results = await p.gather()
    
    assert len(results) == 3
    print("✓ Phase 2 complete")

asyncio.run(test())
```

---

### Phase 3: ExecutionState Integration ⏳
**Time:** 2-3 hours  
**Goal:** Integrate with ExecutionState

#### Tasks:
1. Add to `ExecutionState` in `rh_agents/core/execution.py`:
   ```python
   def parallel(
       self,
       max_workers: int = 5,
       error_strategy: ErrorStrategy = ErrorStrategy.FAIL_SLOW,
       timeout: Optional[float] = None,
       name: Optional[str] = None
   ) -> ParallelExecutionManager:
       from rh_agents.core.parallel import ParallelExecutionManager
       return ParallelExecutionManager(
           execution_state=self,
           max_workers=max_workers,
           error_strategy=error_strategy,
           timeout=timeout
       )
   ```

#### Validation:
```python
async def test():
    state = ExecutionState()
    async with state.parallel(max_workers=2) as p:
        # ... add tasks
        results = await p.gather()
    print("✓ Phase 3 complete")
```

---

### Phase 4: Advanced Result Modes ⏳
**Time:** 3-4 hours  
**Goal:** Multiple result collection strategies

#### Tasks:
1. Implement in `ParallelExecutionManager`:
   ```python
   async def stream(self) -> AsyncGenerator[ExecutionResult, None]:
       # Yield results as tasks complete (asyncio.as_completed)
   
   async def gather_dict(self) -> dict[str, ExecutionResult]:
       # Return named results
   ```

2. Add timeout support with `asyncio.wait_for()`

#### Validation:
```python
# Test streaming
async for result in parallel.stream():
    print(f"Got: {result}")

# Test dict mode
results = await parallel.gather_dict()
assert "task1" in results
```

---

### Phase 5: Error Handling ⏳
**Time:** 2-3 hours  
**Goal:** Configurable error strategies

#### Tasks:
1. Implement fail fast (cancel remaining on error)
2. Enhance fail slow (collect all, wrap errors)
3. Add retry logic with exponential backoff
4. Basic circuit breaker

#### Validation:
```python
# Fail slow (default) - collects all
results = await parallel.gather()
assert results[0].ok == False  # error
assert results[1].ok == True   # success

# Fail fast - raises immediately
try:
    await parallel_fast.gather()
except Exception:
    print("✓ Fail fast works")
```

---

### Phase 6: Basic EventPrinter ⏳
**Time:** 3-4 hours  
**Goal:** Parallel event visualization (real-time mode)

#### Tasks:
1. Create `ParallelEventPrinter` in `bus_handlers.py`:
   ```python
   class ParallelEventPrinter(EventPrinter):
       - __init__(parallel_mode, show_progress)
       - print_event(event)  # override
       - _handle_parallel_event(event)
       - Track groups in self.parallel_groups dict
   ```

2. Implement real-time mode (interleaved output)
3. Add group lifecycle events (GROUP_STARTED, GROUP_COMPLETED)

#### Validation:
Visual inspection of output during parallel execution

---

### Phase 7: Progress Bar ⏳
**Time:** 4-5 hours  
**Goal:** Progress bar visualization

#### Tasks:
1. Implement in `ParallelEventPrinter`:
   ```python
   def _render_progress_bar(self, completed, total, width=10):
       # Return: "████████▒▒ 80%"
   
   def _update_progress_bar(self, group_id):
       # Update and re-render progress bar
   ```

2. Progress mode implementation
3. ANSI escape codes for line updates
4. Group statistics

#### Validation:
Visual inspection - should see updating progress bar

---

### Phase 8: Testing & Examples ⏳
**Time:** 3-4 hours  
**Goal:** Comprehensive tests and docs

#### Tasks:
1. Create `tests/test_parallel_execution.py`:
   - All result modes
   - Error strategies
   - Concurrency limiting
   - Timeout behavior
   - State recovery

2. Create examples:
   - `examples/parallel_basic.py`
   - `examples/parallel_streaming.py`
   - `examples/parallel_error_handling.py`

#### Validation:
```bash
python -m pytest tests/test_parallel_execution.py -v
python examples/parallel_basic.py
```

---

### Phase 9: Polish ⏳
**Time:** 4-5 hours  
**Goal:** Production ready

#### Tasks:
1. Add metrics collection
2. Performance optimization
3. Edge case handling
4. Documentation polish

#### Validation:
- Coverage > 90%
- All examples working
- README updated

---

## 🚦 Current Status

**Phase:** Not Started  
**Next Step:** Begin Phase 1 - Create core data models

---

## 🔧 Quick Commands

```bash
# Run tests
python -m pytest tests/test_parallel_execution.py -v

# Type check
mypy rh_agents/core/parallel.py

# Run example
python examples/parallel_basic.py

# Check coverage
pytest --cov=rh_agents.core.parallel tests/
```

---

## 📝 Notes for Implementation

### Key Design Decisions (Confirmed)
1. ✅ Context Manager API - `async with state.parallel()`
2. ✅ Per-Group Semaphores for concurrency control
3. ✅ Multiple result strategies (list, stream, dict)
4. ✅ Configurable error handling (fail slow default)
5. ✅ Hybrid display modes (progress + realtime)

### Critical Implementation Details
- Use `asyncio.Semaphore` per group (not global)
- Use `asyncio.as_completed()` for streaming mode
- Use `asyncio.gather(*tasks, return_exceptions=True)` for fail slow
- Cancel tasks explicitly in fail fast mode
- ANSI codes: `\033[<n>A` move up n lines, `\r` carriage return

### What We Have (Framework Context)
```python
# Available in codebase:
from rh_agents.core.execution import ExecutionState, EventBus, HistorySet
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.bus_handlers import EventPrinter

# ExecutionEvent already has:
# - actor, address, execution_status, execution_time
# - result, is_replayed, from_cache
# - __call__ method that executes and publishes

# EventBus already has:
# - subscribe(handler), publish(event)
# - Sequential handler calling with await
```

---

## ✅ Definition of Done

Each phase complete when:
- [ ] All code implemented
- [ ] Tests written and passing
- [ ] Validation script succeeds
- [ ] No regressions
- [ ] Committed

Feature complete when:
- [ ] All 9 phases done
- [ ] Integration tests pass
- [ ] Examples work
- [ ] Documentation updated
- [ ] Performance acceptable

---

**Ready to start?** → Begin with Phase 1
