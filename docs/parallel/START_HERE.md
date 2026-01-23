# 🚀 Parallel Execution - Quick Start Implementation Guide

**For:** LLM Implementer  
**Goal:** Implement parallel execution feature in phases  
**Time Estimate:** 28-38 hours total

---

## 📚 Read These Documents First

1. **[PARALLEL_EXECUTION_IMPLEMENTATION.md](./PARALLEL_EXECUTION_IMPLEMENTATION.md)** - Full specification with all details
2. **[IMPLEMENTATION_CHECKLIST.md](./IMPLEMENTATION_CHECKLIST.md)** - Phase-by-phase checklist

---

## 🎯 What We're Building

A parallel execution system that allows users to run independent events concurrently:

```python
# Instead of this (sequential):
result1 = await event1(input1, context, state)
result2 = await event2(input2, context, state)
result3 = await event3(input3, context, state)

# Users can do this (parallel):
async with state.parallel(max_workers=5) as p:
    p.add(event1(input1, context, state))
    p.add(event2(input2, context, state))
    p.add(event3(input3, context, state))
    results = await p.gather()
```

---

## ✅ Prerequisites Check

All required components exist in the codebase:

```python
# Core framework (already available)
from rh_agents.core.execution import ExecutionState, EventBus
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.bus_handlers import EventPrinter
```

**✅ You have everything needed to start!**

---

## 🏗️ Architecture Overview

```
New Components to Create:
├── rh_agents/core/parallel.py (NEW)
│   ├── GroupStatus enum
│   ├── ErrorStrategy enum
│   ├── ParallelDisplayMode enum
│   ├── ParallelEventGroup model
│   ├── ParallelGroupTracker model
│   └── ParallelExecutionManager class
│
Modifications:
├── rh_agents/core/events.py (MODIFY)
│   └── Add 3 fields to ExecutionEvent
│
├── rh_agents/core/execution.py (MODIFY)
│   └── Add parallel() method to ExecutionState
│
└── rh_agents/bus_handlers.py (MODIFY)
    └── Add ParallelEventPrinter class
```

---

## 📋 Implementation Phases (9 Steps)

### Phase 1: Data Models (2-3h) ⭐ START HERE
Create enums and Pydantic models for parallel execution.

**Action:** Create `rh_agents/core/parallel.py` with enums and models  
**Validation:** `python -c "from rh_agents.core.parallel import *"`

---

### Phase 2: Core Manager (4-6h)
Build ParallelExecutionManager with semaphore-based concurrency.

**Action:** Implement context manager, add(), gather()  
**Validation:** Run test script to execute 3 parallel tasks

---

### Phase 3: Integration (2-3h)
Connect ParallelExecutionManager to ExecutionState.

**Action:** Add `parallel()` method to ExecutionState  
**Validation:** `state.parallel()` works

---

### Phase 4: Advanced Modes (3-4h)
Add streaming and dictionary result collection.

**Action:** Implement stream() and gather_dict()  
**Validation:** Both modes return correct results

---

### Phase 5: Error Handling (2-3h)
Implement fail fast/slow strategies and retry logic.

**Action:** Add error strategies and circuit breaker  
**Validation:** Test both error modes

---

### Phase 6: Basic Printer (3-4h)
Add parallel event detection to EventPrinter.

**Action:** Create ParallelEventPrinter with real-time mode  
**Validation:** Visual inspection of output

---

### Phase 7: Progress Bar (4-5h)
Implement progress bar visualization.

**Action:** Add progress rendering and ANSI updates  
**Validation:** Visual inspection of progress bar

---

### Phase 8: Tests & Examples (3-4h)
Create comprehensive tests and usage examples.

**Action:** Write tests and create 3-4 example files  
**Validation:** All tests pass, examples run

---

### Phase 9: Polish (4-5h)
Final touches, optimization, and documentation.

**Action:** Metrics, optimization, edge cases, docs  
**Validation:** Coverage > 90%, all examples work

---

## 🎨 Key Design Patterns

### 1. Context Manager Pattern
```python
class ParallelExecutionManager:
    async def __aenter__(self):
        # Setup
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Auto-gather if not done
        if self.tasks and not self.results:
            await self.gather()
```

### 2. Semaphore for Concurrency Control
```python
self.semaphore = asyncio.Semaphore(max_workers)

async def _execute_with_semaphore(self, coro, index):
    async with self.semaphore:
        # Only max_workers tasks run this simultaneously
        result = await coro
        return result
```

### 3. Fail Slow with gather()
```python
# Collect all results, wrap exceptions
results = await asyncio.gather(*tasks, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        # Wrap in ExecutionResult with ok=False
        pass
```

### 4. Streaming with as_completed()
```python
async def stream(self):
    for coro in asyncio.as_completed(self.tasks):
        result = await coro
        yield result
```

---

## 🔧 Implementation Tips

### For Each Phase:

1. **Read the phase spec carefully** - understand all requirements
2. **Look at existing code** - follow framework patterns
3. **Implement incrementally** - small steps, test often
4. **Write validation test** - confirm it works before moving on
5. **Commit** - clear commit message describing what was done

### Code Style:

- Use Pydantic models for all data structures
- Add type hints to all functions
- Write docstrings for public APIs
- Follow existing code formatting

### Testing:

- Test each component individually
- Test error cases explicitly
- Visual inspection for printer components
- Integration tests for full workflows

---

## 📖 Example: Phase 1 Implementation

**Goal:** Create core data models

**Step 1:** Create file
```bash
touch rh_agents/core/parallel.py
```

**Step 2:** Add imports
```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid
```

**Step 3:** Define enums
```python
class GroupStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ErrorStrategy(str, Enum):
    FAIL_SLOW = "fail_slow"
    FAIL_FAST = "fail_fast"

class ParallelDisplayMode(str, Enum):
    REALTIME = "realtime"
    PROGRESS = "progress"
```

**Step 4:** Define models
```python
class ParallelEventGroup(BaseModel):
    """Metadata for a group of events executing in parallel."""
    
    group_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = Field(default=None)
    total: int = Field(default=0)
    completed: int = Field(default=0)
    failed: int = Field(default=0)
    # ... more fields as per spec
```

**Step 5:** Validate
```bash
python -c "from rh_agents.core.parallel import GroupStatus, ErrorStrategy; print('✓ Works!')"
```

**Step 6:** Commit
```bash
git add rh_agents/core/parallel.py
git commit -m "Phase 1: Add parallel execution data models and enums"
```

---

## ⚠️ Common Pitfalls

1. ❌ **Don't create semaphore per task** → Use one per group
2. ❌ **Don't forget to await coroutines** → Always await in async functions
3. ❌ **Don't ignore exceptions in gather()** → Use `return_exceptions=True`
4. ❌ **Don't forget to cancel tasks** → Cancel explicitly in fail fast mode
5. ❌ **Don't assume completion order** → Tasks finish in arbitrary order
6. ❌ **Don't leak tasks** → Always await or cancel all tasks

---

## 🧪 Testing Strategy

### Per Phase:
- Write small test for each new component
- Run immediately after implementing
- Fix issues before moving to next phase

### Integration:
- Test full workflows in Phase 8
- Test with real ExecutionEvents
- Test state recovery compatibility

### Visual:
- Manually inspect EventPrinter output
- Check progress bar appearance
- Verify formatting looks good

---

## 📊 Progress Tracking

Update this as you complete each phase:

- [ ] Phase 1: Core Data Models
- [ ] Phase 2: Basic ParallelExecutionManager
- [ ] Phase 3: ExecutionState Integration
- [ ] Phase 4: Advanced Result Modes
- [ ] Phase 5: Error Handling
- [ ] Phase 6: Basic EventPrinter
- [ ] Phase 7: Progress Bar
- [ ] Phase 8: Testing & Examples
- [ ] Phase 9: Polish

---

## 🎯 Success Criteria

**You're done when:**

1. ✅ All 9 phases complete
2. ✅ All tests passing
3. ✅ Examples run without errors
4. ✅ Visual output looks good
5. ✅ Code coverage > 90%
6. ✅ Documentation updated
7. ✅ No regressions in existing tests

**Feature Complete:** Users can execute events in parallel with proper concurrency control, error handling, and visualization.

---

## 🚀 Ready to Start?

1. Open [PARALLEL_EXECUTION_IMPLEMENTATION.md](./PARALLEL_EXECUTION_IMPLEMENTATION.md)
2. Read Phase 1 specification
3. Create `rh_agents/core/parallel.py`
4. Implement enums and models
5. Validate with test command
6. Move to Phase 2

**Good luck! Take it one phase at a time.** 🎉
