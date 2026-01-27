# Process Interrupt Specification

## Executive Summary

This specification defines a process interrupt mechanism for the RH-Agents framework that allows users to gracefully interrupt agent processing at any point during execution. The interrupt system will integrate with the existing EventBus architecture to propagate cancellation signals throughout the execution tree, stopping event generators and cleaning up resources.

**Key Features:**
- **Dual Control Modes**: Support both local control (via `state.request_interrupt()`) and distributed control (via custom interrupt checkers)
- **Distributed System Support**: Use `set_interrupt_checker()` to poll external interrupt signals from Redis, databases, APIs, message queues, or any external source
- **Automatic Checking**: Framework automatically checks for interrupts at key execution points
- **Graceful Shutdown**: Save state and clean up resources before termination
- **Hybrid Approach**: Combine local and external interrupt sources in the same execution

---

## 1. Problem Statement

Currently, there is no built-in mechanism to interrupt long-running agent executions. Users need the ability to:
- Stop agent processing immediately or gracefully
- Cancel pending operations in parallel execution groups
- Clean up resources and preserve state before termination
- Receive feedback about interrupted operations
- **Support distributed systems** where interrupt signals come from external sources

### Use Cases
1. **User-initiated cancellation**: User clicks "Stop" button in UI
2. **Timeout enforcement**: Execution exceeds maximum allowed time
3. **Resource limits**: System detects resource exhaustion
4. **Error escalation**: Critical error requires immediate shutdown
5. **Priority override**: Higher priority task needs to preempt current execution
6. **Distributed control**: Interrupt signal from external service (Redis, database, API, message queue)
7. **Kubernetes jobs**: Control plane signals job termination via ConfigMap/Secret

---

## 2. Core Requirements

### Functional Requirements
- ✅ Interrupt can be triggered at any point during execution
- ✅ All event generators must terminate when interrupt is fired
- ✅ Parallel execution groups must cancel all pending tasks
- ✅ Event handlers must be notified of interruption
- ✅ State can be preserved before termination (checkpoint)
- ✅ Interrupt reason should be captured and logged

### Non-Functional Requirements
- ⚡ Interrupt propagation should be near-instantaneous (< 100ms)
- 🔒 Thread-safe and asyncio-safe implementation
- 🧹 No resource leaks (all tasks properly cancelled)
- 📊 Minimal performance overhead when not interrupted
- 🔄 Compatible with state recovery and replay mechanisms

---

## 3. Architecture Overview

### Component Interaction Flow

```
┌─────────────┐
│    USER     │
│  (Trigger)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ InterruptController │◄──────┐
│  (Singleton/State)  │       │
└──────┬──────────────┘       │
       │                      │
       │ 1. Set flag          │ 5. Subscribe
       │ 2. Publish event     │
       ▼                      │
┌─────────────────────┐       │
│     EventBus        │──────┘
│  (INTERRUPTED evt)  │
└──────┬──────────────┘
       │
       ├──────────────────────┬──────────────────┐
       │                      │                  │
       ▼                      ▼                  ▼
┌─────────────┐      ┌─────────────┐    ┌─────────────┐
│ExecutionEvent│     │  Parallel   │    │   Event     │
│  (__call__)  │     │   Manager   │    │  Handlers   │
└─────────────┘      └─────────────┘    └─────────────┘
       │                     │                  │
       │ Check flag         │ Cancel tasks      │ Clean up
       ▼                     ▼                  ▼
   Raise                 Gather              Notify
 Interrupt             Exceptions           Subscribers
```

---

## 4. IMPLEMENTATION DESIGN

## 🎯 RECOMMENDED APPROACH: Flag-Based Interrupt + Generator Kill (Option A + D)

### Overview
This design combines two complementary mechanisms:
1. **Flag-Based Interrupt (Option A)**: Graceful interruption via flag checks at strategic execution points
2. **Generator Kill Switch (Option D)**: Immediate termination of event generators for streaming endpoints

### Why This Approach?
- ✅ **Graceful by default**: Allows state preservation and cleanup
- ✅ **Distributed system support**: Works with external interrupt sources via custom checkers
- ✅ **Streaming-aware**: Properly closes event streams and generators
- ✅ **State-safe**: Can checkpoint before termination
- ✅ **Minimal overhead**: Simple flag checks (~0.1μs)
- ✅ **Testable**: Clear, deterministic behavior
- ✅ **EventBus integrated**: Propagates interrupts throughout execution tree

### Description
Add an interrupt flag to `ExecutionState` that is checked at key execution points. Combine with generator registry for immediate stream termination. Simple, predictable, and compatible with state recovery.

### Components

#### A.1. New ExecutionStatus
```python
# In rh_agents/core/types.py
class ExecutionStatus(str, Enum):
    STARTED = 'started'
    COMPLETED = 'completed'
    FAILED = 'failed'
    AWAITING = 'awaiting'
    HUMAN_INTERVENTION = 'human_intervention'
    RECOVERED = 'recovered'
    INTERRUPTED = 'interrupted'  # NEW
    CANCELLING = 'cancelling'    # NEW - transitional state
```

#### A.2. InterruptReason Model
```python
# In rh_agents/core/types.py or new rh_agents/core/interrupt.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class InterruptReason(str, Enum):
    """Reason for execution interruption."""
    USER_CANCELLED = "user_cancelled"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    ERROR_THRESHOLD = "error_threshold"
    PRIORITY_OVERRIDE = "priority_override"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CUSTOM = "custom"

class InterruptSignal(BaseModel):
    """Signal model for execution interruption."""
    reason: InterruptReason = Field(description="Why execution was interrupted")
    message: Optional[str] = Field(default=None, description="Human-readable message")
    triggered_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    triggered_by: Optional[str] = Field(default=None, description="Who/what triggered interrupt")
    save_checkpoint: bool = Field(default=True, description="Save state before terminating")
```

#### A.3. Enhanced ExecutionState
```python
# In rh_agents/core/execution.py
from typing import Callable, Optional, Union, Awaitable

# Type alias for interrupt checker function
InterruptChecker = Union[
    Callable[[], bool],                                    # Sync function returning bool
    Callable[[], Awaitable[bool]],                        # Async function returning bool
    Callable[[], InterruptSignal],                        # Sync function returning signal details
    Callable[[], Awaitable[InterruptSignal]],             # Async function returning signal details
    Callable[[], Optional[InterruptSignal]],              # Sync function returning signal or None
    Callable[[], Awaitable[Optional[InterruptSignal]]]    # Async function returning signal or None
]

class ExecutionState(BaseModel):
    # ... existing fields ...
    
    # Interrupt management
    _is_interrupted: bool = False  # Internal flag, excluded from serialization
    interrupt_signal: Optional[InterruptSignal] = Field(default=None, exclude=True)
    _interrupt_checker: Optional[InterruptChecker] = Field(default=None, exclude=True)
    
    def set_interrupt_checker(self, checker: InterruptChecker):
        """
        Set a custom function to check for interrupt signals from external sources.
        
        This is useful for distributed systems where the interrupt signal comes
        from an external source (e.g., Redis, database, message queue, API endpoint).
        
        Args:
            checker: Function (sync or async) that returns:
                     - bool: True if interrupted (uses generic interrupt info)
                     - InterruptSignal: Detailed interrupt information with reason, message, triggered_by
                     - None/False: Not interrupted
        
        Example:
            # Simple boolean check - uses generic interrupt info
            def check_redis_interrupt():
                return redis_client.get(f"interrupt:{execution_id}") == "1"
            
            state.set_interrupt_checker(check_redis_interrupt)
            
            # Detailed interrupt info from database
            async def check_db_interrupt():
                result = await db.fetchone(
                    "SELECT interrupted, reason, message, triggered_by FROM jobs WHERE id = ?",
                    (job_id,)
                )
                if result and result['interrupted']:
                    return InterruptSignal(
                        reason=InterruptReason(result['reason']),
                        message=result['message'],
                        triggered_by=result['triggered_by']
                    )
                return None  # or False
            
            state.set_interrupt_checker(check_db_interrupt)
            
            # API-based with detailed info
            def check_api_interrupt():
                response = requests.get(f"https://api.example.com/jobs/{job_id}/status")
                data = response.json()
                if data.get('interrupt_requested'):
                    return InterruptSignal(
                        reason=InterruptReason(data.get('reason', 'custom')),
                        message=data.get('message', 'Interrupted from API'),
                        triggered_by=data.get('triggered_by', 'api')
                    )
                return False
            
            state.set_interrupt_checker(check_api_interrupt)
        """
        self._interrupt_checker = checker
    
    def request_interrupt(
        self,
        reason: InterruptReason,
        message: Optional[str] = None,
        triggered_by: Optional[str] = None,
        save_checkpoint: bool = True
    ):
        """
        Request execution interruption directly (for local control).
        
        Use this when you have direct access to the ExecutionState instance
        and want to interrupt it programmatically.
        
        This method:
        1. Sets the interrupt flag
        2. Creates an interrupt signal
        3. Publishes INTERRUPTED event to EventBus
        4. Optionally saves checkpoint
        
        Args:
            reason: Why execution is being interrupted
            message: Human-readable explanation
            triggered_by: Identifier of who/what triggered interrupt
            save_checkpoint: Whether to save state before terminating
            
        Example:
            # Direct local interrupt
            state.request_interrupt(
                reason=InterruptReason.USER_CANCELLED,
                message="User clicked stop button",
                triggered_by="web_ui"
            )
            
            # Timeout enforcement
            state.request_interrupt(
                reason=InterruptReason.TIMEOUT,
                message="Execution exceeded 5 minutes",
                triggered_by="timeout_monitor"
            )
        """
        # Set interrupt flag
        self._is_interrupted = True
        
        # Create interrupt signal
        self.interrupt_signal = InterruptSignal(
            reason=reason,
            message=message,
            triggered_by=triggered_by,
            save_checkpoint=save_checkpoint
        )
        
        # Save checkpoint if requested
        if save_checkpoint and self.state_backend:
            self.save_checkpoint(
                status=StateStatus.INTERRUPTED,
                metadata=StateMetadata(
                    reason=reason.value,
                    message=message or "Execution interrupted"
                )
            )
        
        # Publish interrupt event to bus (will be handled by subscribers)
        # Create a special interrupt event
        from rh_agents.core.events import InterruptEvent
        interrupt_event = InterruptEvent(
            signal=self.interrupt_signal,
            execution_state=self
        )
        asyncio.create_task(self.event_bus.publish(interrupt_event))
    
    def is_interrupted(self) -> bool:
        """Check if execution has been interrupted (local flag only)."""
        return self._is_interrupted
    
    async def check_interrupt(self):
        """
        Check for interrupt signals and raise exception if interrupted.
        
        This method checks BOTH:
        1. Local interrupt flag (set via request_interrupt())
        2. External interrupt signal (via custom interrupt_checker if configured)
        
        Should be called at safe checkpoint locations in your code.
        
        Raises:
            ExecutionInterrupted: If execution has been interrupted
            
        Example:
            # In agent handlers or long-running loops
            async def process_documents(docs, context, state):
                results = []
                for doc in docs:
                    # Check for interrupt before processing each document
                    await state.check_interrupt()
                    
                    result = await process_single_doc(doc)
                    results.append(result)
                    
                return results
        """
        # First check local flag
        if self._is_interrupted:
            raise ExecutionInterrupted(
                reason=self.interrupt_signal.reason if self.interrupt_signal else InterruptReason.CUSTOM,
                message=self.interrupt_signal.message if self.interrupt_signal else "Execution interrupted"
            )
        
        # Then check external interrupt checker if configured
        if self._interrupt_checker:
            try:
                # Handle both sync and async checkers
                import inspect
                if inspect.iscoroutinefunction(self._interrupt_checker):
                    checker_result = await self._interrupt_checker()
                else:
                    checker_result = self._interrupt_checker()
                
                # Handle different return types
                if checker_result:
                    if isinstance(checker_result, InterruptSignal):
                        # Detailed interrupt signal provided
                        signal = checker_result
                    elif isinstance(checker_result, bool):
                        # Simple boolean - create generic signal
                        signal = InterruptSignal(
                            reason=InterruptReason.CUSTOM,
                            message="Interrupt signal detected from external source",
                            triggered_by="external_checker"
                        )
                    else:
                        # Unexpected type, treat as generic interrupt
                        signal = InterruptSignal(
                            reason=InterruptReason.CUSTOM,
                            message="Interrupt signal detected from external source",
                            triggered_by="external_checker"
                        )
                    
                    # Trigger local interrupt with signal details
                    self.request_interrupt(
                        reason=signal.reason,
                        message=signal.message,
                        triggered_by=signal.triggered_by
                    )
                    # Raise exception immediately
                    raise ExecutionInterrupted(
                        reason=signal.reason,
                        message=signal.message
                    )
            except ExecutionInterrupted:
                raise  # Re-raise our own exception
            except Exception as e:
                # Log but don't fail if checker fails
                import logging
                logging.warning(f"Interrupt checker failed: {e}")
```

#### A.4. Custom Exception
```python
# In rh_agents/core/exceptions.py (new file)
class ExecutionInterrupted(Exception):
    """Exception raised when execution is interrupted by user or system."""
    
    def __init__(self, reason: InterruptReason, message: Optional[str] = None):
        self.reason = reason
        self.message = message or f"Execution interrupted: {reason.value}"
        super().__init__(self.message)
```

#### A.5. InterruptEvent Model
```python
# In rh_agents/core/events.py
class InterruptEvent(BaseModel):
    """Special event published when execution is interrupted."""
    signal: InterruptSignal
    execution_state: Any  # ExecutionState at runtime
    
    model_config = {"arbitrary_types_allowed": True}
```

#### A.6. Modified ExecutionEvent.__call__()
```python
# In rh_agents/core/events.py - ExecutionEvent.__call__() method
async def __call__(self, input_data, extra_context, execution_state: ExecutionState) -> ExecutionResult[T]:
    execution_state.push_context(f'{self.actor.name}{"::" + self.tag if self.tag else ""}')
    
    try:
        # CHECK INTERRUPT AT START
        await execution_state.check_interrupt()
        
        current_address = execution_state.get_current_address(self.actor.event_type)
        
        # ... existing replay logic ...
        
        # CHECK INTERRUPT BEFORE EXECUTION
        await execution_state.check_interrupt()
        
        # Run preconditions
        await self.actor.run_preconditions(input_data, extra_context, execution_state)
        
        # CHECK INTERRUPT AFTER PRECONDITIONS
        await execution_state.check_interrupt()
        
        # Start timer and mark as started
        self.start_timer()
        self.detail = self._serialize_detail(input_data)
        await execution_state.add_event(self, ExecutionStatus.STARTED)
        
        # Execute handler
        result = await self.actor.handler(input_data, extra_context, execution_state)
        
        # CHECK INTERRUPT AFTER EXECUTION
        await execution_state.check_interrupt()
        
        # ... rest of existing code ...
        
    except ExecutionInterrupted as e:
        # Handle interruption gracefully
        self.stop_timer()
        self.message = e.message
        self.detail = f"Interrupted: {e.reason.value}"
        await execution_state.add_event(self, ExecutionStatus.INTERRUPTED)
        
        return ExecutionResult(
            result=None,
            execution_time=self.execution_time,
            ok=False,
            erro_message=e.message
        )
    
    except Exception as e:
        # ... existing error handling ...
```

#### A.7. Parallel Execution Integration
```python
# In rh_agents/core/parallel.py - ParallelExecutionManager
async def gather(self) -> list[ExecutionResult]:
    """Execute all tasks and gather results."""
    try:
        # ... existing code ...
        
        # Monitor interrupt signal during execution
        async def interrupt_monitor():
            while not self.group.status in [GroupStatus.COMPLETED, GroupStatus.FAILED, GroupStatus.CANCELLED]:
                if self.execution_state.is_interrupted():
                    # Cancel all pending tasks
                    for task in self.tasks:
                        if not task.done():
                            task.cancel()
                    self.group.status = GroupStatus.CANCELLED
                    break
                await asyncio.sleep(0.1)  # Check every 100ms
        
        # Start interrupt monitor
        monitor_task = asyncio.create_task(interrupt_monitor())
        
        try:
            results = await asyncio.gather(*self.tasks, return_exceptions=True)
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # ... process results ...
        
    except asyncio.CancelledError:
        self.group.status = GroupStatus.CANCELLED
        raise ExecutionInterrupted(
            reason=InterruptReason.USER_CANCELLED,
            message="Parallel execution cancelled"
        )
```

### Key Features of Combined Approach
- ✅ **Dual control modes**: Local (`request_interrupt()`) and distributed (`set_interrupt_checker()`)
- ✅ **Flexible return types**: Checker can return `bool` or `InterruptSignal` with full details
- ✅ **Graceful shutdown**: Checkpoint state before termination
- ✅ **Stream termination**: Generators and event streams close cleanly
- ✅ **EventBus integration**: Interrupt events notify all subscribers
- ✅ **Minimal overhead**: Simple flag checks at strategic points
- ✅ **No race conditions**: Single-threaded flag access in asyncio
- ✅ **Testable**: Clear, deterministic behavior

### Design Trade-offs
- ⚠️ **Checkpoint granularity**: Interrupts occur at checkpoint locations (by design)
- ⚠️ **Handler awareness**: Long-running handlers should add `check_interrupt()` calls
- ✅ **Mitigated by**: Framework automatically checks at all key execution points

---

## 4.1. ALTERNATIVE APPROACHES (For Reference)

### Option B: AsyncIO Task Cancellation (Not Recommended)

**Summary**: Use native asyncio task cancellation for immediate termination.

**Why Not Recommended**:
- ❌ Too aggressive - no graceful shutdown
- ❌ State loss - hard to save checkpoint properly
- ❌ Complex error handling - CancelledError propagation issues
- ❌ Side effects - may leave resources inconsistent
- ❌ Replay issues - hard to resume from cancelled state

**When to Consider**: Emergency shutdown, critical error recovery

<details>
<summary>Click to see implementation details</summary>

#### B.1. Task-Wrapped Execution
```python
# In rh_agents/core/execution.py
class ExecutionState(BaseModel):
    # ... existing fields ...
    
    _execution_task: Optional[asyncio.Task] = Field(default=None, exclude=True)
    
    def set_execution_task(self, task: asyncio.Task):
        """Store reference to main execution task for cancellation."""
        self._execution_task = task
    
    async def interrupt(
        self,
        reason: InterruptReason,
        message: Optional[str] = None,
        save_checkpoint: bool = True
    ):
        """Cancel the execution task immediately."""
        if save_checkpoint and self.state_backend:
            self.save_checkpoint(status=StateStatus.INTERRUPTED)
        
        if self._execution_task and not self._execution_task.done():
            self._execution_task.cancel(msg=f"{reason.value}: {message or ''}")
```

#### B.2. Usage Pattern
```python
# In user code or agent runners
async def run_agent_with_interrupt(agent, input_data, state):
    # Wrap execution in task
    task = asyncio.create_task(
        ExecutionEvent(actor=agent)(input_data, "", state)
    )
    
    # Store task reference for cancellation
    state.set_execution_task(task)
    
    try:
        return await task
    except asyncio.CancelledError:
        # Handle cancellation
        return ExecutionResult(
            result=None,
            ok=False,
            erro_message="Execution cancelled"
        )
```

### Pros ✅
- ✅ **Immediate**: Cancellation happens instantly
- ✅ **Built-in**: Uses native asyncio mechanism
- ✅ **Comprehensive**: Cancels entire task tree
- ✅ **Resource cleanup**: asyncio handles cleanup

### Cons ❌
- ❌ **Aggressive**: No graceful shutdown
- ❌ **Lost state**: May not save checkpoint properly
- ❌ **Complex error handling**: CancelledError propagation
- ❌ **Side effects**: May leave resources in inconsistent state
- ❌ **Replay issues**: Hard to resume from cancelled state

**Use Cases**: Emergency shutdown, critical error recovery, hard timeout enforcement

</details>

---

### Option C: Hybrid with Force-Cancel Timeout (Optional Enhancement)

**Summary**: Add force-cancellation after grace period if graceful interrupt doesn't complete.

**Status**: Can be added as future enhancement to Option A+D

**Use Cases**: Production systems with strict SLAs, guaranteed termination required

<details>
<summary>Click to see implementation details</summary>

#### Description
Combines Option A (flag-based) with Option B (task cancellation) for graceful shutdown with hard timeout fallback.

### Components

#### C.1. Two-Phase Interrupt
```python
# In rh_agents/core/execution.py
class ExecutionState(BaseModel):
    # ... existing fields ...
    
    _is_interrupted: bool = False
    _interrupt_task: Optional[asyncio.Task] = Field(default=None, exclude=True)
    _grace_period: float = 5.0  # seconds
    
    async def request_interrupt(
        self,
        reason: InterruptReason,
        message: Optional[str] = None,
        grace_period: float = 5.0,
        save_checkpoint: bool = True
    ):
        """
        Request interruption with graceful shutdown period.
        
        Phase 1 (0-grace_period): Set flag, allow graceful shutdown
        Phase 2 (grace_period+): Force cancellation if not stopped
        
        Args:
            reason: Interruption reason
            message: Human-readable message
            grace_period: Seconds to wait before force cancel
            save_checkpoint: Save state during graceful phase
        """
        # Phase 1: Graceful
        self._is_interrupted = True
        self.interrupt_signal = InterruptSignal(
            reason=reason,
            message=message,
            save_checkpoint=save_checkpoint
        )
        
        if save_checkpoint and self.state_backend:
            self.save_checkpoint(status=StateStatus.INTERRUPTED)
        
        # Publish interrupt event
        from rh_agents.core.events import InterruptEvent
        interrupt_event = InterruptEvent(
            signal=self.interrupt_signal,
            execution_state=self
        )
        await self.event_bus.publish(interrupt_event)
        
        # Phase 2: Force cancellation after grace period
        async def force_cancel():
            await asyncio.sleep(grace_period)
            if self._is_interrupted and self._execution_task:
                if not self._execution_task.done():
                    print(f"⚠️  Grace period expired, forcing cancellation...")
                    self._execution_task.cancel(msg="Grace period expired")
        
        self._interrupt_task = asyncio.create_task(force_cancel())
```

### Pros ✅
- ✅ **Best of both**: Graceful by default, forceful as fallback
- ✅ **Configurable**: Grace period can be adjusted
- ✅ **State-safe**: Checkpoint during graceful phase
- ✅ **Guaranteed**: Will terminate eventually

### Cons ❌
- ⚠️ **Complex**: More moving parts
- ⚠️ **Timing sensitive**: Grace period must be tuned
- ⚠️ **Race conditions**: Phase transition timing

**Use Cases**: Production systems requiring reliability, long-running workflows, multi-stage pipelines

</details>

---

## 4.2. GENERATOR KILL SWITCH (Integrated in Main Design)

The generator kill mechanism is **already included** in the recommended approach (Option A+D). This section shows the implementation details.

### Components

#### Generator Registry
```python
# In rh_agents/core/execution.py
class ExecutionState(BaseModel):
    # ... existing fields ...
    
    _active_generators: set[asyncio.Task] = Field(default_factory=set, exclude=True)
    
    def register_generator(self, generator_task: asyncio.Task):
        """Register an active event generator task."""
        self._active_generators.add(generator_task)
    
    def unregister_generator(self, generator_task: asyncio.Task):
        """Remove generator from active set."""
        self._active_generators.discard(generator_task)
    
    async def kill_generators(self):
        """Cancel all active event generators immediately."""
        for gen_task in list(self._active_generators):
            if not gen_task.done():
                gen_task.cancel()
        
        # Wait for all to complete
        if self._active_generators:
            await asyncio.gather(*self._active_generators, return_exceptions=True)
        
        self._active_generators.clear()
```

#### D.2. EventBus Stream Integration
```python
# In rh_agents/core/execution.py - EventBus
class EventBus(BaseModel):
    # ... existing fields ...
    
    _stream_task: Optional[asyncio.Task] = Field(default=None, exclude=True)
    
    async def stream(self) -> AsyncGenerator[ExecutionEvent, None]:
        """Stream events from queue."""
        try:
            while True:
                event = await self.queue.get()
                
                # Check for interrupt event
                if isinstance(event, InterruptEvent):
                    print("🛑 Interrupt signal received, terminating stream...")
                    break
                
                yield event
        except asyncio.CancelledError:
            print("🛑 Event stream cancelled")
            raise
```

### Integration Points
- ✅ **Streaming API endpoints**: Clean termination of SSE/WebSocket streams
- ✅ **EventBus.stream()**: Proper generator cleanup
- ✅ **Real-time updates**: Graceful connection closure
- ✅ **Parallel execution**: Coordinated with main interrupt flag

### Implementation Note
Generators register themselves with ExecutionState and are automatically cancelled when interrupt is triggered. This works seamlessly with the flag-based interrupt mechanism.

---

## 5. IMPLEMENTATION ROADMAP

### Architecture Summary

The recommended design (Option A + D) provides:
1. ✅ **Graceful execution interruption** via flag checks at strategic points
2. ✅ **Immediate generator termination** for streaming endpoints
3. ✅ **State preservation** through checkpoint mechanism
4. ✅ **EventBus integration** for notification propagation
5. ✅ **Distributed system support** via custom interrupt checkers
6. ✅ **Flexible signal format** - bool or InterruptSignal return types

### Implementation Phases

#### Phase 1: Core Infrastructure (Week 1)
- [ ] Add `ExecutionStatus.INTERRUPTED` and `CANCELLING` to ExecutionStatus enum
- [ ] Create `InterruptReason`, `InterruptSignal`, `InterruptEvent` models
- [ ] Add `ExecutionInterrupted` exception class
- [ ] Implement `ExecutionState.request_interrupt()` method for local control
- [ ] Implement `ExecutionState.set_interrupt_checker()` method for external signals
- [ ] Implement `ExecutionState.check_interrupt()` with support for both bool and InterruptSignal returns
- [ ] Add type alias `InterruptChecker` with all supported signatures

#### Phase 2: Integration (Week 2)
- [ ] Modify `ExecutionEvent.__call__()` to check interrupts
- [ ] Add generator registry to `ExecutionState`
- [ ] Update `EventBus.stream()` to handle interrupt events
- [ ] Integrate with `ParallelExecutionManager`

#### Phase 3: Handlers & UI (Week 3)
- [ ] Create `EventPrinter` handler for interrupt events
- [ ] Add interrupt controls to streaming API
- [ ] Implement timeout-based auto-interrupt
- [ ] Add examples and tests

### 5.1. IMPLEMENTATION READINESS CHECKLIST ✅

#### Design Completeness
- ✅ **Architecture defined**: Flag-based + Generator kill approach
- ✅ **Data models specified**: `InterruptReason`, `InterruptSignal`, `InterruptEvent`, `ExecutionInterrupted`
- ✅ **API signatures defined**: `request_interrupt()`, `set_interrupt_checker()`, `check_interrupt()`
- ✅ **Type system complete**: `InterruptChecker` with all variants (bool, InterruptSignal, Optional)
- ✅ **Integration points clear**: ExecutionState, ExecutionEvent, EventBus, ParallelExecutionManager
- ✅ **Error handling specified**: ExecutionInterrupted exception with reason and message

#### Distributed System Support
- ✅ **External checker pattern**: Documented with multiple examples (Redis, DB, API, Kafka, K8s)
- ✅ **Return type flexibility**: Supports bool (simple) and InterruptSignal (detailed)
- ✅ **Error resilience**: Checker failures logged but don't crash execution
- ✅ **Performance guidance**: Caching pattern provided for expensive checks

#### Code Examples Provided
- ✅ **Core implementation**: All components (A.1 through A.7, D.1-D.2)
- ✅ **Usage patterns**: 7+ distributed system examples
- ✅ **Best practices**: 5 patterns + 4 anti-patterns
- ✅ **Integration examples**: FastAPI, timeout enforcement, resource monitoring
- ✅ **Test examples**: Unit tests and integration tests

#### Documentation
- ✅ **API reference**: All methods documented with examples
- ✅ **Event flow diagrams**: Normal vs interrupted execution
- ✅ **Decision matrix**: When to use each pattern
- ✅ **Performance analysis**: Overhead measurements and optimization tips
- ✅ **Security considerations**: Auth, rate limiting, audit logging

#### Implementation Phases Defined
- ✅ **Phase 1 (Week 1)**: Core infrastructure - 7 specific tasks
- ✅ **Phase 2 (Week 2)**: Integration - 4 specific tasks
- ✅ **Phase 3 (Week 3)**: Handlers & UI - 4 specific tasks

#### Files to Create/Modify
- ✅ **New files identified**: 5 files (exceptions.py, interrupt.py or add to types.py, tests, examples)
- ✅ **Files to modify identified**: 6 files with specific changes needed
- ✅ **No ambiguity**: Each file's changes are specified in detail

### 5.2. READY FOR IMPLEMENTATION ✅

**Status**: 🟢 **SPECIFICATION COMPLETE - READY TO IMPLEMENT**

All necessary components, patterns, and examples are defined. An LLM or developer can now:
1. ✅ Implement Phase 1 core infrastructure from sections A.1-A.7, D.1-D.2
2. ✅ Follow integration patterns from section 6.6 for distributed systems
3. ✅ Use test examples from section 8 for validation
4. ✅ Reference best practices from section 7.5 during implementation

---

## 6. INTERFACE SPECIFICATION

### User-Facing API

#### 6.1. Trigger Interrupt (Synchronous Context)
```python
# From user code (e.g., Flask/FastAPI endpoint)
execution_state.request_interrupt(
    reason=InterruptReason.USER_CANCELLED,
    message="User clicked stop button",
    triggered_by="user_123",
    save_checkpoint=True
)
```

#### 6.2. Trigger Interrupt (Async Context)
```python
# From async handler
await execution_state.request_interrupt(
    reason=InterruptReason.TIMEOUT,
    message="Execution exceeded 5 minute limit",
    save_checkpoint=True
)
```

#### 6.3. FastAPI Integration Example
```python
from fastapi import FastAPI, HTTPException
from typing import Dict
import asyncio

app = FastAPI()

# Store active execution states
active_executions: Dict[str, ExecutionState] = {}

@app.post("/agent/execute")
async def execute_agent(query: str):
    state = ExecutionState()
    execution_id = state.state_id
    active_executions[execution_id] = state
    
    # Start execution in background
    task = asyncio.create_task(run_agent(query, state))
    
    return {
        "execution_id": execution_id,
        "status": "started"
    }

@app.post("/agent/interrupt/{execution_id}")
async def interrupt_execution(execution_id: str):
    state = active_executions.get(execution_id)
    if not state:
        raise HTTPException(404, "Execution not found")
    
    state.request_interrupt(
        reason=InterruptReason.USER_CANCELLED,
        message="User requested interruption",
        triggered_by="api_client"
    )
    
    return {"status": "interrupted"}
```

#### 6.4. Event Handler for Interrupts
```python
from rh_agents.bus_handlers import EventPrinter
from rh_agents.core.events import InterruptEvent

class InterruptAwarePrinter(EventPrinter):
    """Printer that handles interrupt events."""
    
    def __call__(self, event):
        if isinstance(event, InterruptEvent):
            self.print_interrupt(event)
        else:
            super().__call__(event)
    
    def print_interrupt(self, event: InterruptEvent):
        print(f"\n{self.RED}{self.BOLD}╔═══════════════════════════════════════╗{self.RESET}")
        print(f"{self.RED}{self.BOLD}║     ⚠️  EXECUTION INTERRUPTED ⚠️      ║{self.RESET}")
        print(f"{self.RED}{self.BOLD}╚═══════════════════════════════════════╝{self.RESET}\n")
        print(f"{self.BOLD}Reason:{self.RESET} {self.YELLOW}{event.signal.reason.value}{self.RESET}")
        if event.signal.message:
            print(f"{self.BOLD}Message:{self.RESET} {event.signal.message}")
        if event.signal.triggered_by:
            print(f"{self.BOLD}Triggered by:{self.RESET} {event.signal.triggered_by}")
        print(f"{self.BOLD}Time:{self.RESET} {event.signal.triggered_at}\n")
```

#### 6.5. Timeout Enforcement
```python
async def execute_with_timeout(agent, input_data, state, timeout_seconds=300):
    """Execute agent with automatic timeout."""
    
    async def timeout_monitor():
        await asyncio.sleep(timeout_seconds)
        state.request_interrupt(
            reason=InterruptReason.TIMEOUT,
            message=f"Execution exceeded {timeout_seconds}s timeout",
            save_checkpoint=True
        )
    
    monitor = asyncio.create_task(timeout_monitor())
    
    try:
        result = await ExecutionEvent(actor=agent)(input_data, "", state)
        monitor.cancel()
        return result
    except ExecutionInterrupted:
        return ExecutionResult(
            result=None,
            ok=False,
            erro_message="Execution timed out"
        )
```

#### 6.6. Distributed System Patterns (External Interrupt Sources)

When running agents in distributed systems, you can't always use `state.request_interrupt()` directly because you don't have access to the ExecutionState instance. Use `set_interrupt_checker()` to poll external interrupt signals.

##### 6.6.1. Redis-Based Interrupt Check
```python
import redis
import asyncio
import json

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Pattern 1: Simple boolean check
def check_redis_interrupt_simple(execution_id: str):
    """Check if interrupt signal is set in Redis (simple)."""
    return redis_client.get(f"interrupt:{execution_id}") == "1"

state = ExecutionState()
state.set_interrupt_checker(lambda: check_redis_interrupt_simple(state.state_id))

# From another process/service, trigger interrupt:
# redis_client.set(f"interrupt:{execution_id}", "1")

# Pattern 2: Detailed interrupt info
def check_redis_interrupt_detailed(execution_id: str):
    """Check Redis for detailed interrupt information."""
    data = redis_client.get(f"interrupt:{execution_id}")
    if data:
        interrupt_data = json.loads(data)
        return InterruptSignal(
            reason=InterruptReason(interrupt_data.get('reason', 'custom')),
            message=interrupt_data.get('message', 'Interrupted by external signal'),
            triggered_by=interrupt_data.get('triggered_by', 'external_service')
        )
    return None

state = ExecutionState()
state.set_interrupt_checker(lambda: check_redis_interrupt_detailed(state.state_id))

# From another process/service, trigger interrupt with details:
# redis_client.set(f"interrupt:{execution_id}", json.dumps({
#     "reason": "user_cancelled",
#     "message": "User clicked stop button",
#     "triggered_by": "web_ui_user_123"
# }))

# Now run your agent - it will check Redis periodically
result = await ExecutionEvent(actor=agent)(input_data, "", state)
```

##### 6.6.2. Database-Based Interrupt Check (Async)
```python
import aiosqlite

# Pattern 1: Simple boolean check
async def check_db_interrupt_simple(job_id: str):
    """Check database for interrupt signal (async)."""
    async with aiosqlite.connect("jobs.db") as db:
        cursor = await db.execute(
            "SELECT interrupted FROM jobs WHERE id = ?", 
            (job_id,)
        )
        row = await cursor.fetchone()
        return row[0] == 1 if row else False

state = ExecutionState()
state.set_interrupt_checker(lambda: check_db_interrupt_simple(state.state_id))

# From another process, trigger interrupt:
# UPDATE jobs SET interrupted = 1 WHERE id = '<job_id>';

# Pattern 2: Detailed interrupt info from database
async def check_db_interrupt_detailed(job_id: str):
    """Check database for detailed interrupt information (async)."""
    async with aiosqlite.connect("jobs.db") as db:
        cursor = await db.execute(
            """SELECT interrupted, reason, message, triggered_by 
               FROM jobs WHERE id = ?""", 
            (job_id,)
        )
        row = await cursor.fetchone()
        if row and row[0] == 1:  # interrupted = 1
            return InterruptSignal(
                reason=InterruptReason(row[1]) if row[1] else InterruptReason.CUSTOM,
                message=row[2] or "Interrupted by external signal",
                triggered_by=row[3] or "database"
            )
        return None

state = ExecutionState()
state.set_interrupt_checker(lambda: check_db_interrupt_detailed(state.state_id))

# From another process, trigger interrupt with details:
# UPDATE jobs SET 
#   interrupted = 1, 
#   reason = 'user_cancelled',
#   message = 'User requested stop',
#   triggered_by = 'admin_user_456'
# WHERE id = '<job_id>';
```

##### 6.6.3. REST API Interrupt Check
```python
import requests

# Pattern 1: Simple boolean check
def check_api_interrupt_simple(execution_id: str):
    """Poll API endpoint for interrupt status."""
    try:
        response = requests.get(
            f"https://control-plane.example.com/jobs/{execution_id}/status",
            timeout=2
        )
        return response.json().get('interrupt_requested', False)
    except Exception:
        return False  # Continue if check fails

state = ExecutionState()
state.set_interrupt_checker(lambda: check_api_interrupt_simple(state.state_id))

# Pattern 2: Detailed interrupt info from API
def check_api_interrupt_detailed(execution_id: str):
    """Poll API endpoint for detailed interrupt information."""
    try:
        response = requests.get(
            f"https://control-plane.example.com/jobs/{execution_id}/status",
            timeout=2
        )
        data = response.json()
        if data.get('interrupt_requested'):
            return InterruptSignal(
                reason=InterruptReason(data.get('reason', 'custom')),
                message=data.get('message', 'Interrupted from control plane'),
                triggered_by=data.get('triggered_by', 'api')
            )
        return None
    except Exception:
        return None  # Continue if check fails

state = ExecutionState()
state.set_interrupt_checker(lambda: check_api_interrupt_detailed(state.state_id))

# From control plane API, set interrupt with details:
# POST /jobs/{execution_id}/interrupt
# {
#   "reason": "timeout",
#   "message": "Job exceeded maximum runtime",
#   "triggered_by": "timeout_monitor"
# }
```

##### 6.6.4. Message Queue (Kafka/RabbitMQ) Pattern
```python
from kafka import KafkaConsumer
from queue import Queue
import threading

class InterruptSignalListener:
    """Background listener for interrupt signals from message queue."""
    
    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.interrupt_requested = False
        self.queue = Queue()
        self._consumer = None
        self._thread = None
    
    def start(self):
        """Start listening for interrupt signals in background thread."""
        self._consumer = KafkaConsumer(
            f'interrupts.{self.execution_id}',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='latest'
        )
        
        def listen():
            for message in self._consumer:
                if message.value == b'interrupt':
                    self.interrupt_requested = True
                    break
        
        self._thread = threading.Thread(target=listen, daemon=True)
        self._thread.start()
    
    def is_interrupted(self) -> bool:
        return self.interrupt_requested
    
    def stop(self):
        if self._consumer:
            self._consumer.close()

# Usage
listener = InterruptSignalListener(execution_id="job-123")
listener.start()

state = ExecutionState()
state.set_interrupt_checker(listener.is_interrupted)

try:
    result = await ExecutionEvent(actor=agent)(input_data, "", state)
finally:
    listener.stop()

# From another service, publish interrupt:
# producer.send(f'interrupts.{execution_id}', b'interrupt')
```

##### 6.6.5. Shared Memory / File-Based (Simple but Effective)
```python
import os
from pathlib import Path

def check_interrupt_file(execution_id: str):
    """Check if interrupt file exists."""
    interrupt_file = Path(f"/tmp/interrupts/{execution_id}.signal")
    return interrupt_file.exists()

# Setup
state = ExecutionState()
state.set_interrupt_checker(lambda: check_interrupt_file(state.state_id))

# From another process, create interrupt file:
# touch /tmp/interrupts/{execution_id}.signal
```

##### 6.6.6. Combined Pattern: Local + External
```python
# Best of both worlds: support both local and external interrupts

# Setup external checker for distributed control
state = ExecutionState()
state.set_interrupt_checker(lambda: check_redis_interrupt(state.state_id))

# Local code can still use direct interrupt
async def timeout_monitor():
    await asyncio.sleep(300)
    # Direct local interrupt for timeout
    state.request_interrupt(
        reason=InterruptReason.TIMEOUT,
        message="Local timeout enforced"
    )

asyncio.create_task(timeout_monitor())

# Agent will check BOTH:
# 1. Local interrupt flag (via request_interrupt)
# 2. External Redis signal (via interrupt_checker)
result = await ExecutionEvent(actor=agent)(input_data, "", state)
```

##### 6.6.7. Kubernetes ConfigMap/Secret Pattern
```python
from kubernetes import client, config

def check_k8s_interrupt(namespace: str, execution_id: str):
    """Check Kubernetes ConfigMap for interrupt signal."""
    try:
        config.load_incluster_config()  # Or load_kube_config() for local
        v1 = client.CoreV1Api()
        
        configmap = v1.read_namespaced_config_map(
            name=f"execution-{execution_id}",
            namespace=namespace
        )
        return configmap.data.get('interrupt', 'false') == 'true'
    except Exception:
        return False

# Usage in Kubernetes pod
state = ExecutionState()
state.set_interrupt_checker(
    lambda: check_k8s_interrupt("default", state.state_id)
)

# From kubectl or control plane:
# kubectl patch configmap execution-{id} -p '{"data":{"interrupt":"true"}}'
```

#### 6.7. Resource Limit Enforcement
```python
class ResourceMonitor:
    """Monitor system resources and interrupt on limits."""
    
    def __init__(self, state: ExecutionState, max_memory_mb=1000):
        self.state = state
        self.max_memory_mb = max_memory_mb
    
    async def monitor(self):
        while not self.state.is_interrupted():
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb:
                self.state.request_interrupt(
                    reason=InterruptReason.RESOURCE_LIMIT,
                    message=f"Memory usage {memory_mb:.0f}MB exceeds limit {self.max_memory_mb}MB",
                    save_checkpoint=True
                )
                break
            
            await asyncio.sleep(1)
```

---

## 7. EVENT FLOW DIAGRAM

### Normal Execution vs Interrupted Execution

```
┌─────────────────────────────────────────────────────────────┐
│                     NORMAL EXECUTION                         │
└─────────────────────────────────────────────────────────────┘

User Input
    │
    ▼
┌──────────────┐
│ Agent Start  │ ──► ExecutionEvent[STARTED]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LLM Call    │ ──► ExecutionEvent[STARTED]
└──────┬───────┘     ExecutionEvent[COMPLETED]
       │
       ▼
┌──────────────┐
│  Tool Call   │ ──► ExecutionEvent[STARTED]
└──────┬───────┘     ExecutionEvent[COMPLETED]
       │
       ▼
┌──────────────┐
│ Agent End    │ ──► ExecutionEvent[COMPLETED]
└──────────────┘

═══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│                  INTERRUPTED EXECUTION                       │
└─────────────────────────────────────────────────────────────┘

User Input
    │
    ▼
┌──────────────┐
│ Agent Start  │ ──► ExecutionEvent[STARTED]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LLM Call    │ ──► ExecutionEvent[STARTED]
└──────┬───────┘     ExecutionEvent[COMPLETED]
       │
       ▼
┌──────────────┐
│USER CLICKS   │ ──► state.request_interrupt()
│   STOP!      │
└──────┬───────┘
       │
       ├─────────────┐
       │             ▼
       │        ┌────────────────┐
       │        │ InterruptEvent │ ──► EventBus.publish()
       │        └────────────────┘
       │                  │
       │                  ├──► EventPrinter: shows interrupt
       │                  ├──► Streamer: closes connection
       │                  └──► Custom handlers: cleanup
       │
       ▼
┌──────────────┐
│  Tool Call   │ ──► check_interrupt() ──► ExecutionInterrupted!
└──────┬───────┘                                  │
       │                                          ▼
       │                              ExecutionEvent[INTERRUPTED]
       ▼
┌──────────────┐
│ Agent End    │ ──► ExecutionEvent[INTERRUPTED]
└──────────────┘     ExecutionResult(ok=False)
```

---

## 7.5. BEST PRACTICES & DESIGN PATTERNS

### Pattern 1: Local Control (Direct Access to State)
**Use when:** You have direct access to the ExecutionState instance (same process/thread)

```python
# ✅ Good: Direct local interrupt
state = ExecutionState()

# Set timeout monitor
async def timeout_handler():
    await asyncio.sleep(300)
    state.request_interrupt(
        reason=InterruptReason.TIMEOUT,
        message="Execution exceeded 5 minutes"
    )

asyncio.create_task(timeout_handler())

# UI click handler
@app.post("/stop")
def stop_execution():
    state.request_interrupt(
        reason=InterruptReason.USER_CANCELLED,
        message="User clicked stop button",
        triggered_by="web_ui"
    )
```

### Pattern 2: Distributed Control (External Signal Source)
**Use when:** Interrupt signal comes from external system (different process, machine, or service)

```python
# ✅ Good: External interrupt checker
state = ExecutionState()

# Set up checker for external signal
def check_external_interrupt():
    return redis_client.get(f"interrupt:{state.state_id}") == "1"

state.set_interrupt_checker(check_external_interrupt)

# Another service/process triggers interrupt:
# redis_client.set(f"interrupt:{execution_id}", "1")
```

### Pattern 3: Hybrid (Best of Both Worlds)
**Use when:** You want both local and external interrupt capabilities

```python
# ✅ Best: Supports both local and external interrupts
state = ExecutionState()

# Configure external checker for distributed control
state.set_interrupt_checker(
    lambda: check_redis_interrupt(state.state_id)
)

# Local code can still trigger directly
async def local_timeout():
    await asyncio.sleep(300)
    state.request_interrupt(reason=InterruptReason.TIMEOUT)

asyncio.create_task(local_timeout())

# Agents check BOTH sources automatically via check_interrupt()
```

### Pattern 4: Graceful Shutdown in Long-Running Loops
**Use when:** Processing collections or long-running iterations

```python
# ✅ Good: Regular interrupt checks in loops
@agent("process_documents")
async def process_docs(documents: list[Document], context, state):
    results = []
    
    for i, doc in enumerate(documents):
        # Check for interrupt before each document
        await state.check_interrupt()
        
        result = await process_single_document(doc)
        results.append(result)
        
        # Optional: check after expensive operations too
        await state.check_interrupt()
    
    return results
```

### Pattern 5: Interrupt-Aware Event Handlers
**Use when:** Custom handlers need interrupt awareness

```python
# ✅ Good: Check interrupts at key points
@agent("complex_analysis")
async def analyze(data: Data, context, state):
    # Phase 1
    await state.check_interrupt()
    preprocessed = await preprocess(data)
    
    # Phase 2
    await state.check_interrupt()
    analyzed = await run_analysis(preprocessed)
    
    # Phase 3
    await state.check_interrupt()
    result = await post_process(analyzed)
    
    return result
```

### ❌ Anti-Patterns to Avoid

#### ❌ Don't: Forget to check interrupts in long-running operations
```python
# BAD: No interrupt checks
async def process_large_dataset(data, context, state):
    for item in huge_dataset:  # Could run for hours
        result = expensive_operation(item)  # No way to stop!
    return result

# GOOD: Regular checks
async def process_large_dataset(data, context, state):
    for item in huge_dataset:
        await state.check_interrupt()  # ✅ Can stop gracefully
        result = expensive_operation(item)
    return result
```

#### ❌ Don't: Use global variables for distributed systems
```python
# BAD: Global variable won't work across processes/machines
global_interrupt_flag = False

def check_interrupt():
    return global_interrupt_flag

# GOOD: Use external checker
def check_interrupt():
    return redis_client.get(f"interrupt:{execution_id}") == "1"
```

#### ❌ Don't: Ignore interrupt checker failures silently
```python
# BAD: Fails silently if checker raises exception
def unreliable_checker():
    response = requests.get("http://api.example.com/status")  # No timeout!
    return response.json()['interrupted']

# GOOD: Handle errors gracefully
def reliable_checker():
    try:
        response = requests.get("http://api.example.com/status", timeout=2)
        return response.json().get('interrupted', False)
    except Exception as e:
        logging.warning(f"Interrupt check failed: {e}")
        return False  # Continue on check failure
```

#### ❌ Don't: Make expensive calls in interrupt checker
```python
# BAD: Expensive operation called frequently
def slow_checker():
    # This gets called many times per second!
    results = database.query("SELECT * FROM huge_table WHERE ...")
    return results[0]['interrupted']

# GOOD: Fast, indexed lookup
def fast_checker():
    # Simple key lookup, very fast
    return redis_client.get(f"interrupt:{execution_id}") == "1"
```

### Performance Considerations

1. **Interrupt Check Frequency**
   - `check_interrupt()` is called automatically at key execution points
   - Add manual checks in long-running loops (e.g., every 100 iterations)
   - Balance responsiveness vs overhead

2. **External Checker Performance**
   - Keep checker functions FAST (< 10ms)
   - Use caching if needed
   - Consider async checkers for I/O operations

3. **Avoid Checker Storms**
   ```python
   # ✅ Good: Cache external checks
   class CachedInterruptChecker:
       def __init__(self, check_interval=1.0):
           self.last_check = 0
           self.last_result = False
           self.check_interval = check_interval
       
       def check(self):
           now = time.time()
           if now - self.last_check < self.check_interval:
               return self.last_result
           
           self.last_check = now
           self.last_result = self._actual_check()
           return self.last_result
       
       def _actual_check(self):
           return redis_client.get(f"interrupt:{execution_id}") == "1"
   
   checker = CachedInterruptChecker(check_interval=1.0)
   state.set_interrupt_checker(checker.check)
   ```

### Decision Matrix: When to Use Each Pattern

| Scenario | Pattern | Example |
|----------|---------|---------|
| Single process, local control | `state.request_interrupt()` | CLI tool, single-machine app |
| Distributed system | `set_interrupt_checker()` | Microservices, Kubernetes jobs |
| Web UI with backend | `set_interrupt_checker()` | Check database/Redis from UI |
| Both local and remote | Hybrid | Production system with multiple triggers |
| Long-running jobs | Add manual `check_interrupt()` | Data processing pipelines |
| Real-time requirements | Use caching in checker | High-frequency operations |

---

## 8. TESTING STRATEGY

### Unit Tests
```python
# tests/test_interrupt.py

async def test_basic_interrupt():
    """Test that interrupt flag stops execution."""
    state = ExecutionState()
    
    # Trigger interrupt
    state.request_interrupt(
        reason=InterruptReason.USER_CANCELLED,
        message="Test interrupt"
    )
    
    # Check flag
    assert state.is_interrupted()
    
    # Should raise exception
    with pytest.raises(ExecutionInterrupted):
        state.check_interrupt()

async def test_interrupt_preserves_state():
    """Test that checkpoint is saved on interrupt."""
    backend = FileSystemStateBackend(base_path="/tmp/test_states")
    state = ExecutionState(state_backend=backend)
    
    # Add some data
    state.storage.set("key", "value")
    
    # Interrupt with checkpoint
    state.request_interrupt(
        reason=InterruptReason.USER_CANCELLED,
        save_checkpoint=True
    )
    
    # Verify checkpoint exists
    snapshot = backend.load_state(state.state_id)
    assert snapshot is not None
    assert snapshot.status == StateStatus.INTERRUPTED

async def test_parallel_cancellation():
    """Test that parallel tasks are cancelled on interrupt."""
    state = ExecutionState()
    
    async def slow_task():
        await asyncio.sleep(10)
        return "done"
    
    async with state.parallel(max_workers=3) as p:
        p.add(slow_task())
        p.add(slow_task())
        p.add(slow_task())
        
        # Interrupt after 0.1s
        await asyncio.sleep(0.1)
        state.request_interrupt(InterruptReason.USER_CANCELLED)
        
        # Should raise or return incomplete results
        results = await p.gather()
    
    assert state.is_interrupted()
```

### Integration Tests
```python
async def test_end_to_end_interrupt():
    """Test interrupt in full agent execution."""
    llm = MockLLM()
    agent = TestAgent(llm=llm)
    state = ExecutionState()
    
    # Start execution in background
    task = asyncio.create_task(
        ExecutionEvent(actor=agent)("test input", "", state)
    )
    
    # Interrupt after 0.5s
    await asyncio.sleep(0.5)
    state.request_interrupt(InterruptReason.USER_CANCELLED)
    
    # Wait for completion
    result = await task
    
    # Verify interrupted
    assert not result.ok
    assert "interrupted" in result.erro_message.lower()
```

---

## 9. DOCUMENTATION & EXAMPLES

### Example 1: Basic Interrupt
```python
"""
examples/interrupt_basic.py
Basic interrupt example.
"""
import asyncio
from rh_agents import ExecutionState, InterruptReason

async def main():
    state = ExecutionState()
    
    # Simulate long-running operation
    async def long_operation():
        for i in range(100):
            state.check_interrupt()  # Check at each step
            await asyncio.sleep(0.1)
            print(f"Step {i}")
    
    # Start operation
    task = asyncio.create_task(long_operation())
    
    # Interrupt after 1 second
    await asyncio.sleep(1)
    state.request_interrupt(
        reason=InterruptReason.USER_CANCELLED,
        message="User stopped the process"
    )
    
    try:
        await task
    except ExecutionInterrupted as e:
        print(f"Interrupted: {e.message}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Web API with Interrupt
```python
"""
examples/interrupt_api.py
FastAPI endpoint with interrupt support.
"""
# (See section 6.3 above)
```

---

## 10. MIGRATION GUIDE

### For Existing Code
1. **No breaking changes**: Interrupt is opt-in
2. **Existing handlers**: Continue to work unchanged
3. **To add interrupt support**:
   - Add `state.check_interrupt()` calls in long-running handlers
   - Subscribe interrupt-aware event handlers
   - Expose interrupt endpoint in API

### For Custom Handlers
```python
# Before
async def my_handler(args, context, state):
    result = do_work()
    return result

# After (with interrupt support)
async def my_handler(args, context, state):
    state.check_interrupt()  # Check before work
    result = do_work()
    state.check_interrupt()  # Check after work
    return result
```

---

## 11. PERFORMANCE IMPACT

### Overhead Analysis
- **Flag check**: ~0.1μs (negligible)
- **Checkpoint save**: ~10-100ms (only when interrupted)
- **Event publish**: ~1-5ms (only when interrupted)
- **Generator monitoring**: ~0.1ms per check (100ms interval)

### Optimization Tips
1. Place `check_interrupt()` at strategic points (not in tight loops)
2. Use async checkpointing to avoid blocking
3. Batch interrupt checks (e.g., every N iterations)

---

## 12. SECURITY CONSIDERATIONS

### Potential Issues
1. **Denial of Service**: Malicious interrupts could disrupt service
2. **State tampering**: Unauthorized access to interrupt API
3. **Resource leaks**: Improper cleanup on interrupt

### Mitigations
1. **Authentication**: Require auth for interrupt endpoint
2. **Rate limiting**: Limit interrupt requests per user/IP
3. **Audit logging**: Log all interrupt events with user ID
4. **Cleanup hooks**: Ensure resources are freed on interrupt

```python
# Example: Authenticated interrupt endpoint
@app.post("/agent/interrupt/{execution_id}")
async def interrupt_execution(
    execution_id: str,
    current_user: User = Depends(get_current_user)
):
    # Verify user owns this execution
    state = active_executions.get(execution_id)
    if not state or state.metadata.user_id != current_user.id:
        raise HTTPException(403, "Unauthorized")
    
    # Log interrupt
    logger.info(f"User {current_user.id} interrupted execution {execution_id}")
    
    state.request_interrupt(
        reason=InterruptReason.USER_CANCELLED,
        triggered_by=current_user.id
    )
    
    return {"status": "interrupted"}
```

---

## 13. FUTURE ENHANCEMENTS

### V2 Features
1. **Interrupt priorities**: Different interrupt levels (soft, hard, critical)
2. **Resumable interrupts**: Ability to resume after graceful interrupt
3. **Interrupt callbacks**: Custom cleanup functions per actor
4. **Interrupt propagation control**: Fine-grained control over which events are interrupted
5. **Interrupt analytics**: Track interrupt patterns and reasons

### Example: Interrupt Priorities
```python
class InterruptPriority(int, Enum):
    SOFT = 1      # Request stop, can be ignored
    NORMAL = 5    # Should stop at next checkpoint
    HARD = 9      # Force stop immediately
    CRITICAL = 10 # Kill process if needed
```

---

## 14. DESIGN DECISIONS ✅ (All Resolved)

All design questions have been resolved during specification development:

### Q1: Interrupt Strategy ✅ **DECIDED**
**Decision**: Option A (Flag-Based) + Option D (Generator Kill)
- Graceful interruption via flag checks at strategic points
- Immediate generator termination for streaming endpoints
- Distributed system support via custom interrupt checkers

### Q2: Checkpoint Behavior ✅ **DECIDED**
**Decision**: User decides per-interrupt via `save_checkpoint` parameter
- Default: `save_checkpoint=True` (saves state by default)
- User can override per-interrupt: `state.request_interrupt(..., save_checkpoint=False)`
- Provides maximum flexibility for different use cases

### Q3: Generator Handling ✅ **DECIDED**
**Decision**: Kill generators immediately on interrupt
- Generators register with ExecutionState
- Automatic cancellation when interrupt is triggered
- Clean termination of streaming endpoints (SSE, WebSocket)

### Q4: API Exposure ✅ **DECIDED**
**Decision**: Comprehensive examples provided
- FastAPI integration example (section 6.3)
- Distributed system patterns (section 6.6.1-6.6.7)
- Timeout enforcement example (section 6.5)
- Resource monitoring example (section 6.7)
- Basic interrupt example (section 9)

### Q5: Error Handling ✅ **DECIDED**
**Decision**: Raise `ExecutionInterrupted`, then convert to `ExecutionResult(ok=False)`
- Framework raises `ExecutionInterrupted` exception internally
- `ExecutionEvent.__call__()` catches and converts to `ExecutionResult`
- User receives clean error result: `ExecutionResult(result=None, ok=False, erro_message="...")`
- Maintains compatibility with existing error handling patterns

### Q6: External Checker Return Type ✅ **DECIDED**
**Decision**: Support both `bool` and `InterruptSignal` returns
- Simple case: Return `True`/`False` (uses generic interrupt info)
- Detailed case: Return `InterruptSignal` object with reason, message, triggered_by
- Provides flexibility for simple and complex distributed systems

---

## 15. IMPLEMENTATION CHECKLIST

### Core Files to Create/Modify

#### New Files
- [ ] `rh_agents/core/exceptions.py` - Custom exceptions
- [ ] `rh_agents/core/interrupt.py` - Interrupt models (or add to types.py)
- [ ] `tests/test_interrupt.py` - Unit tests
- [ ] `examples/interrupt_basic.py` - Basic example
- [ ] `examples/interrupt_api.py` - API example

#### Files to Modify
- [ ] `rh_agents/core/types.py` - Add INTERRUPTED status, InterruptReason
- [ ] `rh_agents/core/execution.py` - Add interrupt methods to ExecutionState
- [ ] `rh_agents/core/events.py` - Add InterruptEvent, modify ExecutionEvent.__call__()
- [ ] `rh_agents/core/parallel.py` - Add interrupt monitoring to ParallelExecutionManager
- [ ] `rh_agents/bus_handlers.py` - Add interrupt event handling to printers
- [ ] `examples/streaming_api.py` - Add interrupt endpoint

---

## CONCLUSION

This specification defines a complete interrupt system for the RH-Agents framework using **Flag-Based Interrupt + Generator Kill (Option A + D)** approach.

### ✅ Implementation Ready

The specification includes:
- ✅ Complete implementation details (sections A.1-A.7, D.1-D.2)
- ✅ Distributed system patterns (section 6.6)
- ✅ Best practices and anti-patterns (section 7.5)
- ✅ Test strategy (section 8)
- ✅ Three-phase implementation roadmap (section 5)
- ✅ File-by-file change list (section 15)

### 🚀 Next Steps

**Ready to start Phase 1 implementation:**
1. Create `InterruptReason`, `InterruptSignal`, `InterruptEvent` models
2. Add `ExecutionInterrupted` exception
3. Implement `ExecutionState` interrupt methods:
   - `request_interrupt()` - for local control
   - `set_interrupt_checker()` - for distributed control  
   - `check_interrupt()` - checks both sources, supports bool and InterruptSignal returns
4. Add `InterruptChecker` type alias with all variants
5. Add `ExecutionStatus.INTERRUPTED` enum value

**Follow with Phase 2 and 3** as defined in section 5.

### 🎯 Key Design Principles

- ✅ **Graceful shutdown by default** - checkpoint before termination
- ✅ **Distributed system support** - external interrupt checkers
- ✅ **Flexible signal format** - bool or detailed InterruptSignal
- ✅ **State preservation** - save progress on interrupt
- ✅ **EventBus integration** - notify all subscribers
- ✅ **Minimal performance impact** - flag checks < 0.1μs
- ✅ **Backward compatible** - opt-in feature
- ✅ **Easy to test and debug** - deterministic behavior

### 📋 Design Decisions Summary

All design questions resolved:

**Q1: Interrupt Strategy** ✅ Flag-based (Option A) + Generator Kill (Option D)  
**Q2: Checkpoint Behavior** ✅ User decides per-interrupt via `save_checkpoint` parameter (default: True)  
**Q3: Generator Handling** ✅ Kill generators immediately on interrupt  
**Q4: API Exposure** ✅ Comprehensive examples provided (sections 6.3, 6.5-6.7, 9)  
**Q5: Error Handling** ✅ Raise `ExecutionInterrupted`, caught and converted to `ExecutionResult(ok=False)`  
**Q6: External Checker Return** ✅ Support both `bool` (simple) and `InterruptSignal` (detailed)

See section 14 for detailed rationale on each decision.

---

## 📦 DELIVERABLES SUMMARY

### Core Components (Phase 1)
1. `InterruptReason` enum (7 reasons: user_cancelled, timeout, resource_limit, etc.)
2. `InterruptSignal` model (reason, message, triggered_at, triggered_by)
3. `InterruptEvent` model (signal + execution_state)
4. `ExecutionInterrupted` exception (reason, message)
5. `ExecutionState.request_interrupt()` - local control
6. `ExecutionState.set_interrupt_checker()` - external signals
7. `ExecutionState.check_interrupt()` - unified checking (bool or InterruptSignal)
8. `InterruptChecker` type alias (6 variants: sync/async × bool/signal/optional)

### Integrations (Phase 2)
9. `ExecutionEvent.__call__()` - add 4 interrupt check points
10. Generator registry in `ExecutionState`
11. `EventBus.stream()` - handle interrupt events
12. `ParallelExecutionManager` - interrupt monitoring

### UI & Examples (Phase 3)
13. `EventPrinter` interrupt handler
14. FastAPI interrupt endpoint example
15. Distributed system examples (Redis, DB, API, Kafka, K8s, file-based)
16. Test suite (unit + integration)

**Total**: 16 deliverables across 3 phases, fully specified and ready for implementation.
