# Retry Mechanism - Implementation Specification

**Status**: ✅ Approved - Ready for Implementation  
**Version**: 2.0 (Consolidated)  
**Date**: February 10, 2026

---

## Quick Implementation Summary

### What We're Building
A retry mechanism that automatically retries failed execution events with configurable backoff strategies and smart error filtering.

### Key Features
- ✅ **Multi-level configuration**: Event → Actor-type → Global → Built-in defaults
- ✅ **Smart error filtering**: Default transient errors + whitelist/blacklist override
- ✅ **Flexible backoff**: Constant, Linear, Exponential, Fibonacci strategies
- ✅ **Full observability**: RETRYING events + retry metadata + beautiful console output
- ✅ **State-aware**: Persists retry history, fast deterministic replay
- ✅ **Interrupt-safe**: Responsive cancellation during retry delays
- ✅ **Timeout control**: Per-attempt and total retry timeout

### Implementation Order
1. **Phase 1** (1-2 days): Core retry loop, backoff, error filtering
2. **Phase 2** (0.5-1 day): Multi-level configuration & precedence
3. **Phase 3** (0.5-1 day): Beautiful printer display with ↻ symbol
4. **Phase 4** (1 day): State persistence & replay support
5. **Phase 5** (1 day): Interrupt & timeout integration

### Files to Create/Modify
- **New**: `rh_agents/core/retry.py` - RetryConfig, backoff logic, error filtering
- **Modify**: `rh_agents/core/events.py` - Add retry loop to ExecutionEvent.__call__()
- **Modify**: `rh_agents/core/execution.py` - Add retry config fields to ExecutionState
- **Modify**: `rh_agents/core/types.py` - Add BackoffStrategy enum, RETRYING status
- **Modify**: `rh_agents/bus_handlers.py` - Update EventPrinter for retry display
- **Modify**: `rh_agents/decorators.py` - Add retry_config parameter to decorators
- **New**: 13+ test files for comprehensive testing

---

## Overview

This specification defines a retry mechanism for execution events in the rh_agents framework. The system will allow configurable retry behavior at multiple levels (global defaults, actor-type specific, and per-event) with intelligent backoff strategies and error filtering.

## Design Decisions Summary

| Area | Decision | Rationale |
|------|----------|-----------|
| **Backoff Strategy** | Enum-based with built-in strategies | Simple, type-safe, covers common cases |
| **Error Filtering** | Smart defaults + whitelist/blacklist override | Safe defaults, full control when needed |
| **Event Emission** | RETRYING event + retry metadata in STARTED | Maximum observability |
| **Config Precedence** | Simple override (most specific wins) | Clear, predictable, performant |
| **State Persistence** | Persist state, skip retry logic on replay | Accurate history, fast replay |
| **Retry Scope** | Handler + postconditions (skip preconditions) | Efficient, balanced coverage |
| **Interrupt Check** | Check interrupts during retry delays | Responsive, better UX |
| **Timeout Handling** | Separate per-attempt and total timeouts | Flexible and correct |
| **Printer Display** | Dual mode: compact for success, detailed for failure | Best UX, use ↻ symbol |
| **Default Behavior** | Different defaults per actor type | Smart defaults, explicit override |
| **Phase 1 Scope** | Simple retry only (no circuit breakers yet) | Iterative development |

---

## Core Components


### 1. BackoffStrategy Enum

Built-in backoff strategies for retry delays:

```python
class BackoffStrategy(str, Enum):
    CONSTANT = "constant"      # Fixed delay between retries
    LINEAR = "linear"          # Linearly increasing delay (delay * attempt)
    EXPONENTIAL = "exponential" # Exponentially increasing delay (delay * multiplier^attempt)
    FIBONACCI = "fibonacci"    # Fibonacci sequence delays (1, 1, 2, 3, 5, 8, ...)
```

### 2. Default Exception Lists

Smart defaults for transient vs permanent errors:

```python
# Default transient errors that should be retried
DEFAULT_RETRYABLE_EXCEPTIONS = [
    TimeoutError,
    asyncio.TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionRefusedError,
    # Add HTTP-specific errors as needed (429, 503, etc.)
]

# Default permanent errors that should NOT be retried
DEFAULT_NON_RETRYABLE_EXCEPTIONS = [
    ValidationError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    # Add auth/permission errors as needed
]
```

### 3. RetryConfig Model

Configuration for retry behavior:

```python
class RetryConfig(BaseModel):
    # Core retry settings
    max_attempts: int = Field(default=3, ge=1, description="Maximum retry attempts (including initial)")
    backoff_strategy: BackoffStrategy = Field(default=BackoffStrategy.EXPONENTIAL)
    initial_delay: float = Field(default=1.0, ge=0, description="Initial delay in seconds")
    max_delay: float = Field(default=60.0, ge=0, description="Maximum delay between retries")
    backoff_multiplier: float = Field(default=2.0, ge=1.0, description="Multiplier for exponential/linear backoff")
    jitter: bool = Field(default=True, description="Add random jitter to prevent thundering herd")
    
    # Error filtering (override defaults)
    retry_on_exceptions: Optional[list[type[Exception]]] = Field(
        default=None, 
        description="Whitelist of exceptions to retry (overrides defaults if set)"
    )
    exclude_exceptions: Optional[list[type[Exception]]] = Field(
        default=None, 
        description="Blacklist of exceptions to never retry (merged with defaults)"
    )
    
    # Timeout settings
    retry_timeout: Optional[float] = Field(
        default=None, 
        ge=0, 
        description="Total timeout for all retry attempts combined (seconds)"
    )
    
    # Control
    enabled: bool = Field(default=True, description="Enable/disable retry for this config")
    
    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Logic:
        1. If retry_on_exceptions is set (whitelist), exception must be in it
        2. Exception must NOT be in exclude_exceptions (merged with defaults)
        3. If both are None, use DEFAULT_RETRYABLE_EXCEPTIONS
        """
        # Build effective exclude list (user's + defaults)
        exclude_list = DEFAULT_NON_RETRYABLE_EXCEPTIONS.copy()
        if self.exclude_exceptions:
            exclude_list.extend(self.exclude_exceptions)
        
        # Check exclusions first
        if any(isinstance(exception, exc_type) for exc_type in exclude_list):
            return False
        
        # If whitelist is set, exception must be in it
        if self.retry_on_exceptions is not None:
            return any(isinstance(exception, exc_type) for exc_type in self.retry_on_exceptions)
        
        # Otherwise, check default retryable list
        return any(isinstance(exception, exc_type) for exc_type in DEFAULT_RETRYABLE_EXCEPTIONS)
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay before retry attempt.
        
        Args:
            attempt: Retry attempt number (1-based)
        
        Returns:
            Delay in seconds (with jitter if enabled)
        """
        if self.backoff_strategy == BackoffStrategy.CONSTANT:
            delay = self.initial_delay
        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.initial_delay * attempt
        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))
        elif self.backoff_strategy == BackoffStrategy.FIBONACCI:
            # Generate fibonacci number for attempt
            fib = self._fibonacci(attempt)
            delay = self.initial_delay * fib
        else:
            delay = self.initial_delay
        
        # Apply max delay cap
        delay = min(delay, self.max_delay)
        
        # Apply jitter if enabled
        if self.jitter:
            import random
            # Add ±25% jitter
            jitter_factor = 0.75 + (random.random() * 0.5)
            delay = delay * jitter_factor
        
        return delay
    
    @staticmethod
    def _fibonacci(n: int) -> int:
        """Calculate nth fibonacci number (1-based)."""
        if n <= 1:
            return 1
        a, b = 1, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return a

# Default retry configurations by actor type
DEFAULT_RETRY_CONFIG_BY_ACTOR_TYPE = {
    EventType.TOOL_CALL: RetryConfig(
        max_attempts=3,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        initial_delay=1.0,
        max_delay=30.0,
        enabled=True
    ),
    EventType.LLM_CALL: RetryConfig(
        max_attempts=3,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        initial_delay=1.0,
        max_delay=30.0,
        enabled=True
    ),
    EventType.AGENT_CALL: RetryConfig(
        enabled=False  # Agents handle their own logic
    )
}
```


### 4. ExecutionEvent Changes

Add retry-related fields to `ExecutionEvent`:

```python
class ExecutionEvent(BaseModel, Generic[T]):
    # ... existing fields ...
    
    # Retry configuration and state
    retry_config: Optional[RetryConfig] = Field(
        default=None, 
        description="Retry configuration for this event"
    )
    retry_attempt: int = Field(
        default=0, 
        description="Current retry attempt number (0 = first attempt, 1 = first retry)"
    )
    is_retry: bool = Field(
        default=False, 
        description="True if this is a retry attempt"
    )
    original_error: Optional[str] = Field(
        default=None, 
        description="Error from previous attempt that triggered retry"
    )
    retry_delay: Optional[float] = Field(
        default=None,
        description="Delay before this retry (seconds)"
    )
```

### 5. ExecutionState Changes

Add retry configuration management to `ExecutionState`:

```python
class ExecutionState(BaseModel):
    # ... existing fields ...
    
    # Retry configuration (Phase 2)
    default_retry_config: Optional[RetryConfig] = Field(
        default=None, 
        description="Default retry config for all events"
    )
    retry_config_by_actor_type: dict[EventType, RetryConfig] = Field(
        default_factory=dict, 
        description="Retry configs per actor type (overrides default)"
    )
    
    def get_effective_retry_config(
        self, 
        event: ExecutionEvent, 
        actor_type: EventType
    ) -> Optional[RetryConfig]:
        """
        Get effective retry config using precedence rules:
        1. Event's retry_config (highest priority)
        2. Actor-type specific config
        3. Global default config
        4. Built-in defaults by actor type (lowest priority)
        
        Returns None if retry is disabled at all levels.
        """
        # Check event-level config first
        if event.retry_config is not None:
            return event.retry_config if event.retry_config.enabled else None
        
        # Check actor-type config
        if actor_type in self.retry_config_by_actor_type:
            config = self.retry_config_by_actor_type[actor_type]
            return config if config.enabled else None
        
        # Check global default
        if self.default_retry_config is not None:
            return self.default_retry_config if self.default_retry_config.enabled else None
        
        # Fall back to built-in defaults
        if actor_type in DEFAULT_RETRY_CONFIG_BY_ACTOR_TYPE:
            config = DEFAULT_RETRY_CONFIG_BY_ACTOR_TYPE[actor_type]
            return config if config.enabled else None
        
        return None
```

### 6. New Event Types

Add retry-specific execution status:

```python
class ExecutionStatus(str, Enum):
    # ... existing statuses ...
    RETRYING = "retrying"  # Event is being retried after failure
```

---

## Implementation Details

### Retry Logic Flow

The retry mechanism is implemented in `ExecutionEvent.__call__()` with the following flow:

```
1. Get effective retry config (precedence: event → actor-type → global → default)
2. If no retry config or disabled, execute normally (existing behavior)
3. Run preconditions (once, no retry)
4. Enter retry loop:
   a. Execute handler + postconditions
   b. On success: break loop, return result
   c. On failure:
      - Check if should retry (error filtering + attempt count)
      - If yes: emit RETRYING event, wait with backoff, continue loop
      - If no: break loop, return failure
5. Check interrupt during retry delays (responsive cancellation)
6. Respect total retry timeout (cumulative across all attempts)
```

### Event Emission Strategy

**Flow with retries:**
```
STARTED (attempt=0) 
  → FAILED (attempt=0, will retry) 
    → RETRYING (delay=2.0s, next_attempt=1)
      → STARTED (attempt=1, is_retry=True) 
        → FAILED (attempt=1, will retry)
          → RETRYING (delay=4.0s, next_attempt=2)
            → STARTED (attempt=2, is_retry=True)
              → COMPLETED (attempt=2, is_retry=True)
```

**New event details:**
- `RETRYING` event includes: attempt number, delay, reason, next attempt
- `STARTED` events after first attempt: marked with `is_retry=True` and `retry_attempt`
- `FAILED` events: include whether retry will happen
- Final `COMPLETED`/`FAILED`: includes total attempts

### Configuration Precedence

**Simple override (most specific wins):**

```python
effective_config = (
    event.retry_config 
    OR state.retry_config_by_actor_type[actor_type]
    OR state.default_retry_config
    OR DEFAULT_RETRY_CONFIG_BY_ACTOR_TYPE[actor_type]
)
```

No merging - each level provides a complete config or defers to next level.

### State Persistence & Replay

**Behavior:**
- Retry state (attempt count, errors) IS persisted in snapshots
- On replay: Events show retry history but DON'T re-execute retry logic
- Replay is fast (no backoff waits)
- If event failed after all retries, it stays failed on replay
- If event succeeded after retries, it stays succeeded

### Retry Scope

**What gets retried:**
1. ✅ Handler execution
2. ✅ Postconditions
3. ✅ Artifact storage
4. ❌ Preconditions (run once, no retry)

**Rationale:** Preconditions are typically validations that shouldn't change between retries. If they fail, the event should fail immediately.

### Interrupt & Timeout Handling

**Interrupt checks:**
- Before retry delay: Check if interrupted
- During retry delay: Use interruptible sleep (check every 100ms or on interrupt signal)
- After retry delay: Check if interrupted before next attempt

**Timeout handling:**
- Each attempt respects per-event timeout (if set via `execution_state.set_timeout()`)
- All retries share `retry_timeout` from RetryConfig (cumulative)
- If retry_timeout exceeded, stop retrying even if max_attempts not reached

```python
# Example timing:
retry_config = RetryConfig(
    max_attempts=5,
    retry_timeout=30.0  # Total time limit for all retries
)

# If attempts take: 3s, 3s, 3s, 25s
# Fourth attempt won't start (would exceed 30s total)
```

---

## Printer & Display

### Console Output (EventPrinter)

**Dual mode strategy:**

**Success after retries (compact):**
```
🔧 fetch_data [STARTED]
✔ fetch_data [COMPLETED after 3 attempts] (5.2s total)
  ↻ Recovered after 2 retries
```

**Failure with retries (detailed):**
```
🔧 fetch_data [STARTED]
✖ fetch_data [FAILED] (0.5s) - Connection timeout
  ↻ Retrying (attempt 2/3) in 2.0s...
🔧 fetch_data [STARTED] (retry 2/3)
✖ fetch_data [FAILED] (0.5s) - Connection timeout
  ↻ Retrying (attempt 3/3) in 4.0s...
🔧 fetch_data [STARTED] (retry 3/3)
✖ fetch_data [FAILED] (0.5s) - Connection timeout
✖ fetch_data [EXHAUSTED all 3 attempts] (7.0s total)
```

**Key visual elements:**
- Use ↻ symbol for retry indication
- Indent retry messages under failed events
- Show attempt counter: `(retry 2/3)`
- Show delay: `in 2.0s...`
- Show total time for completed retries

### Summary Statistics

Add to `EventPrinter.print_summary()`:

```
📊 RETRY STATISTICS
├─ Total events with retries: 5
├─ Succeeded after retry: 4 (80%)
├─ Failed after all retries: 1 (20%)
├─ Total retry attempts: 12
└─ Avg retries per event: 2.4
```

### SSE Streaming (EventStreamer)

Include retry events in SSE stream:
- Emit `RETRYING` events as separate SSE messages
- Include retry metadata in all events
- Clients can filter/display as needed

---



## Usage Examples

### Example 1: Global default retry for all events
```python
execution_state = ExecutionState(
    default_retry_config=RetryConfig(
        max_attempts=3,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        initial_delay=1.0,
        max_delay=30.0
    )
)
```

### Example 2: Actor-type specific retry
```python
execution_state = ExecutionState(
    retry_config_by_actor_type={
        EventType.TOOL_CALL: RetryConfig(
            max_attempts=5,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            retry_on_exceptions=[TimeoutError, ConnectionError]
        ),
        EventType.LLM_CALL: RetryConfig(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            exclude_exceptions=[ValidationError]
        )
    }
)
```

### Example 3: Per-event retry configuration
```python
@tool(
    retry_config=RetryConfig(
        max_attempts=5,
        initial_delay=2.0,
        backoff_multiplier=3.0
    )
)
async def flaky_tool(input_data, context, state):
    # This tool will retry up to 5 times with aggressive backoff
    pass
```

### Example 4: Disable retry for specific event
```python
@agent(retry_config=RetryConfig(enabled=False))
async def no_retry_agent(input_data, context, state):
    # This agent will never retry (override default)
    pass
```

### Example 5: Custom error filtering
```python
from requests.exceptions import HTTPError

@tool(
    retry_config=RetryConfig(
        max_attempts=5,
        retry_on_exceptions=[HTTPError],  # Only retry HTTP errors
        exclude_exceptions=[HTTPError],  # But not if response is available
        retry_timeout=60.0  # Max 60s total for all retries
    )
)
async def api_call(input_data, context, state):
    # Custom retry logic fo+r API calls
    pass
```

---

## Implementation Phases

### Phase 1: Core Retry Infrastructure ✅
**Goal**: Basic retry mechanism with essential features  
**Estimated Time**: 1-2 days

**Tasks**:
1. Add `BackoffStrategy` enum to `types.py`
2. Add DEFAULT exception lists to `types.py` or new `retry.py` module
3. Create `RetryConfig` Pydantic model in `core/retry.py`
4. Add retry fields to `ExecutionEvent` in `core/events.py`
5. Add `RETRYING` status to `ExecutionStatus` enum
6. Implement retry loop in `ExecutionEvent.__call__()`:
   - Get effective retry config
   - Wrap handler + postconditions in retry loop
   - Implement backoff calculation
   - Implement error filtering (`should_retry()`)
   - Emit RETRYING events
   - Check interrupts during delays
   - Handle retry_timeout
7. Create unit tests:
   - `test_retry_config.py`: Config validation, backoff calculations
   - `test_retry_logic.py`: Retry loop with mock failures
   - `test_retry_error_filtering.py`: Whitelist/blacklist logic

**Deliverables**:
- ✅ `RetryConfig` model with backoff and error filtering
- ✅ Retry loop in `ExecutionEvent`
- ✅ RETRYING event emission
- ✅ Basic unit tests passing
- ✅ Can manually test with simple retry scenario

**Acceptance Criteria**:
- Event retries on transient errors up to max_attempts
- Exponential backoff works correctly
- Error filtering respects whitelist/blacklist
- RETRYING events are emitted
- Non-retryable errors fail immediately

---

### Phase 2: Configuration & Precedence ✅
**Goal**: Multi-level retry configuration  
**Estimated Time**: 0.5-1 day

**Tasks**:
1. Add retry fields to `ExecutionState`:
   - `default_retry_config`
   - `retry_config_by_actor_type`
   - `get_effective_retry_config()` method
2. Add `DEFAULT_RETRY_CONFIG_BY_ACTOR_TYPE` constant
3. Update decorators (`@agent`, `@tool`) to accept `retry_config` parameter
4. Update `BaseActor` model to include `retry_config` field
5. Create integration tests:
   - `test_retry_precedence.py`: Test all precedence levels
   - `test_retry_defaults.py`: Test default configs by actor type
6. Update documentation with configuration examples

**Deliverables**:
- ✅ Configuration precedence working (event → actor-type → global → default)
- ✅ Decorators support retry_config parameter
- ✅ Integration tests for all config levels
- ✅ Documentation updated

**Acceptance Criteria**:
- Can set retry config at global, actor-type, and event level
- Most specific config wins (no merging)
- Defaults work as expected (tools/LLMs retry, agents don't)
- Can disable retry by passing `RetryConfig(enabled=False)`

---

### Phase 3: Printer & Display ✅
**Goal**: Beautiful retry visualization  
**Estimated Time**: 0.5-1 day

**Tasks**:
1. Update `EventPrinter.print_event()` to handle RETRYING events
2. Implement dual-mode display:
   - Compact for successful retries
   - Detailed for failed retries
3. Add retry fields to event display:
   - Show `(retry 2/3)` for retry attempts
   - Show `↻ Retrying in 2.0s...` for RETRYING events
   - Show total attempts for final events
4. Add retry statistics to `EventPrinter.print_summary()`:
   - Total events with retries
   - Success/fail rates after retry
   - Average retries per event
5. Update `ParallelEventPrinter` to handle retry events
6. Update `EventStreamer` for SSE to include retry events
7. Create visual test examples

**Deliverables**:
- ✅ Beautiful console output for retries
- ✅ Retry stats in summary
- ✅ SSE streams include retry events
- ✅ Example screenshots/output

**Acceptance Criteria**:
- RETRYING events display with ↻ symbol and delay
- Retry attempts show as `(retry N/M)`
- Successful retries show compact summary
- Failed retries show detailed breakdown
- Summary includes retry statistics

---

### Phase 4: State Persistence & Replay ✅
**Goal**: Handle retry state in save/restore  
**Estimated Time**: 1 day

**Tasks**:
1. Update `ExecutionEvent` serialization to include retry state
2. Update `to_snapshot()` to persist retry state
3. Update `from_snapshot()` to restore retry state
4. Implement replay behavior:
   - Skip retry logic during replay
   - Preserve retry history in replayed events
   - Mark replayed retry events appropriately
5. Create persistence tests:
   - `test_retry_persistence.py`: Save/restore with retries
   - `test_retry_replay.py`: Replay behavior
   - `test_retry_validation_mode.py`: Validation mode with retries
6. Update documentation for replay behavior

**Deliverables**:
- ✅ Retry state survives save/restore
- ✅ Replay skips retry logic (fast, deterministic)
- ✅ Replay preserves retry history
- ✅ Tests demonstrate correct behavior

**Acceptance Criteria**:
- Saved state includes retry attempt counts and errors
- Restored state shows retry history
- Replay doesn't re-execute retry logic
- Replay doesn't wait for backoff delays
- Failed-after-retries stays failed on replay

---

### Phase 5: Interrupt & Timeout Integration ✅
**Goal**: Retries work correctly with interrupts  
**Estimated Time**: 1 day

**Tasks**:
1. Implement interrupt checking during retry delays:
   - Check before delay
   - Check during delay (interruptible sleep)
   - Check after delay
2. Implement retry_timeout tracking:
   - Track cumulative retry time
   - Stop retrying if retry_timeout exceeded
   - Emit appropriate events on timeout
3. Handle interaction with per-event timeout (from `set_timeout()`)
4. Create interrupt tests:
   - `test_retry_interrupt.py`: Interrupt during retry
   - `test_retry_timeout.py`: Timeout during retry sequence
5. Update documentation for timeout behavior

**Deliverables**:
- ✅ Can interrupt during retry delays
- ✅ retry_timeout enforced correctly
- ✅ Per-attempt timeout still works
- ✅ Tests verify behavior

**Acceptance Criteria**:
- Interrupt during retry delay stops execution gracefully
- retry_timeout prevents infinite retries
- Both per-attempt and total timeouts work together
- Appropriate events emitted on interrupt/timeout

---

### Phase 6 (Future): Advanced Resilience 🔮
**Goal**: Production-grade resilience patterns  
**Estimated Time**: 2-3 days (future work)

**Potential Scope**:
- Implement retry budgets (rate limiting retries)
- Implement circuit breakers per actor
- Add metrics/monitoring hooks
- Custom backoff strategies (callable support)
- Retry scope configuration
- Advanced error filtering patterns

**Note**: Not part of initial implementation. Will be designed based on Phase 1-5 usage and feedback.

---

## Testing Strategy

### Unit Tests
- ✅ `RetryConfig` validation and defaults
- ✅ Backoff calculation with all strategies (constant, linear, exponential, fibonacci)
- ✅ Error filtering (whitelist/blacklist logic, smart defaults)
- ✅ Configuration precedence resolution
- ✅ Retry counter and state management

### Integration Tests
- ✅ Full retry sequence with mock failures
- ✅ Configuration at different levels (event, actor-type, global, default)
- ✅ Event emission during retries (STARTED, FAILED, RETRYING)
- ✅ State persistence with retries
- ✅ Replay with retry history

### End-to-End Tests
- ✅ Real workflow with transient failures
- ✅ Retry with actual tool/LLM calls (using test doubles)
- ✅ Interrupt during retry
- ✅ Timeout during retry sequence
- ✅ Parallel execution with retries

### Test Files Structure
```
tests/
├── test_retry_config.py          # Unit: Config model
├── test_retry_logic.py           # Unit: Retry loop
├── test_retry_error_filtering.py # Unit: Error filtering
├── test_retry_backoff.py         # Unit: Backoff strategies
├── test_retry_precedence.py      # Integration: Config levels
├── test_retry_defaults.py        # Integration: Default configs
├── test_retry_events.py          # Integration: Event emission
├── test_retry_printer.py         # Integration: Display
├── test_retry_persistence.py     # Integration: Save/restore
├── test_retry_replay.py          # Integration: Replay behavior
├── test_retry_interrupt.py       # Integration: Interrupt handling
├── test_retry_timeout.py         # Integration: Timeout handling
└── test_retry_e2e.py             # E2E: Full workflows
```

---

## Implementation Checklist

### Phase 1: Core ✅ 
- [ ] Create `core/retry.py` module
- [ ] Add `BackoffStrategy` enum
- [ ] Add DEFAULT exception lists
- [ ] Implement `RetryConfig` model with `should_retry()` and `calculate_delay()`
- [ ] Add retry fields to `ExecutionEvent`
- [ ] Add `RETRYING` to `ExecutionStatus`
- [ ] Implement retry loop in `ExecutionEvent.__call__()`
- [ ] Emit RETRYING events
- [ ] Implement interruptible sleep
- [ ] Write unit tests
- [ ] Manual testing

### Phase 2: Configuration ✅
- [ ] Add retry fields to `ExecutionState`
- [ ] Implement `get_effective_retry_config()`
- [ ] Add `DEFAULT_RETRY_CONFIG_BY_ACTOR_TYPE`
- [ ] Update decorators (`@agent`, `@tool`, `@llm`)
- [ ] Update `BaseActor` model
- [ ] Write integration tests
- [ ] Update documentation

### Phase 3: Display ✅
- [ ] Update `EventPrinter.print_event()` for RETRYING
- [ ] Implement dual-mode display logic
- [ ] Add retry statistics to summary
- [ ] Update `ParallelEventPrinter`
- [ ] Update `EventStreamer` (SSE)
- [ ] Visual testing
- [ ] Screenshots/examples

### Phase 4: Persistence ✅
- [ ] Update event serialization
- [ ] Update `to_snapshot()` / `from_snapshot()`
- [ ] Implement replay skip logic
- [ ] Handle incomplete retry sequences
- [ ] Write persistence tests
- [ ] Write replay tests
- [ ] Update documentation

### Phase 5: Interrupts ✅
- [ ] Implement interrupt checks in retry loop
- [ ] Implement `retry_timeout` tracking
- [ ] Handle per-event timeout interaction
- [ ] Write interrupt tests
- [ ] Write timeout tests
- [ ] Update documentation

---

## Open Questions & Future Considerations

1. **Idempotency Validation**: Should we add an `@idempotent` decorator to explicitly mark handlers that are safe to retry?

2. **Retry Metrics**: Should we expose retry metrics through a metrics interface (e.g., for Prometheus)?

3. **Cost Tracking**: For LLM calls, should each retry attempt be tracked separately in cost calculations?

4. **Parallel Execution**: 
   - Should parallel groups have a shared retry budget?
   - How do retries affect parallel group timing stats?

5. **Human Intervention**: Should `ExecutionStatus.HUMAN_INTERVENTION` disable/pause retries automatically?

6. **Exponential Backoff with Jitter**: Fine-tune jitter algorithm (current: ±25%, could use decorrelated jitter)

7. **Retry Headers**: For HTTP tools, should we automatically add retry metadata to headers (e.g., `X-Retry-Attempt: 2`)?

8. **Custom Backoff**: Future phase - allow users to provide custom backoff callables?

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Retry storms (too many retries) | Medium | High | Implement conservative defaults, add retry_timeout |
| Non-idempotent operations retried | Medium | High | Document best practices, error filtering, future: @idempotent decorator |
| Long retry delays block execution | Low | Medium | Interruptible delays, clear timeout handling |
| Complex interaction with replay | Medium | Medium | Thorough testing, clear documentation |
| Performance overhead | Low | Low | Minimal overhead (only on failures), config precedence is O(1) |

---

## Success Criteria

✅ **Phase 1 Complete When**:
- Tool/LLM calls automatically retry on transient errors
- Configurable backoff strategies work correctly
- Error filtering prevents retry of permanent errors
- Tests pass

✅ **Phase 2 Complete When**:
- Can configure retry at all levels (global, actor-type, event)
- Precedence rules are clear and work correctly
- Defaults provide good out-of-box experience
- Documentation is clear

✅ **Phase 3 Complete When**:
- Console output beautifully shows retry progress
- Summary includes retry statistics
- SSE streams include retry events
- Visual examples demonstrate functionality

✅ **Phase 4 Complete When**:
- State save/restore preserves retry history
- Replay is fast and deterministic
- Retry state doesn't cause replay issues
- Tests validate all scenarios

✅ **Phase 5 Complete When**:
- Can interrupt during retry delays
- Timeouts work correctly (per-attempt and total)
- No deadlocks or blocking
- Tests validate interrupt/timeout behavior

---

**Document Version**: 2.0 (Consolidated)  
**Status**: ✅ Approved - Implementation Ready  
**Next Action**: Begin Phase 1 Implementation
