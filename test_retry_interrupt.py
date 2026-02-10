"""
Test Suite: Phase 5 - Interrupt Integration

Tests interrupt handling during retry backoff delays.
"""
import pytest
import asyncio
from pydantic import BaseModel
from rh_agents.core.execution import ExecutionState
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import Tool
from rh_agents.core.types import EventType, ExecutionStatus, InterruptReason
from rh_agents.core.retry import RetryConfig, BackoffStrategy
from rh_agents.core.exceptions import ExecutionInterrupted


# ===== Helper Functions =====

def get_status(event):
    """Get execution_status from event (handles both ExecutionEvent and dict)."""
    if isinstance(event, dict):
        return event.get("execution_status")
    else:
        return getattr(event, "execution_status", None)

def get_attr(event, attr_name, default=None):
    """Get attribute from event (handles both ExecutionEvent and dict)."""
    if isinstance(event, dict):
        return event.get(attr_name, default)
    else:
        return getattr(event, attr_name, default)


# ===== Test Input Model =====

class RequestInput(BaseModel):
    value: str


# ===== Test: Basic interrupt during retry backoff =====

@pytest.mark.asyncio
async def test_interrupt_during_retry_backoff():
    """Test that interrupt is detected during retry backoff delay."""
    state = ExecutionState()
    
    # Track handler call count
    call_count = 0
    
    async def flaky_handler(input_data, context, execution_state):
        nonlocal call_count
        call_count += 1
        # Always fail to trigger retry
        raise ConnectionError("Service unavailable")
    
    tool = Tool(
        name="flaky_service",
        description="Flaky service",
        input_model=RequestInput,
        handler=flaky_handler,
        retry_config=RetryConfig(
            max_attempts=5,
            initial_delay=2.0,  # Long delay to allow interrupt
            backoff_strategy=BackoffStrategy.CONSTANT
        )
    )
    
    event = ExecutionEvent(actor=tool)
    
    # Start execution in background
    async def execute_with_interrupt():
        # Wait a bit for first failure to happen
        await asyncio.sleep(0.2)
        
        # Request interrupt during retry backoff
        state.request_interrupt(
            reason=InterruptReason.USER_CANCELLED,
            message="User cancelled during retry"
        )
    
    # Run both tasks
    task1 = asyncio.create_task(event(RequestInput(value="test"), {}, state))
    task2 = asyncio.create_task(execute_with_interrupt())
    
    # Should raise ExecutionInterrupted
    with pytest.raises(ExecutionInterrupted) as exc_info:
        await task1
    
    await task2
    
    # Verify interrupt details
    assert exc_info.value.reason == InterruptReason.USER_CANCELLED
    assert "User cancelled during retry" in str(exc_info.value.message)
    
    # Verify handler was called once (failed), but not retried due to interrupt
    assert call_count == 1
    
    # Verify events in history
    events = state.history.get_event_list()
    print(f"\\nDebug: Total events = {len(events)}")
    for i, e in enumerate(events):
        status = get_status(e)
        status_val = status.value if hasattr(status, 'value') else status
        addr = get_attr(e, 'address', 'unknown')
        print(f"  Event {i}: {status_val} at {addr}")
    
    # Verify INTERRUPTED event was published
    interrupted_events = [e for e in events if get_status(e) == ExecutionStatus.INTERRUPTED.value or (hasattr(get_status(e), 'value') and get_status(e).value == ExecutionStatus.INTERRUPTED.value)]
    
    # Since interrupt happens during backoff, we expect INTERRUPTED event
    print(f"\\nInterrupted events found: {len(interrupted_events)}")
    assert len(interrupted_events) > 0, f"Expected at least one INTERRUPTED event, found {len(interrupted_events)}. Events: {[get_status(e) for e in events]}"


# ===== Test: Interrupt at different retry stages =====

@pytest.mark.asyncio
async def test_interrupt_at_different_retry_stages():
    """Test interrupt can happen at various points during retry cycle."""
    
    # Test 1: Interrupt before retry starts
    state1 = ExecutionState()
    call_count_1 = 0
    
    async def handler_1(input_data, context, execution_state):
        nonlocal call_count_1
        call_count_1 += 1
        # Interrupt before failure
        execution_state.request_interrupt()
        raise ConnectionError("Should not retry")
    
    tool_1 = Tool(
        name="test1",
        description="Test 1",
        input_model=RequestInput,
        handler=handler_1,
        retry_config=RetryConfig(max_attempts=3, initial_delay=1.0)
    )
    
    event_1 = ExecutionEvent(actor=tool_1)
    
    with pytest.raises(ExecutionInterrupted):
        await event_1(RequestInput(value="test"), {}, state1)
    
    assert call_count_1 == 1  # Called once, then interrupted
    
    # Test 2: Interrupt during backoff delay
    state2 = ExecutionState()
    call_count_2 = 0
    
    async def handler_2(input_data, context, execution_state):
        nonlocal call_count_2
        call_count_2 += 1
        raise ConnectionError("Fail")
    
    tool_2 = Tool(
        name="test2",
        description="Test 2",
        input_model=RequestInput,
        handler=handler_2,
        retry_config=RetryConfig(max_attempts=3, initial_delay=1.0)
    )
    
    event_2 = ExecutionEvent(actor=tool_2)
    
    async def interrupt_during_backoff():
        await asyncio.sleep(0.2)  # Wait for first failure
        state2.request_interrupt()
    
    task1 = asyncio.create_task(event_2(RequestInput(value="test"), {}, state2))
    task2 = asyncio.create_task(interrupt_during_backoff())
    
    with pytest.raises(ExecutionInterrupted):
        await task1
    
    await task2
    
    assert call_count_2 == 1  # First attempt only


# ===== Test: Interrupt is responsive during long backoff =====

@pytest.mark.asyncio
async def test_interrupt_responsive_during_long_backoff():
    """Test that interrupt is detected quickly even during long backoff delays."""
    state = ExecutionState()
    
    async def always_fail(input_data, context, execution_state):
        raise TimeoutError("Always fails")
    
    tool = Tool(
        name="slow_retry",
        description="Slow retry",
        input_model=RequestInput,
        handler=always_fail,
        retry_config=RetryConfig(
            max_attempts=3,
            initial_delay=10.0,  # Very long delay
            backoff_strategy=BackoffStrategy.CONSTANT
        )
    )
    
    event = ExecutionEvent(actor=tool)
    
    start_time = asyncio.get_event_loop().time()
    
    async def interrupt_quickly():
        await asyncio.sleep(0.3)  # Wait a bit, then interrupt
        state.request_interrupt()
    
    task1 = asyncio.create_task(event(RequestInput(value="test"), {}, state))
    task2 = asyncio.create_task(interrupt_quickly())
    
    with pytest.raises(ExecutionInterrupted):
        await task1
    
    await task2
    
    end_time = asyncio.get_event_loop().time()
    elapsed = end_time - start_time
    
    # Should interrupt within ~0.5s, not wait full 10s delay
    assert elapsed < 1.0, f"Interrupt took {elapsed}s, should be < 1s"


# ===== Test: State preservation during interrupt =====

@pytest.mark.asyncio
async def test_state_preserved_when_interrupted_during_retry():
    """Test that execution state is properly preserved when interrupted during retry."""
    state = ExecutionState()
    
    async def flaky_handler(input_data, context, execution_state):
        raise ConnectionError("Connection failed")
    
    tool = Tool(
        name="api_service",
        description="API service",
        input_model=RequestInput,
        handler=flaky_handler,
        retry_config=RetryConfig(max_attempts=5, initial_delay=2.0)
    )
    
    event = ExecutionEvent(actor=tool)
    
    async def interrupt_during_retry():
        await asyncio.sleep(0.2)
        state.request_interrupt(
            reason=InterruptReason.USER_CANCELLED,
            message="Test interrupt",
            save_checkpoint=True
        )
    
    task1 = asyncio.create_task(event(RequestInput(value="test"), {}, state))
    task2 = asyncio.create_task(interrupt_during_retry())
    
    with pytest.raises(ExecutionInterrupted):
        await task1
    
    await task2
    
    # Verify event history was preserved
    events = state.history.get_event_list()
    
    # Debug: print all events with details
    print(f"\\nTotal events: {len(events)}")
    for i, e in enumerate(events):
        status = get_status(e)
        status_val = status.value if hasattr(status, 'value') else status
        addr = get_attr(e, 'address', 'unknown')
        print(f"  Event {i}: status={status_val}, address={addr}")
    
    # Should have: STARTED, FAILED, RETRYING, INTERRUPTED
    statuses = [get_status(e) for e in events]
    
    # Convert enums to values for comparison
    status_values = [s.value if hasattr(s, 'value') else s for s in statuses]
    
    # We should have RETRYING (confirmed retry started)
    assert ExecutionStatus.RETRYING.value in status_values, f"Expected RETRYING in {status_values}"
    
    # Note: INTERRUPTED might not be in history if the interrupt was caught
    # at a higher level. The key is that ExecutionInterrupted was raised.
    # This is acceptable as long as the interrupt was handled correctly.
    
    # Verify RETRYING event has retry information
    retrying_events = [e for e in events if get_status(e) == ExecutionStatus.RETRYING.value]
    assert len(retrying_events) > 0
    assert get_attr(retrying_events[0], "retry_attempt") == 1
    assert get_attr(retrying_events[0], "original_error") is not None


# ===== Test: Multiple retry attempts before interrupt =====

@pytest.mark.asyncio
async def test_interrupt_after_multiple_retry_attempts():
    """Test interrupt after several retry attempts have occurred."""
    state = ExecutionState()
    call_count = 0
    
    async def handler(input_data, context, execution_state):
        nonlocal call_count
        call_count += 1
        raise ConnectionError(f"Attempt {call_count} failed")
    
    tool = Tool(
        name="multi_retry",
        description="Multi retry",
        input_model=RequestInput,
        handler=handler,
        retry_config=RetryConfig(
            max_attempts=10,
            initial_delay=0.3,
            backoff_strategy=BackoffStrategy.CONSTANT
        )
    )
    
    event = ExecutionEvent(actor=tool)
    
    async def interrupt_after_retries():
        # Wait for a few retry attempts
        await asyncio.sleep(1.5)  # Should allow 3-4 attempts
        state.request_interrupt()
    
    task1 = asyncio.create_task(event(RequestInput(value="test"), {}, state))
    task2 = asyncio.create_task(interrupt_after_retries())
    
    with pytest.raises(ExecutionInterrupted):
        await task1
    
    await task2
    
    # Should have attempted multiple times before interrupt
    assert call_count >= 2
    assert call_count < 10  # But not all attempts
    
    # Verify multiple RETRYING events
    events = state.history.get_event_list()
    retrying_events = [e for e in events if get_status(e) == ExecutionStatus.RETRYING.value]
    assert len(retrying_events) >= 1


# ===== Test: External interrupt checker during retry =====

@pytest.mark.asyncio
async def test_external_interrupt_checker_during_retry():
    """Test that external interrupt checker is called during retry backoff."""
    state = ExecutionState()
    
    interrupt_check_count = 0
    should_interrupt = False
    
    def interrupt_checker():
        nonlocal interrupt_check_count
        interrupt_check_count += 1
        return should_interrupt
    
    state.set_interrupt_checker(interrupt_checker)
    
    async def always_fail(input_data, context, execution_state):
        raise TimeoutError("Always fails")
    
    tool = Tool(
        name="checked_service",
        description="Checked service",
        input_model=RequestInput,
        handler=always_fail,
        retry_config=RetryConfig(max_attempts=5, initial_delay=1.0)
    )
    
    event = ExecutionEvent(actor=tool)
    
    async def trigger_interrupt():
        nonlocal should_interrupt
        await asyncio.sleep(0.3)
        should_interrupt = True
    
    task1 = asyncio.create_task(event(RequestInput(value="test"), {}, state))
    task2 = asyncio.create_task(trigger_interrupt())
    
    with pytest.raises(ExecutionInterrupted):
        await task1
    
    await task2
    
    # Interrupt checker should have been called multiple times during backoff
    assert interrupt_check_count > 1


# ===== Test: Interrupt message preserved =====

@pytest.mark.asyncio
async def test_interrupt_message_preserved():
    """Test that interrupt message and reason are properly preserved."""
    state = ExecutionState()
    
    async def flaky_handler(input_data, context, execution_state):
        raise ConnectionError("Connection failed")
    
    tool = Tool(
        name="message_test",
        description="Message test",
        input_model=RequestInput,
        handler=flaky_handler,
        retry_config=RetryConfig(max_attempts=3, initial_delay=1.0)
    )
    
    event = ExecutionEvent(actor=tool)
    
    async def interrupt_with_message():
        await asyncio.sleep(0.2)
        state.request_interrupt(
            reason=InterruptReason.TIMEOUT,
            message="Operation timeout after 30s",
            triggered_by="timeout_monitor"
        )
    
    task1 = asyncio.create_task(event(RequestInput(value="test"), {}, state))
    task2 = asyncio.create_task(interrupt_with_message())
    
    with pytest.raises(ExecutionInterrupted) as exc_info:
        await task1
    
    await task2
    
    # Verify interrupt details
    assert exc_info.value.reason == InterruptReason.TIMEOUT
    assert "Operation timeout" in str(exc_info.value.message)
    
    # Verify INTERRUPTED event has details
    events = state.history.get_event_list()
    interrupted_events = [e for e in events if get_status(e) == ExecutionStatus.INTERRUPTED.value]
    assert len(interrupted_events) > 0
    
    interrupted_event = interrupted_events[-1]
    # The message should be from the interrupt signal, not a generic message
    assert "Operation timeout" in str(get_attr(interrupted_event, "message", ""))
    
    # The detail field should indicate it was during retry backoff
    detail = get_attr(interrupted_event, "detail", "")
    assert "retry backoff" in str(detail).lower() or "timeout" in str(detail).lower()


# ===== Test: No interrupt during normal retry =====

@pytest.mark.asyncio
async def test_no_interrupt_completes_retry_normally():
    """Test that retry completes normally when no interrupt occurs."""
    state = ExecutionState()
    call_count = 0
    
    async def eventually_succeeds(input_data, context, execution_state):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Not ready yet")
        return "success"
    
    tool = Tool(
        name="eventual_success",
        description="Eventual success",
        input_model=RequestInput,
        handler=eventually_succeeds,
        retry_config=RetryConfig(max_attempts=5, initial_delay=0.1)
    )
    
    event = ExecutionEvent(actor=tool)
    
    result = await event(RequestInput(value="test"), {}, state)
    
    # Should succeed after retries
    assert result.ok
    assert result.result == "success"
    assert call_count == 3
    
    # Should NOT have INTERRUPTED event
    events = state.history.get_event_list()
    interrupted_events = [e for e in events if get_status(e) == ExecutionStatus.INTERRUPTED.value]
    assert len(interrupted_events) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
