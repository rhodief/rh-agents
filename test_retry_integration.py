"""
Integration test for retry loop in ExecutionEvent.

Tests:
- Full retry sequence with mock failures
- Successful retry after failures
- Maximum attempts exhaustion
- Error filtering
- RETRYING event emission
"""
import pytest
import asyncio
import time
from pydantic import BaseModel
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.actors import BaseActor
from rh_agents.core.execution import ExecutionState
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.retry import RetryConfig, BackoffStrategy


# Simple input model for testing
class SimpleInput(BaseModel):
    value: int = 0


class TestRetryLoopBasics:
    """Test basic retry loop functionality."""
    
    @pytest.mark.asyncio
    async def test_success_on_first_attempt_no_retry(self):
        """If handler succeeds on first attempt, no retry occurs."""
        # Setup
        state = ExecutionState()
        events_received = []
        
        def track_event(event):
            events_received.append(event)
        
        state.event_bus.subscribe(track_event)
        
        handler_calls = []
        
        async def successful_handler(input_data, context, state):
            handler_calls.append(1)
            return "success"
        
        actor = BaseActor(
            name="test_actor",
            description="Test actor",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=successful_handler
        )
        
        event = ExecutionEvent(
            actor=actor,
            retry_config=RetryConfig(max_attempts=3)
        )
        
        # Execute
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Verify
        assert result.ok is True
        assert result.result == "success"
        assert len(handler_calls) == 1  # Called only once
        
        # Check events
        started_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.STARTED]
        completed_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.COMPLETED]
        retrying_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.RETRYING]
        
        assert len(started_events) == 1
        assert len(completed_events) == 1
        assert len(retrying_events) == 0  # No retries
    
    @pytest.mark.asyncio
    async def test_retry_after_transient_failure(self):
        """Handler fails with retryable error, then succeeds on retry."""
        # Setup
        state = ExecutionState()
        events_received = []
        
        def track_event(event):
            events_received.append(event)
        
        state.event_bus.subscribe(track_event)
        
        attempt_count = [0]
        
        async def flaky_handler(input_data, context, state):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                raise ConnectionError("Transient network error")
            return "success_after_retry"
        
        actor = BaseActor(
            name="flaky_actor",
            description="Flaky test actor",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=flaky_handler
        )
        
        event = ExecutionEvent(
            actor=actor,
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=0.01,  # Fast retry for testing
                jitter=False
            )
        )
        
        # Execute
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Verify
        assert result.ok is True
        assert result.result == "success_after_retry"
        assert attempt_count[0] == 2  # Failed once, succeeded on retry
        
        # Check events
        started_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.STARTED]
        failed_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.FAILED]
        retrying_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.RETRYING]
        completed_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.COMPLETED]
        
        assert len(started_events) == 2  # Initial + 1 retry
        assert len(retrying_events) == 1  # One RETRYING event
        assert len(completed_events) == 1  # Final success
    
    @pytest.mark.asyncio
    async def test_exhaust_all_retries(self):
        """All retries exhausted, return failure."""
        # Setup
        state = ExecutionState()
        events_received = []
        
        def track_event(event):
            events_received.append(event)
        
        state.event_bus.subscribe(track_event)
        
        attempt_count = [0]
        
        async def always_failing_handler(input_data, context, state):
            attempt_count[0] += 1
            raise ConnectionError(f"Attempt {attempt_count[0]} failed")
        
        actor = BaseActor(
            name="failing_actor",
            description="Always failing test actor",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=always_failing_handler
        )
        
        event = ExecutionEvent(
            actor=actor,
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=0.01,
                jitter=False
            )
        )
        
        # Execute
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Verify
        assert result.ok is False
        assert attempt_count[0] == 3  # All 3 attempts used
        
        # Check events
        started_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.STARTED]
        retrying_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.RETRYING]
        failed_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.FAILED]
        
        assert len(started_events) == 3  # All 3 attempts
        assert len(retrying_events) == 2  # 2 RETRYING events (after attempts 1 and 2)
        # Note: Each failed attempt emits a FAILED event before retry (if retry is possible)
        # The final failed attempt also emits FAILED
        # So we can have multiple FAILED events - one per attempt that actually failed
    
    @pytest.mark.asyncio
    async def test_non_retryable_error_no_retry(self):
        """Errors can be excluded from retry using exclude_exceptions."""
        # Setup
        state = ExecutionState()
        events_received = []
        
        def track_event(event):
            events_received.append(event)
        
        state.event_bus.subscribe(track_event)
        
        attempt_count = [0]
        
        async def validation_handler(input_data, context, state):
            attempt_count[0] += 1
            raise ValueError("Invalid input - non-retryable")
        
        actor = BaseActor(
            name="validation_actor",
            description="Validation error test actor",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=validation_handler
        )
        
        event = ExecutionEvent(
            actor=actor,
            retry_config=RetryConfig(
                max_attempts=3,
                exclude_exceptions=[ValueError]  # Opt-out ValueError from retry
            )
        )
        
        # Execute
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Verify
        assert result.ok is False
        assert attempt_count[0] == 1  # Only one attempt
        
        # Check events
        started_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.STARTED]
        retrying_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.RETRYING]
        failed_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.FAILED]
        
        assert len(started_events) == 1
        assert len(retrying_events) == 0  # No retries
        assert len(failed_events) == 1


class TestRetryEventEmission:
    """Test that retry events are emitted correctly."""
    
    @pytest.mark.asyncio
    async def test_retrying_event_details(self):
        """RETRYING event should include retry details."""
        # Setup
        state = ExecutionState()
        events_received = []
        
        def track_event(event):
            events_received.append(event)
        
        state.event_bus.subscribe(track_event)
        
        attempt_count = [0]
        
        async def fail_then_succeed(input_data, context, state):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                raise TimeoutError("First attempt timeout")
            return "success"
        
        actor = BaseActor(
            name="test_actor",
            description="Test actor for event emission",
            input_model=SimpleInput,
            event_type=EventType.LLM_CALL,
            handler=fail_then_succeed
        )
        
        event = ExecutionEvent(
            actor=actor,
            retry_config=RetryConfig(
                max_attempts=2,
                initial_delay=0.01,
                backoff_strategy=BackoffStrategy.CONSTANT,
                jitter=False
            )
        )
        
        # Execute
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Verify
        assert result.ok is True
        
        # Check RETRYING event
        retrying_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.RETRYING]
        assert len(retrying_events) == 1
        
        retry_event = retrying_events[0]
        # retry_attempt==1 means we're about to start the second attempt (first retry)
        assert retry_event.retry_attempt == 1  # Next attempt will be attempt 1 (retry)
        assert retry_event.original_error is not None
        assert "timeout" in str(retry_event.original_error).lower()
    
    @pytest.mark.asyncio
    async def test_retry_attempt_tracking(self):
        """Retry attempt number should be tracked correctly."""
        # Setup
        state = ExecutionState()
        events_received = []
        
        def track_event(event):
            events_received.append(event)
        
        state.event_bus.subscribe(track_event)
        
        attempt_count = [0]
        
        async def fail_twice_then_succeed(input_data, context, state):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise TimeoutError(f"Attempt {attempt_count[0]} timeout")
            return "success"
        
        actor = BaseActor(
            name="test_actor",
            description="Test actor for attempt tracking",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=fail_twice_then_succeed
        )
        
        event = ExecutionEvent(
            actor=actor,
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=0.01,
                jitter=False
            )
        )
        
        # Execute
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Verify
        assert result.ok is True
        assert attempt_count[0] == 3
        
        # Check attempt numbers in STARTED events
        started_events = [e for e in events_received if hasattr(e, 'execution_status') and e.execution_status == ExecutionStatus.STARTED]
        assert len(started_events) == 3
        assert started_events[0].retry_attempt == 0
        assert started_events[1].retry_attempt == 1
        assert started_events[2].retry_attempt == 2


class TestRetryBackoff:
    """Test backoff delay calculation and application."""
    
    @pytest.mark.asyncio
    async def test_backoff_delay_applied(self):
        """Verify that backoff delay is actually waited."""
        # Setup
        state = ExecutionState()
        
        attempt_times = []
        
        async def track_timing_handler(input_data, context, state):
            attempt_times.append(time.time())
            if len(attempt_times) < 2:
                raise TimeoutError("Forced failure")
            return "success"
        
        actor = BaseActor(
            name="timing_actor",
            description="Test actor for timing verification",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=track_timing_handler
        )
        
        event = ExecutionEvent(
            actor=actor,
            retry_config=RetryConfig(
                max_attempts=2,
                initial_delay=0.1,  # 100ms delay
                jitter=False
            )
        )
        
        # Execute
        start_time = time.time()
        result = await event(input_data="test", extra_context={}, execution_state=state)
        end_time = time.time()
        
        # Verify
        assert result.ok is True
        assert len(attempt_times) == 2
        
        # Check that delay was applied (at least 100ms between attempts)
        time_between_attempts = attempt_times[1] - attempt_times[0]
        assert time_between_attempts >= 0.1  # At least 100ms
        
        # Total execution should be at least 100ms (for the delay)
        total_time = end_time - start_time
        assert total_time >= 0.1


class TestRetryTimeout:
    """Test retry timeout functionality."""
    
    @pytest.mark.asyncio
    async def test_retry_timeout_stops_retries(self):
        """Retry timeout prevents more retries even if attempts remain."""
        # Setup
        state = ExecutionState()
        
        attempt_count = [0]
        
        async def slow_failing_handler(input_data, context, state):
            attempt_count[0] += 1
            await asyncio.sleep(0.2)  # Each attempt takes 200ms
            raise TimeoutError(f"Attempt {attempt_count[0]} timeout")
        
        actor = BaseActor(
            name="slow_actor",
            description="Slow actor for timeout testing",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=slow_failing_handler
        )
        
        event = ExecutionEvent(
            actor=actor,
            retry_config=RetryConfig(
                max_attempts=10,  # Plenty of attempts
                initial_delay=0.01,
                retry_timeout=0.5,  # But only 500ms total
                jitter=False
            )
        )
        
        # Execute
        result = await event(input_data="test", extra_context={}, execution_state=state)
        
        # Verify - should stop before all 10 attempts due to timeout
        assert result.ok is False
        assert attempt_count[0] < 10  # Didn't use all attempts
        assert attempt_count[0] >= 1  # At least one attempt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
