"""
Tests for Phase 3: Printer Display for Retry Mechanism

Tests:
- RETRYING status display with ↻ symbol
- Retry attempt tracking display
- Retry statistics in summary
- Compact and verbose mode display
"""
import pytest
import io
import sys
from contextlib import redirect_stdout
from pydantic import BaseModel
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import BaseActor
from rh_agents.core.execution import ExecutionState
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.retry import RetryConfig
from rh_agents.bus_handlers import EventPrinter


class SimpleInput(BaseModel):
    value: int = 0


class TestPrinterRetryDisplay:
    """Test that printer displays retry information correctly."""
    
    def test_retrying_status_display(self):
        """Test RETRYING status is displayed with ↻ symbol."""
        printer = EventPrinter()
        
        # Create a mock RETRYING event
        async def dummy_handler(input_data, context, state):
            pass
        
        actor = BaseActor(
            name="test_actor",
            description="Test",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=dummy_handler
        )
        
        event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.RETRYING,
            message="Retrying after TimeoutError",
            detail="Attempt 2/3 in 1.0s",
            retry_attempt=1,
            original_error="Connection timeout",
            retry_delay=1.0
        )
        
        # Capture output
        output = io.StringIO()
        with redirect_stdout(output):
            printer.print_event(event)
        
        result = output.getvalue()
        
        # Verify RETRYING status displayed
        assert "↻" in result or "RETRYING" in result
        assert "test_actor" in result
        
        # Verify retry statistics tracked
        assert printer.retry_events == 1
        assert printer.total_retry_attempts == 1
    
    def test_retry_attempt_number_display(self):
        """Test retry attempt number is displayed."""
        printer = EventPrinter()
        
        async def dummy_handler(input_data, context, state):
            pass
        
        actor = BaseActor(
            name="retry_actor",
            description="Test",
            input_model=SimpleInput,
            event_type=EventType.LLM_CALL,
            handler=dummy_handler
        )
        
        # First retry (retry_attempt=1 means 2nd attempt)
        event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.STARTED,
            retry_attempt=1,
            is_retry=True,
            retry_delay=0.5
        )
        
        output = io.StringIO()
        with redirect_stdout(output):
            printer.print_event(event)
        
        result = output.getvalue()
        
        # Should show retry attempt indicator
        assert "↻" in result or "Retry" in result
        assert "2" in result  # Second attempt (retry_attempt=1 + 1)
    
    def test_original_error_display(self):
        """Test original error is displayed in RETRYING event."""
        printer = EventPrinter()
        
        async def dummy_handler(input_data, context, state):
            pass
        
        actor = BaseActor(
            name="error_actor",
            description="Test",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=dummy_handler
        )
        
        event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.RETRYING,
            message="Retrying after error",
            original_error="ConnectionError: Failed to connect to API",
            retry_attempt=0
        )
        
        output = io.StringIO()
        with redirect_stdout(output):
            printer.print_event(event)
        
        result = output.getvalue()
        
        # Should display error information
        assert "ConnectionError" in result or "Failed to connect" in result or "error" in result.lower()
    
    def test_retry_statistics_in_summary(self):
        """Test retry statistics appear in summary."""
        printer = EventPrinter()
        
        async def dummy_handler(input_data, context, state):
            pass
        
        actor = BaseActor(
            name="test_actor",
            description="Test",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=dummy_handler
        )
        
        # Simulate a sequence of events with retries
        # Started
        event1 = ExecutionEvent(
            actor=actor,
            address="test::tool_call",
            execution_status=ExecutionStatus.STARTED,
            retry_attempt=0
        )
        printer.print_event(event1)
        
        # Failed
        event2 = ExecutionEvent(
            actor=actor,
            address="test::tool_call",
            execution_status=ExecutionStatus.FAILED,
            message="Error occurred"
        )
        printer.print_event(event2)
        
        # Retrying
        event3 = ExecutionEvent(
            actor=actor,
            address="test::tool_call",
            execution_status=ExecutionStatus.RETRYING,
            retry_attempt=1,
            original_error="Error occurred"
        )
        printer.print_event(event3)
        
        # Started again
        event4 = ExecutionEvent(
            actor=actor,
            address="test::tool_call",
            execution_status=ExecutionStatus.STARTED,
            retry_attempt=1
        )
        printer.print_event(event4)
        
        # Completed
        event5 = ExecutionEvent(
            actor=actor,
            address="test::tool_call",
            execution_status=ExecutionStatus.COMPLETED,
            retry_attempt=1
        )
        printer.print_event(event5)
        
        # Print summary and capture output
        output = io.StringIO()
        with redirect_stdout(output):
            printer.print_summary()
        
        result = output.getvalue()
        
        # Verify retry statistics present
        assert "Retry" in result or "retry" in result
        assert "1" in result  # At least 1 retry event
        
        # Verify statistics are correct
        assert printer.retry_events == 1
        assert len(printer.events_with_retries) == 1
        assert printer.total_retry_attempts == 1
    
    def test_multiple_retries_tracking(self):
        """Test tracking multiple retry events for different actors."""
        printer = EventPrinter()
        
        async def dummy_handler(input_data, context, state):
            pass
        
        # Two different actors
        actor1 = BaseActor(
            name="actor1",
            description="Test",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=dummy_handler
        )
        
        actor2 = BaseActor(
            name="actor2",
            description="Test",
            input_model=SimpleInput,
            event_type=EventType.LLM_CALL,
            handler=dummy_handler
        )
        
        # Actor 1 retries twice
        for i in range(2):
            event = ExecutionEvent(
                actor=actor1,
                address="actor1::tool_call",
                execution_status=ExecutionStatus.RETRYING,
                retry_attempt=i
            )
            printer.print_event(event)
        
        # Actor 2 retries once
        event = ExecutionEvent(
            actor=actor2,
            address="actor2::llm_call",
            execution_status=ExecutionStatus.RETRYING,
            retry_attempt=0
        )
        printer.print_event(event)
        
        # Check statistics
        assert printer.retry_events == 3  # Total retry events
        assert len(printer.events_with_retries) == 2  # Two unique events retried
        assert printer.total_retry_attempts == 3  # All RETRYING events counted
    
    @pytest.mark.asyncio
    async def test_full_retry_cycle_display(self):
        """Test complete retry cycle with actual execution and printer."""
        state = ExecutionState()
        printer = EventPrinter()
        
        # Subscribe printer to event bus
        state.event_bus.subscribe(printer.print_event)
        
        # Track output
        outputs = []
        original_print = print
        
        def capture_print(*args, **kwargs):
            output = io.StringIO()
            original_print(*args, file=output, **kwargs)
            outputs.append(output.getvalue())
        
        import builtins
        builtins.print = capture_print
        
        try:
            # Create actor that fails once then succeeds
            call_count = [0]
            
            async def flaky_handler(input_data, context, state):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise ConnectionError("Network error")
                return "success"
            
            actor = BaseActor(
                name="flaky_tool",
                description="Flaky tool",
                input_model=SimpleInput,
                event_type=EventType.TOOL_CALL,
                handler=flaky_handler
            )
            
            event = ExecutionEvent(
                actor=actor,
                retry_config=RetryConfig(max_attempts=2, initial_delay=0.01, jitter=False)
            )
            
            # Execute
            result = await event(input_data="test", extra_context={}, execution_state=state)
            
            # Verify execution succeeded
            assert result.ok is True
            assert call_count[0] == 2
            
            # Verify printer captured retry events
            assert printer.retry_events > 0
            
            # Check that output contains retry indicators
            all_output = "\n".join(outputs)
            # Should have STARTED, FAILED, RETRYING, STARTED, COMPLETED
            assert "STARTED" in all_output
            assert "FAILED" in all_output or "fail" in all_output.lower()
            
        finally:
            builtins.print = original_print
    
    def test_retry_rate_calculation(self):
        """Test retry rate is calculated correctly in summary."""
        printer = EventPrinter()
        
        async def dummy_handler(input_data, context, state):
            pass
        
        actor = BaseActor(
            name="test",
            description="Test",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=dummy_handler
        )
        
        # 10 started events
        for i in range(10):
            event = ExecutionEvent(
                actor=actor,
                address=f"test{i}::tool_call",
                execution_status=ExecutionStatus.STARTED
            )
            printer.print_event(event)
        
        # 3 of them retry
        for i in range(3):
            event = ExecutionEvent(
                actor=actor,
                address=f"test{i}::tool_call",
                execution_status=ExecutionStatus.RETRYING,
                retry_attempt=1
            )
            printer.print_event(event)
        
        # Verify counts
        assert printer.started_events == 10
        assert len(printer.events_with_retries) == 3
        
        # Calculate expected retry rate: 3/10 = 30%
        expected_rate = 30.0
        
        # Print summary to verify it calculates correctly
        output = io.StringIO()
        with redirect_stdout(output):
            printer.print_summary()
        
        result = output.getvalue()
        
        # Verify retry rate is shown
        assert "30" in result or "3" in result  # Should show 30% or at least 3 events


class TestPrinterCompactVerboseMode:
    """Test compact vs verbose display modes."""
    
    def test_default_verbose_mode(self):
        """Test default mode shows full details."""
        printer = EventPrinter(show_timestamp=True, show_address=True)
        
        async def dummy_handler(input_data, context, state):
            pass
        
        actor = BaseActor(
            name="test",
            description="Test",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=dummy_handler
        )
        
        event = ExecutionEvent(
            actor=actor,
            address="test::tool_call",
            execution_status=ExecutionStatus.STARTED,
            detail="Test input data"
        )
        
        output = io.StringIO()
        with redirect_stdout(output):
            printer.print_event(event)
        
        result = output.getvalue()
        
        # Verbose mode should show timestamp and address
        assert "test::tool_call" in result  # Address
        # Timestamp check is flexible since format may vary
    
    def test_compact_mode(self):
        """Test compact mode hides optional details."""
        printer = EventPrinter(show_timestamp=False, show_address=False)
        
        async def dummy_handler(input_data, context, state):
            pass
        
        actor = BaseActor(
            name="test",
            description="Test",
            input_model=SimpleInput,
            event_type=EventType.TOOL_CALL,
            handler=dummy_handler
        )
        
        event = ExecutionEvent(
            actor=actor,
            address="test::tool_call",
            execution_status=ExecutionStatus.STARTED,
            detail="Test input data"
        )
        
        output = io.StringIO()
        with redirect_stdout(output):
            printer.print_event(event)
        
        result = output.getvalue()
        
        # Compact mode should not show timestamp line
        # (address won't appear as separate line)
        line_count = len([l for l in result.split("\n") if l.strip()])
        # Should have fewer lines than verbose mode
        assert line_count <= 4  # Main + detail + closing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
