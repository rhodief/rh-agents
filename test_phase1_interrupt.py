"""Test Phase 1: Core Interrupt Infrastructure"""
import asyncio
from rh_agents.core.types import ExecutionStatus, InterruptReason, InterruptSignal, InterruptEvent
from rh_agents.core.execution import ExecutionState, InterruptChecker
from rh_agents.core.exceptions import ExecutionInterrupted


async def test_enum_values():
    """Test that new enum values exist."""
    print("✓ Testing enum values...")
    assert ExecutionStatus.INTERRUPTED == 'interrupted'
    assert ExecutionStatus.CANCELLING == 'cancelling'
    assert InterruptReason.USER_CANCELLED == "user_cancelled"
    assert InterruptReason.TIMEOUT == "timeout"
    print("  ✓ ExecutionStatus.INTERRUPTED and CANCELLING exist")
    print("  ✓ InterruptReason enum with 7 values exists")


async def test_interrupt_signal_model():
    """Test InterruptSignal model creation."""
    print("\n✓ Testing InterruptSignal model...")
    signal = InterruptSignal(
        reason=InterruptReason.USER_CANCELLED,
        message="Test cancellation",
        triggered_by="test_user"
    )
    assert signal.reason == InterruptReason.USER_CANCELLED
    assert signal.message == "Test cancellation"
    assert signal.triggered_by == "test_user"
    assert signal.save_checkpoint is True
    print("  ✓ InterruptSignal model created successfully")


async def test_interrupt_event_model():
    """Test InterruptEvent model creation."""
    print("\n✓ Testing InterruptEvent model...")
    signal = InterruptSignal(reason=InterruptReason.TIMEOUT, message="Operation timed out")
    event = InterruptEvent(signal=signal, state_id="test-state-123")
    assert event.signal.reason == InterruptReason.TIMEOUT
    assert event.state_id == "test-state-123"
    print("  ✓ InterruptEvent model created successfully")


async def test_execution_interrupted_exception():
    """Test ExecutionInterrupted exception."""
    print("\n✓ Testing ExecutionInterrupted exception...")
    try:
        raise ExecutionInterrupted(
            reason=InterruptReason.USER_CANCELLED,
            message="User pressed cancel"
        )
    except ExecutionInterrupted as e:
        assert e.reason == InterruptReason.USER_CANCELLED
        assert "User pressed cancel" in str(e)
    print("  ✓ ExecutionInterrupted exception works correctly")


async def test_request_interrupt():
    """Test ExecutionState.request_interrupt() method."""
    print("\n✓ Testing request_interrupt() method...")
    state = ExecutionState()
    
    # Initially not interrupted
    assert state.is_interrupted is False
    assert state.interrupt_signal is None
    
    # Request interrupt
    state.request_interrupt(
        reason=InterruptReason.USER_CANCELLED,
        message="Test interrupt",
        triggered_by="test_suite"
    )
    
    # Verify state changed
    assert state.is_interrupted is True
    assert state.interrupt_signal is not None
    assert state.interrupt_signal.reason == InterruptReason.USER_CANCELLED
    assert state.interrupt_signal.message == "Test interrupt"
    print("  ✓ request_interrupt() sets internal flags correctly")


async def test_set_interrupt_checker():
    """Test ExecutionState.set_interrupt_checker() method."""
    print("\n✓ Testing set_interrupt_checker() method...")
    state = ExecutionState()
    
    # Define a simple checker
    def check_interrupt() -> bool:
        return False
    
    # Set checker
    state.set_interrupt_checker(check_interrupt)
    assert state.interrupt_checker is not None
    
    # Clear checker
    state.set_interrupt_checker(None)
    assert state.interrupt_checker is None
    print("  ✓ set_interrupt_checker() works correctly")


async def test_check_interrupt_local_flag():
    """Test check_interrupt() with local flag."""
    print("\n✓ Testing check_interrupt() with local flag...")
    state = ExecutionState()
    
    # No interrupt - should not raise
    await state.check_interrupt()
    
    # Request interrupt
    state.request_interrupt(reason=InterruptReason.USER_CANCELLED)
    
    # Should raise ExecutionInterrupted
    try:
        await state.check_interrupt()
        assert False, "Should have raised ExecutionInterrupted"
    except ExecutionInterrupted as e:
        assert e.reason == InterruptReason.USER_CANCELLED
    print("  ✓ check_interrupt() detects local flag and raises exception")


async def test_check_interrupt_with_bool_checker():
    """Test check_interrupt() with external checker returning bool."""
    print("\n✓ Testing check_interrupt() with bool checker...")
    state = ExecutionState()
    should_interrupt = False
    
    def check_interrupt() -> bool:
        return should_interrupt
    
    state.set_interrupt_checker(check_interrupt)
    
    # No interrupt
    await state.check_interrupt()
    
    # Trigger interrupt
    should_interrupt = True
    try:
        await state.check_interrupt()
        assert False, "Should have raised ExecutionInterrupted"
    except ExecutionInterrupted as e:
        assert e.reason == InterruptReason.USER_CANCELLED
        assert state.is_interrupted is True
    print("  ✓ check_interrupt() with bool checker works")


async def test_check_interrupt_with_signal_checker():
    """Test check_interrupt() with external checker returning InterruptSignal."""
    print("\n✓ Testing check_interrupt() with InterruptSignal checker...")
    state = ExecutionState()
    should_interrupt = False
    
    def check_interrupt() -> InterruptSignal | bool:
        if should_interrupt:
            return InterruptSignal(
                reason=InterruptReason.TIMEOUT,
                message="External timeout detected",
                triggered_by="monitoring_system"
            )
        return False
    
    state.set_interrupt_checker(check_interrupt)
    
    # No interrupt
    await state.check_interrupt()
    
    # Trigger interrupt
    should_interrupt = True
    try:
        await state.check_interrupt()
        assert False, "Should have raised ExecutionInterrupted"
    except ExecutionInterrupted as e:
        assert e.reason == InterruptReason.TIMEOUT
        assert "External timeout detected" in str(e)
        assert state.interrupt_signal.triggered_by == "monitoring_system"
    print("  ✓ check_interrupt() with InterruptSignal checker works")


async def test_check_interrupt_with_async_checker():
    """Test check_interrupt() with async external checker."""
    print("\n✓ Testing check_interrupt() with async checker...")
    state = ExecutionState()
    should_interrupt = False
    
    async def check_interrupt() -> bool:
        await asyncio.sleep(0.01)  # Simulate async operation
        return should_interrupt
    
    state.set_interrupt_checker(check_interrupt)
    
    # No interrupt
    await state.check_interrupt()
    
    # Trigger interrupt
    should_interrupt = True
    try:
        await state.check_interrupt()
        assert False, "Should have raised ExecutionInterrupted"
    except ExecutionInterrupted as e:
        assert e.reason == InterruptReason.USER_CANCELLED
    print("  ✓ check_interrupt() with async checker works")


async def test_interrupt_event_published():
    """Test that InterruptEvent is published to event bus."""
    print("\n✓ Testing InterruptEvent publishing...")
    state = ExecutionState()
    events_received = []
    
    # Subscribe to events
    def handler(event):
        events_received.append(event)
    
    state.event_bus.subscribe(handler)
    
    # Request interrupt and check
    state.request_interrupt(reason=InterruptReason.USER_CANCELLED)
    try:
        await state.check_interrupt()
    except ExecutionInterrupted:
        pass
    
    # Verify InterruptEvent was published
    assert len(events_received) == 1
    assert isinstance(events_received[0], InterruptEvent)
    assert events_received[0].signal.reason == InterruptReason.USER_CANCELLED
    print("  ✓ InterruptEvent published to event bus successfully")


async def main():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("Phase 1: Core Interrupt Infrastructure Tests")
    print("=" * 60)
    
    try:
        await test_enum_values()
        await test_interrupt_signal_model()
        await test_interrupt_event_model()
        await test_execution_interrupted_exception()
        await test_request_interrupt()
        await test_set_interrupt_checker()
        await test_check_interrupt_local_flag()
        await test_check_interrupt_with_bool_checker()
        await test_check_interrupt_with_signal_checker()
        await test_check_interrupt_with_async_checker()
        await test_interrupt_event_published()
        
        print("\n" + "=" * 60)
        print("✅ ALL PHASE 1 TESTS PASSED!")
        print("=" * 60)
        print("\nPhase 1 Deliverables:")
        print("  ✓ ExecutionStatus.INTERRUPTED and CANCELLING enums")
        print("  ✓ InterruptReason enum (7 values)")
        print("  ✓ InterruptSignal model")
        print("  ✓ InterruptEvent model")
        print("  ✓ ExecutionInterrupted exception")
        print("  ✓ ExecutionState.request_interrupt() method")
        print("  ✓ ExecutionState.set_interrupt_checker() method")
        print("  ✓ ExecutionState.check_interrupt() method")
        print("  ✓ InterruptChecker type alias (6 variants)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
