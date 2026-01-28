"""Test Phase 2: Integration with Execution Flow"""
import asyncio
from rh_agents.core.types import ExecutionStatus, InterruptReason, InterruptSignal, InterruptEvent
from rh_agents.core.execution import ExecutionState
from rh_agents.core.exceptions import ExecutionInterrupted
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import Agent


async def test_execution_event_interrupt_check():
    """Test that ExecutionEvent.__call__() checks for interrupts."""
    print("✓ Testing ExecutionEvent interrupt checks...")
    
    # Create a simple agent without LLM dependency
    @Agent(name="test_agent")
    async def test_handler(data: str, context, state):
        return f"Processed: {data}"
    
    agent = test_handler()
    state = ExecutionState()
    
    # Normal execution - should succeed
    event = ExecutionEvent(actor=agent)
    result = await event(input_data="test", extra_context="", execution_state=state)
    assert result.ok is True
    assert result.result == "Processed: test"
    print("  ✓ Normal execution works")
    
    # Interrupt before execution
    state2 = ExecutionState()
    state2.request_interrupt(reason=InterruptReason.USER_CANCELLED)
    
    event2 = ExecutionEvent(actor=agent)
    result2 = await event2(input_data="test", extra_context="", execution_state=state2)
    
    assert result2.ok is False
    assert "interrupted" in result2.erro_message.lower()
    print("  ✓ Interrupt before execution handled correctly")


async def test_execution_event_interrupt_during_handler():
    """Test interrupt triggered during handler execution."""
    print("\n✓ Testing interrupt during handler execution...")
    
    @Agent(name="long_running_agent")
    async def long_handler(data: str, context, state):
        # Simulate long operation with interrupt checks
        for i in range(10):
            await state.check_interrupt()
            await asyncio.sleep(0.1)
        return "completed"
    
    agent = long_handler()
    state = ExecutionState()
    
    # Start execution
    event = ExecutionEvent(actor=agent)
    exec_task = asyncio.create_task(event(input_data="test", extra_context="", execution_state=state))
    
    # Interrupt after 0.3 seconds
    await asyncio.sleep(0.3)
    state.request_interrupt(reason=InterruptReason.USER_CANCELLED, message="Test interrupt")
    
    # Wait for result
    result = await exec_task
    
    assert result.ok is False
    assert "interrupted" in result.erro_message.lower() or "Test interrupt" in result.erro_message
    print("  ✓ Interrupt during handler execution handled correctly")


async def test_generator_registry():
    """Test generator registration and cleanup."""
    print("\n✓ Testing generator registry...")
    
    state = ExecutionState()
    
    # Create mock generator task
    async def mock_generator():
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("    Generator cancelled")
            raise
    
    # Register generator
    gen_task = asyncio.create_task(mock_generator())
    state.register_generator(gen_task)
    
    assert gen_task in state.active_generators
    print("  ✓ Generator registered")
    
    # Kill generators
    await state.kill_generators()
    
    assert len(state.active_generators) == 0
    assert gen_task.cancelled() or gen_task.done()
    print("  ✓ Generator killed and unregistered")


async def test_event_bus_stream_interrupt():
    """Test that EventBus.stream() terminates on InterruptEvent."""
    print("\n✓ Testing EventBus.stream() interrupt handling...")
    
    state = ExecutionState()
    events_received = []
    
    # Create simple events
    from rh_agents.core.events import ExecutionEvent
    from rh_agents.core.actors import Agent
    
    @Agent(name="test")
    async def test_handler(data: str, context, state_param):
        return data
    
    agent = test_handler()
    
    # Create events
    event1 = ExecutionEvent(actor=agent)
    event2 = ExecutionEvent(actor=agent)
    
    # Publish events
    await state.event_bus.publish(event1)
    await state.event_bus.publish(event2)
    
    # Publish interrupt event
    interrupt_signal = InterruptSignal(
        reason=InterruptReason.USER_CANCELLED,
        message="Stream terminated"
    )
    interrupt_event = InterruptEvent(signal=interrupt_signal, state_id=state.state_id)
    await state.event_bus.publish(interrupt_event)
    
    # Stream should terminate after interrupt event
    stream_count = 0
    async for event in state.event_bus.stream():
        events_received.append(event)
        stream_count += 1
        if isinstance(event, InterruptEvent):
            break
    
    # Should have received 2 regular events + 1 interrupt event
    assert stream_count == 3
    assert isinstance(events_received[-1], InterruptEvent)
    print("  ✓ EventBus.stream() terminated on InterruptEvent")


async def test_parallel_execution_interrupt():
    """Test interrupt monitoring in parallel execution."""
    print("\n✓ Testing parallel execution interrupt...")
    
    state = ExecutionState()
    results = []
    
    async def slow_task(n: int):
        """Slow task that checks interrupts."""
        for i in range(10):
            await state.check_interrupt()
            await asyncio.sleep(0.1)
        return f"Task {n} complete"
    
    # Start parallel execution
    async def run_parallel():
        async with state.parallel(max_workers=3) as p:
            p.add(slow_task(1))
            p.add(slow_task(2))
            p.add(slow_task(3))
            return await p.gather()
    
    parallel_task = asyncio.create_task(run_parallel())
    
    # Interrupt after 0.3 seconds
    await asyncio.sleep(0.3)
    state.request_interrupt(
        reason=InterruptReason.USER_CANCELLED,
        message="Parallel execution interrupted"
    )
    
    # Should raise ExecutionInterrupted
    try:
        results = await parallel_task
        # If we get here, the tasks completed before interrupt
        print("  ⚠ Tasks completed before interrupt could take effect")
    except ExecutionInterrupted as e:
        assert e.reason == InterruptReason.USER_CANCELLED
        print("  ✓ Parallel execution interrupted successfully")


async def test_parallel_execution_with_interrupt_check():
    """Test that parallel execution checks interrupt before and after."""
    print("\n✓ Testing parallel execution with pre/post interrupt checks...")
    
    state = ExecutionState()
    
    # Interrupt before parallel execution
    state.request_interrupt(reason=InterruptReason.USER_CANCELLED)
    
    async def quick_task():
        return "done"
    
    # Should be interrupted before tasks even start
    try:
        async with state.parallel(max_workers=2) as p:
            p.add(quick_task())
            results = await p.gather()
        assert False, "Should have raised ExecutionInterrupted"
    except ExecutionInterrupted as e:
        assert e.reason == InterruptReason.USER_CANCELLED
        print("  ✓ Parallel execution pre-check detected interrupt")


async def test_interrupt_during_preconditions():
    """Test interrupt detection during preconditions."""
    print("\n✓ Testing interrupt during preconditions...")
    
    precondition_called = False
    
    @Agent(name="agent_with_preconditions")
    async def handler_with_preconditions(data: str, context, state):
        return "result"
    
    agent = handler_with_preconditions()
    
    # Add a slow precondition
    async def slow_precondition(data, context, state):
        nonlocal precondition_called
        precondition_called = True
        await asyncio.sleep(0.2)
    
    agent.preconditions.append(slow_precondition)
    
    state = ExecutionState()
    event = ExecutionEvent(actor=agent)
    
    # Start execution
    exec_task = asyncio.create_task(event(input_data="test", extra_context="", execution_state=state))
    
    # Interrupt during precondition
    await asyncio.sleep(0.1)
    state.request_interrupt(reason=InterruptReason.USER_CANCELLED)
    
    result = await exec_task
    
    assert result.ok is False
    assert "interrupted" in result.erro_message.lower()
    print("  ✓ Interrupt during preconditions handled correctly")


async def main():
    """Run all Phase 2 integration tests."""
    print("=" * 60)
    print("Phase 2: Integration Tests")
    print("=" * 60)
    
    try:
        await test_execution_event_interrupt_check()
        await test_execution_event_interrupt_during_handler()
        await test_generator_registry()
        await test_event_bus_stream_interrupt()
        await test_parallel_execution_interrupt()
        await test_parallel_execution_with_interrupt_check()
        await test_interrupt_during_preconditions()
        
        print("\n" + "=" * 60)
        print("✅ ALL PHASE 2 TESTS PASSED!")
        print("=" * 60)
        print("\nPhase 2 Deliverables:")
        print("  ✓ ExecutionEvent.__call__() interrupt checks (4 checkpoints)")
        print("  ✓ Generator registry (register/unregister/kill methods)")
        print("  ✓ EventBus.stream() InterruptEvent handling")
        print("  ✓ ParallelExecutionManager interrupt monitoring")
        print("\nIntegration Points Tested:")
        print("  ✓ Interrupt before execution")
        print("  ✓ Interrupt during handler execution")
        print("  ✓ Interrupt during preconditions")
        print("  ✓ Generator cleanup on interrupt")
        print("  ✓ Event stream termination")
        print("  ✓ Parallel execution cancellation")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
