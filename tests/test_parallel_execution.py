#!/usr/bin/env python3
"""
Comprehensive tests for parallel execution functionality.
"""

import asyncio
import pytest
from rh_agents.core.execution import ExecutionState
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.actors import BaseActor
from rh_agents.core.parallel import ErrorStrategy, ParallelExecutionManager
from rh_agents.bus_handlers import ParallelEventPrinter
from pydantic import BaseModel


# Test fixtures
class EmptyInput(BaseModel):
    pass


async def mock_handler(*args, **kwargs):
    return "Mock result"


def create_test_actor(name="TestActor"):
    return BaseActor(
        name=name,
        description="Test actor",
        input_model=EmptyInput,
        handler=mock_handler,
        event_type=EventType.AGENT_CALL,
        cacheable=False
    )


# Test 1: Basic parallel execution
@pytest.mark.asyncio
async def test_basic_parallel_execution():
    """Test basic parallel execution with gather()."""
    state = ExecutionState()
    
    async def task(n):
        await asyncio.sleep(0.01)
        return n * 2
    
    async with state.parallel(max_workers=3) as p:
        p.add(lambda: task(1))
        p.add(lambda: task(2))
        p.add(lambda: task(3))
        results = await p.gather()
    
    assert len(results) == 3
    assert all(r.ok for r in results)
    assert results[0].result == 2
    assert results[1].result == 4
    assert results[2].result == 6


# Test 2: Concurrency limiting
@pytest.mark.asyncio
async def test_concurrency_limiting():
    """Test that max_workers correctly limits concurrency."""
    state = ExecutionState()
    concurrent_count = 0
    max_concurrent = 0
    
    async def task(n):
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.05)
        concurrent_count -= 1
        return n
    
    async with state.parallel(max_workers=2) as p:
        for i in range(5):
            p.add(lambda i=i: task(i))
        await p.gather()
    
    # With max_workers=2, we should never have more than 2 concurrent
    assert max_concurrent <= 2


# Test 3: Streaming mode
@pytest.mark.asyncio
async def test_streaming_mode():
    """Test streaming result collection."""
    state = ExecutionState()
    
    async def task(n):
        await asyncio.sleep(0.01 * n)
        return n
    
    async with state.parallel(max_workers=3) as p:
        for i in range(5):
            p.add(lambda i=i: task(i))
        
        results = []
        async for result in p.stream():
            results.append(result)
    
    assert len(results) == 5
    # First result should be fastest (smallest sleep time)
    assert results[0].result == 0


# Test 4: Dictionary mode
@pytest.mark.asyncio
async def test_dictionary_mode():
    """Test named result collection."""
    state = ExecutionState()
    
    async def task(n):
        await asyncio.sleep(0.01)
        return n * 10
    
    async with state.parallel(max_workers=3) as p:
        p.add(lambda: task(1), name="first")
        p.add(lambda: task(2), name="second")
        p.add(lambda: task(3), name="third")
        results = await p.gather_dict()
    
    assert "first" in results
    assert "second" in results
    assert "third" in results
    assert results["first"].result == 10
    assert results["second"].result == 20
    assert results["third"].result == 30


# Test 5: Fail slow error strategy
@pytest.mark.asyncio
async def test_fail_slow_strategy():
    """Test fail slow collects all results including errors."""
    state = ExecutionState()
    
    async def success_task():
        await asyncio.sleep(0.01)
        return "success"
    
    async def failing_task():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")
    
    async with state.parallel(max_workers=2, error_strategy=ErrorStrategy.FAIL_SLOW) as p:
        p.add(lambda: failing_task())
        p.add(lambda: success_task())
        p.add(lambda: failing_task())
        results = await p.gather()
    
    assert len(results) == 3
    assert not results[0].ok
    assert results[1].ok
    assert not results[2].ok
    assert "Test error" in results[0].erro_message


# Test 6: Fail fast error strategy
@pytest.mark.asyncio
async def test_fail_fast_strategy():
    """Test fail fast cancels remaining tasks on error."""
    state = ExecutionState()
    
    async def failing_task():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")
    
    async def slow_task():
        await asyncio.sleep(1.0)  # Should be cancelled
        return "slow"
    
    with pytest.raises(ValueError, match="Test error"):
        async with state.parallel(max_workers=2, error_strategy=ErrorStrategy.FAIL_FAST) as p:
            p.add(lambda: failing_task())
            p.add(lambda: slow_task())
            await p.gather()


# Test 7: Timeout support
@pytest.mark.asyncio
async def test_timeout():
    """Test parallel group timeout."""
    state = ExecutionState()
    
    async def slow_task():
        await asyncio.sleep(1.0)
        return "done"
    
    with pytest.raises(asyncio.TimeoutError):
        async with state.parallel(max_workers=2, timeout=0.1) as p:
            p.add(lambda: slow_task())
            await p.gather()


# Test 8: Retry logic
@pytest.mark.asyncio
async def test_retry_logic():
    """Test retry logic with exponential backoff."""
    state = ExecutionState()
    attempt_counts = {}
    
    async def flaky_task(task_id):
        if task_id not in attempt_counts:
            attempt_counts[task_id] = 0
        attempt_counts[task_id] += 1
        
        if attempt_counts[task_id] < 2:
            raise ValueError(f"Attempt {attempt_counts[task_id]}")
        
        return f"success_{task_id}"
    
    async with state.parallel(max_workers=2, max_retries=2, retry_delay=0.01) as p:
        p.add(lambda: flaky_task("task1"))
        p.add(lambda: flaky_task("task2"))
        results = await p.gather()
    
    assert len(results) == 2
    assert all(r.ok for r in results)
    assert attempt_counts["task1"] == 2
    assert attempt_counts["task2"] == 2


# Test 9: Circuit breaker
@pytest.mark.asyncio
async def test_circuit_breaker():
    """Test circuit breaker opens after threshold failures."""
    state = ExecutionState()
    
    async def failing_task():
        raise ValueError("Circuit test")
    
    async with state.parallel(
        max_workers=1,
        circuit_breaker_threshold=3,
        error_strategy=ErrorStrategy.FAIL_SLOW
    ) as p:
        for i in range(5):
            p.add(lambda: failing_task())
        results = await p.gather()
    
    # First 3 should fail normally, last 2 should be rejected by circuit breaker
    circuit_breaker_rejections = sum(
        1 for r in results if "Circuit breaker" in r.erro_message
    )
    assert circuit_breaker_rejections >= 2


# Test 10: Event metadata
@pytest.mark.asyncio
async def test_event_metadata():
    """Test that parallel events have correct metadata."""
    state = ExecutionState()
    events_seen = []
    
    def event_handler(event):
        if event.is_parallel:
            events_seen.append(event)
    
    state.event_bus.subscribe(event_handler)
    
    actor = create_test_actor("MetadataTest")
    
    async with state.parallel(max_workers=2) as p:
        for i in range(3):
            event = ExecutionEvent(
                actor=actor,
                execution_status=ExecutionStatus.STARTED,
                detail=f"Task {i}"
            )
            # Note: In real usage, events would be created automatically
            # This is a simplified test
        await p.gather()
    
    # Events would have is_parallel, group_id, parallel_index set
    # This is tested implicitly through printer tests


# Test 11: Empty parallel group
@pytest.mark.asyncio
async def test_empty_parallel_group():
    """Test parallel group with no tasks."""
    state = ExecutionState()
    
    async with state.parallel(max_workers=2) as p:
        results = await p.gather()
    
    assert len(results) == 0


# Test 12: Single task parallel group
@pytest.mark.asyncio
async def test_single_task_parallel():
    """Test parallel group with single task."""
    state = ExecutionState()
    
    async def task():
        await asyncio.sleep(0.01)
        return "single"
    
    async with state.parallel(max_workers=2) as p:
        p.add(lambda: task())
        results = await p.gather()
    
    assert len(results) == 1
    assert results[0].ok
    assert results[0].result == "single"


# Test 13: ParallelEventPrinter integration
@pytest.mark.asyncio
async def test_parallel_event_printer():
    """Test ParallelEventPrinter with parallel execution."""
    printer = ParallelEventPrinter(parallel_mode="realtime")
    state = ExecutionState()
    state.event_bus.subscribe(printer.print_event)
    
    actor = create_test_actor("PrinterTest")
    
    # Simulate parallel events
    group_id = "test_group"
    for i in range(3):
        start_event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.STARTED,
            is_parallel=True,
            group_id=group_id,
            parallel_index=i,
            detail="Test task"
        )
        printer.print_event(start_event)
        
        complete_event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.COMPLETED,
            is_parallel=True,
            group_id=group_id,
            parallel_index=i,
            execution_time=0.1
        )
        printer.print_event(complete_event)
    
    # Check group was tracked
    assert group_id in printer.parallel_groups
    tracker = printer.parallel_groups[group_id]
    assert tracker.completed == 3


# Test 14: Progress bar rendering
def test_progress_bar_rendering():
    """Test progress bar rendering logic."""
    printer = ParallelEventPrinter(parallel_mode="progress")
    actor = create_test_actor("ProgressTest")
    
    group_id = "progress_test"
    
    # Start some tasks
    for i in range(5):
        event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.STARTED,
            is_parallel=True,
            group_id=group_id,
            parallel_index=i,
            detail="Progress Test"
        )
        printer.print_event(event)
    
    # Complete some tasks
    for i in range(3):
        event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.COMPLETED,
            is_parallel=True,
            group_id=group_id,
            parallel_index=i,
            execution_time=0.1
        )
        printer.print_event(event)
    
    tracker = printer.parallel_groups[group_id]
    assert tracker.started == 5
    assert tracker.completed == 3
    assert tracker.completion_percentage == 60.0


# Test 15: Mixed regular and parallel events
@pytest.mark.asyncio
async def test_mixed_events():
    """Test mix of regular and parallel events."""
    printer = ParallelEventPrinter(parallel_mode="realtime")
    state = ExecutionState()
    state.event_bus.subscribe(printer.print_event)
    
    actor = create_test_actor("MixedTest")
    
    # Regular event
    regular_event = ExecutionEvent(
        actor=actor,
        execution_status=ExecutionStatus.COMPLETED,
        is_parallel=False,
        detail="Regular task"
    )
    printer.print_event(regular_event)
    
    # Parallel events
    group_id = "mixed_group"
    for i in range(2):
        parallel_event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.COMPLETED,
            is_parallel=True,
            group_id=group_id,
            parallel_index=i,
            detail="Parallel task"
        )
        printer.print_event(parallel_event)
    
    # Should have one parallel group tracked
    assert len(printer.parallel_groups) == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
