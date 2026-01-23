#!/usr/bin/env python3
"""
Visualization Example with ParallelEventPrinter

This example demonstrates the visual output options for parallel execution:
- Real-time mode: Shows events as they happen (interleaved)
- Progress mode: Shows progress bars with statistics
"""

import asyncio
import random
from rh_agents.core.execution import ExecutionState
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.actors import BaseActor
from rh_agents.bus_handlers import ParallelEventPrinter
from pydantic import BaseModel


# Setup test actor
class EmptyInput(BaseModel):
    pass


async def mock_handler(*args, **kwargs):
    return "Result"


def create_actor(name="TaskActor"):
    return BaseActor(
        name=name,
        description="Task processor",
        input_model=EmptyInput,
        handler=mock_handler,
        event_type=EventType.AGENT_CALL,
        cacheable=False
    )


async def simulate_task(task_id: int, delay: float, fail_rate: float = 0.2):
    """Simulate a task with variable duration and failure rate."""
    await asyncio.sleep(delay)
    
    if random.random() < fail_rate:
        raise RuntimeError(f"Task {task_id} encountered an error")
    
    return f"Task {task_id} completed in {delay:.2f}s"


async def example_realtime_mode():
    """Demonstrate real-time visualization mode."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Real-Time Mode")
    print("=" * 70)
    print("Shows events as they occur, interleaved output.")
    print()
    
    # Create printer in real-time mode
    printer = ParallelEventPrinter(parallel_mode="realtime")
    state = ExecutionState()
    state.event_bus.subscribe(printer.print_event)
    
    actor = create_actor("RealtimeTask")
    
    # Simulate parallel execution by manually creating events
    group_id = "realtime_demo"
    num_tasks = 5
    
    # Start events
    for i in range(num_tasks):
        event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.STARTED,
            is_parallel=True,
            group_id=group_id,
            parallel_index=i,
            detail="Processing items in parallel"
        )
        await state.event_bus.publish(event)
    
    # Complete events with delays
    for i in range(num_tasks):
        await asyncio.sleep(0.2)
        
        if i == 2:  # Simulate one failure
            event = ExecutionEvent(
                actor=actor,
                execution_status=ExecutionStatus.FAILED,
                is_parallel=True,
                group_id=group_id,
                parallel_index=i,
                message="Simulated error for demo",
                execution_time=0.2
            )
        else:
            event = ExecutionEvent(
                actor=actor,
                execution_status=ExecutionStatus.COMPLETED,
                is_parallel=True,
                group_id=group_id,
                parallel_index=i,
                execution_time=0.2
            )
        
        await state.event_bus.publish(event)
    
    print()
    print("Real-time mode features:")
    print("  • Events appear immediately as they happen")
    print("  • Shows individual task progress")
    print("  • Good for debugging and monitoring")
    print("  • See exact sequence of operations")


async def example_progress_mode():
    """Demonstrate progress bar visualization mode."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Progress Mode")
    print("=" * 70)
    print("Shows aggregated progress with visual bars.")
    print()
    
    # Create printer in progress mode
    printer = ParallelEventPrinter(parallel_mode="progress")
    state = ExecutionState()
    state.event_bus.subscribe(printer.print_event)
    
    actor = create_actor("ProgressTask")
    
    # Simulate parallel execution
    group_id = "progress_demo"
    num_tasks = 15
    
    # Start all tasks
    for i in range(num_tasks):
        event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.STARTED,
            is_parallel=True,
            group_id=group_id,
            parallel_index=i,
            detail="Batch Processing Job"
        )
        await state.event_bus.publish(event)
    
    # Complete tasks gradually
    for i in range(num_tasks):
        await asyncio.sleep(0.15)
        
        # Simulate some failures
        if i in [3, 7, 11]:
            event = ExecutionEvent(
                actor=actor,
                execution_status=ExecutionStatus.FAILED,
                is_parallel=True,
                group_id=group_id,
                parallel_index=i,
                message=f"Failed on validation check",
                execution_time=0.15
            )
        else:
            event = ExecutionEvent(
                actor=actor,
                execution_status=ExecutionStatus.COMPLETED,
                is_parallel=True,
                group_id=group_id,
                parallel_index=i,
                execution_time=0.15
            )
        
        await state.event_bus.publish(event)
    
    print()
    print("Progress mode features:")
    print("  • Visual progress bar with percentage")
    print("  • Aggregated statistics (✓ completed, ✖ failed)")
    print("  • Real-time updates (on TTY terminals)")
    print("  • Clean output for production logs")
    print("  • Shows timing information")


async def example_comparison():
    """Side-by-side comparison of both modes."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Mode Comparison")
    print("=" * 70)
    print()
    
    print("Testing with 8 tasks, some failures...")
    print()
    
    # Both modes with same data
    for mode_name, mode in [("realtime", "realtime"), ("progress", "progress")]:
        print(f"\n--- {mode_name.upper()} MODE ---\n")
        
        printer = ParallelEventPrinter(parallel_mode=mode)
        state = ExecutionState()
        state.event_bus.subscribe(printer.print_event)
        
        actor = create_actor(f"{mode_name}Actor")
        group_id = f"compare_{mode_name}"
        num_tasks = 8
        
        # Start and complete tasks
        for i in range(num_tasks):
            start_event = ExecutionEvent(
                actor=actor,
                execution_status=ExecutionStatus.STARTED,
                is_parallel=True,
                group_id=group_id,
                parallel_index=i,
                detail="Comparison Test"
            )
            await state.event_bus.publish(start_event)
        
        await asyncio.sleep(0.1)
        
        for i in range(num_tasks):
            if i % 4 == 0:  # Every 4th task fails
                event = ExecutionEvent(
                    actor=actor,
                    execution_status=ExecutionStatus.FAILED,
                    is_parallel=True,
                    group_id=group_id,
                    parallel_index=i,
                    message="Demo failure",
                    execution_time=0.1
                )
            else:
                event = ExecutionEvent(
                    actor=actor,
                    execution_status=ExecutionStatus.COMPLETED,
                    is_parallel=True,
                    group_id=group_id,
                    parallel_index=i,
                    execution_time=0.1
                )
            
            await state.event_bus.publish(event)
            await asyncio.sleep(0.05)
    
    print()


async def main():
    print("=" * 70)
    print("PARALLEL EXECUTION VISUALIZATION")
    print("=" * 70)
    print()
    print("This example demonstrates different visualization modes")
    print("for parallel execution progress.")
    
    await example_realtime_mode()
    await example_progress_mode()
    await example_comparison()
    
    print("\n" + "=" * 70)
    print("CHOOSING A VISUALIZATION MODE")
    print("=" * 70)
    print()
    print("REALTIME MODE:")
    print("  Best for:")
    print("    • Development and debugging")
    print("    • Detailed execution monitoring")
    print("    • Understanding event sequences")
    print("  Pros:")
    print("    • Full detail on each event")
    print("    • Easy to debug failures")
    print("  Cons:")
    print("    • Verbose output")
    print("    • Hard to gauge overall progress")
    print()
    print("PROGRESS MODE:")
    print("  Best for:")
    print("    • Production environments")
    print("    • Batch processing jobs")
    print("    • User-facing progress displays")
    print("  Pros:")
    print("    • Clean, compact output")
    print("    • Clear progress indication")
    print("    • Good for logs")
    print("  Cons:")
    print("    • Less detail on individual tasks")
    print("    • Requires TTY for live updates")
    print()
    print("Usage:")
    print("  printer = ParallelEventPrinter(parallel_mode='realtime')  # or 'progress'")
    print("  state.event_bus.subscribe(printer.print_event)")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
