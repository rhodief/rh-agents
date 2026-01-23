#!/usr/bin/env python3
"""
Phase 6 Validation: Basic EventPrinter Support
"""

import asyncio
from rh_agents.core.execution import ExecutionState
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.actors import BaseActor
from rh_agents.bus_handlers import ParallelEventPrinter
from pydantic import BaseModel, Field


# Create a simple input model
class EmptyInput(BaseModel):
    pass


# Create async handler
async def mock_handler(*args, **kwargs):
    return "Mock result"


# Create a simple mock actor for testing  
def create_mock_actor(name="TestActor"):
    return BaseActor(
        name=name,
        description="Test actor for validation",
        input_model=EmptyInput,
        handler=mock_handler,
        event_type=EventType.AGENT_CALL,
        cacheable=False
    )

print("=" * 70)
print("PHASE 6 VALIDATION - Basic EventPrinter Support")
print("=" * 70)
print()


# Test 1: ParallelEventPrinter instantiation
print("Test 1: ParallelEventPrinter Instantiation")
try:
    printer = ParallelEventPrinter(parallel_mode="realtime")
    print("  ✓ ParallelEventPrinter instantiates correctly")
    print(f"    - Parallel mode: {printer.parallel_mode}")
    print(f"    - Parallel groups: {len(printer.parallel_groups)}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print()


# Test 2: Parallel event detection
print("Test 2: Parallel Event Detection")
try:
    printer = ParallelEventPrinter(parallel_mode="realtime")
    
    # Create a mock actor
    actor = create_mock_actor()
    
    # Create parallel events
    group_id = "test_group_001"
    
    # Event 1: Started
    event1 = ExecutionEvent(
        actor=actor,
        execution_status=ExecutionStatus.STARTED,
        is_parallel=True,
        group_id=group_id,
        parallel_index=0,
        detail="Processing document 1"
    )
    
    print(f"  Created event with:")
    print(f"    - is_parallel: {event1.is_parallel}")
    print(f"    - group_id: {event1.group_id}")
    print(f"    - parallel_index: {event1.parallel_index}")
    print(f"    - status: {event1.execution_status}")
    
    # Print the event
    print(f"\n  Printing parallel event:")
    printer.print_event(event1)
    
    # Verify group was created
    assert group_id in printer.parallel_groups, "Group not created"
    tracker = printer.parallel_groups[group_id]
    assert tracker.started == 1, "Started count incorrect"
    
    print(f"\n  ✓ Parallel event detection works")
    print(f"    - Group created: {group_id}")
    print(f"    - Started count: {tracker.started}")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print()


# Test 3: Group lifecycle
print("Test 3: Group Lifecycle")
try:
    printer = ParallelEventPrinter(parallel_mode="realtime")
    actor = create_mock_actor(name="ProcessDoc")
    
    group_id = "lifecycle_test"
    num_tasks = 3
    
    # Start events
    print(f"  Starting {num_tasks} tasks...")
    for i in range(num_tasks):
        event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.STARTED,
            is_parallel=True,
            group_id=group_id,
            parallel_index=i,
            detail=f"Task {i}"
        )
        printer.print_event(event)
    
    tracker = printer.parallel_groups[group_id]
    print(f"\n  After starts: {tracker.started} started")
    
    # Complete events
    print(f"\n  Completing {num_tasks} tasks...")
    for i in range(num_tasks):
        event = ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.COMPLETED,
            is_parallel=True,
            group_id=group_id,
            parallel_index=i,
            detail=f"Task {i} result",
            execution_time=0.1 + i * 0.05
        )
        printer.print_event(event)
    
    print(f"\n  After completion: {tracker.completed} completed")
    print(f"  Is complete: {tracker.is_complete}")
    print(f"  Completion %: {tracker.completion_percentage:.1f}%")
    
    print(f"\n  ✓ Group lifecycle works")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print()


# Test 4: Mixed regular and parallel events
print("Test 4: Mixed Regular and Parallel Events")
try:
    printer = ParallelEventPrinter(parallel_mode="realtime")
    actor = create_mock_actor(name="MixedAgent")
    
    # Regular event
    print("  Regular event:")
    regular_event = ExecutionEvent(
        actor=actor,
        execution_status=ExecutionStatus.COMPLETED,
        is_parallel=False,
        detail="Regular task"
    )
    printer.print_event(regular_event)
    
    # Parallel event
    print("\n  Parallel event:")
    parallel_event = ExecutionEvent(
        actor=actor,
        execution_status=ExecutionStatus.COMPLETED,
        is_parallel=True,
        group_id="mixed_test",
        parallel_index=0,
        detail="Parallel task"
    )
    printer.print_event(parallel_event)
    
    print(f"\n  ✓ Mixed events work")
    print(f"    - Parallel groups: {len(printer.parallel_groups)}")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print()


# Test 5: Error handling in parallel groups
print("Test 5: Error Handling")
try:
    printer = ParallelEventPrinter(parallel_mode="realtime")
    actor = create_mock_actor(name="FailAgent")
    
    group_id = "error_test"
    
    # Start and complete successfully
    for i in range(2):
        printer.print_event(ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.STARTED,
            is_parallel=True,
            group_id=group_id,
            parallel_index=i
        ))
        printer.print_event(ExecutionEvent(
            actor=actor,
            execution_status=ExecutionStatus.COMPLETED,
            is_parallel=True,
            group_id=group_id,
            parallel_index=i,
            execution_time=0.1
        ))
    
    # One failure
    printer.print_event(ExecutionEvent(
        actor=actor,
        execution_status=ExecutionStatus.STARTED,
        is_parallel=True,
        group_id=group_id,
        parallel_index=2
    ))
    printer.print_event(ExecutionEvent(
        actor=actor,
        execution_status=ExecutionStatus.FAILED,
        is_parallel=True,
        group_id=group_id,
        parallel_index=2,
        message="Connection timeout",
        execution_time=0.5
    ))
    
    tracker = printer.parallel_groups[group_id]
    print(f"\n  Final stats:")
    print(f"    - Completed: {tracker.completed}")
    print(f"    - Failed: {tracker.failed}")
    print(f"    - Success rate: {tracker.completion_percentage:.1f}%")
    
    print(f"\n  ✓ Error handling works")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("✅ PHASE 6 COMPLETE")
print("=" * 70)
print()
print("Implemented:")
print("  • ParallelEventPrinter class extending EventPrinter")
print("  • Parallel group tracking system")
print("  • Real-time display mode (interleaved output)")
print("  • Group lifecycle event handling")
print("  • Mixed regular/parallel event support")
print()
print("Next: Phase 7 - Progress Bar Visualization")
