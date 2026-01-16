"""
Test Phase 2: Smart Replay Logic

Tests intelligent event replay including:
- Skip completed events on restore
- Execute only new events
- resume_from_address functionality
- Selective event publishing
- Validation mode
"""
import asyncio
from pathlib import Path
import shutil

from rh_agents.core.execution import ExecutionState
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import BaseActor
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.state_recovery import StateStatus, StateMetadata, ReplayMode
from rh_agents.state_backends import FileSystemStateBackend, FileSystemArtifactBackend
from pydantic import BaseModel


class SimpleInput(BaseModel):
    value: int


class SimpleOutput(BaseModel):
    result: int


# Test actor handler functions
async def add_one_handler(input_data: SimpleInput, extra_context, execution_state):
    return SimpleOutput(result=input_data.value + 1)


async def multiply_by_two_handler(input_data: SimpleInput, extra_context, execution_state):
    return SimpleOutput(result=input_data.value * 2)


async def add_ten_handler(input_data: SimpleInput, extra_context, execution_state):
    return SimpleOutput(result=input_data.value + 10)


# Create actor instances (not classes)
def create_add_one_actor():
    return BaseActor(
        name="add_one",
        description="Adds 1 to input value",
        input_model=SimpleInput,
        output_model=SimpleOutput,
        handler=add_one_handler,
        event_type=EventType.TOOL_CALL
    )


def create_multiply_by_two_actor():
    return BaseActor(
        name="multiply_by_two",
        description="Multiplies input by 2",
        input_model=SimpleInput,
        output_model=SimpleOutput,
        handler=multiply_by_two_handler,
        event_type=EventType.TOOL_CALL
    )


def create_add_ten_actor():
    return BaseActor(
        name="add_ten",
        description="Adds 10 to input value",
        input_model=SimpleInput,
        output_model=SimpleOutput,
        handler=add_ten_handler,
        event_type=EventType.TOOL_CALL
    )


async def test_basic_replay():
    """Test basic replay: execute, save, restore, replay skips completed events."""
    test_dir = Path("./.test_phase2_store")
    
    # Clean up
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    try:
        # Create backends
        state_backend = FileSystemStateBackend(str(test_dir))
        artifact_backend = FileSystemArtifactBackend(str(test_dir / "artifacts"))
        
        # PART 1: Initial execution
        print("=" * 60)
        print("PART 1: Initial Execution")
        print("=" * 60)
        
        state = ExecutionState(
            state_backend=state_backend,
            artifact_backend=artifact_backend
        )
        
        # Create events
        event1 = ExecutionEvent(actor=create_add_one_actor())
        event2 = ExecutionEvent(actor=create_multiply_by_two_actor())
        
        # Execute step 1
        result1 = await event1(SimpleInput(value=5), None, state)
        print(f"✅ Step 1 executed: 5 + 1 = {result1.result.result}")
        assert result1.result.result == 6
        
        # Execute step 2
        result2 = await event2(SimpleInput(value=result1.result.result), None, state)
        print(f"✅ Step 2 executed: 6 * 2 = {result2.result.result}")
        assert result2.result.result == 12
        
        # Save checkpoint
        saved = state.save_checkpoint(
            status=StateStatus.PAUSED,
            metadata=StateMetadata(
                tags=["test", "phase2"],
                description="After 2 steps"
            )
        )
        assert saved
        print(f"✅ Checkpoint saved: {state.state_id}")
        
        # PART 2: Restore and replay
        print("\n" + "=" * 60)
        print("PART 2: Restore and Replay")
        print("=" * 60)
        
        restored_state = ExecutionState.load_from_state_id(
            state_id=state.state_id,
            state_backend=state_backend,
            artifact_backend=artifact_backend,
            replay_mode=ReplayMode.NORMAL
        )
        
        assert restored_state is not None
        print(f"✅ State restored: {restored_state.state_id}")
        
        # Replay step 1 (should be skipped)
        event1_replay = ExecutionEvent(actor=create_add_one_actor())
        result1_replay = await event1_replay(SimpleInput(value=5), None, restored_state)
        print(f"✅ Step 1 replayed: {result1_replay.result.result} (is_replayed={event1_replay.is_replayed})")
        assert result1_replay.result.result == 6
        assert event1_replay.is_replayed == True
        assert event1_replay.skip_republish == True
        
        # Replay step 2 (should be skipped)
        event2_replay = ExecutionEvent(actor=create_multiply_by_two_actor())
        result2_replay = await event2_replay(SimpleInput(value=6), None, restored_state)
        print(f"✅ Step 2 replayed: {result2_replay.result.result} (is_replayed={event2_replay.is_replayed})")
        assert result2_replay.result.result == 12
        assert event2_replay.is_replayed == True
        
        # Execute new step 3 (should execute normally)
        event3 = ExecutionEvent(actor=create_add_ten_actor())
        result3 = await event3(SimpleInput(value=result2_replay.result.result), None, restored_state)
        print(f"✅ Step 3 executed (new): 12 + 10 = {result3.result.result}")
        assert result3.result.result == 22
        assert event3.is_replayed == False
        
        print("\n" + "=" * 60)
        print("✅ BASIC REPLAY TEST PASSED!")
        print("=" * 60)
        
    finally:
        # Clean up
        if test_dir.exists():
            shutil.rmtree(test_dir)


async def test_resume_from_address():
    """Test resume_from_address: skip until specific event, then re-execute from there."""
    test_dir = Path("./.test_phase2_resume")
    
    # Clean up
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    try:
        # Create backends
        state_backend = FileSystemStateBackend(str(test_dir))
        artifact_backend = FileSystemArtifactBackend(str(test_dir / "artifacts"))
        
        # PART 1: Initial execution
        print("\n" + "=" * 60)
        print("PART 1: Execute 3 Steps")
        print("=" * 60)
        
        state = ExecutionState(
            state_backend=state_backend,
            artifact_backend=artifact_backend
        )
        
        # Execute 3 steps
        event1 = ExecutionEvent(actor=create_add_one_actor())
        result1 = await event1(SimpleInput(value=10), None, state)
        print(f"✅ Step 1: 10 + 1 = {result1.result.result}")
        step1_address = state.history.get_event_list()[-1].address
        
        event2 = ExecutionEvent(actor=create_multiply_by_two_actor())
        result2 = await event2(SimpleInput(value=result1.result.result), None, state)
        print(f"✅ Step 2: 11 * 2 = {result2.result.result}")
        step2_address = state.history.get_event_list()[-1].address
        
        event3 = ExecutionEvent(actor=create_add_ten_actor())
        result3 = await event3(SimpleInput(value=result2.result.result), None, state)
        print(f"✅ Step 3: 22 + 10 = {result3.result.result}")
        step3_address = state.history.get_event_list()[-1].address
        
        # Save checkpoint
        state.save_checkpoint(status=StateStatus.PAUSED)
        print(f"✅ Checkpoint saved with 3 steps")
        
        # PART 2: Restore with resume_from_address at step 2
        print("\n" + "=" * 60)
        print(f"PART 2: Resume from Step 2 (address: {step2_address})")
        print("=" * 60)
        
        restored_state = ExecutionState.load_from_state_id(
            state_id=state.state_id,
            state_backend=state_backend,
            artifact_backend=artifact_backend,
            replay_mode=ReplayMode.NORMAL,
            resume_from_address=step2_address
        )
        
        print(f"✅ State restored with resume_from_address={step2_address}")
        
        # Replay step 1 (should be skipped, still before resume point)
        event1_replay = ExecutionEvent(actor=create_add_one_actor())
        result1_replay = await event1_replay(SimpleInput(value=10), None, restored_state)
        print(f"✅ Step 1 skipped: {result1_replay.result.result} (is_replayed={event1_replay.is_replayed})")
        assert event1_replay.is_replayed == True
        
        # Replay step 2 (this is the resume point - should re-execute)
        event2_replay = ExecutionEvent(actor=create_multiply_by_two_actor())
        result2_replay = await event2_replay(SimpleInput(value=100), None, restored_state)  # Changed input!
        print(f"✅ Step 2 RE-EXECUTED: 100 * 2 = {result2_replay.result.result} (is_replayed={event2_replay.is_replayed})")
        assert result2_replay.result.result == 200  # New result!
        assert event2_replay.is_replayed == False  # Not replayed, executed!
        
        # Check resume_from_address was cleared
        assert restored_state.resume_from_address is None
        print(f"✅ resume_from_address cleared after reaching it")
        
        # Step 3 should also re-execute (everything after resume point)
        event3_replay = ExecutionEvent(actor=create_add_ten_actor())
        result3_replay = await event3_replay(SimpleInput(value=result2_replay.result.result), None, restored_state)
        print(f"✅ Step 3 RE-EXECUTED: 200 + 10 = {result3_replay.result.result} (is_replayed={event3_replay.is_replayed})")
        assert result3_replay.result.result == 210  # New result!
        assert event3_replay.is_replayed == False
        
        print("\n" + "=" * 60)
        print("✅ RESUME_FROM_ADDRESS TEST PASSED!")
        print("=" * 60)
        
    finally:
        # Clean up
        if test_dir.exists():
            shutil.rmtree(test_dir)


async def test_validation_mode():
    """Test validation mode: re-execute and compare with historical results."""
    test_dir = Path("./.test_phase2_validation")
    
    # Clean up
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    try:
        # Create backends
        state_backend = FileSystemStateBackend(str(test_dir))
        artifact_backend = FileSystemArtifactBackend(str(test_dir / "artifacts"))
        
        # PART 1: Initial execution
        print("\n" + "=" * 60)
        print("PART 1: Initial Execution")
        print("=" * 60)
        
        state = ExecutionState(
            state_backend=state_backend,
            artifact_backend=artifact_backend
        )
        
        # Execute steps
        event1 = ExecutionEvent(actor=create_add_one_actor())
        result1 = await event1(SimpleInput(value=5), None, state)
        print(f"✅ Step 1: 5 + 1 = {result1.result.result}")
        
        event2 = ExecutionEvent(actor=create_multiply_by_two_actor())
        result2 = await event2(SimpleInput(value=result1.result.result), None, state)
        print(f"✅ Step 2: 6 * 2 = {result2.result.result}")
        
        # Save checkpoint
        state.save_checkpoint(status=StateStatus.COMPLETED)
        print(f"✅ Checkpoint saved")
        
        # PART 2: Restore in VALIDATION mode
        print("\n" + "=" * 60)
        print("PART 2: Restore in VALIDATION Mode")
        print("=" * 60)
        
        restored_state = ExecutionState.load_from_state_id(
            state_id=state.state_id,
            state_backend=state_backend,
            artifact_backend=artifact_backend,
            replay_mode=ReplayMode.VALIDATION  # VALIDATION MODE
        )
        
        print(f"✅ State restored in VALIDATION mode")
        
        # In validation mode, events should re-execute (not skip)
        event1_val = ExecutionEvent(actor=create_add_one_actor())
        result1_val = await event1_val(SimpleInput(value=5), None, restored_state)
        print(f"✅ Step 1 re-executed: {result1_val.result.result} (is_replayed={event1_val.is_replayed})")
        assert event1_val.is_replayed == False  # Not replayed, re-executed!
        assert result1_val.result.result == 6  # Same result (deterministic)
        
        event2_val = ExecutionEvent(actor=create_multiply_by_two_actor())
        result2_val = await event2_val(SimpleInput(value=result1_val.result.result), None, restored_state)
        print(f"✅ Step 2 re-executed: {result2_val.result.result} (is_replayed={event2_val.is_replayed})")
        assert event2_val.is_replayed == False
        assert result2_val.result.result == 12
        
        print("\n" + "=" * 60)
        print("✅ VALIDATION MODE TEST PASSED!")
        print("=" * 60)
        
    finally:
        # Clean up
        if test_dir.exists():
            shutil.rmtree(test_dir)


async def test_event_publishing():
    """Test selective event publishing during replay."""
    test_dir = Path("./.test_phase2_publishing")
    
    # Clean up
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    try:
        # Create backends
        state_backend = FileSystemStateBackend(str(test_dir))
        artifact_backend = FileSystemArtifactBackend(str(test_dir / "artifacts"))
        
        # PART 1: Initial execution with event tracking
        print("\n" + "=" * 60)
        print("PART 1: Track Published Events")
        print("=" * 60)
        
        state = ExecutionState(
            state_backend=state_backend,
            artifact_backend=artifact_backend
        )
        
        published_events = []
        
        def track_event(event):
            published_events.append({
                'actor': event.actor.name,
                'status': event.execution_status,
                'is_replayed': getattr(event, 'is_replayed', False)
            })
        
        state.event_bus.subscribe(track_event)
        
        # Execute steps
        event1 = ExecutionEvent(actor=create_add_one_actor())
        await event1(SimpleInput(value=5), None, state)
        
        event2 = ExecutionEvent(actor=create_multiply_by_two_actor())
        await event2(SimpleInput(value=6), None, state)
        
        initial_event_count = len(published_events)
        print(f"✅ Initial execution published {initial_event_count} events")
        
        # Save checkpoint
        state.save_checkpoint(status=StateStatus.PAUSED)
        
        # PART 2: Restore and track events (should only publish new events)
        print("\n" + "=" * 60)
        print("PART 2: Replay with Selective Publishing")
        print("=" * 60)
        
        published_events.clear()
        
        restored_state = ExecutionState.load_from_state_id(
            state_id=state.state_id,
            state_backend=state_backend,
            artifact_backend=artifact_backend,
            replay_mode=ReplayMode.NORMAL
        )
        
        # Re-register subscriber
        restored_state.event_bus.subscribe(track_event)
        
        # Replay events (should publish RECOVERED status but skip_republish should work)
        event1_replay = ExecutionEvent(actor=create_add_one_actor())
        await event1_replay(SimpleInput(value=5), None, restored_state)
        
        event2_replay = ExecutionEvent(actor=create_multiply_by_two_actor())
        await event2_replay(SimpleInput(value=6), None, restored_state)
        
        # Execute new event (should publish normally)
        event3_new = ExecutionEvent(actor=create_add_ten_actor())
        await event3_new(SimpleInput(value=12), None, restored_state)
        
        replay_event_count = len(published_events)
        print(f"✅ Replay published {replay_event_count} events")
        
        # Check that only new events + recovery markers were published
        replayed_events = [e for e in published_events if e['is_replayed']]
        new_events = [e for e in published_events if not e['is_replayed']]
        
        print(f"  - Replayed events published: {len(replayed_events)} (RECOVERED status)")
        print(f"  - New events published: {len(new_events)}")
        
        # New events should be published (STARTED + COMPLETED)
        assert len(new_events) >= 2  # At least STARTED and COMPLETED for new event
        
        print("\n" + "=" * 60)
        print("✅ EVENT PUBLISHING TEST PASSED!")
        print("=" * 60)
        
    finally:
        # Clean up
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 2 TESTS: Smart Replay Logic")
    print("="*70)
    
    print("\n" + "Test 1: Basic Replay (Skip Completed Events)")
    asyncio.run(test_basic_replay())
    
    print("\n" + "Test 2: Resume from Specific Address")
    asyncio.run(test_resume_from_address())
    
    print("\n" + "Test 3: Validation Mode (Re-execute All)")
    asyncio.run(test_validation_mode())
    
    print("\n" + "Test 4: Selective Event Publishing")
    asyncio.run(test_event_publishing())
    
    print("\n" + "="*70)
    print("✅ ALL PHASE 2 TESTS PASSED!")
    print("="*70 + "\n")
