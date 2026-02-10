"""
Demo Script: Phase 4 - State Persistence & Replay Integration

Demonstrates:
1. Saving execution state with retry configuration
2. Restoring state and continuing execution
3. Replay with retry events
4. Resume from specific retry event
5. Retry statistics preservation across save/restore
"""
import asyncio
from pydantic import BaseModel
from rh_agents.core.execution import ExecutionState
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import Tool
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.retry import RetryConfig, BackoffStrategy
from rh_agents.core.state_recovery import StateStatus, ReplayMode, StateMetadata
from rh_agents.bus_handlers import EventPrinter


# ===== In-Memory State Backend for Demo =====

class InMemoryStateBackend:
    """Simple in-memory state backend for demonstration."""
    
    def __init__(self):
        self.snapshots = {}
    
    def save_state(self, snapshot):
        self.snapshots[snapshot.state_id] = snapshot
        print(f"✅ Saved state: {snapshot.state_id[:8]}... (status={snapshot.status.value})")
        return True
    
    def load_state(self, state_id):
        snapshot = self.snapshots.get(state_id)
        if snapshot:
            print(f"✅ Loaded state: {state_id[:8]}... (status={snapshot.status.value})")
        return snapshot


# ===== Demo Helpers =====

class APIRequest(BaseModel):
    endpoint: str
    data: str


async def flaky_api_handler(input_data, context, state):
    """API that fails first, succeeds on retry."""
    # Check if this is a retry attempt
    if state.current_execution and state.current_execution.retry_attempt > 0:
        return f"Success: {input_data.data}"
    else:
        raise ConnectionError("API temporarily unavailable")


async def unreliable_service_handler(input_data, context, state):
    """Service that always fails (for demo purposes)."""
    attempt = state.current_execution.retry_attempt if state.current_execution else 0
    raise TimeoutError(f"Service timeout (attempt {attempt})")


def print_separator(title: str):
    """Print a formatted section separator."""
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80 + "\n")


def print_state_info(state: ExecutionState):
    """Print information about execution state."""
    print(f"📊 State ID: {state.state_id[:8]}...")
    print(f"📦 Events in history: {len(state.history.get_event_list())}")
    
    # Count retry events
    event_list = state.history.get_event_list()
    retrying_events = [e for e in event_list if e.get("execution_status") == ExecutionStatus.RETRYING.value]
    print(f"↻  Retry events: {len(retrying_events)}")
    
    # Show retry configs
    if state.default_retry_config:
        config = state.default_retry_config
        if isinstance(config, dict):
            print(f"⚙️  Default retry config: max_attempts={config.get('max_attempts')}")
        else:
            print(f"⚙️  Default retry config: max_attempts={config.max_attempts}")
    print()


# ===== Demo 1: Basic State Persistence with Retry Config =====

async def demo1_basic_persistence():
    print_separator("Demo 1: Basic State Persistence with Retry Configuration")
    
    # Create state with retry configuration
    state = ExecutionState()
    backend = InMemoryStateBackend()
    state.state_backend = backend
    
    # Configure retry at state level
    state.default_retry_config = RetryConfig(max_attempts=3, initial_delay=1.0)
    state.retry_config_by_actor_type[EventType.TOOL_CALL] = RetryConfig(
        max_attempts=5,
        backoff_strategy=BackoffStrategy.EXPONENTIAL
    )
    
    print("📝 Configured state-level retry settings:")
    print(f"   • Default: max_attempts=3")
    print(f"   • TOOL_CALL: max_attempts=5, strategy=EXPONENTIAL\n")
    
    # Save checkpoint
    state.save_checkpoint(status=StateStatus.RUNNING)
    
    print(f"\n💾 State saved to backend")
    
    # Restore from backend
    print(f"\n🔄 Restoring state from backend...\n")
    restored_state = ExecutionState.load_from_state_id(state.state_id, backend)
    
    # Verify retry configs were restored
    print("✅ State restored successfully!")
    print(f"   • Default retry config preserved: {restored_state.default_retry_config is not None}")
    print(f"   • Type-specific configs preserved: {EventType.TOOL_CALL in restored_state.retry_config_by_actor_type}")
    print()


# ===== Demo 2: Save/Restore with Retry Events =====

async def demo2_retry_events_persistence():
    print_separator("Demo 2: Save/Restore with Retry Events in History")
    
    state = ExecutionState()
    backend = InMemoryStateBackend()
    state.state_backend = backend
    
    # Create tool with retry config
    tool = Tool(
        name="flaky_api",
        description="Flaky API client",
        input_model=APIRequest,
        handler=flaky_api_handler,
        retry_config=RetryConfig(max_attempts=3, initial_delay=0.5)
    )
    
    print("🔧 Simulating execution with retry...\n")
    
    # Simulate execution: STARTED -> FAILED -> RETRYING -> STARTED -> COMPLETED
    events = [
        ExecutionEvent(
            actor=tool,
            address="flaky_api::tool_call",
            execution_status=ExecutionStatus.STARTED,
            retry_attempt=0
        ),
        ExecutionEvent(
            actor=tool,
            address="flaky_api::tool_call",
            execution_status=ExecutionStatus.FAILED,
            retry_attempt=0,
            message="API temporarily unavailable"
        ),
        ExecutionEvent(
            actor=tool,
            address="flaky_api::tool_call",
            execution_status=ExecutionStatus.RETRYING,
            retry_attempt=1,
            is_retry=True,
            original_error="API temporarily unavailable",
            retry_delay=0.5
        ),
        ExecutionEvent(
            actor=tool,
            address="flaky_api::tool_call",
            execution_status=ExecutionStatus.STARTED,
            retry_attempt=1,
            is_retry=True
        ),
        ExecutionEvent(
            actor=tool,
            address="flaky_api::tool_call",
            execution_status=ExecutionStatus.COMPLETED,
            retry_attempt=1,
            is_retry=True,
            result="Success: test data"
        )
    ]
    
    for event in events:
        state.history.add(event)
    
    print(f"📊 Execution complete:")
    print(f"   • Total events: {len(state.history.get_event_list())}")
    print(f"   • Result: {events[-1].result}\n")
    
    # Save state
    state.save_checkpoint(status=StateStatus.COMPLETED)
    
    # Restore state
    print(f"\n🔄 Restoring state from backend...\n")
    restored_state = ExecutionState.load_from_state_id(state.state_id, backend)
    
    # Verify all retry events are preserved
    event_list = restored_state.history.get_event_list()
    retrying_events = [e for e in event_list if e.get("execution_status") == ExecutionStatus.RETRYING.value]
    
    print("✅ State restored successfully!")
    print(f"   • Total events preserved: {len(event_list)}")
    print(f"   • Retry events preserved: {len(retrying_events)}")
    
    # Show retry details
    if retrying_events:
        retry = retrying_events[0]
        print(f"\n📋 Retry event details:")
        print(f"   • Retry attempt: {retry.get('retry_attempt')}")
        print(f"   • Original error: {retry.get('original_error')}")
        print(f"   • Retry delay: {retry.get('retry_delay')}s")
    print()


# ===== Demo 3: Resume from Retry Event =====

async def demo3_resume_from_retry():
    print_separator("Demo 3: Resume from Retry Event")
    
    state = ExecutionState()
    backend = InMemoryStateBackend()
    state.state_backend = backend
    
    # Create unreliable service
    tool = Tool(
        name="unreliable_service",
        description="Unreliable service",
        input_model=APIRequest,
        handler=unreliable_service_handler,
        retry_config=RetryConfig(max_attempts=3)
    )
    
    print("🔧 Simulating interrupted execution during retry wait...\n")
    
    # Simulate execution interrupted during retry: STARTED -> FAILED -> RETRYING
    events = [
        ExecutionEvent(
            actor=tool,
            address="unreliable_service::tool_call",
            execution_status=ExecutionStatus.STARTED,
            retry_attempt=0
        ),
        ExecutionEvent(
            actor=tool,
            address="unreliable_service::tool_call",
            execution_status=ExecutionStatus.FAILED,
            retry_attempt=0,
            message="Service unavailable"
        ),
        ExecutionEvent(
            actor=tool,
            address="unreliable_service::tool_call",
            execution_status=ExecutionStatus.RETRYING,
            retry_attempt=1,
            is_retry=True,
            original_error="Service unavailable",
            retry_delay=1.0
        )
    ]
    
    for event in events:
        state.history.add(event)
    
    print(f"📊 Execution interrupted:")
    print(f"   • Events before interrupt: {len(state.history.get_event_list())}")
    print(f"   • Last event: RETRYING (waiting for retry delay)\n")
    
    # Save checkpoint (interrupted state)
    state.save_checkpoint(status=StateStatus.PAUSED)
    
    # Restore with resume_from_address
    print(f"\n🔄 Restoring with resume_from_address...\n")
    restored_state = ExecutionState.load_from_state_id(
        state.state_id,
        backend,
        replay_mode=ReplayMode.NORMAL,
        resume_from_address="unreliable_service::tool_call"
    )
    
    print("✅ State restored with resume point!")
    print(f"   • Resume from: {restored_state.resume_from_address}")
    print(f"   • Replay mode: {restored_state.replay_mode.value}")
    
    # Verify retry event is accessible
    event_list = restored_state.history.get_event_list()
    retrying_event = [e for e in event_list if e.get("execution_status") == ExecutionStatus.RETRYING.value][0]
    
    print(f"\n📋 Retry state preserved:")
    print(f"   • Retry attempt: {retrying_event.get('retry_attempt')}")
    print(f"   • Original error: {retrying_event.get('original_error')}")
    print(f"   • Ready to continue from: attempt {retrying_event.get('retry_attempt') + 1}")
    print()


# ===== Demo 4: Full Workflow with EventPrinter =====

async def demo4_full_workflow_with_printer():
    print_separator("Demo 4: Complete Save/Restore Workflow with Visual Display")
    
    # Create state with printer
    state = ExecutionState()
    backend = InMemoryStateBackend()
    state.state_backend = backend
    
    printer = EventPrinter(show_address=True, show_timestamp=False)
    state.event_bus.subscribe(printer.print_event)
    
    # Configure retry
    state.default_retry_config = RetryConfig(max_attempts=3, initial_delay=0.5)
    
    # Create tool
    tool = Tool(
        name="demo_api",
        description="Demo API",
        input_model=APIRequest,
        handler=flaky_api_handler,
        retry_config=RetryConfig(max_attempts=2, initial_delay=0.3)
    )
    
    print("🎬 Executing tool with retry (will fail once, then succeed)...\n")
    
    # Simulate execution with events published to printer
    events = [
        ExecutionEvent(
            actor=tool,
            address="demo_api::tool_call",
            execution_status=ExecutionStatus.STARTED,
            retry_attempt=0
        ),
        ExecutionEvent(
            actor=tool,
            address="demo_api::tool_call",
            execution_status=ExecutionStatus.FAILED,
            retry_attempt=0,
            message="Connection error"
        ),
        ExecutionEvent(
            actor=tool,
            address="demo_api::tool_call",
            execution_status=ExecutionStatus.RETRYING,
            retry_attempt=1,
            is_retry=True,
            original_error="Connection error",
            retry_delay=0.3
        ),
        ExecutionEvent(
            actor=tool,
            address="demo_api::tool_call",
            execution_status=ExecutionStatus.STARTED,
            retry_attempt=1,
            is_retry=True
        ),
        ExecutionEvent(
            actor=tool,
            address="demo_api::tool_call",
            execution_status=ExecutionStatus.COMPLETED,
            retry_attempt=1,
            is_retry=True,
            result="API success"
        )
    ]
    
    for event in events:
        state.history.add(event)
        await state.event_bus.publish(event)
    
    # Save state
    print(f"\n💾 Saving state...\n")
    state.save_checkpoint(status=StateStatus.COMPLETED)
    
    # Print summary
    printer.print_summary()
    
    # Restore and verify
    print(f"\n🔄 Restoring state...\n")
    restored_state = ExecutionState.load_from_state_id(state.state_id, backend)
    
    print_state_info(restored_state)
    
    print("✅ Complete workflow demonstrated!")
    print(f"   • Retry events preserved across save/restore")
    print(f"   • Retry statistics available for analysis")
    print(f"   • State can be resumed from any retry point")
    print()


# ===== Main Demo Runner =====

async def main():
    print("\n" + "=" * 80)
    print("PHASE 4 DEMO: State Persistence & Replay Integration")
    print("=" * 80)
    
    await demo1_basic_persistence()
    await asyncio.sleep(0.5)
    
    await demo2_retry_events_persistence()
    await asyncio.sleep(0.5)
    
    await demo3_resume_from_retry()
    await asyncio.sleep(0.5)
    
    await demo4_full_workflow_with_printer()
    
    print("=" * 80)
    print("✨ Demo Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("• Retry configurations are fully preserved in state snapshots")
    print("• All retry events (RETRYING, attempts, errors) are saved and restored")
    print("• State can be resumed from any retry event")
    print("• Retry statistics are preserved across save/restore cycles")
    print("• Complete integration with EventPrinter for visual feedback")
    print()


if __name__ == "__main__":
    asyncio.run(main())
