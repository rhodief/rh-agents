"""
Test Suite: Phase 4 - State Persistence & Replay Integration

Tests retry data serialization, state save/restore, and replay scenarios.
"""
import pytest
import asyncio
from typing import Optional
from rh_agents.core.execution import ExecutionState
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import Tool
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.retry import RetryConfig, BackoffStrategy
from rh_agents.core.state_recovery import StateSnapshot, StateStatus, ReplayMode
from rh_agents.core.state_backend import StateBackend


# ===== In-Memory State Backend for Testing =====

class InMemoryStateBackend(StateBackend):
    """Simple in-memory state backend for testing."""
    
    def __init__(self):
        self.snapshots: dict[str, StateSnapshot] = {}
    
    def save_state(self, snapshot: StateSnapshot) -> bool:
        """Save snapshot to memory."""
        self.snapshots[snapshot.state_id] = snapshot
        return True
    
    def load_state(self, state_id: str) -> Optional[StateSnapshot]:
        """Load snapshot from memory."""
        return self.snapshots.get(state_id)
    
    def update_state(self, snapshot: StateSnapshot) -> bool:
        """Update existing snapshot in memory."""
        if snapshot.state_id in self.snapshots:
            self.snapshots[snapshot.state_id] = snapshot
            return True
        return False
    
    def list_states(
        self,
        status: Optional[StateStatus] = None,
        tags: Optional[list[str]] = None,
        limit: int = 100
    ) -> list[StateSnapshot]:
        """List all snapshots in memory."""
        results = list(self.snapshots.values())
        
        if status:
            results = [s for s in results if s.status == status]
        
        if tags:
            results = [s for s in results if any(tag in s.metadata.tags for tag in tags)]
        
        return results[:limit]
    
    def delete_state(self, state_id: str) -> bool:
        """Delete snapshot from memory."""
        if state_id in self.snapshots:
            del self.snapshots[state_id]
            return True
        return False


# ===== Test: Basic retry config serialization =====

def test_retry_config_serialization():
    """Test that RetryConfig serializes and deserializes correctly."""
    config = RetryConfig(
        max_attempts=5,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        initial_delay=2.0,
        max_delay=120.0,
        jitter=True,
        enabled=True
    )
    
    # Serialize to dict
    config_dict = config.model_dump()
    
    # Verify key fields
    assert config_dict["max_attempts"] == 5
    assert config_dict["backoff_strategy"] == "exponential"
    assert config_dict["initial_delay"] == 2.0
    assert config_dict["max_delay"] == 120.0
    assert config_dict["jitter"] is True
    assert config_dict["enabled"] is True
    
    # Deserialize back
    restored_config = RetryConfig(**config_dict)
    assert restored_config.max_attempts == 5
    assert restored_config.backoff_strategy == BackoffStrategy.EXPONENTIAL
    assert restored_config.initial_delay == 2.0


# ===== Test: ExecutionState retry fields in snapshot =====

def test_execution_state_retry_fields_in_snapshot():
    """Test that ExecutionState retry config fields are included in snapshots."""
    state = ExecutionState()
    
    # Set retry configurations
    state.default_retry_config = RetryConfig(max_attempts=5, initial_delay=2.0)
    state.retry_config_by_actor_type[EventType.TOOL_CALL] = RetryConfig(
        max_attempts=3,
        backoff_strategy=BackoffStrategy.LINEAR,
        initial_delay=1.0
    )
    
    # Create snapshot
    snapshot = state.to_snapshot(status=StateStatus.RUNNING)
    
    # Verify retry fields are in serialized state
    exec_state = snapshot.execution_state
    assert "default_retry_config" in exec_state
    assert "retry_config_by_actor_type" in exec_state
    
    # Verify default_retry_config was serialized
    default_config = exec_state["default_retry_config"]
    assert default_config["max_attempts"] == 5
    assert default_config["initial_delay"] == 2.0
    
    # Verify retry_config_by_actor_type was serialized
    type_configs = exec_state["retry_config_by_actor_type"]
    assert EventType.TOOL_CALL.value in type_configs
    tool_config = type_configs[EventType.TOOL_CALL.value]
    assert tool_config["max_attempts"] == 3
    assert tool_config["backoff_strategy"] == "linear"
    assert tool_config["initial_delay"] == 1.0


# ===== Test: ExecutionEvent retry fields in snapshot =====

@pytest.mark.asyncio
async def test_execution_event_retry_fields_in_snapshot():
    """Test that ExecutionEvent retry fields are preserved in snapshots."""
    state = ExecutionState()
    
    # Create a tool with retry config
    from pydantic import BaseModel as PydanticBaseModel
    class DummyInput(PydanticBaseModel):
        text: str
    
    async def dummy_handler(x, ctx, st):
        return "result"
    
    tool = Tool(
        name="test_tool",
        description="Test tool",
        input_model=DummyInput,
        handler=dummy_handler,
        retry_config=RetryConfig(max_attempts=3, initial_delay=1.0)
    )
    
    # Create event with retry data
    event = ExecutionEvent(
        actor=tool,
        retry_config=tool.retry_config,
        retry_attempt=2,
        is_retry=True,
        original_error="Connection timeout",
        retry_delay=2.0
    )
    event.address = "test_tool::tool_call"
    event.execution_status = ExecutionStatus.RETRYING
    
    # Add to history
    state.history.add(event)
    
    # Create snapshot
    snapshot = state.to_snapshot()
    
    # Verify event is in snapshot with retry fields
    events = snapshot.execution_state["history"]["events"]
    assert len(events) == 1
    
    event_dict = events[0]
    assert event_dict["retry_attempt"] == 2
    assert event_dict["is_retry"] is True
    assert event_dict["original_error"] == "Connection timeout"
    assert event_dict["retry_delay"] == 2.0
    
    # Verify retry_config was serialized
    assert "retry_config" in event_dict
    retry_config_dict = event_dict["retry_config"]
    assert retry_config_dict["max_attempts"] == 3
    assert retry_config_dict["initial_delay"] == 1.0


# ===== Test: State restoration with retry configs =====

def test_state_restoration_with_retry_configs():
    """Test that retry configs are properly restored from snapshot."""
    # Create original state with retry configs
    original_state = ExecutionState()
    original_state.default_retry_config = RetryConfig(max_attempts=5, initial_delay=2.0)
    original_state.retry_config_by_actor_type[EventType.TOOL_CALL] = RetryConfig(
        max_attempts=3,
        backoff_strategy=BackoffStrategy.EXPONENTIAL
    )
    original_state.retry_config_by_actor_type[EventType.LLM_CALL] = RetryConfig(
        max_attempts=4,
        backoff_strategy=BackoffStrategy.LINEAR
    )
    
    # Create snapshot
    snapshot = original_state.to_snapshot()
    
    # Restore from snapshot
    restored_state = ExecutionState.from_snapshot(snapshot)
    
    # Verify default_retry_config was restored (as dict)
    assert restored_state.default_retry_config is not None
    default_config = restored_state.default_retry_config
    # After restoration, it's a dict, not a RetryConfig object
    assert isinstance(default_config, dict)
    assert default_config["max_attempts"] == 5
    assert default_config["initial_delay"] == 2.0
    
    # Verify retry_config_by_actor_type was restored
    assert EventType.TOOL_CALL in restored_state.retry_config_by_actor_type
    assert EventType.LLM_CALL in restored_state.retry_config_by_actor_type
    
    tool_config = restored_state.retry_config_by_actor_type[EventType.TOOL_CALL]
    assert isinstance(tool_config, dict)
    assert tool_config["max_attempts"] == 3
    assert tool_config["backoff_strategy"] == "exponential"


# ===== Test: Replay with retry events =====

@pytest.mark.asyncio
async def test_replay_with_retry_events():
    """Test that retry events are properly handled during replay."""
    state = ExecutionState()
    backend = InMemoryStateBackend()
    state.state_backend = backend
    
    # Create tool with retry config
    from pydantic import BaseModel as PydanticBaseModel
    class DummyInput(PydanticBaseModel):
        text: str
    
    async def service_handler(x, ctx, st):
        return "success"
    
    tool = Tool(
        name="flaky_service",
        description="Flaky test service",
        input_model=DummyInput,
        handler=service_handler,
        retry_config=RetryConfig(max_attempts=3, initial_delay=0.5)
    )
    
    # Simulate execution with retry: STARTED -> FAILED -> RETRYING -> STARTED -> COMPLETED
    events = [
        ExecutionEvent(
            actor=tool,
            address="flaky_service::tool_call",
            execution_status=ExecutionStatus.STARTED,
            retry_config=tool.retry_config,
            retry_attempt=0
        ),
        ExecutionEvent(
            actor=tool,
            address="flaky_service::tool_call",
            execution_status=ExecutionStatus.FAILED,
            retry_config=tool.retry_config,
            retry_attempt=0,
            message="API timeout"
        ),
        ExecutionEvent(
            actor=tool,
            address="flaky_service::tool_call",
            execution_status=ExecutionStatus.RETRYING,
            retry_config=tool.retry_config,
            retry_attempt=1,
            is_retry=True,
            original_error="API timeout",
            retry_delay=0.5
        ),
        ExecutionEvent(
            actor=tool,
            address="flaky_service::tool_call",
            execution_status=ExecutionStatus.STARTED,
            retry_config=tool.retry_config,
            retry_attempt=1,
            is_retry=True
        ),
        ExecutionEvent(
            actor=tool,
            address="flaky_service::tool_call",
            execution_status=ExecutionStatus.COMPLETED,
            retry_config=tool.retry_config,
            retry_attempt=1,
            is_retry=True,
            result="success"
        )
    ]
    
    # Add all events to history
    for event in events:
        state.history.add(event)
    
    # Save checkpoint
    state.save_checkpoint(status=StateStatus.COMPLETED)
    
    # Load from backend
    restored_state = ExecutionState.load_from_state_id(
        state.state_id,
        backend,
        replay_mode=ReplayMode.NORMAL
    )
    
    assert restored_state is not None
    
    # Verify all retry events are in history
    event_list = restored_state.history.get_event_list()
    assert len(event_list) == 5
    
    # Verify retry information is preserved
    retrying_event = [e for e in event_list if e.get("execution_status") == ExecutionStatus.RETRYING.value][0]
    assert retrying_event["retry_attempt"] == 1
    assert retrying_event["is_retry"] is True
    assert retrying_event["original_error"] == "API timeout"
    assert retrying_event["retry_delay"] == 0.5
    
    # Verify final completion
    final_event = event_list[-1]
    assert final_event["execution_status"] == ExecutionStatus.COMPLETED.value
    assert final_event["result"] == "success"


# ===== Test: Resume from retry event =====

@pytest.mark.asyncio
async def test_resume_from_retry_event():
    """Test resuming execution from a RETRYING event."""
    state = ExecutionState()
    backend = InMemoryStateBackend()
    state.state_backend = backend
    
    from pydantic import BaseModel as PydanticBaseModel
    class DummyInput(PydanticBaseModel):
        text: str
    
    async def unreliable_handler(x, ctx, st):
        return "result"
    
    tool = Tool(
        name="unreliable_service",
        description="Unreliable test service",
        input_model=DummyInput,
        handler=unreliable_handler,
        retry_config=RetryConfig(max_attempts=3)
    )
    
    # Simulate execution interrupted during retry wait: STARTED -> FAILED -> RETRYING
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
    
    # Save checkpoint (interrupted during retry wait)
    state.save_checkpoint(status=StateStatus.PAUSED)
    
    # Restore with resume_from_address pointing to the retry event
    restored_state = ExecutionState.load_from_state_id(
        state.state_id,
        backend,
        replay_mode=ReplayMode.NORMAL,
        resume_from_address="unreliable_service::tool_call"
    )
    
    assert restored_state is not None
    assert restored_state.resume_from_address == "unreliable_service::tool_call"
    
    # Verify retry event is in history with retry data
    event_list = restored_state.history.get_event_list()
    retrying_event = [e for e in event_list if e.get("execution_status") == ExecutionStatus.RETRYING.value][0]
    
    assert retrying_event["retry_attempt"] == 1
    assert retrying_event["original_error"] == "Service unavailable"
    assert retrying_event["retry_delay"] == 1.0


# ===== Test: Retry statistics preservation =====

@pytest.mark.asyncio
async def test_retry_statistics_preservation():
    """Test that retry statistics can be computed from restored state."""
    state = ExecutionState()
    backend = InMemoryStateBackend()
    state.state_backend = backend
    
    from pydantic import BaseModel as PydanticBaseModel
    class DummyInput(PydanticBaseModel):
        text: str
    
    async def handler_a(x, ctx, st):
        return "ok"
    
    async def handler_b(x, ctx, st):
        return "ok"
    
    tool1 = Tool(name="service_a", description="Service A", input_model=DummyInput, handler=handler_a)
    tool2 = Tool(name="service_b", description="Service B", input_model=DummyInput, handler=handler_b)
    
    # Service A: 2 failures, 2 retries, then success
    for i in range(2):
        state.history.add(ExecutionEvent(
            actor=tool1,
            address="service_a::tool_call",
            execution_status=ExecutionStatus.FAILED,
            retry_attempt=i
        ))
        state.history.add(ExecutionEvent(
            actor=tool1,
            address="service_a::tool_call",
            execution_status=ExecutionStatus.RETRYING,
            retry_attempt=i+1,
            is_retry=True
        ))
    
    state.history.add(ExecutionEvent(
        actor=tool1,
        address="service_a::tool_call",
        execution_status=ExecutionStatus.COMPLETED,
        retry_attempt=2,
        result="ok"
    ))
    
    # Service B: 1 failure, 1 retry, then success
    state.history.add(ExecutionEvent(
        actor=tool2,
        address="service_b::tool_call",
        execution_status=ExecutionStatus.FAILED,
        retry_attempt=0
    ))
    state.history.add(ExecutionEvent(
        actor=tool2,
        address="service_b::tool_call",
        execution_status=ExecutionStatus.RETRYING,
        retry_attempt=1,
        is_retry=True
    ))
    state.history.add(ExecutionEvent(
        actor=tool2,
        address="service_b::tool_call",
        execution_status=ExecutionStatus.COMPLETED,
        retry_attempt=1,
        result="ok"
    ))
    
    # Save and restore
    state.save_checkpoint()
    restored_state = ExecutionState.load_from_state_id(state.state_id, backend)
    
    # Compute retry statistics from restored state
    event_list = restored_state.history.get_event_list()
    retrying_events = [e for e in event_list if e.get("execution_status") == ExecutionStatus.RETRYING.value]
    
    assert len(retrying_events) == 3  # Total retry attempts
    
    # Count unique addresses that retried
    retried_addresses = set(e.get("address") for e in retrying_events)
    assert len(retried_addresses) == 2  # service_a and service_b
    
    # Verify service_a had 2 retries
    service_a_retries = [e for e in retrying_events if e.get("address") == "service_a::tool_call"]
    assert len(service_a_retries) == 2
    
    # Verify service_b had 1 retry
    service_b_retries = [e for e in retrying_events if e.get("address") == "service_b::tool_call"]
    assert len(service_b_retries) == 1


# ===== Test: Full end-to-end persistence workflow =====

@pytest.mark.asyncio
async def test_full_persistence_workflow_with_retry():
    """Test complete save/restore workflow with retry configuration."""
    # Setup initial state
    state = ExecutionState()
    backend = InMemoryStateBackend()
    state.state_backend = backend
    
    # Configure retry at state level
    state.default_retry_config = RetryConfig(max_attempts=3, initial_delay=1.0)
    state.retry_config_by_actor_type[EventType.TOOL_CALL] = RetryConfig(
        max_attempts=5,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        initial_delay=2.0
    )
    
    # Create tool with actor-level retry config
    from pydantic import BaseModel as PydanticBaseModel
    class DummyInput(PydanticBaseModel):
        text: str
    
    async def api_handler(x, ctx, st):
        return "response"
    
    tool = Tool(
        name="api_client",
        description="API client tool",
        input_model=DummyInput,
        handler=api_handler,
        retry_config=RetryConfig(
            max_attempts=4,
            backoff_strategy=BackoffStrategy.LINEAR,
            initial_delay=1.5
        )
    )
    
    # Simulate execution with retry
    state.history.add(ExecutionEvent(
        actor=tool,
        address="api_client::tool_call",
        execution_status=ExecutionStatus.STARTED,
        retry_config=tool.retry_config,
        retry_attempt=0
    ))
    
    state.history.add(ExecutionEvent(
        actor=tool,
        address="api_client::tool_call",
        execution_status=ExecutionStatus.FAILED,
        retry_config=tool.retry_config,
        retry_attempt=0,
        message="Network error"
    ))
    
    state.history.add(ExecutionEvent(
        actor=tool,
        address="api_client::tool_call",
        execution_status=ExecutionStatus.RETRYING,
        retry_config=tool.retry_config,
        retry_attempt=1,
        is_retry=True,
        original_error="Network error",
        retry_delay=1.5
    ))
    
    # Save checkpoint
    assert state.save_checkpoint(status=StateStatus.RUNNING) is True
    
    # Verify snapshot exists in backend
    snapshot = backend.load_state(state.state_id)
    assert snapshot is not None
    assert snapshot.status == StateStatus.RUNNING
    
    # Restore from backend
    restored_state = ExecutionState.load_from_state_id(state.state_id, backend)
    assert restored_state is not None
    
    # Verify state-level retry configs
    assert restored_state.default_retry_config is not None
    assert restored_state.default_retry_config["max_attempts"] == 3
    
    assert EventType.TOOL_CALL in restored_state.retry_config_by_actor_type
    tool_config = restored_state.retry_config_by_actor_type[EventType.TOOL_CALL]
    assert tool_config["max_attempts"] == 5
    assert tool_config["backoff_strategy"] == "exponential"
    
    # Verify event history with retry data
    event_list = restored_state.history.get_event_list()
    assert len(event_list) == 3
    
    # Check RETRYING event has all retry fields
    retrying_event = [e for e in event_list if e.get("execution_status") == ExecutionStatus.RETRYING.value][0]
    assert retrying_event["retry_attempt"] == 1
    assert retrying_event["is_retry"] is True
    assert retrying_event["original_error"] == "Network error"
    assert retrying_event["retry_delay"] == 1.5
    
    # Check retry_config is present in event
    assert "retry_config" in retrying_event
    event_retry_config = retrying_event["retry_config"]
    assert event_retry_config["max_attempts"] == 4
    assert event_retry_config["backoff_strategy"] == "linear"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
