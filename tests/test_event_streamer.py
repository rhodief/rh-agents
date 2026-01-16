"""
Test EventStreamer functionality
"""
import asyncio
import pytest
from pydantic import BaseModel
from rh_agents.bus_handlers import EventStreamer
from rh_agents.core.execution import EventBus, ExecutionState
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.types import EventType, ExecutionStatus
from rh_agents.core.actors import BaseActor
from rh_agents.models import Message, AuthorType


class DummyInput(BaseModel):
    text: str = "test"


class DummyActor(BaseActor):
    """Simple actor for testing"""
    def __init__(self):
        async def dummy_handler(*args, **kwargs):
            return "test result"
        
        super().__init__(
            name="DummyActor",
            description="Test actor",
            input_model=DummyInput,
            handler=dummy_handler,
            event_type=EventType.AGENT_CALL
        )


@pytest.mark.asyncio
async def test_event_streamer_basic():
    """Test that EventStreamer can be subscribed and receives events"""
    # Create streamer and bus
    streamer = EventStreamer()
    bus = EventBus()
    bus.subscribe(streamer)
    
    # Create execution state
    state = ExecutionState(event_bus=bus)
    
    # Create a test agent and event
    actor = DummyActor()
    message = Message(content="Test input", author=AuthorType.USER)
    
    # Create and publish event
    event = ExecutionEvent[Message](actor=actor)
    event.address = "test::agent_call"
    event.execution_status = ExecutionStatus.STARTED
    event.detail = "Test detail"
    
    # Publish event
    await bus.publish(event)
    
    # Give it a moment to process
    await asyncio.sleep(0.1)
    
    # Check that event was received
    assert not streamer.queue.empty()
    received_event = await streamer.queue.get()
    assert received_event.address == "test::agent_call"
    assert received_event.execution_status == ExecutionStatus.STARTED


@pytest.mark.asyncio
async def test_event_streamer_stream_generator():
    """Test that stream() generates SSE-formatted output"""
    # Create streamer
    streamer = EventStreamer(heartbeat_interval=0.1)
    
    # Create a dummy execution task that completes immediately
    async def dummy_execution():
        await asyncio.sleep(0.05)
    
    execution_task = asyncio.create_task(dummy_execution())
    
    # Create a test event
    actor = DummyActor()
    event = ExecutionEvent[Message](actor=actor)
    event.address = "test::agent_call"
    event.execution_status = ExecutionStatus.COMPLETED
    event.execution_time = 0.5
    
    # Put event in queue
    await streamer.queue.put(event)
    
    # Collect stream output
    output = []
    async for chunk in streamer.stream(execution_task=execution_task):
        output.append(chunk)
        # Stop after getting a few chunks
        if len(output) >= 4:
            break
    
    # Verify SSE format
    assert any(": stream-start" in chunk for chunk in output)
    assert any("data: " in chunk for chunk in output)
    assert any("event_type" in chunk or "address" in chunk for chunk in output)


@pytest.mark.asyncio  
async def test_event_streamer_with_cache_stats():
    """Test that cache stats are included when enabled"""
    
    class DummyCacheBackend:
        def get_stats(self):
            return {"hits": 5, "misses": 3, "hit_rate": 0.625}
    
    # Create streamer with cache stats enabled
    streamer = EventStreamer(include_cache_stats=True, heartbeat_interval=0.05)
    cache_backend = DummyCacheBackend()
    
    # Create a quick execution task
    async def dummy_execution():
        await asyncio.sleep(0.02)
    
    execution_task = asyncio.create_task(dummy_execution())
    
    # Collect all stream output
    output = []
    async for chunk in streamer.stream(execution_task=execution_task, cache_backend=cache_backend):
        output.append(chunk)
    
    # Join all output
    full_output = "".join(output)
    
    # Verify completion event with cache stats
    assert "complete" in full_output
    assert "cache_stats" in full_output or "hits" in full_output


@pytest.mark.asyncio
async def test_event_streamer_error_handling():
    """Test that errors are caught and sent as error events"""
    streamer = EventStreamer(heartbeat_interval=0.05)
    
    # Create a task that raises an error
    async def failing_execution():
        await asyncio.sleep(0.02)
        raise ValueError("Test error")
    
    execution_task = asyncio.create_task(failing_execution())
    
    # Collect stream output
    output = []
    async for chunk in streamer.stream(execution_task=execution_task):
        output.append(chunk)
    
    full_output = "".join(output)
    
    # Verify error event
    assert "error" in full_output.lower()
    assert "Test error" in full_output or "ValueError" in full_output


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_event_streamer_basic())
    print("✅ test_event_streamer_basic passed")
    
    asyncio.run(test_event_streamer_stream_generator())
    print("✅ test_event_streamer_stream_generator passed")
    
    asyncio.run(test_event_streamer_with_cache_stats())
    print("✅ test_event_streamer_with_cache_stats passed")
    
    asyncio.run(test_event_streamer_error_handling())
    print("✅ test_event_streamer_error_handling passed")
    
    print("\n✨ All tests passed!")
