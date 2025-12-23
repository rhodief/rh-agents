"""
Test script to verify that events are immutable (read-only) when passed to subscribers.
"""

from rh_agents.core.execution import EventBus, ExecutionState
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import Tool
from rh_agents.core.types import EventType, ExecutionStatus
from pydantic import BaseModel

# Create a simple test actor
class DummyArgs(BaseModel):
    test: str = "test"

async def dummy_handler(args, context, execution_state):
    return "test result"

# Create a test tool actor
test_actor = Tool(
    name="test_actor",
    description="A test actor",
    input_model=DummyArgs,
    handler=dummy_handler,
    cacheable=False
)

def test_event_immutability():
    """Test that events cannot be mutated by subscribers."""
    
    # Create event bus
    bus = EventBus()
    
    # Track original and received events
    original_detail = "Original detail"
    received_events = []
    
    # Create a subscriber that tries to mutate the event
    def malicious_subscriber(event):
        """This subscriber tries to modify the event."""
        received_events.append(event)
        # Try to mutate the event
        event.detail = "MUTATED BY SUBSCRIBER"
        event.address = "MUTATED ADDRESS"
    
    # Subscribe the malicious subscriber
    bus.subscribe(malicious_subscriber)
    
    # Create and publish an event
    event = ExecutionEvent(
        actor=test_actor,
        detail=original_detail,
        address="test::address",
        execution_status=ExecutionStatus.STARTED
    )
    
    # Publish the event
    await bus.publish(event)
    
    # Verify that the original event stored in the bus is NOT mutated
    print(f"✓ Original event in bus.events:")
    print(f"  - detail: {bus.events[0].detail}")
    print(f"  - address: {bus.events[0].address}")
    
    # Verify what the subscriber received
    print(f"\n✓ Event received by subscriber:")
    print(f"  - detail: {received_events[0].detail}")
    print(f"  - address: {received_events[0].address}")
    
    # Check if mutation affected the original
    if bus.events[0].detail == original_detail:
        print(f"\n✅ SUCCESS: Original event is protected from mutation!")
        print(f"   The subscriber received a copy and couldn't mutate the original.")
    else:
        print(f"\n❌ FAILURE: Original event was mutated by subscriber!")
        print(f"   Expected: {original_detail}")
        print(f"   Got: {bus.events[0].detail}")
    
    # Also check if the subscriber's changes are isolated
    if received_events[0].detail == "MUTATED BY SUBSCRIBER":
        print(f"\n✓ The subscriber CAN modify its own copy (as expected)")
    
    return bus.events[0].detail == original_detail

if __name__ == "__main__":
    print("Testing Event Immutability in EventBus")
    print("=" * 50)
    print()
    
    success = test_event_immutability()
    
    print()
    print("=" * 50)
    if success:
        print("✅ All tests passed! Events are immutable.")
    else:
        print("❌ Tests failed! Events can be mutated.")
