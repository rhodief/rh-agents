"""
Quick resume example - shows smart replay with automatic skipping
"""
import asyncio
from rh_agents.core.execution import ExecutionState, EventBus
from rh_agents.core.events import ExecutionEvent
from rh_agents.models import Message
from rh_agents.core.actors import Actor
from rh_agents.state_backends import FileSystemStateBackend, FileSystemArtifactBackend
from rh_agents.core.state_recovery import ReplayMode
import os
from datetime import datetime

# Same actors as checkpoint example
class SimpleActor(Actor):
    """Simple actor that processes a message"""
    
    def __call__(self, input: Message) -> Message:
        return Message(
            content=input.content.upper(),
            role="assistant"
        )

class ProcessorActor(Actor):
    """Processor that adds a prefix"""
    
    def __call__(self, input: Message) -> Message:
        return Message(
            content=f"PROCESSED: {input.content}",
            role="assistant"
        )

class FinalActor(Actor):
    """New actor for additional step"""
    
    def __call__(self, input: Message) -> Message:
        return Message(
            content=f"{input.content} - FINALIZED AT {datetime.now().strftime('%H:%M:%S')}",
            role="assistant"
        )

async def main():
    print("\n" + "="*80)
    print("🔄 QUICK RESUME EXAMPLE")
    print("="*80)
    print("\n📝 This will load the checkpoint and replay with smart skipping")
    print("🎯 We'll add a third step that wasn't in the original\n")
    
    # Load state_id
    state_id_file = ".state_store/latest_state_id.txt"
    if not os.path.exists(state_id_file):
        print(f"❌ Error: {state_id_file} not found!")
        print("   Please run quick_checkpoint.py first")
        return
    
    with open(state_id_file, "r") as f:
        state_id = f.read().strip()
    
    print(f"📍 Loading state: {state_id}\n")
    
    # Initialize backends
    state_backend = FileSystemStateBackend(".state_store")
    artifact_backend = FileSystemArtifactBackend(".state_store/artifacts")
    
    # Create event bus
    bus = EventBus()
    
    print("="*80)
    print("🔄 LOADING CHECKPOINT")
    print("="*80 + "\n")
    
    # Load execution state from checkpoint
    agent_execution_state = ExecutionState.load_from_state_id(
        state_id=state_id,
        event_bus=bus,
        state_backend=state_backend,
        artifact_backend=artifact_backend,
        replay_mode=ReplayMode.NORMAL  # Smart skipping enabled
    )
    
    snapshot = agent_execution_state.to_snapshot()
    print(f"✅ Checkpoint loaded!")
    print(f"📊 Loaded {len(agent_execution_state.history)} events from history")
    print(f"🏷️  Tags: {snapshot.metadata.tags if snapshot.metadata else 'None'}")
    print(f"📝  Description: {snapshot.metadata.description if snapshot.metadata else 'None'}")
    print(f"⏰  Original timestamp: {snapshot.timestamp}")
    
    # Create simple message (same as original)
    message = Message(content="hello world", role="user")
    
    # Create actors
    simple_actor = SimpleActor(name="SimpleActor")
    processor_actor = ProcessorActor(name="ProcessorActor")
    final_actor = FinalActor(name="FinalActor")
    
    print("\n" + "="*80)
    print("▶️  REPLAYING PIPELINE")
    print("="*80 + "\n")
    
    # Track timing
    events_skipped = 0
    
    print("🔹 Step 1: SimpleActor")
    start = datetime.now()
    result1 = await ExecutionEvent[Message](actor=simple_actor)(
        message, "", agent_execution_state
    )
    duration1 = (datetime.now() - start).total_seconds()
    
    # Check if replayed (very fast = from cache)
    if duration1 < 0.01:
        print(f"  [REPLAYED] Result: {result1.content} (~0ms)")
        events_skipped += 1
    else:
        print(f"  [EXECUTED] Result: {result1.content} ({duration1:.2f}s)")
    
    print("\n🔹 Step 2: ProcessorActor")
    start = datetime.now()
    result2 = await ExecutionEvent[Message](actor=processor_actor)(
        result1, "", agent_execution_state
    )
    duration2 = (datetime.now() - start).total_seconds()
    
    if duration2 < 0.01:
        print(f"  [REPLAYED] Result: {result2.content} (~0ms)")
        events_skipped += 1
    else:
        print(f"  [EXECUTED] Result: {result2.content} ({duration2:.2f}s)")
    
    print("\n🔹 Step 3: FinalActor (NEW - not in checkpoint)")
    start = datetime.now()
    result3 = await ExecutionEvent[Message](actor=final_actor)(
        result2, "", agent_execution_state
    )
    duration3 = (datetime.now() - start).total_seconds()
    
    if duration3 < 0.01:
        print(f"  [REPLAYED] Result: {result3.content} (~0ms)")
        events_skipped += 1
    else:
        print(f"  [EXECUTED] Result: {result3.content} ({duration3:.2f}s)")
    
    print("\n" + "="*80)
    print("📊 REPLAY STATISTICS")
    print("="*80)
    print(f"\n✅ Events skipped (from checkpoint): {events_skipped}")
    print(f"🆕 Events executed (new work): {3 - events_skipped}")
    print(f"⚡ Time saved by replaying: {events_skipped * 100:.0f}ms estimated")
    
    print("\n" + "="*80)
    print("💡 KEY INSIGHT")
    print("="*80)
    print("""
Steps 1 and 2 were loaded from the checkpoint (~0ms each)
Step 3 was executed fresh because it wasn't in the original checkpoint

This demonstrates smart replay:
- Completed events are automatically skipped
- New work executes normally
- No code changes needed - addresses match automatically!
""")
    
    print("="*80)
    print("✨ Replay complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
