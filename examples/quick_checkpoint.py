"""
Quick checkpoint example - simplified version that completes quickly
"""
import asyncio
from rh_agents.core.execution import ExecutionState, EventBus
from rh_agents.core.events import ExecutionEvent
from rh_agents.models import Message
from rh_agents.core.actors import Actor
from rh_agents.state_backends import FileSystemStateBackend, FileSystemArtifactBackend
from rh_agents.core.state_recovery import StateStatus, StateMetadata
import os

# Simple actors for quick demo
class SimpleActor(Actor):
    """Simple actor that processes a message"""
    
    def __call__(self, input: Message) -> Message:
        # Just uppercase the content
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

async def main():
    print("\n" + "="*80)
    print("🚀 QUICK CHECKPOINT EXAMPLE")
    print("="*80)
    print("\n📝 Running a simple two-step pipeline")
    print("💾 State will be saved to: .state_store/\n")
    
    # Initialize backends
    state_backend = FileSystemStateBackend(".state_store")
    artifact_backend = FileSystemArtifactBackend(".state_store/artifacts")
    
    # Create event bus and execution state
    bus = EventBus()
    agent_execution_state = ExecutionState(
        event_bus=bus,
        state_backend=state_backend,
        artifact_backend=artifact_backend
    )
    
    # Create simple message
    message = Message(content="hello world", role="user")
    
    # Create actors
    simple_actor = SimpleActor(name="SimpleActor")
    processor_actor = ProcessorActor(name="ProcessorActor")
    
    print("🔹 Step 1: SimpleActor")
    result1 = await ExecutionEvent[Message](actor=simple_actor)(
        message, "", agent_execution_state
    )
    print(f"  Result: {result1.content}\n")
    
    print("🔹 Step 2: ProcessorActor")
    result2 = await ExecutionEvent[Message](actor=processor_actor)(
        result1, "", agent_execution_state
    )
    print(f"  Result: {result2.content}\n")
    
    print("="*80)
    print("💾 SAVING CHECKPOINT")
    print("="*80)
    
    # Save checkpoint with metadata
    state_id = agent_execution_state.save_checkpoint(
        status=StateStatus.COMPLETED,
        metadata=StateMetadata(
            tags=["quick-test", "complete"],
            description="Quick two-step pipeline execution",
            pipeline_name="simple_pipeline"
        )
    )
    
    print(f"\n✅ Checkpoint saved!")
    print(f"📍 State ID: {state_id}")
    print(f"📊 Events in history: {len(agent_execution_state.history)}")
    
    # Save state_id to file for easy access
    state_id_file = ".state_store/latest_state_id.txt"
    with open(state_id_file, "w") as f:
        f.write(str(state_id))
    print(f"💾 State ID written to: {state_id_file}")
    
    # Display storage info
    state_dir = ".state_store/states"
    if os.path.exists(state_dir):
        state_files = [f for f in os.listdir(state_dir) if f.endswith('.json')]
        print(f"\n📂 Storage:")
        print(f"  States: {len(state_files)} file(s) in {state_dir}/")
    
    artifact_dir = ".state_store/artifacts"
    if os.path.exists(artifact_dir):
        artifact_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pkl')]
        print(f"  Artifacts: {len(artifact_files)} file(s) in {artifact_dir}/")
    
    print("\n" + "="*80)
    print("✨ Next: Run quick_resume.py to restore and replay")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
