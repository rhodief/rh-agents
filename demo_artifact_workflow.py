"""
Demonstration of Artifact System with Doctrine-like workflow
"""
import asyncio
from rh_agents.core.execution import ExecutionState
from rh_agents.core.actors import Agent, ToolSet
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.types import EventType
from rh_agents.models import Message, AuthorType
from rh_agents.agents import Doctrine, DoctrineStep
from pydantic import BaseModel


class MockDoctrine(BaseModel):
    """Mock Doctrine for testing without OpenAI"""
    goal: str
    steps: list[str]


async def test_doctrine_artifact_workflow():
    """
    Simulates the Doctrine workflow:
    1. User sends message
    2. DoctrineReceiver creates Doctrine (artifact)
    3. Subsequent calls retrieve Doctrine from artifacts (no recreation)
    """
    
    print("="*70)
    print("ARTIFACT WORKFLOW DEMONSTRATION - Doctrine-like Pattern")
    print("="*70 + "\n")
    
    # Create execution state (no cache backend needed for artifacts!)
    execution_state = ExecutionState()
    
    # Create a mock Doctrine-producing agent
    async def doctrine_handler(input_data, context, state):
        """Simulates creating a Doctrine from user input"""
        print("   🔄 Creating Doctrine (simulating LLM call)...")
        await asyncio.sleep(0.1)  # Simulate LLM latency
        return MockDoctrine(
            goal="Process legal document",
            steps=["Step 1: Extract text", "Step 2: Analyze", "Step 3: Generate report"]
        )
    
    doctrine_agent = Agent(
        name="MockDoctrineAgent",
        description="Creates execution plan from user input",
        input_model=Message,
        output_model=MockDoctrine,
        handler=doctrine_handler,
        event_type=EventType.AGENT_CALL,
        tools=ToolSet(),
        is_artifact=True,  # Mark as artifact
        cacheable=True     # Enable caching
    )
    
    # Test 1: First execution - creates and stores artifact
    print("📝 Test 1: First Execution (creates Doctrine artifact)")
    print("-" * 70)
    
    event1 = ExecutionEvent[MockDoctrine](actor=doctrine_agent)
    user_input = Message(content="Create legal report", author=AuthorType.USER)
    
    import time
    start = time.time()
    result1 = await event1(user_input, "", execution_state)
    elapsed1 = time.time() - start
    
    print(f"   ✅ Doctrine created: {result1.result.goal}")
    print(f"   ⏱️  Time: {elapsed1*1000:.2f}ms")
    print(f"   📦 From cache: {event1.from_cache}")
    print(f"   📊 Artifacts in storage: {len(execution_state.storage.artifacts)}")
    print()
    
    # Test 2: Second execution with same input - retrieves from artifacts
    print("📝 Test 2: Second Execution (retrieves from artifact storage)")
    print("-" * 70)
    
    event2 = ExecutionEvent[MockDoctrine](actor=doctrine_agent)
    
    start = time.time()
    result2 = await event2(user_input, "", execution_state)
    elapsed2 = time.time() - start
    
    print(f"   ✅ Doctrine retrieved: {result2.result.goal}")
    print(f"   ⏱️  Time: {elapsed2*1000:.2f}ms")
    print(f"   📦 From cache: {event2.from_cache}")
    print(f"   💬 Message: {event2.message}")
    print()
    
    # Test 3: Performance comparison
    print("📊 Performance Analysis")
    print("-" * 70)
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float('inf')
    print(f"   First execution:  {elapsed1*1000:.2f}ms (with LLM simulation)")
    print(f"   Second execution: {elapsed2*1000:.2f}ms (from artifact)")
    print(f"   Speedup:          {speedup:.1f}x faster")
    print()
    
    # Test 4: Verify artifact content
    print("🔍 Artifact Storage Inspection")
    print("-" * 70)
    print(f"   Total artifacts: {len(execution_state.storage.artifacts)}")
    for i, (key, artifact) in enumerate(execution_state.storage.artifacts.items(), 1):
        print(f"   Artifact #{i}:")
        print(f"     Key: {key[:32]}...")
        print(f"     Type: {type(artifact.result).__name__}")
        print(f"     Goal: {artifact.result.goal}")
        print(f"     Steps: {len(artifact.result.steps)}")
    print()
    
    # Test 5: Different input creates new artifact
    print("📝 Test 3: Different Input (creates new artifact)")
    print("-" * 70)
    
    event3 = ExecutionEvent[MockDoctrine](actor=doctrine_agent)
    different_input = Message(content="Analyze contract", author=AuthorType.USER)
    
    result3 = await event3(different_input, "", execution_state)
    
    print(f"   ✅ New doctrine created: {result3.result.goal}")
    print(f"   📦 From cache: {event3.from_cache}")
    print(f"   📊 Total artifacts now: {len(execution_state.storage.artifacts)}")
    print()
    
    print("="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("• Artifacts are stored in ExecutionState (no cache backend needed)")
    print("• Same input retrieves cached artifact (no LLM call)")
    print("• Different input creates new artifact")
    print("• Significant performance improvement on cache hits")


async def main():
    await test_doctrine_artifact_workflow()


if __name__ == "__main__":
    asyncio.run(main())
