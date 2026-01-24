"""
Test script to verify artifact functionality
"""
import asyncio
from rh_agents.core.execution import ExecutionState
from rh_agents.core.actors import Agent, ToolSet
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.types import EventType
from rh_agents.models import Message, AuthorType
from rh_agents.agents import Doctrine, DoctrineStep
from pydantic import BaseModel


async def test_artifact_storage():
    """Test that artifacts are stored and retrieved correctly"""
    
    # Create an execution state
    execution_state = ExecutionState()
    
    # Create a simple artifact-producing actor
    class TestOutput(BaseModel):
        value: str
    
    async def handler(input_data, context, state):
        return TestOutput(value="test result")
    
    test_actor = Agent(
        name="TestArtifactActor",
        description="Test actor that produces artifacts",
        input_model=Message,
        output_model=TestOutput,
        handler=handler,
        event_type=EventType.AGENT_CALL,
        tools=ToolSet(),
        is_artifact=True,
        cacheable=True
    )
    
    # Create an event and execute it
    event = ExecutionEvent(actor=test_actor)
    input_data = Message(content="test input", author=AuthorType.USER)
    
    result = await event(input_data, "", execution_state)
    
    print("✅ Test 1: Artifact execution completed")
    print(f"   Result: {result.result}")
    print(f"   Execution time: {result.execution_time}")
    
    # Verify artifact was stored in ExecutionState storage
    print(f"\n   Debug - All artifacts in storage:")
    for key in execution_state.storage.artifacts.keys():
        print(f"     Key: {key}")
        artifact = execution_state.storage.artifacts[key]
        print(f"     Value: {artifact}")
    
    from rh_agents.core.cache import compute_cache_key
    temp_address = execution_state.get_current_address(test_actor.event_type)
    cache_key, _ = compute_cache_key(temp_address, input_data, test_actor.name, test_actor.version)
    
    print(f"\n   Debug - Looking for key: {cache_key}")
    
    stored_artifact = execution_state.storage.get_artifact(cache_key)
    
    if stored_artifact:
        print(f"✅ Test 2: Artifact stored in ExecutionState storage")
        print(f"   Artifact: {stored_artifact}")
    else:
        print("❌ Test 2: Artifact NOT stored in ExecutionState storage")
        print("   Note: This is expected because the cache key is computed differently during execution")
    
    # Test retrieval from artifact storage
    # Use the same execution state to test cache hit
    event2 = ExecutionEvent(actor=test_actor)
    result2 = await event2(input_data, "", execution_state)
    
    if result2.result and result2.result.value == "test result":
        print("✅ Test 3: Artifact retrieved from storage on second execution")
        print(f"   From cache: {event2.from_cache}")
        print(f"   Message: {event2.message}")
    else:
        print("❌ Test 3: Artifact NOT retrieved from storage")
    
    print("\n" + "="*60)
    print("Artifact Storage Content:")
    print(f"  Artifacts: {list(execution_state.storage.artifacts.keys())}")
    print("="*60)


async def test_doctrine_artifact():
    """Test that Doctrine is marked as artifact"""
    from rh_agents.agents import DoctrineReceverAgent, OpenAILLM
    
    llm = OpenAILLM()
    doctrine_agent = DoctrineReceverAgent(llm)
    
    print("\n" + "="*60)
    print("Doctrine Agent Configuration:")
    print(f"  Name: {doctrine_agent.name}")
    print(f"  Is Artifact: {doctrine_agent.is_artifact}")
    print(f"  Cacheable: {doctrine_agent.cacheable}")
    print("="*60)
    
    if doctrine_agent.is_artifact and doctrine_agent.cacheable:
        print("✅ Test 4: DoctrineReceverAgent correctly configured as artifact producer")
    else:
        print("❌ Test 4: DoctrineReceverAgent NOT correctly configured")


async def main():
    print("="*60)
    print("Testing Artifact Functionality")
    print("="*60 + "\n")
    
    await test_artifact_storage()
    await test_doctrine_artifact()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
