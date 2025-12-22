"""
Simple test to verify the caching system works correctly.
"""
import asyncio
from pathlib import Path
from rh_agents.core.actors import LLM, Tool
from rh_agents.core.events import ExecutionEvent, ExecutionResult
from rh_agents.core.execution import ExecutionState
from rh_agents.cache_backends import FileCacheBackend, InMemoryCacheBackend
from rh_agents.core.result_types import LLM_Result, Tool_Result
from rh_agents.bus_handlers import EventPrinter
from rh_agents.core.execution import EventBus
from pydantic import BaseModel


class TestInput(BaseModel):
    value: str


class TestOutput(BaseModel):
    result: str


async def test_llm_handler(input_data: TestInput, context: str, execution_state) -> LLM_Result:
    """Simulated LLM handler that takes time."""
    print(f"  [LLM] Processing: {input_data.value}")
    await asyncio.sleep(0.1)  # Simulate API call
    return LLM_Result(
        content=f"Processed: {input_data.value}",
        is_content=True
    )


async def test_tool_handler(input_data: TestInput, context: str, execution_state) -> Tool_Result:
    """Simulated Tool handler."""
    print(f"  [TOOL] Processing: {input_data.value}")
    await asyncio.sleep(0.05)
    return Tool_Result(
        output=f"Tool result: {input_data.value}",
        tool_name="test_tool"
    )


async def test_caching():
    """Test the caching system."""
    print("=" * 70)
    print("CACHING SYSTEM TEST")
    print("=" * 70)
    
    # Create cache backend
    cache_backend = InMemoryCacheBackend()
    
    # Create event bus with printer
    printer = EventPrinter(show_timestamp=False, show_address=False)
    bus = EventBus()
    bus.subscribe(printer)
    
    # Create execution state with cache
    execution_state = ExecutionState(
        event_bus=bus,
        cache_backend=cache_backend
    )
    
    # Test 1: LLM with caching enabled (default)
    print("\n" + "─" * 70)
    print("TEST 1: LLM with caching (default)")
    print("─" * 70)
    
    llm = LLM(
        name="TestLLM",
        description="Test LLM",
        input_model=TestInput,
        output_model=LLM_Result,
        handler=test_llm_handler,
        cacheable=True,
        version="1.0.0"
    )
    
    print(f"LLM cacheable: {llm.cacheable}, TTL: {llm.cache_ttl}s")
    
    # First execution (cache miss)
    print("\nExecution 1 (should cache miss):")
    event1 = ExecutionEvent(actor=llm)
    input_data = TestInput(value="Hello World")
    result1 = await event1(input_data, "", execution_state)
    print(f"Result: {result1.result.content if result1.ok else 'FAILED'}")
    print(f"From cache: {event1.from_cache}")
    
    # Second execution with same input (cache hit)
    print("\nExecution 2 (should cache hit):")
    event2 = ExecutionEvent(actor=llm)
    result2 = await event2(input_data, "", execution_state)
    print(f"Result: {result2.result.content if result2.ok else 'FAILED'}")
    print(f"From cache: {event2.from_cache}")
    
    # Third execution with different input (cache miss)
    print("\nExecution 3 with different input (should cache miss):")
    event3 = ExecutionEvent(actor=llm)
    input_data2 = TestInput(value="Different Input")
    result3 = await event3(input_data2, "", execution_state)
    print(f"Result: {result3.result.content if result3.ok else 'FAILED'}")
    print(f"From cache: {event3.from_cache}")
    
    # Test 2: Tool with caching disabled (default)
    print("\n" + "─" * 70)
    print("TEST 2: Tool with caching disabled (default)")
    print("─" * 70)
    
    tool = Tool(
        name="TestTool",
        description="Test Tool",
        input_model=TestInput,
        handler=test_tool_handler,
        cacheable=False
    )
    
    print(f"Tool cacheable: {tool.cacheable}")
    
    # Execute twice with same input (should not use cache)
    print("\nExecution 1:")
    event4 = ExecutionEvent(actor=tool)
    result4 = await event4(input_data, "", execution_state)
    print(f"From cache: {event4.from_cache}")
    
    print("\nExecution 2 (should NOT cache):")
    event5 = ExecutionEvent(actor=tool)
    result5 = await event5(input_data, "", execution_state)
    print(f"From cache: {event5.from_cache}")
    
    # Test 3: Version invalidation
    print("\n" + "─" * 70)
    print("TEST 3: Version invalidation")
    print("─" * 70)
    
    # Create LLM with different version
    llm_v2 = LLM(
        name="TestLLM",
        description="Test LLM v2",
        input_model=TestInput,
        output_model=LLM_Result,
        handler=test_llm_handler,
        cacheable=True,
        version="2.0.0"  # Different version
    )
    
    print(f"LLM v2 version: {llm_v2.version}")
    
    # Execute with same input as before (should cache miss due to version change)
    print("\nExecution with v2.0.0 (should cache miss):")
    event6 = ExecutionEvent(actor=llm_v2)
    result6 = await event6(input_data, "", execution_state)
    print(f"From cache: {event6.from_cache}")
    
    # Cache statistics
    print("\n" + "─" * 70)
    print("CACHE STATISTICS")
    print("─" * 70)
    stats = cache_backend.get_stats()
    print(f"Backend: {stats['backend']}")
    print(f"Cached entries: {stats['size']}")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    
    # Event summary
    printer.print_summary()
    
    # Verify results
    print("\n" + "─" * 70)
    print("VERIFICATION")
    print("─" * 70)
    assert not event1.from_cache, "First LLM execution should not be from cache"
    assert event2.from_cache, "Second LLM execution should be from cache"
    assert not event3.from_cache, "LLM with different input should not be from cache"
    assert not event4.from_cache, "Tool execution should not use cache"
    assert not event5.from_cache, "Tool second execution should not use cache"
    assert not event6.from_cache, "LLM with different version should not be from cache"
    print("✅ All assertions passed!")


if __name__ == "__main__":
    asyncio.run(test_caching())
