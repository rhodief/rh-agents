"""
Demo script showing Phase 3 retry display features.

Demonstrates:
- RETRYING status with ↻ symbol
- Retry attempt tracking
- Retry statistics in summary
- Compact vs verbose modes
"""
import asyncio
from pydantic import BaseModel
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import BaseActor
from rh_agents.core.execution import ExecutionState
from rh_agents.core.types import EventType
from rh_agents.core.retry import RetryConfig, BackoffStrategy
from rh_agents.bus_handlers import EventPrinter


class MessageInput(BaseModel):
    text: str


async def main():
    print("\n" + "=" * 80)
    print("PHASE 3 DEMO: Retry Display Features")
    print("=" * 80 + "\n")
    
    # Create execution state with printer
    state = ExecutionState()
    printer = EventPrinter(show_timestamp=True, show_address=True)
    state.event_bus.subscribe(printer.print_event)
    
    print("\n🎯 Demo 1: Tool with transient failure (succeeds on retry)")
    print("-" * 80)
    
    # Create a flaky tool that fails once then succeeds
    attempt_count_1 = [0]
    
    async def flaky_api_call(input_data, context, state):
        attempt_count_1[0] += 1
        if attempt_count_1[0] == 1:
            raise ConnectionError("API temporarily unavailable")
        return f"API Response: Processed '{input_data.text}'"
    
    flaky_tool = BaseActor(
        name="api_client",
        description="Calls external API",
        input_model=MessageInput,
        event_type=EventType.TOOL_CALL,
        handler=flaky_api_call,
        retry_config=RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            backoff_strategy=BackoffStrategy.LINEAR,
            jitter=False
        )
    )
    
    event1 = ExecutionEvent(actor=flaky_tool)
    result1 = await event1(
        input_data=MessageInput(text="Hello World"),
        extra_context={},
        execution_state=state
    )
    
    print(f"\n✅ Result: {result1.result if result1.ok else 'Failed'}")
    
    # Demo 2: Tool that exhausts retries
    print("\n\n🎯 Demo 2: Tool that exhausts all retries")
    print("-" * 80)
    
    attempt_count_2 = [0]
    
    async def always_fails(input_data, context, state):
        attempt_count_2[0] += 1
        raise TimeoutError(f"Connection timeout (attempt {attempt_count_2[0]})")
    
    failing_tool = BaseActor(
        name="unreliable_service",
        description="Service with persistent issues",
        input_model=MessageInput,
        event_type=EventType.TOOL_CALL,
        handler=always_fails,
        retry_config=RetryConfig(
            max_attempts=3,
            initial_delay=0.3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            jitter=False
        )
    )
    
    event2 = ExecutionEvent(actor=failing_tool)
    result2 = await event2(
        input_data=MessageInput(text="Test request"),
        extra_context={},
        execution_state=state
    )
    
    print(f"\n❌ Result: {result2.erro_message}")
    
    # Demo 3: LLM call with retry
    print("\n\n🎯 Demo 3: LLM call with exponential backoff")
    print("-" * 80)
    
    attempt_count_3 = [0]
    
    async def flaky_llm(input_data, context, state):
        attempt_count_3[0] += 1
        if attempt_count_3[0] < 3:
            raise TimeoutError("Rate limit exceeded")
        return f"LLM Response: {input_data.text.upper()}"
    
    llm_actor = BaseActor(
        name="gpt4_model",
        description="OpenAI GPT-4 model",
        input_model=MessageInput,
        event_type=EventType.LLM_CALL,
        handler=flaky_llm,
        retry_config=RetryConfig(
            max_attempts=5,
            initial_delay=1.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
            jitter=False
        )
    )
    
    event3 = ExecutionEvent(actor=llm_actor)
    result3 = await event3(
        input_data=MessageInput(text="translate to spanish"),
        extra_context={},
        execution_state=state
    )
    
    print(f"\n✅ Result: {result3.result if result3.ok else 'Failed'}")
    
    # Print comprehensive summary
    print("\n\n" + "=" * 80)
    printer.print_summary()
    
    # Demo compact mode
    print("\n\n🎯 Demo 4: Compact mode display")
    print("-" * 80)
    print("(Same execution with minimal output)\n")
    
    state2 = ExecutionState()
    compact_printer = EventPrinter(show_timestamp=False, show_address=False)
    state2.event_bus.subscribe(compact_printer.print_event)
    
    async def quick_success(input_data, context, state):
        return "OK"
    
    simple_tool = BaseActor(
        name="quick_tool",
        description="Fast tool",
        input_model=MessageInput,
        event_type=EventType.TOOL_CALL,
        handler=quick_success
    )
    
    event4 = ExecutionEvent(actor=simple_tool)
    await event4(
        input_data=MessageInput(text="test"),
        extra_context={},
        execution_state=state2
    )
    
    print("\n✨ Compact mode shows less detail but same essential information")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
