"""
Basic Interrupt Example

Demonstrates core interrupt functionality:
1. Direct interrupt using request_interrupt()
2. Timeout-based interrupt using set_timeout()
3. EventPrinter handling of interrupt events
"""
import asyncio
from rh_agents import ExecutionState, ExecutionEvent, Message, AuthorType
from rh_agents.agents import OpenAILLM, StepExecutorAgent
from rh_agents.bus_handlers import EventPrinter
from rh_agents.core.types import InterruptReason
from rh_agents.core.exceptions import ExecutionInterrupted


async def example_direct_interrupt():
    """Example 1: Direct interrupt using request_interrupt()"""
    print("\n" + "="*70)
    print("Example 1: Direct Interrupt")
    print("="*70 + "\n")
    
    # Create event bus with printer that handles interrupts
    from rh_agents.core.execution import EventBus
    bus = EventBus()
    printer = EventPrinter(show_timestamp=True, show_address=True)
    bus.subscribe(printer)
    
    # Create execution state
    state = ExecutionState(event_bus=bus)
    
    # Create a simple agent
    llm = OpenAILLM()
    agent = StepExecutorAgent(llm=llm, tools=[])
    
    # Start execution in background
    message = Message(content="Count to 10", author=AuthorType.USER)
    
    async def run_agent():
        try:
            result = await ExecutionEvent(actor=agent)(message, "", state)
            return result
        except ExecutionInterrupted as e:
            print(f"\n✅ Caught interrupt: {e.message}")
            return None
    
    # Start agent
    task = asyncio.create_task(run_agent())
    
    # Wait a bit then interrupt
    await asyncio.sleep(0.5)
    print("\n🔴 Triggering interrupt...\n")
    
    state.request_interrupt(
        reason=InterruptReason.USER_CANCELLED,
        message="User clicked stop button",
        triggered_by="example_code"
    )
    
    # Wait for completion
    result = await task
    
    # Print summary
    printer.print_summary()


async def example_timeout_interrupt():
    """Example 2: Timeout-based automatic interrupt"""
    print("\n" + "="*70)
    print("Example 2: Timeout-Based Interrupt")
    print("="*70 + "\n")
    
    # Create event bus with printer
    from rh_agents.core.execution import EventBus
    bus = EventBus()
    printer = EventPrinter(show_timestamp=True, show_address=False)
    bus.subscribe(printer)
    
    # Create execution state with 2-second timeout
    state = ExecutionState(event_bus=bus)
    state.set_timeout(2.0, "Operation must complete within 2 seconds")
    
    # Create agent
    llm = OpenAILLM()
    agent = StepExecutorAgent(llm=llm, tools=[])
    
    # Start execution
    message = Message(content="Analyze this complex document...", author=AuthorType.USER)
    
    print("⏱️  Starting execution with 2-second timeout...\n")
    
    try:
        result = await ExecutionEvent(actor=agent)(message, "", state)
        print("\n✅ Execution completed successfully")
        state.cancel_timeout()  # Cancel timeout on success
    except ExecutionInterrupted as e:
        print(f"\n⏰ Execution timed out: {e.message}")
    
    # Print summary
    printer.print_summary()


async def example_interrupt_with_cleanup():
    """Example 3: Interrupt with proper cleanup"""
    print("\n" + "="*70)
    print("Example 3: Interrupt with Cleanup")
    print("="*70 + "\n")
    
    # Create event bus with printer
    from rh_agents.core.execution import EventBus
    bus = EventBus()
    printer = EventPrinter(show_timestamp=False, show_address=True)
    bus.subscribe(printer)
    
    # Create execution state
    state = ExecutionState(event_bus=bus)
    
    # Register a mock generator (simulating streaming endpoint)
    async def mock_stream():
        """Simulate an event stream"""
        try:
            for i in range(100):
                await asyncio.sleep(0.1)
                print(f"  📡 Stream event {i}")
        except asyncio.CancelledError:
            print("  🛑 Stream cancelled cleanly")
            raise
    
    stream_task = asyncio.create_task(mock_stream())
    state.register_generator(stream_task)
    
    # Create agent
    llm = OpenAILLM()
    agent = StepExecutorAgent(llm=llm, tools=[])
    
    # Start execution
    message = Message(content="Process data", author=AuthorType.USER)
    
    async def run_with_cleanup():
        try:
            result = await ExecutionEvent(actor=agent)(message, "", state)
            return result
        except ExecutionInterrupted as e:
            print(f"\n⚠️  Interrupt detected: {e.message}")
            print("🧹 Cleaning up generators...")
            await state.kill_generators()
            print("✅ Cleanup complete")
            return None
    
    # Start execution
    task = asyncio.create_task(run_with_cleanup())
    
    # Wait then interrupt
    await asyncio.sleep(0.3)
    print("\n🔴 Triggering interrupt with generator cleanup...\n")
    
    state.request_interrupt(
        reason=InterruptReason.USER_CANCELLED,
        message="Interrupt with cleanup",
        triggered_by="example_code"
    )
    
    # Wait for completion
    result = await task
    
    # Verify generator is stopped
    await asyncio.sleep(0.2)
    print(f"\n📊 Stream task done: {stream_task.done()}")
    
    # Print summary
    printer.print_summary()


async def main():
    """Run all examples"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*20 + "INTERRUPT EXAMPLES" + " "*30 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Run examples
    await example_direct_interrupt()
    await asyncio.sleep(1)
    
    await example_timeout_interrupt()
    await asyncio.sleep(1)
    
    await example_interrupt_with_cleanup()
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*20 + "EXAMPLES COMPLETE!" + " "*29 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
