"""
Distributed Interrupt Example

Demonstrates interrupt checking from external sources:
1. Redis-based interrupt checking
2. Database-based interrupt checking
3. API-based interrupt checking (simulated)
4. File-based interrupt checking
"""
import asyncio
import json
from pathlib import Path
from rh_agents import ExecutionState, ExecutionEvent, Message, AuthorType
from rh_agents.agents import OpenAILLM, StepExecutorAgent
from rh_agents.bus_handlers import EventPrinter
from rh_agents.core.types import InterruptReason, InterruptSignal
from rh_agents.core.exceptions import ExecutionInterrupted


async def example_file_based_interrupt():
    """Example 1: File-based interrupt checking (simplest distributed pattern)"""
    print("\n" + "="*70)
    print("Example 1: File-Based Interrupt")
    print("="*70 + "\n")
    
    # Create event bus with printer
    from rh_agents.core.execution import EventBus
    bus = EventBus()
    printer = EventPrinter(show_timestamp=False, show_address=True)
    bus.subscribe(printer)
    
    # Create execution state
    state = ExecutionState(event_bus=bus)
    execution_id = state.state_id
    
    # Setup interrupt file path
    interrupt_dir = Path("/tmp/interrupts")
    interrupt_dir.mkdir(exist_ok=True)
    interrupt_file = interrupt_dir / f"{execution_id}.signal"
    
    # Define interrupt checker
    def check_interrupt_file():
        """Check if interrupt file exists"""
        return interrupt_file.exists()
    
    # Set the checker
    state.set_interrupt_checker(check_interrupt_file)
    print(f"📁 Watching interrupt file: {interrupt_file}\n")
    
    # Create agent
    llm = OpenAILLM()
    agent = StepExecutorAgent(llm=llm, tools=[])
    
    # Start execution
    message = Message(content="Process data", author=AuthorType.USER)
    
    async def run_agent():
        try:
            result = await ExecutionEvent(actor=agent)(message, "", state)
            return result
        except ExecutionInterrupted as e:
            print(f"\n✅ Caught interrupt from file: {e.message}")
            return None
    
    # Start agent
    task = asyncio.create_task(run_agent())
    
    # Simulate external process creating interrupt file
    await asyncio.sleep(0.5)
    print("🔴 External process creating interrupt file...\n")
    interrupt_file.write_text("interrupt")
    
    # Wait for completion
    result = await task
    
    # Cleanup
    if interrupt_file.exists():
        interrupt_file.unlink()
    
    # Print summary
    printer.print_summary()


async def example_detailed_interrupt_signal():
    """Example 2: File-based with detailed InterruptSignal"""
    print("\n" + "="*70)
    print("Example 2: Detailed Interrupt Signal (JSON File)")
    print("="*70 + "\n")
    
    # Create event bus with printer
    from rh_agents.core.execution import EventBus
    bus = EventBus()
    printer = EventPrinter(show_timestamp=False, show_address=False)
    bus.subscribe(printer)
    
    # Create execution state
    state = ExecutionState(event_bus=bus)
    execution_id = state.state_id
    
    # Setup interrupt file path
    interrupt_dir = Path("/tmp/interrupts")
    interrupt_dir.mkdir(exist_ok=True)
    interrupt_file = interrupt_dir / f"{execution_id}.json"
    
    # Define interrupt checker that returns InterruptSignal
    def check_interrupt_detailed():
        """Check for detailed interrupt information"""
        if not interrupt_file.exists():
            return None
        
        try:
            data = json.loads(interrupt_file.read_text())
            return InterruptSignal(
                reason=InterruptReason(data.get('reason', 'custom')),
                message=data.get('message', 'Interrupted by external signal'),
                triggered_by=data.get('triggered_by', 'external_service')
            )
        except Exception:
            return None
    
    # Set the checker
    state.set_interrupt_checker(check_interrupt_detailed)
    print(f"📁 Watching interrupt file: {interrupt_file}\n")
    
    # Create agent
    llm = OpenAILLM()
    agent = StepExecutorAgent(llm=llm, tools=[])
    
    # Start execution
    message = Message(content="Process data", author=AuthorType.USER)
    
    async def run_agent():
        try:
            result = await ExecutionEvent(actor=agent)(message, "", state)
            return result
        except ExecutionInterrupted as e:
            print(f"\n✅ Caught detailed interrupt: {e.message}")
            return None
    
    # Start agent
    task = asyncio.create_task(run_agent())
    
    # Simulate external process creating detailed interrupt signal
    await asyncio.sleep(0.5)
    print("🔴 External process creating detailed interrupt signal...\n")
    
    interrupt_data = {
        "reason": "timeout",
        "message": "Job exceeded maximum runtime of 5 minutes",
        "triggered_by": "kubernetes_job_controller"
    }
    interrupt_file.write_text(json.dumps(interrupt_data, indent=2))
    
    # Wait for completion
    result = await task
    
    # Cleanup
    if interrupt_file.exists():
        interrupt_file.unlink()
    
    # Print summary
    printer.print_summary()


async def example_memory_based_interrupt():
    """Example 3: In-memory interrupt checking (simulating Redis/database)"""
    print("\n" + "="*70)
    print("Example 3: In-Memory Interrupt (Simulates Redis/DB)")
    print("="*70 + "\n")
    
    # Simulate external interrupt storage (like Redis)
    interrupt_store = {}
    
    # Create event bus with printer
    from rh_agents.core.execution import EventBus
    bus = EventBus()
    printer = EventPrinter(show_timestamp=False, show_address=True)
    bus.subscribe(printer)
    
    # Create execution state
    state = ExecutionState(event_bus=bus)
    execution_id = state.state_id
    
    # Define interrupt checker (simulates Redis GET)
    def check_redis_interrupt():
        """Simulate checking Redis for interrupt signal"""
        return interrupt_store.get(execution_id, False)
    
    # Set the checker
    state.set_interrupt_checker(check_redis_interrupt)
    print(f"🔴 Watching interrupt store for execution: {execution_id}\n")
    
    # Create agent
    llm = OpenAILLM()
    agent = StepExecutorAgent(llm=llm, tools=[])
    
    # Start execution
    message = Message(content="Process large dataset", author=AuthorType.USER)
    
    async def run_agent():
        try:
            result = await ExecutionEvent(actor=agent)(message, "", state)
            return result
        except ExecutionInterrupted as e:
            print(f"\n✅ Caught interrupt from store: {e.message}")
            return None
    
    # Start agent
    task = asyncio.create_task(run_agent())
    
    # Simulate external service setting interrupt flag (like Redis SET)
    await asyncio.sleep(0.5)
    print("🔴 External service setting interrupt flag...\n")
    interrupt_store[execution_id] = True
    
    # Wait for completion
    result = await task
    
    # Cleanup
    interrupt_store.clear()
    
    # Print summary
    printer.print_summary()


async def example_combined_interrupt():
    """Example 4: Combined local + external interrupt"""
    print("\n" + "="*70)
    print("Example 4: Combined Local + External Interrupt")
    print("="*70 + "\n")
    
    # External interrupt storage
    interrupt_store = {}
    
    # Create event bus with printer
    from rh_agents.core.execution import EventBus
    bus = EventBus()
    printer = EventPrinter(show_timestamp=False, show_address=False)
    bus.subscribe(printer)
    
    # Create execution state
    state = ExecutionState(event_bus=bus)
    execution_id = state.state_id
    
    # Set external checker
    state.set_interrupt_checker(lambda: interrupt_store.get(execution_id, False))
    
    # Also set local timeout
    state.set_timeout(3.0, "Local timeout of 3 seconds")
    
    print(f"⏱️  Timeout: 3 seconds (local)")
    print(f"🔴 External store: watching {execution_id}")
    print("💡 Whichever triggers first will interrupt\n")
    
    # Create agent
    llm = OpenAILLM()
    agent = StepExecutorAgent(llm=llm, tools=[])
    
    # Start execution
    message = Message(content="Long running task", author=AuthorType.USER)
    
    async def run_agent():
        try:
            result = await ExecutionEvent(actor=agent)(message, "", state)
            state.cancel_timeout()  # Cancel timeout on success
            return result
        except ExecutionInterrupted as e:
            print(f"\n⚠️  Interrupted: {e.message}")
            print(f"📍 Reason: {e.reason.value}")
            return None
    
    # Start agent
    task = asyncio.create_task(run_agent())
    
    # Simulate external interrupt (before timeout)
    await asyncio.sleep(1.5)
    print("🔴 External service triggering interrupt (before timeout)...\n")
    interrupt_store[execution_id] = True
    
    # Wait for completion
    result = await task
    
    # Cleanup
    interrupt_store.clear()
    
    # Print summary
    printer.print_summary()


async def main():
    """Run all examples"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "DISTRIBUTED INTERRUPT EXAMPLES" + " "*24 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Run examples
    await example_file_based_interrupt()
    await asyncio.sleep(1)
    
    await example_detailed_interrupt_signal()
    await asyncio.sleep(1)
    
    await example_memory_based_interrupt()
    await asyncio.sleep(1)
    
    await example_combined_interrupt()
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*20 + "EXAMPLES COMPLETE!" + " "*29 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
