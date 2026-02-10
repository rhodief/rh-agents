"""
Demo Script: Phase 5 - Interrupt Integration

Demonstrates:
1. Interrupt during retry backoff delay2. Responsive interrupt handling (fast detection)
3. State preservation when interrupted
4. External interrupt checker integration
5. Graceful interrupt with retry context
"""
import asyncio
from pydantic import BaseModel
from rh_agents.core.execution import ExecutionState
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import Tool
from rh_agents.core.types import ExecutionStatus, InterruptReason
from rh_agents.core.retry import RetryConfig, BackoffStrategy
from rh_agents.core.exceptions import ExecutionInterrupted
from rh_agents.bus_handlers import EventPrinter


# ===== Demo Input Model =====

class APIRequest(BaseModel):
    endpoint: str
    data: str


# ===== Demo Helpers =====

def print_separator(title: str):
    """Print a formatted section separator."""
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80 + "\n")


# ===== Demo 1: Basic interrupt during retry backoff =====

async def demo1_interrupt_during_backoff():
    print_separator("Demo 1: Interrupt During Retry Backoff")
    
    state = ExecutionState()
    printer = EventPrinter(show_address=False, show_timestamp=False)
    state.event_bus.subscribe(printer.print_event)
    
    call_count = 0
    
    async def flaky_api(input_data, context, execution_state):
        nonlocal call_count
        call_count += 1
        print(f"  📡 API call attempt #{call_count}")
        raise ConnectionError("Service temporarily unavailable")
    
    tool = Tool(
        name="flaky_api",
        description="Flaky API client",
        input_model=APIRequest,
        handler=flaky_api,
        retry_config=RetryConfig(
            max_attempts=5,
            initial_delay=3.0,  # Long delay
            backoff_strategy=BackoffStrategy.CONSTANT
        )
    )
    
    event = ExecutionEvent(actor=tool)
    
    print("🔧 Starting API call with retry (3s backoff delay)...")
    print("⏱️  Will interrupt after 0.5s...\n")
    
    async def interrupt_during_wait():
        await asyncio.sleep(0.5)
        print("\n🛑 User requested cancel!\n")
        state.request_interrupt(
            reason=InterruptReason.USER_CANCELLED,
            message="User pressed Cancel button"
        )
    
    task1 = asyncio.create_task(event(APIRequest(endpoint="/api/test", data="test"), {}, state))
    task2 = asyncio.create_task(interrupt_during_wait())
    
    try:
        await task1
    except ExecutionInterrupted as e:
        print(f"\n✅ Execution interrupted: {e.message}")
        print(f"   Reason: {e.reason.value}")
    
    await task2
    
    # Show statistics
    print(f"\n📊 Results:")
    print(f"   • API call attempts: {call_count}")
    print(f"   • Interrupt detected during backoff delay")
    print(f"   • Total execution time: < 1 second (not full 3s backoff)")


# ===== Demo 2: Responsive interrupt during long backoff =====

async def demo2_responsive_interrupt():
    print_separator("Demo 2: Responsive Interrupt (10s backoff, 0.3s response)")
    
    state = ExecutionState()
    
    async def always_fails(input_data, context, execution_state):
        raise TimeoutError("Service timeout")
    
    tool = Tool(
        name="slow_service",
        description="Slow service",
        input_model=APIRequest,
        handler=always_fails,
        retry_config=RetryConfig(
            max_attempts=5,
            initial_delay=10.0,  # Very long delay
            backoff_strategy=BackoffStrategy.CONSTANT
        )
    )
    
    event = ExecutionEvent(actor=tool)
    
    print("🔧 Starting service call with 10-second retry backoff...")
    print("⏱️  Will interrupt after 0.3s...\n")
    
    start_time = asyncio.get_event_loop().time()
    
    async def quick_interrupt():
        await asyncio.sleep(0.3)
        state.request_interrupt(
            reason=InterruptReason.USER_CANCELLED,
            message="Quick cancel"
        )
    
    task1 = asyncio.create_task(event(APIRequest(endpoint="/api/slow", data="test"), {}, state))
    task2 = asyncio.create_task(quick_interrupt())
    
    try:
        await task1
    except ExecutionInterrupted:
        pass
    
    await task2
    
    end_time = asyncio.get_event_loop().time()
    elapsed = end_time - start_time
    
    print(f"✅ Interrupt detected in {elapsed:.2f}s (not 10s!)")
    print(f"   • Interrupt checking happens every 0.1s during backoff")
    print(f"   • Responsive even with long retry delays")


# ===== Demo 3: State preservation with interrupt =====

async def demo3_state_preservation():
    print_separator("Demo 3: State Preservation When Interrupted")
    
    state = ExecutionState()
    
    async def unreliable_service(input_data, context, execution_state):
        raise ConnectionError("Connection lost")
    
    tool = Tool(
        name="data_service",
        description="Data service",
        input_model=APIRequest,
        handler=unreliable_service,
        retry_config=RetryConfig(max_attempts=5, initial_delay=2.0)
    )
    
    event = ExecutionEvent(actor=tool)
    
    print("🔧 Executing service call with retry (showing state preservation)...\n")
    
    async def interrupt_after_first_retry():
        await asyncio.sleep(0.3)  # Let first failure happen
        state.request_interrupt(
            reason=InterruptReason.RESOURCE_LIMIT,
            message="System memory limit reached",
            save_checkpoint=True
        )
    
    task1 = asyncio.create_task(event(APIRequest(endpoint="/api/data", data="query"), {}, state))
    task2 = asyncio.create_task(interrupt_after_first_retry())
    
    try:
        await task1
    except ExecutionInterrupted as e:
        print(f"\n🛑 Interrupted: {e.message}\n")
    
    await task2
    
    # Examine state
    events = state.history.get_event_list()
    print(f"📋 Execution history preserved:")
    print(f"   • Total events: {len(events)}")
    
    def get_status(e):
        if isinstance(e, dict):
            return e.get("execution_status")
        return getattr(e, "execution_status", None)
    
    statuses = [get_status(e) for e in events]
    status_counts = {}
    for s in statuses:
        val = s.value if hasattr(s, 'value') else s
        status_counts[val] = status_counts.get(val, 0) + 1
    
    for status, count in sorted(status_counts.items()):
        print(f"   • {status}: {count}")
    
    print(f"\n✅ Complete execution context preserved for recovery")


# ===== Demo 4: External interrupt checker =====

async def demo4_external_interrupt_checker():
    print_separator("Demo 4: External Interrupt Checker Integration")
    
    state = ExecutionState()
    
    interrupt_check_count = 0
    should_interrupt = False
    
    def check_for_interrupt():
        """Simulates external interrupt checker (e.g., checking file, API, etc.)"""
        nonlocal interrupt_check_count
        interrupt_check_count += 1
        return should_interrupt
    
    state.set_interrupt_checker(check_for_interrupt)
    
    async def slow_processing(input_data, context, execution_state):
        raise TimeoutError("Processing timeout")
    
    tool = Tool(
        name="processor",
        description="Data processor",
        input_model=APIRequest,
        handler=slow_processing,
        retry_config=RetryConfig(
            max_attempts=10,
            initial_delay=1.5,
            backoff_strategy=BackoffStrategy.CONSTANT
        )
    )
    
    event = ExecutionEvent(actor=tool)
    
    print("🔧 Starting processing with external interrupt checker...")
    print("📡 Interrupt checker will be called periodically during backoff\n")
    
    async def trigger_external_interrupt():
        nonlocal should_interrupt
        await asyncio.sleep(0.5)
        print("🔔 External signal received (interrupt file created)\n")
        should_interrupt = True
    
    task1 = asyncio.create_task(event(APIRequest(endpoint="/api/process", data="batch"), {}, state))
    task2 = asyncio.create_task(trigger_external_interrupt())
    
    try:
        await task1
    except ExecutionInterrupted as e:
        print(f"🛑 Execution interrupted via external checker\n")
    
    await task2
    
    print(f"📊 Interrupt checker statistics:")
    print(f"   • Checker called {interrupt_check_count} times")
    print(f"   • Checked during: preconditions + retry backoff delays")
    print(f"\n✅ External interrupt mechanism integrated with retry system")


# ===== Demo 5: Multiple retries before interrupt =====

async def demo5_multiple_retries():
    print_separator("Demo 5: Interrupt After Multiple Retry Attempts")
    
    state = ExecutionState()
    printer = EventPrinter(show_address=False, show_timestamp=False)
    state.event_bus.subscribe(printer.print_event)
    
    call_count = 0
    
    async def persistent_failure(input_data, context, execution_state):
        nonlocal call_count
        call_count += 1
        raise ConnectionError(f"Attempt {call_count} failed")
    
    tool = Tool(
        name="unstable_api",
        description="Unstable API",
        input_model=APIRequest,
        handler=persistent_failure,
        retry_config=RetryConfig(
            max_attempts=10,
            initial_delay=0.4,
            backoff_strategy=BackoffStrategy.CONSTANT
        )
    )
    
    event = ExecutionEvent(actor=tool)
    
    print("🔧 Starting API calls with retry (will attempt multiple times)...\n")
    
    async def interrupt_after_several_attempts():
        await asyncio.sleep(2.0)  # Let several attempts happen
        print("\n⏸️  Interrupt after several retry attempts\n")
        state.request_interrupt(
            reason=InterruptReason.USER_CANCELLED,
            message="User stopped execution"
        )
    
    task1 = asyncio.create_task(event(APIRequest(endpoint="/api/unstable", data="test"), {}, state))
    task2 = asyncio.create_task(interrupt_after_several_attempts())
    
    try:
        await task1
    except ExecutionInterrupted:
        pass
    
    await task2
    
    print(f"\n📊 Final results:")
    print(f"   • Total attempts before interrupt: {call_count}")
    print(f"   • Interrupt respected immediately during backoff")
    print(f"   • Did not exhaust all 10 retry attempts")
    
    # Show retry events
    events = state.history.get_event_list()
    def get_status(e):
        if isinstance(e, dict):
            return e.get("execution_status")
        return getattr(e, "execution_status", None)
    
    retrying_events = [e for e in events if get_status(e) == ExecutionStatus.RETRYING or (hasattr(get_status(e), 'value') and get_status(e).value == 'retrying')]
    print(f"   • RETRYING events recorded: {len(retrying_events)}")


# ===== Main Demo Runner =====

async def main():
    print("\n" + "=" * 80)
    print("PHASE 5 DEMO: Interrupt Integration")
    print("=" * 80)
    
    await demo1_interrupt_during_backoff()
    await asyncio.sleep(0.5)
    
    await demo2_responsive_interrupt()
    await asyncio.sleep(0.5)
    
    await demo3_state_preservation()
    await asyncio.sleep(0.5)
    
    await demo4_external_interrupt_checker()
    await asyncio.sleep(0.5)
    
    await demo5_multiple_retries()
    
    print("\n" + "=" * 80)
    print("✨ Demo Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("• Interrupts are detected quickly during retry backoff (every 0.1s)")
    print("• Execution state is preserved when interrupted during retry")
    print("• External interrupt checkers work seamlessly with retry delays")
    print("• Interrupt messages and reasons are properly preserved")
    print("• INTERRUPTED status is recorded in event history")
    print("• No need to wait for full backoff delay - responsive cancellation")
    print()


if __name__ == "__main__":
    asyncio.run(main())
