"""
Demo: New Intuitive Retry Behavior

Shows how retry now works in a more intuitive way:
- By default, ALL exceptions are retried when retry_config is set
- Use exclude_exceptions to opt-out specific errors
"""
import asyncio
from rh_agents import ExecutionEvent, ExecutionState
from rh_agents.core.retry import RetryConfig
from rh_agents.core.types import EventType
from rh_agents.core.actors import BaseActor
from pydantic import BaseModel


class SimpleInput(BaseModel):
    value: str


async def demo_1_retry_all_by_default():
    """Demo 1: With retry_config, ALL exceptions are retried by default."""
    print("\n" + "="*70)
    print("DEMO 1: Default behavior - Retry ALL exceptions")
    print("="*70)
    
    state = ExecutionState()
    attempt = [0]
    
    async def handler(input_data, context, exec_state):
        attempt[0] += 1
        print(f"  Attempt {attempt[0]}")
        if attempt[0] < 3:
            # Raise ANY exception - it will be retried!
            raise ValueError(f"Random error on attempt {attempt[0]}")
        return "Success!"
    
    actor = BaseActor(
        name="test_actor",
        description="Test actor",
        input_model=SimpleInput,
        event_type=EventType.TOOL_CALL,
        handler=handler
    )
    
    # Just set retry_config - ALL exceptions will be retried!
    event = ExecutionEvent(
        actor=actor,
        retry_config=RetryConfig(max_attempts=3, initial_delay=0.1)
    )
    
    result = await event(input_data="test", extra_context={}, execution_state=state)
    
    print(f"\n✅ Result: {result.result}")
    print(f"   Total attempts: {attempt[0]}")
    print(f"   ValueError was retried even though it's a 'permanent' error!")


async def demo_2_exclude_specific_errors():
    """Demo 2: Use exclude_exceptions to opt-out specific errors."""
    print("\n" + "="*70)
    print("DEMO 2: Exclude specific errors from retry")
    print("="*70)
    
    state = ExecutionState()
    attempt = [0]
    
    async def handler(input_data, context, exec_state):
        attempt[0] += 1
        print(f"  Attempt {attempt[0]}")
        # Raise ValueError - which is in exclude list
        raise ValueError("This should NOT be retried")
    
    actor = BaseActor(
        name="validation_actor",
        description="Validation actor",
        input_model=SimpleInput,
        event_type=EventType.TOOL_CALL,
        handler=handler
    )
    
    # Exclude ValueError from retry
    event = ExecutionEvent(
        actor=actor,
        retry_config=RetryConfig(
            max_attempts=3, 
            exclude_exceptions=[ValueError]  # Opt-out
        )
    )
    
    result = await event(input_data="test", extra_context={}, execution_state=state)
    
    print(f"\n❌ Result: {result.ok}")
    print(f"   Total attempts: {attempt[0]}")
    print(f"   ValueError was NOT retried (as excluded)")


async def demo_3_whitelist_mode():
    """Demo 3: Use retry_on_exceptions for opt-in mode."""
    print("\n" + "="*70)
    print("DEMO 3: Opt-in mode - ONLY retry specific exceptions")
    print("="*70)
    
    state = ExecutionState()
    attempt = [0]
    error_types = [TimeoutError, ValueError]  # Try different errors
    
    async def handler(input_data, context, exec_state):
        attempt[0] += 1
        error = error_types[min(attempt[0]-1, len(error_types)-1)]
        print(f"  Attempt {attempt[0]}: Raising {error.__name__}")
        raise error(f"Test error {attempt[0]}")
    
    actor = BaseActor(
        name="timeout_actor",
        description="Timeout actor",
        input_model=SimpleInput,
        event_type=EventType.TOOL_CALL,
        handler=handler
    )
    
    # ONLY retry TimeoutError (opt-in mode)
    event = ExecutionEvent(
        actor=actor,
        retry_config=RetryConfig(
            max_attempts=3,
            retry_on_exceptions=[TimeoutError]  # Opt-in
        )
    )
    
    result = await event(input_data="test", extra_context={}, execution_state=state)
    
    print(f"\n❌ Result: {result.ok}")
    print(f"   Total attempts: {attempt[0]}")
    print(f"   TimeoutError was retried, but ValueError stopped execution")


async def demo_4_comparison():
    """Demo 4: Compare old vs new behavior."""
    print("\n" + "="*70)
    print("DEMO 4: Why the new behavior is better")
    print("="*70)
    
    print("\n📊 OLD BEHAVIOR (complex, non-intuitive):")
    print("  - Had DEFAULT_RETRYABLE_EXCEPTIONS (TimeoutError, ConnectionError, etc.)")
    print("  - Had DEFAULT_NON_RETRYABLE_EXCEPTIONS (ValueError, TypeError, etc.)")
    print("  - Users had to whitelist Exception to retry everything")
    print("  - Example: raise Exception('test') → NOT retried ❌")
    print("  - Fix required: retry_on_exceptions=[Exception]")
    
    print("\n✨ NEW BEHAVIOR (simple, intuitive):")
    print("  - If retry_config is set → retry ALL exceptions by default ✅")
    print("  - Use exclude_exceptions=[ValueError] to opt-out specific errors")
    print("  - Use retry_on_exceptions=[TimeoutError] for fine-grained control")
    print("  - Example: raise Exception('test') → retried automatically ✅")
    print("  - No fix needed!")
    
    print("\n💡 Key Insight:")
    print("  When you set retry_config, you WANT to retry errors.")
    print("  It makes no sense to require whitelisting every exception type!")


async def main():
    """Run all demos."""
    print("\n" + "🎯 " + "="*66 + " 🎯")
    print("   NEW INTUITIVE RETRY BEHAVIOR - DEMO")
    print("🎯 " + "="*66 + " 🎯")
    
    await demo_1_retry_all_by_default()
    await demo_2_exclude_specific_errors()
    await demo_3_whitelist_mode()
    await demo_4_comparison()
    
    print("\n" + "="*70)
    print("✅ All demos complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
