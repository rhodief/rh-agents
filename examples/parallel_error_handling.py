#!/usr/bin/env python3
"""
Error Handling Strategies Example

This example demonstrates the different error handling strategies:
- FAIL_SLOW: Collect all results, including errors (default)
- FAIL_FAST: Stop on first error, cancel remaining tasks
- Retry logic with exponential backoff
- Circuit breaker pattern
"""

import asyncio
import random
from rh_agents.core.execution import ExecutionState
from rh_agents.core.parallel import ErrorStrategy


async def unreliable_task(task_id: int, failure_rate: float = 0.3) -> str:
    """
    Simulate an unreliable task that sometimes fails.
    """
    await asyncio.sleep(0.2)
    
    if random.random() < failure_rate:
        raise ConnectionError(f"Task {task_id} failed - network error")
    
    return f"Task {task_id} completed successfully"


async def always_failing_task(task_id: int) -> str:
    """Task that always fails."""
    await asyncio.sleep(0.1)
    raise ValueError(f"Task {task_id} always fails")


async def slow_task(task_id: int) -> str:
    """Slow task to demonstrate cancellation."""
    await asyncio.sleep(5.0)
    return f"Task {task_id} completed (slow)"


async def example_fail_slow():
    """Demonstrate FAIL_SLOW strategy (default)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: FAIL_SLOW Strategy (Default)")
    print("=" * 70)
    print("Collects all results, even if some tasks fail.")
    print()
    
    state = ExecutionState()
    
    async with state.parallel(
        max_workers=3,
        error_strategy=ErrorStrategy.FAIL_SLOW,
        name="Fail Slow Demo"
    ) as p:
        for i in range(5):
            p.add(lambda i=i: unreliable_task(i, failure_rate=0.4))
        
        results = await p.gather()
    
    successful = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    
    print(f"Results: {len(successful)} succeeded, {len(failed)} failed")
    print()
    print("Successful tasks:")
    for r in successful:
        print(f"  ✓ {r.result}")
    
    print()
    print("Failed tasks:")
    for r in failed:
        print(f"  ✗ {r.erro_message}")
    
    print()
    print("Use FAIL_SLOW when:")
    print("  • You want partial results even if some tasks fail")
    print("  • Tasks are independent and failures don't invalidate others")
    print("  • You need visibility into all failures for debugging")


async def example_fail_fast():
    """Demonstrate FAIL_FAST strategy."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: FAIL_FAST Strategy")
    print("=" * 70)
    print("Stops immediately on first error, cancels remaining tasks.")
    print()
    
    state = ExecutionState()
    
    try:
        async with state.parallel(
            max_workers=2,
            error_strategy=ErrorStrategy.FAIL_FAST,
            name="Fail Fast Demo"
        ) as p:
            # Add a failing task and slow tasks
            p.add(lambda: always_failing_task(1))
            p.add(lambda: slow_task(2))
            p.add(lambda: slow_task(3))
            
            results = await p.gather()
        
        print("Should not reach here!")
    
    except ValueError as e:
        print(f"✓ Execution stopped on first error: {e}")
        print()
        print("Remaining tasks were cancelled (didn't wait 5 seconds)")
    
    print()
    print("Use FAIL_FAST when:")
    print("  • One failure invalidates all other work")
    print("  • You want to save resources and stop immediately")
    print("  • Tasks have dependencies or are part of a transaction")


async def example_retry():
    """Demonstrate retry logic with exponential backoff."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Retry Logic with Exponential Backoff")
    print("=" * 70)
    print("Automatically retries failed tasks with increasing delays.")
    print()
    
    state = ExecutionState()
    attempt_count = {}
    
    async def flaky_task(task_id: int):
        """Task that fails first 2 attempts, succeeds on 3rd."""
        if task_id not in attempt_count:
            attempt_count[task_id] = 0
        attempt_count[task_id] += 1
        
        await asyncio.sleep(0.1)
        
        if attempt_count[task_id] < 3:
            print(f"  Task {task_id} attempt {attempt_count[task_id]} - FAILED")
            raise ConnectionError(f"Task {task_id} attempt {attempt_count[task_id]} failed")
        
        print(f"  Task {task_id} attempt {attempt_count[task_id]} - SUCCESS")
        return f"Task {task_id} succeeded after {attempt_count[task_id]} attempts"
    
    async with state.parallel(
        max_workers=2,
        max_retries=3,
        retry_delay=0.1,  # Base delay: 0.1s, then 0.2s, then 0.4s (exponential)
        name="Retry Demo"
    ) as p:
        for i in range(3):
            p.add(lambda i=i: flaky_task(i))
        
        results = await p.gather()
    
    print()
    for r in results:
        if r.ok:
            print(f"✓ {r.result}")
    
    print()
    print("Retry backoff: 0.1s → 0.2s → 0.4s (exponential)")
    print()
    print("Use retries when:")
    print("  • Failures are transient (network glitches, rate limits)")
    print("  • External service might recover quickly")
    print("  • Cost of retry is lower than cost of failure")


async def example_circuit_breaker():
    """Demonstrate circuit breaker pattern."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Circuit Breaker Pattern")
    print("=" * 70)
    print("Stops calling failing service after threshold to prevent cascading failures.")
    print()
    
    state = ExecutionState()
    
    async with state.parallel(
        max_workers=1,  # Sequential to see circuit breaker clearly
        circuit_breaker_threshold=3,
        error_strategy=ErrorStrategy.FAIL_SLOW,
        name="Circuit Breaker Demo"
    ) as p:
        for i in range(6):
            p.add(lambda i=i: always_failing_task(i))
        
        results = await p.gather()
    
    print()
    print("Results:")
    for i, r in enumerate(results):
        if "Circuit breaker" in r.erro_message:
            print(f"  Task {i}: ⚡ REJECTED BY CIRCUIT BREAKER")
        else:
            print(f"  Task {i}: ✗ Failed (circuit still closed)")
    
    print()
    print(f"First 3 tasks tried and failed")
    print(f"Circuit breaker opened - remaining tasks rejected immediately")
    print()
    print("Use circuit breaker when:")
    print("  • Calling external services that might go down")
    print("  • Want to prevent cascading failures")
    print("  • Need to fail fast when service is known to be down")


async def main():
    print("=" * 70)
    print("ERROR HANDLING STRATEGIES")
    print("=" * 70)
    
    # Run all examples
    await example_fail_slow()
    await example_fail_fast()
    await example_retry()
    await example_circuit_breaker()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Choose your error strategy based on requirements:")
    print()
    print("FAIL_SLOW (default):")
    print("  ✓ Get partial results even with failures")
    print("  ✓ See all errors for debugging")
    print("  ✗ Wastes resources on doomed work")
    print()
    print("FAIL_FAST:")
    print("  ✓ Save resources, stop immediately")
    print("  ✓ Good for transactional work")
    print("  ✗ Lose partial results")
    print()
    print("RETRY:")
    print("  ✓ Handle transient failures automatically")
    print("  ✓ Improve success rate")
    print("  ✗ Adds latency")
    print()
    print("CIRCUIT BREAKER:")
    print("  ✓ Protect against cascading failures")
    print("  ✓ Fast failure when service is down")
    print("  ✗ May reject valid requests during recovery")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    asyncio.run(main())
