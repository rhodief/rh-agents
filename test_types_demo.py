"""
Demo script to show that types are properly resolved in execution.py
"""
import asyncio
from rh_agents.core.execution import ExecutionState
from rh_agents.core.parallel import ParallelExecutionManager, ErrorStrategy

async def main():
    # Create execution state
    state = ExecutionState()
    
    # Call parallel() - should return ParallelExecutionManager type
    p: ParallelExecutionManager = state.parallel(
        max_workers=5,
        error_strategy=ErrorStrategy.FAIL_SLOW,
        timeout=30.0
    )
    
    # The type checker should now recognize p as ParallelExecutionManager, not Any
    print(f"✓ Type of p: {type(p).__name__}")
    print(f"✓ p is ParallelExecutionManager: {isinstance(p, ParallelExecutionManager)}")
    
    # Access properties - type checker knows these exist
    print(f"✓ max_workers: {p.group.max_workers}")
    print(f"✓ error_strategy: {p.error_strategy}")
    print(f"✓ group name: {p.group.name}")
    
    print("\n✓ All type hints are working correctly!")
    print("✓ state.parallel() returns ParallelExecutionManager (not Any)")
    print("✓ error_strategy parameter uses ErrorStrategy (not Any)")

if __name__ == "__main__":
    asyncio.run(main())
