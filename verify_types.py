"""
Type checker verification - this should have NO type errors
"""
from rh_agents.core.execution import ExecutionState
from rh_agents.core.parallel import ParallelExecutionManager, ErrorStrategy

async def example_usage():
    state = ExecutionState()
    
    # state.parallel() returns ParallelExecutionManager, not Any
    parallel_manager: ParallelExecutionManager = state.parallel(
        max_workers=10,
        error_strategy=ErrorStrategy.FAIL_FAST,
        timeout=60.0,
        name="Test Group"
    )
    
    # Type checker knows these properties exist
    assert parallel_manager.error_strategy == ErrorStrategy.FAIL_FAST
    assert parallel_manager.timeout == 60.0
    assert parallel_manager.group.name == "Test Group"
    assert parallel_manager.group.max_workers == 10
    
    # Type checker knows these methods exist
    async with parallel_manager as p:
        # p is also typed as ParallelExecutionManager
        assert isinstance(p, ParallelExecutionManager)
    
    print("✓ All type hints verified successfully!")

# This demonstrates the key improvement:
# Before: p would be typed as Any, providing no autocomplete or type safety
# After: p is properly typed as ParallelExecutionManager with full type checking
