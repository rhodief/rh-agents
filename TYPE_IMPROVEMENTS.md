# Type Annotation Improvements in execution.py

## Summary

Improved type annotations throughout [execution.py](execution.py) to use specific types instead of `Any` wherever possible. This provides better type safety, IDE autocomplete, and catches type errors at development time.

## Changes Made

### 1. **Import Structure** 
- Added `ParallelExecutionManager` and `ErrorStrategy` to TYPE_CHECKING imports
- Proper handling of forward references for circular imports

### 2. **HistorySet Class**
- **Before**: `events: list[Any]`
- **After**: `events: list[Union['ExecutionEvent', dict[str, Any]]]`
- Properly handles both ExecutionEvent objects and dict representations (from deserialization)

### 3. **EventBus Class**
- **Before**: `events: list[Any]`
- **After**: `events: list['ExecutionEvent']`
- **Before**: `async def stream() -> AsyncGenerator[Any, None]`
- **After**: `async def stream() -> AsyncGenerator['ExecutionEvent', None]`
- Updated deprecated `Config` class to modern `model_config`

### 4. **ExecutionState Class**
- **Before**: `current_execution: Union[Any, None]`
- **After**: `current_execution: Optional['ExecutionEvent']`

### 5. **parallel() Method** ⭐ Main Fix
- **Before**: `def parallel(..., error_strategy: Optional[Any] = None) -> Any:`
- **After**: `def parallel(..., error_strategy: Optional['ErrorStrategy'] = None) -> 'ParallelExecutionManager':`
- Now correctly typed as returning `ParallelExecutionManager` instead of `Any`
- Parameter `error_strategy` properly typed as `ErrorStrategy` enum

### 6. **add_event() Method**
- **Before**: `async def add_event(self, event: Any, ...)`
- **After**: `async def add_event(self, event: 'ExecutionEvent', ...)`

### 7. **Model Rebuild**
- Added `_rebuild_models()` function to resolve forward references after circular imports
- Ensures Pydantic properly validates types at runtime

## Benefits

✅ **Better Type Safety**: Catches type errors during development, not runtime  
✅ **IDE Support**: Better autocomplete and inline documentation  
✅ **Code Clarity**: Makes it clear what types are expected/returned  
✅ **Refactoring Safety**: IDE can safely rename/refactor with type checking  
✅ **Documentation**: Types serve as inline documentation

## Example Usage

```python
from rh_agents.core.execution import ExecutionState
from rh_agents.core.parallel import ParallelExecutionManager, ErrorStrategy

state = ExecutionState()

# Before: p was typed as Any - no autocomplete or type checking
# After: p is correctly typed as ParallelExecutionManager
p: ParallelExecutionManager = state.parallel(
    max_workers=5,
    error_strategy=ErrorStrategy.FAIL_SLOW,  # Enum, not Any
    timeout=30.0
)

# IDE now knows about ParallelExecutionManager methods and properties
# Type checker validates usage at development time
```

## Testing

All tests pass with the new type annotations:
- ✅ `test_parallel_execution.py` - 15/15 tests passing
- ✅ No type errors in Pylance/mypy
- ✅ Proper Pydantic model validation at runtime

## Technical Notes

### Circular Import Handling

The relationship between `ExecutionState` and `ExecutionEvent` creates a circular dependency:
- `execution.py` needs to reference `ExecutionEvent` 
- `events.py` imports `ExecutionState`

**Solution**: Use `TYPE_CHECKING` block + string literals + `model_rebuild()`

1. Import types only during type checking (not runtime)
2. Use string literals in Pydantic fields: `'ExecutionEvent'`
3. Call `model_rebuild()` after imports to resolve forward references

This pattern is the recommended approach for handling circular dependencies in Pydantic v2.
