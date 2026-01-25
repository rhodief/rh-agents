# .pyi Stub Files - Complete Solution Summary

## Problem You Had
When using `.pyi` files for dynamic typing without generics in main files, you were getting "unknown" return types that your IDE couldn't autocomplete.

## Root Cause
The issue is that Python type checkers **cannot infer generic type parameters from runtime values**. When you write:

```python
event = ExecutionEvent(actor=my_llm)
result = await event(input, "", state)
```

The type checker sees `actor` as a runtime value and can't extract the type parameter `R` from `my_llm.output_model` to determine that `result` should be `ExecutionResult[LLM_Result]`.

## What Was Fixed

### 1. Enhanced `.pyi` Files with Proper Generics

Updated [rh_agents/core/actors.pyi](rh_agents/core/actors.pyi):
- Added `Generic[T, R]` to `LLM`
- Added `Generic[T]` to `Tool`
- Added `Generic[T, R]` to `Agent`
- Added overloads for `__init__` methods
- Made `output_model` properly typed

Updated [rh_agents/core/events.pyi](rh_agents/core/events.pyi):
- Added overloads for `ExecutionEvent.__init__` based on actor type
- Added proper type parameter to `__call__` method
- Kept `Generic[T]` for type checkers

### 2. How It Works Now

**Runtime (`.py` files):**
```python
# Simple, no generics needed
my_llm = LLM(
    name="test",
    input_model=MyInput,
    output_model=LLM_Result,
    handler=handler
)

event = ExecutionEvent(actor=my_llm)
result = await event(input, "", state)
```

**Type Checking (`.pyi` files seen by IDE/type checker):**
```python
# Type checker sees generics
class LLM(BaseActor, Generic[T, R]):
    output_model: type[R]
    ...

class ExecutionEvent(Generic[T]):
    def __call__(self: ExecutionEvent[R], ...) -> ExecutionResult[R]: ...
```

### 3. Best Practices for Users

#### Option A: Explicit Type Annotation (RECOMMENDED ✅)
```python
# Tell the type checker what type you expect
result: ExecutionResult[LLM_Result] = await event(input, "", state)

# Now IDE knows result.result is LLM_Result | None
if result.result:
    content = result.result.content  # ✅ Full autocomplete!
```

#### Option B: Runtime Type Check + Type Narrowing
```python
result = await event(input, "", state)

if result.result and isinstance(result.result, LLM_Result):
    # Type narrowing - IDE knows this is LLM_Result
    content = result.result.content  # ✅ Autocomplete works
```

#### Option C: Access output_model from Actor
```python
# Document the expected type
# Type: ExecutionResult[actor.output_model]
result = await event(input, "", state)
```

## Why This Approach?

### ✅ Advantages:
1. **Clean runtime code** - No `Generic[T]` noise in actual usage
2. **Type safety when needed** - Add annotations where important
3. **IDE support** - Full autocomplete with explicit types
4. **Backward compatible** - Existing code works unchanged
5. **Follows Pydantic best practices** - Generics can be tricky with Pydantic v2

### ❌ Why Not Generics Everywhere?
```python
# This would be more verbose:
event = ExecutionEvent[LLM_Result](actor=my_llm)  # Need type param
result: ExecutionResult[LLM_Result] = await event(...)  # Redundant

# Issues:
# - More typing required everywhere
# - Pydantic v2 generic issues
# - Runtime overhead
# - Doesn't match "simple API" design goal
```

## Examples

See these files for working examples:
- [test_typing_check.py](test_typing_check.py) - Basic verification
- [examples/typing_best_practices.py](examples/typing_best_practices.py) - Comprehensive examples
- [docs/TYPING_GUIDE.md](docs/TYPING_GUIDE.md) - Detailed guide

## Running Examples

```bash
# Run the typing example
python examples/typing_best_practices.py

# Run with type checker
pyright examples/typing_best_practices.py
# or
mypy examples/typing_best_practices.py
```

## Summary

Your `.pyi` files ARE working correctly! The "unknown" types issue is resolved by:

1. **For Development**: Add explicit type annotations where you need autocomplete
   ```python
   result: ExecutionResult[LLM_Result] = await event(...)
   ```

2. **For Production**: Code works identically with or without annotations
   ```python
   result = await event(...)  # Works fine, just less IDE support
   ```

3. **Type Checkers**: See generic types from `.pyi` files and validate correctly

The key insight: **Type checkers can't infer types from runtime values**, so you need to either:
- Add explicit annotations (best for IDE experience)
- Use runtime type checks with `isinstance()` (for type narrowing)
- Accept that IDE won't have full autocomplete (still works at runtime)

This is actually the **recommended pattern** for Python libraries that want:
- Simple, clean runtime API
- Rich type information for those who want it
- Flexibility for different use cases
