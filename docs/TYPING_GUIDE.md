"""
GUIDE: Using .pyi Stub Files for Better Type Hints

The problem you had: "unknown" return types when using ExecutionEvent

## How .pyi Files Work

.pyi files are "stub files" that provide type information to type checkers (like Pylance, 
mypy, pyright) WITHOUT changing runtime behavior. This lets you:

1. Keep runtime code simple (no generics)
2. Provide rich type information for IDEs and type checkers
3. Get autocomplete and type safety

## The Current Setup

### Runtime (.py files):
- `ExecutionEvent` - no Generic, simple to use
- `LLM`, `Tool`, `Agent` - no Generic, simple to use
- You create instances without type parameters

### Type Checking (.pyi files):
- `ExecutionEvent[T]` - Generic for type checkers
- `LLM[T, R]`, `Tool[T]`, `Agent[T, R]` - Generic for type checkers
- Type checkers see the generic versions

## The Problem with Automatic Inference

Type checkers CANNOT automatically infer the output type from `actor.output_model` 
because that's a runtime value, not a type-level value.

## The Solution: Type Variables or Explicit Typing

### Option 1: Use reveal_type to check what type checker sees
```python
from rh_agents.core.events import ExecutionEvent
from rh_agents.core.actors import LLM

my_llm = LLM(
    name="test",
    input_model=MyInput,
    output_model=LLM_Result,
    handler=handler
)

event = ExecutionEvent(actor=my_llm)
result = await event(input, "", state)

# Use reveal_type to see what the type checker infers
reveal_type(result)  # ExecutionResult[Unknown] or ExecutionResult[Any]
```

### Option 2: Explicit Type Annotations (RECOMMENDED)
```python
# Tell the type checker explicitly what type you expect
result: ExecutionResult[LLM_Result] = await event(input, "", state)

# Now type checker knows result.result is LLM_Result | None
if result.result:
    content: str = result.result.content  # ✅ Autocomplete works!
```

### Option 3: Use TypeVar with factory functions
```python
from typing import TypeVar

R = TypeVar('R')

def create_llm_event(llm: LLM[Any, R]) -> ExecutionEvent[R]:
    return ExecutionEvent(actor=llm)

# Now type is inferred
event = create_llm_event(my_llm)  # ExecutionEvent[LLM_Result]
result = await event(input, "", state)  # ExecutionResult[LLM_Result]
```

## Why Not Just Add Generics to Runtime Code?

We COULD add `Generic` to the runtime classes:

```python
class ExecutionEvent(BaseModel, Generic[T]):
    ...

# Then you'd use:
event = ExecutionEvent[LLM_Result](actor=my_llm)
```

But this has downsides:
1. More verbose - always need type parameters
2. Pydantic v2 has issues with Generic[T] 
3. Runtime overhead
4. Doesn't match the "simple" design goal

## What DOES Work Well

The current approach works great when you:

1. **Add explicit type annotations:**
   ```python
   result: ExecutionResult[MyType] = await event(...)
   ```

2. **Use with decorators:**
   ```python
   @llm(name="test", output_model=LLM_Result)
   async def my_llm(input: MyInput, ctx: str, state: ExecutionState) -> LLM_Result:
       return LLM_Result(content="...")
   
   # Type is clear from return annotation
   result = await ExecutionEvent(actor=my_llm)(input, "", state)
   ```

3. **Check types at development time:**
   Type checkers like Pylance will warn you if types don't match

## Summary

Your .pyi files ARE working - they provide generic types to type checkers.
But type checkers can't magically infer `R` from `actor.output_model` at runtime.

**Best practice:** Add explicit type annotations when you need them:

```python
# Instead of:
result = await event(...)  # result: ExecutionResult[Unknown]

# Write:
result: ExecutionResult[LLM_Result] = await event(...)  # ✅ Clear types
```

This gives you:
- ✅ Clean runtime code (no Generic[T] noise)
- ✅ Type safety when you need it
- ✅ Autocomplete and IDE support
- ✅ Type checker validation
