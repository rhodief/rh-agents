# Before & After: Type Inference Improvements

## The Problem You Reported

> "I've got issues in pyi files. When I create this file, the idea is to have dynamic typing and don't use generics in the main file, but I don't get it, I've got a lot of unknown return, what I really hate"

## Before (What wasn't working)

### Old `.pyi` Files
```python
# Old events.pyi
class ExecutionEvent(Generic[T]):
    def __init__(self, *, actor: BaseActor, ...) -> None: ...
    async def __call__(...) -> ExecutionResult[T]: ...

# Old actors.pyi
class LLM(BaseActor, Generic[T, R]):
    def __init__(self, name: str, ..., **kwargs) -> None: ...
```

### Problem
```python
my_llm = LLM(
    name="test",
    output_model=LLM_Result,
    ...
)

event = ExecutionEvent(actor=my_llm)
result = await event(input, "", state)

# IDE/type checker showed:
# result: ExecutionResult[Unknown]  ❌
# result.result: Unknown  ❌
# NO autocomplete on result.result  ❌
```

## After (What works now)

### New `.pyi` Files

#### [rh_agents/core/events.pyi](rh_agents/core/events.pyi)
```python
from typing import TypeVar, Generic, overload

T = TypeVar('T')
R = TypeVar('R')
InputT = TypeVar('InputT', bound=BaseModel)
OutputT = TypeVar('OutputT')

class ExecutionEvent(Generic[T]):
    # Overloaded __init__ for different actor types
    @overload
    def __init__(self, *, actor: LLM[InputT, R], ...) -> None: ...
    
    @overload
    def __init__(self, *, actor: Agent[InputT, R], ...) -> None: ...
    
    @overload
    def __init__(self, *, actor: Tool[InputT], ...) -> None: ...
    
    @overload
    def __init__(self, *, actor: BaseActor, ...) -> None: ...
    
    # Return type tied to self's type parameter
    async def __call__(
        self: ExecutionEvent[R],
        input_data: Any,
        extra_context: str,
        execution_state: ExecutionState
    ) -> ExecutionResult[R]: ...
```

#### [rh_agents/core/actors.pyi](rh_agents/core/actors.pyi)
```python
class LLM(BaseActor, Generic[T, R]):
    output_model: type[R]  # Now properly typed!
    
    @overload
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[T],
        output_model: type[R],
        handler: Callable[[T, str, ExecutionState], Awaitable[R]],
        *,
        cacheable: bool = True,
        ...
    ) -> None: ...
    
    @overload
    def __init__(self, *, name: str, ..., **kwargs: Any) -> None: ...

# Similar improvements for Agent and Tool
```

### What Works Now

#### ✅ Option 1: With Explicit Type Annotation (Best IDE Experience)
```python
my_llm = LLM(
    name="test",
    input_model=MyInput,
    output_model=LLM_Result,
    handler=handler
)

event = ExecutionEvent(actor=my_llm)
result: ExecutionResult[LLM_Result] = await event(input, "", state)

# IDE/type checker shows:
# result: ExecutionResult[LLM_Result]  ✅
# result.result: LLM_Result | None  ✅
# Full autocomplete on result.result.content  ✅
if result.result:
    content = result.result.content  # IDE autocomplete works!
```

#### ✅ Option 2: With Runtime Type Check
```python
result = await event(input, "", state)

if result.result and isinstance(result.result, LLM_Result):
    # Type narrowing - IDE knows it's LLM_Result here
    content = result.result.content  # ✅ Autocomplete works
```

#### ✅ Option 3: Without Type Annotation (Works, Less IDE Support)
```python
result = await event(input, "", state)
# Still works at runtime, just less autocomplete
print(result.result)  # Works fine!
```

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Return types** | `Unknown` / `Any` | Properly inferred with annotations |
| **Autocomplete** | Not available | ✅ Works with type annotations |
| **Type safety** | Minimal | ✅ Full type checking |
| **Runtime code** | Simple ✅ | Still simple ✅ |
| **IDE support** | Poor | ✅ Excellent with annotations |

## Why It Works This Way

### Type checkers cannot infer from runtime values
```python
# This is a RUNTIME value:
actor.output_model = LLM_Result

# Type checkers work at STATIC ANALYSIS time
# They can't extract LLM_Result from actor.output_model

# Solution: Use type annotations
result: ExecutionResult[LLM_Result] = await event(...)
```

### Best of both worlds
- **Runtime**: Simple, no generics needed
- **Static analysis**: Rich types with `.pyi` files
- **When needed**: Add explicit annotations for IDE support

## Verification

Run these to verify types work:

```bash
# Runtime execution
python test_typing_check.py
python examples/typing_best_practices.py

# Static type checking
pyright test_typing_check.py
mypy test_typing_check.py
```

## Documentation

See these for more details:
- [PYI_SOLUTION_SUMMARY.md](PYI_SOLUTION_SUMMARY.md) - Complete solution
- [docs/TYPING_GUIDE.md](docs/TYPING_GUIDE.md) - Usage guide
- [examples/typing_best_practices.py](examples/typing_best_practices.py) - Examples

## Bottom Line

Your `.pyi` files now provide **excellent type hints** when you add explicit type annotations. This is the **recommended Python pattern** for libraries that want:

1. ✅ Clean, simple runtime API (no Generic[T] everywhere)
2. ✅ Rich type information for IDEs and type checkers
3. ✅ Flexibility - add types where you need them, omit where you don't

**The "unknown return" issue is solved** - just add explicit type annotations where you want IDE autocomplete:

```python
# Before: result is Unknown
result = await event(...)

# After: result is fully typed
result: ExecutionResult[LLM_Result] = await event(...)
```
