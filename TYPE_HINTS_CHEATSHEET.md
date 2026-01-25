# Type Hints Cheat Sheet

## Quick Reference: Getting Good Types

### ✅ DO: Add explicit type annotations
```python
result: ExecutionResult[LLM_Result] = await event(input, "", state)
```

### ❌ DON'T: Expect automatic type inference
```python
result = await event(input, "", state)  # Type will be Unknown
```

## Common Patterns

### Pattern 1: LLM Calls
```python
from rh_agents.core.result_types import LLM_Result
from rh_agents.core.events import ExecutionResult

my_llm = LLM(
    name="gpt4",
    input_model=QueryInput,
    output_model=LLM_Result,
    handler=handler
)

event = ExecutionEvent(actor=my_llm)

# ✅ Add type annotation
result: ExecutionResult[LLM_Result] = await event(input, "", state)

if result.result:
    content: str = result.result.content  # Autocomplete works!
```

### Pattern 2: Agent Calls with Custom Output
```python
class CustomOutput(BaseModel):
    answer: str
    confidence: float

my_agent = Agent(
    name="analyzer",
    input_model=QueryInput,
    output_model=CustomOutput,
    handler=handler
)

event = ExecutionEvent(actor=my_agent)

# ✅ Add type annotation
result: ExecutionResult[CustomOutput] = await event(input, "", state)

if result.result:
    answer: str = result.result.answer  # Autocomplete works!
```

### Pattern 3: Tool Calls (Returns Any)
```python
my_tool = Tool(
    name="calculator",
    input_model=CalcInput,
    handler=handler
)

event = ExecutionEvent(actor=my_tool)

# Tools return Any - no specific type
result: ExecutionResult[Any] = await event(input, "", state)

# Use runtime checks
if result.result:
    value = result.result  # Any type
```

### Pattern 4: Runtime Type Checking
```python
# No type annotation, use isinstance
result = await event(input, "", state)

if result.result and isinstance(result.result, LLM_Result):
    # Type narrowing - IDE knows it's LLM_Result here
    content = result.result.content  # Works!
```

### Pattern 5: Multiple Event Calls
```python
# Type each result
result1: ExecutionResult[LLM_Result] = await event1(...)
result2: ExecutionResult[CustomOutput] = await event2(...)
result3: ExecutionResult[LLM_Result] = await event3(...)

# Now all have proper types
if result1.result:
    print(result1.result.content)  # ✅

if result2.result:
    print(result2.result.answer)  # ✅
```

## IDE Autocomplete

### With type annotation ✅
```python
result: ExecutionResult[LLM_Result] = await event(...)

if result.result:
    # IDE shows all LLM_Result fields:
    result.result.content          # ✅
    result.result.tools            # ✅
    result.result.tokens_used      # ✅
    result.result.model_name       # ✅
    result.result.error_message    # ✅
```

### Without type annotation ❌
```python
result = await event(...)

if result.result:
    # IDE shows: Unknown
    result.result.???  # No autocomplete ❌
```

## Type Checking Commands

```bash
# Check types with Pyright (recommended)
pyright your_file.py

# Check types with Mypy
mypy your_file.py

# VS Code with Pylance
# Types are checked automatically in the editor
```

## Common Issues

### Issue 1: "Unknown" return type
**Problem:** `result = await event(...)`  
**Solution:** Add type annotation: `result: ExecutionResult[LLM_Result] = await event(...)`

### Issue 2: No autocomplete on result.result
**Problem:** Type annotation missing  
**Solution:** Add: `result: ExecutionResult[YourType] = ...`

### Issue 3: "Cannot parameterize LLM" error
**Problem:** `LLM[Input, Output](...)`  
**Solution:** Remove type parameters: `LLM(...)` (they're only in .pyi files)

### Issue 4: Type mismatch warnings
**Problem:** Type checker shows errors  
**Solution:** Verify output_model matches your type annotation

## Remember

1. **Runtime code is simple** - no generics needed
2. **Add types for IDE support** - explicit annotations
3. **Type checkers are helpful** - run pyright/mypy
4. **Types don't affect runtime** - code works either way

## Quick Test

```python
# Copy this to test your setup:
async def test_types():
    state = ExecutionState()
    event = ExecutionEvent(actor=my_llm)
    
    # Add type annotation here:
    result: ExecutionResult[LLM_Result] = await event(input, "", state)
    
    # This should have autocomplete:
    if result.result:
        print(result.result.content)  # ✅
```

## More Info

- Full guide: [docs/TYPING_GUIDE.md](docs/TYPING_GUIDE.md)
- Examples: [examples/typing_best_practices.py](examples/typing_best_practices.py)
- Summary: [PYI_SOLUTION_SUMMARY.md](PYI_SOLUTION_SUMMARY.md)
