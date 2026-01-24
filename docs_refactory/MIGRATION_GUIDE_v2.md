# Migration Guide: v1.x to v2.0

## Overview

RH-Agents v2.0 is a major refactoring that removes deprecated code, simplifies APIs, and improves type safety. This guide helps you migrate from v1.x to v2.0.

**Key Changes:**
- ✅ Removed deprecated cache system
- ✅ Simplified generic types in public API
- ✅ Added decorator-based actor creation
- ✅ Added validation helpers
- ✅ Added builder pattern for complex actors
- ✅ Improved type safety with `.pyi` stub files

---

## Breaking Changes

### 1. Removed Deprecated Cache System

**What was removed:**
- `rh_agents.core.cache` module
- `rh_agents.cache_backends` module
- `CacheBackend` class
- `InMemoryCacheBackend` class
- `FileCacheBackend` class

**Why:**
The old hash-based cache system was replaced by the state recovery system, which provides better persistence, replay capabilities, and artifact management.

**Migration:**

```python
# ❌ OLD (deprecated, no longer works)
from rh_agents.cache_backends import FileCacheBackend
cache = FileCacheBackend(".cache")
state = ExecutionState(cache_backend=cache)

# ✅ NEW (use state recovery system)
from rh_agents import FileSystemStateBackend, FileSystemArtifactBackend

state_backend = FileSystemStateBackend(".state_store")
artifact_backend = FileSystemArtifactBackend(".artifacts")
state = ExecutionState(
    state_backend=state_backend,
    artifact_backend=artifact_backend
)

# Save checkpoint
await state.save_checkpoint()

# Restore from checkpoint
restored_state = ExecutionState.load_from_state_id(
    state_id,
    state_backend=state_backend
)
```

**Important:** The `cacheable` field in actors is **NOT deprecated**. It's used by the state recovery system to determine if actor results should be cached during replay.

### 2. Generic Types Removed from Public API

**What changed:**
- `ExecutionEvent[OutputT]` → `ExecutionEvent`
- `ExecutionResult[T]` → `ExecutionResult`
- `LLM[T]` → `LLM`

**Why:**
Simplified instantiation and reduced boilerplate while maintaining type safety through `.pyi` stub files for type checkers.

**Migration:**

```python
# ❌ OLD
event = ExecutionEvent[LLM_Result](actor=llm)
result: ExecutionResult[LLM_Result] = await event(input_data, state, context)

# ✅ NEW
event = ExecutionEvent(actor=llm)
result: ExecutionResult = await event(input_data, state, context)

# Type information still available:
output_type = llm.output_model  # LLM_Result
```

**Note:** Type stub files (`.pyi`) still provide full generic types for static type checkers (mypy, pyright), so IDE autocomplete and type checking continue to work perfectly.

### 3. Agent_Result Removed

**What was removed:**
`rh_agents.core.result_types.Agent_Result`

**Why:**
It was never used in the codebase. Agents return domain-specific result types directly.

**Migration:**
If you were using it (unlikely), define your own domain-specific result types:

```python
# ❌ OLD
class MyOutput(Agent_Result):
    data: str

# ✅ NEW
class MyOutput(BaseModel):
    data: str
    success: bool = True
    error: str | None = None
```

### 4. ToolSet API Simplified

**What changed:**
- Removed `get_tool_list()` method
- Simplified internal implementation (single list storage instead of dual list+dict)

**Migration:**

```python
# ❌ OLD
tools = tool_set.get_tool_list()

# ✅ NEW (multiple options)
tools = list(tool_set)           # Convert to list
tools = tool_set.tools           # Access tools property directly
tool = tool_set["tool_name"]     # Access by name still works
```

### 5. EventStreamer API Updated

**What changed:**
Removed `cache_backend` parameter from `stream()` method in EventStreamer.

**Migration:**

```python
# ❌ OLD
stream = streamer.stream(execution_task=task, cache_backend=cache)

# ✅ NEW
stream = streamer.stream(execution_task=task)
```

---

## Non-Breaking Improvements

### 1. Public API Exports (Added in v1.1.0)

You can now import from top-level `rh_agents` package:

```python
# ✅ NEW (recommended)
from rh_agents import Agent, Tool, LLM, ExecutionState, EventPrinter

# ✅ OLD (still works for backward compatibility)
from rh_agents.core.actors import Agent, Tool, LLM
from rh_agents.core.execution import ExecutionState
from rh_agents.bus_handlers import EventPrinter
```

### 2. Decorator-Based Actor Creation (Added in v1.5.0)

New FastAPI-like decorator syntax for creating actors:

```python
from rh_agents import tool_decorator, agent_decorator, ExecutionState
from rh_agents.core.result_types import Tool_Result
from pydantic import BaseModel

class CalculatorInput(BaseModel):
    a: float
    b: float

@tool_decorator(name="calculator", description="Adds two numbers")
async def calculate(input: CalculatorInput, context: str, state: ExecutionState) -> Tool_Result:
    return Tool_Result(output=input.a + input.b, tool_name="calculator")

# calculate is now a Tool instance
print(calculate.name)  # "calculator"
```

### 3. Validation Helpers (Added in v1.5.0)

Validate actor configurations and execution state:

```python
from rh_agents import validate_actor, validate_state, ActorValidationError

try:
    validate_actor(my_tool)
    validate_state(execution_state)
except ActorValidationError as e:
    print(f"Configuration error: {e}")
```

### 4. Builder Pattern (Added in v1.5.0)

Fluent API for constructing complex actors:

```python
from rh_agents import AgentBuilder, ToolBuilder

agent = (
    AgentBuilder()
    .name("MyAgent")
    .description("Does cool stuff")
    .input_model(InputModel)
    .output_model(OutputModel)
    .handler(my_handler)
    .with_llm(my_llm)
    .with_tools([tool1, tool2])
    .cacheable(True)
    .build()
)
```

### 5. Improved Type Safety

Type stub files (`.pyi`) provide full generic type information for type checkers without runtime overhead:

```python
# Type checkers see full generic types
event: ExecutionEvent[LLM_Result] = ExecutionEvent(actor=llm)

# Runtime is simple (no generic parameters)
event = ExecutionEvent(actor=llm)
```

### 6. ActorOutput Protocol (Added in v1.5.0)

Protocol for generic code that works with any actor output:

```python
from rh_agents.core.result_types import ActorOutput

def process_result(result: ActorOutput):
    if result.success:
        print("Success!")
    else:
        print(f"Error: {result.error}")

# Works with LLM_Result and Tool_Result
process_result(llm_result)
process_result(tool_result)
```

---

## Automated Migration Script

Use this script to automatically update your codebase:

```python
#!/usr/bin/env python3
"""
Automated migration script for RH-Agents v1.x to v2.0
"""
import re
from pathlib import Path
import sys


def migrate_file(file_path: Path) -> bool:
    """Migrate a single Python file. Returns True if changes were made."""
    try:
        content = file_path.read_text()
        original = content
        
        # 1. Remove deprecated cache imports
        content = re.sub(
            r'from rh_agents\.cache_backends import [^\n]+\n',
            '',
            content
        )
        content = re.sub(
            r'from rh_agents\.core\.cache import [^\n]+\n',
            '',
            content
        )
        
        # 2. Update to use state backends
        content = content.replace(
            'FileCacheBackend',
            'FileSystemStateBackend  # MIGRATED: use state recovery'
        )
        content = content.replace(
            'InMemoryCacheBackend',
            '# MIGRATED: InMemoryCacheBackend removed, use FileSystemStateBackend'
        )
        
        # 3. Remove generic type parameters from ExecutionEvent/Result
        content = re.sub(r'ExecutionEvent\[[^\]]+\]', 'ExecutionEvent', content)
        content = re.sub(r'ExecutionResult\[[^\]]+\]', 'ExecutionResult', content)
        
        # 4. Replace get_tool_list() calls
        content = content.replace('.get_tool_list()', '.tools  # or list(toolset)')
        
        # 5. Update imports to use top-level package
        old_imports = [
            ('from rh_agents.core.actors import', 'from rh_agents import'),
            ('from rh_agents.core.execution import', 'from rh_agents import'),
            ('from rh_agents.core.result_types import', 'from rh_agents import'),
        ]
        for old, new in old_imports:
            content = content.replace(old, new)
        
        # Write back if changed
        if content != original:
            file_path.write_text(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error migrating {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Run migration on all Python files in current directory."""
    python_files = list(Path('.').rglob('*.py'))
    
    print(f"Found {len(python_files)} Python files")
    print("Starting migration...\n")
    
    migrated = []
    for py_file in python_files:
        # Skip virtual environments and build directories
        if any(part in py_file.parts for part in ['venv', '.venv', 'env', 'build', 'dist', '__pycache__']):
            continue
        
        if migrate_file(py_file):
            migrated.append(py_file)
            print(f"✓ Migrated: {py_file}")
    
    print(f"\nMigration complete!")
    print(f"Modified {len(migrated)} files")
    
    if migrated:
        print("\n⚠️  IMPORTANT: Review changes and test thoroughly!")
        print("Some migrations may need manual adjustment.")


if __name__ == "__main__":
    main()
```

Save as `migrate_to_v2.py` and run:
```bash
python migrate_to_v2.py
```

---

## Testing Your Migration

After migrating, follow these steps:

### 1. Run Type Checker
```bash
mypy your_project/
```

### 2. Run Tests
```bash
pytest tests/
```

### 3. Manual Testing Checklist

- [ ] All imports resolve correctly
- [ ] No references to `CacheBackend` or `cache_backends`
- [ ] State recovery works (save/restore checkpoints)
- [ ] Agent execution produces expected results
- [ ] Event streaming works (if using EventStreamer)
- [ ] Tool and LLM calls execute correctly

### 4. Common Issues

**Issue:** Import error for `FileCacheBackend`
```python
ModuleNotFoundError: No module named 'rh_agents.cache_backends'
```
**Solution:** Replace with `FileSystemStateBackend`:
```python
from rh_agents import FileSystemStateBackend
state_backend = FileSystemStateBackend(".state_store")
```

**Issue:** Type errors with generic types
```python
error: "ExecutionEvent" expects no type arguments, but 1 given
```
**Solution:** Remove type parameters:
```python
# Before: event: ExecutionEvent[LLM_Result] = ...
# After:  event: ExecutionEvent = ...
```

**Issue:** `get_tool_list()` not found
```python
AttributeError: 'ToolSet' object has no attribute 'get_tool_list'
```
**Solution:** Use `.tools` property or convert to list:
```python
# Before: tools = toolset.get_tool_list()
# After:  tools = toolset.tools
```

---

## Version Summary

### v2.0.0 (Breaking Changes)
- ❌ Removed deprecated cache system (`cache.py`, `cache_backends.py`)
- ❌ Removed `Agent_Result` class
- ✅ Updated `EventStreamer.stream()` signature
- ✅ Simplified `ToolSet` API

### v1.5.0 (Non-Breaking)
- ✅ Removed generic types from public API (runtime)
- ✅ Added `.pyi` stub files for type safety
- ✅ Added decorator-based actor creation
- ✅ Added validation helpers
- ✅ Added builder pattern
- ✅ Created `ActorOutput` protocol

### v1.1.0 (Non-Breaking)
- ✅ Added public API exports to top-level package
- ✅ Enhanced documentation

---

## Need Help?

- **Documentation:** See `docs/` directory
- **Examples:** See `examples/` directory for updated usage patterns
- **Issues:** Report migration problems on GitHub

---

**Migration completed? Welcome to RH-Agents v2.0! 🎉**
