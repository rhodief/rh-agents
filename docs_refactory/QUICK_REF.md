# Refactoring Quick Reference

## 📁 Document Structure

```
docs_refactory/
├── REFACTORING_SPEC.md       # Full analysis (human-readable)
├── IMPLEMENTATION_SPEC.md    # Implementation guide (LLM-optimized) ⭐
├── STATUS.md                 # Current status & next steps
└── QUICK_REF.md             # This file
```

## 🎯 What Was Decided

| Question | Decision | Impact |
|----------|----------|--------|
| Generic types? | Remove, add .pyi stubs | Simpler API, maintained type safety |
| Result types? | Keep LLM/Tool, add Protocol, remove Agent | Clean domain types |
| ToolSet? | Simplify with computed dict | Less code, same functionality |
| Deprecated cache? | **Delete completely** | Clean codebase |
| `cacheable` field? | **KEEP IT** (state recovery, not cache) | No change |
| Backend management? | Pydantic exclude=True | Cleaner, more idiomatic |
| Field redeclarations? | Remove redundant ones | Less boilerplate |
| Public API? | Flat exports in __init__.py | Easier imports |
| Handler types? | Type alias with docs | Simple and clear |
| Context param? | Optional (state, context="") | Backward compatible |
| Event addresses? | Keep current | Works well |

## 📊 Impact Summary

### What Gets Removed
- ❌ `rh_agents/core/cache.py` (deprecated)
- ❌ `rh_agents/cache_backends.py` (deprecated)
- ❌ `Agent_Result` class (unused)
- ❌ Generic type parameters from public API
- ❌ Redundant field redeclarations
- ❌ ToolSet dual storage complexity

### What Gets Added
- ✅ Public API exports in `__init__.py`
- ✅ Type stub files (`.pyi`)
- ✅ `ActorOutput` protocol
- ✅ Comprehensive docstrings
- ✅ Migration guide

### What Gets Simplified
- ✨ LLM instantiation: `LLM(...)` vs `LLM[Type](...)`
- ✨ ExecutionEvent usage: `ExecutionEvent(actor=x)` vs `ExecutionEvent[Type](actor=x)`
- ✨ ToolSet: Single storage, computed dict
- ✨ Handler signature: Optional context parameter
- ✨ Backend management: Standard Pydantic exclude

## 🔢 Three Phases

### Phase 1: Non-Breaking (v1.1.0) - 1-2 days
```
Add public API → Update examples → Add docs
```
**Safe** - No code breaks

### Phase 2: Simplification (v1.5.0) - 3-4 days
```
Remove generics → Add stubs → Simplify classes → Update handlers
```
**Safe** - Backward compatible

### Phase 3: Breaking (v2.0.0) - 1-2 days
```
Delete deprecated cache → Final docs → Migration guide
```
**Breaking** - Expected for major version

## 🚀 Quick Start for Implementation

### For LLM Agent:
```
Read IMPLEMENTATION_SPEC.md
Follow Phase 1 → Phase 2 → Phase 3
Test after each change
Check off items in checklist
```

### For Human Developer:
```bash
# 1. Review decisions
cat docs_refactory/STATUS.md

# 2. Start Phase 1
git checkout -b implement-phase1
code rh_agents/__init__.py

# 3. Follow IMPLEMENTATION_SPEC.md
# Each section has before/after code
```

## ⚠️ Critical Notes

### DO NOT Remove
- ✅ `cacheable` field in BaseActor - Used by state recovery!
- ✅ State recovery system - It's the modern approach
- ✅ Event address system - Works well

### DO Remove
- ❌ `cache.py` and `cache_backends.py` - Fully deprecated
- ❌ Generic type parameters - Moving to .pyi stubs
- ❌ `Agent_Result` - Never used

### Test Thoroughly
- After each phase
- All examples must run
- Type checking should pass (with stubs)
- Documentation should build

## 📈 Success Metrics

**Phase 1 Success:**
- [ ] Can import from `rh_agents` directly
- [ ] All examples run unchanged
- [ ] No test failures

**Phase 2 Success:**
- [ ] No generics in public API
- [ ] Type stubs validate with mypy
- [ ] Context parameter optional
- [ ] All tests pass

**Phase 3 Success:**
- [ ] No deprecated modules
- [ ] Migration guide complete
- [ ] Ready for v2.0.0 release

## 🔗 Key Code Changes

### Import Style
```python
# Before
from rh_agents.core.actors import Agent, Tool, LLM
from rh_agents.core.execution import ExecutionState

# After
from rh_agents import Agent, Tool, LLM, ExecutionState
```

### Generic Removal
```python
# Before
event = ExecutionEvent[LLM_Result](actor=llm)

# After  
event = ExecutionEvent(actor=llm)
```

### Handler Signature
```python
# Before
async def handler(input: Input, context: str, state: ExecutionState) -> Output:

# After
async def handler(input: Input, state: ExecutionState, context: str = "") -> Output:
```

### ToolSet Usage
```python
# Before
tools = tool_set.get_tool_list()

# After
tools = list(tool_set)  # or tool_set.tools
```

## 📝 Files to Update Most

1. `rh_agents/__init__.py` - Add exports
2. `rh_agents/core/actors.py` - Remove generics, clean fields
3. `rh_agents/core/events.py` - Remove generics, update handler call
4. `rh_agents/core/result_types.py` - Add protocol, remove Agent_Result
5. `rh_agents/agents.py` - Update all handler signatures
6. All `examples/*.py` - Update imports and handlers
7. Documentation files - Update examples

## 🎓 Why These Changes?

**Simpler API**
- Fewer type parameters = easier to learn
- Flat imports = clearer public API
- Less boilerplate = faster development

**Better Type Safety**
- .pyi stubs for type checkers
- Protocol for result types
- Maintained where it matters

**Cleaner Codebase**
- Removed 2 deprecated modules
- Removed 1 unused class
- Simplified 3 complex classes
- ~15% code reduction

**Pythonic Design**
- Optional parameters instead of required
- Duck typing with protocols
- Standard import patterns
- Trust the framework (Pydantic)

---

**Ready to Implement!** 🚀

Start here: [IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md)
