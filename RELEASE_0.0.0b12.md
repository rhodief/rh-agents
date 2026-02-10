# RH-Agents v0.0.0b12 Release Notes

**Release Date:** February 10, 2026  
**Type:** Major Feature Release

## 🎉 What's New: Comprehensive Retry Mechanism

We're excited to announce a major enhancement to RH-Agents with the introduction of a **comprehensive retry mechanism** that provides intelligent error handling, state persistence, and seamless integration with the existing interrupt system.

## ✨ Key Features

### 1. **Intelligent Retry Configuration**
- Configurable retry attempts with exponential backoff
- Error filtering to retry only specific exceptions
- Timeout settings per retry attempt
- Maximum delay caps for backoff strategies

```python
from rh_agents.core.retry import RetryConfig

retry_config = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=10.0,
    backoff_factor=2.0,
    retry_on_errors=[ConnectionError, TimeoutError]
)
```

### 2. **Multi-Level Retry Configuration**
- Actor-level default retry settings
- Event-level retry overrides
- Flexible inheritance and override patterns
- Default retry for all exceptions with opt-out options

### 3. **State Persistence & Replay Integration**
- Full retry state serialization and restoration
- Chronologically ordered event replay
- Preservation of retry attempts, delays, and error information
- Seamless resume from failed executions

```python
# Save state with retry information
await state.save_checkpoint("/path/to/checkpoint")

# Restore and replay with full retry context
state = ExecutionState.from_checkpoint("/path/to/checkpoint")
```

### 4. **Interrupt Integration**
- Responsive cancellation during retry delays
- Graceful shutdown between retry attempts
- State preservation on interrupt
- No resource leaks during long retry sequences

### 5. **Enhanced Event Tracking**
- New `RETRYING` execution status
- Retry events with original error information
- Retry attempt counting and delay tracking
- Complete execution timeline preservation

### 6. **Intuitive Default Behavior**
- Retry enabled by default for all exceptions
- Simple opt-out with `retry_config=None`
- Sensible default configuration
- Easy customization for specific needs

## 🚀 Quick Start

### Basic Retry Usage
```python
from rh_agents import ExecutionEvent, ExecutionState
from rh_agents.core.retry import RetryConfig

# Use default retry behavior (retries all exceptions)
state = ExecutionState()
result = await ExecutionEvent(actor=my_actor)(input_data, "", state)

# Custom retry configuration
retry_config = RetryConfig(
    max_attempts=5,
    initial_delay=2.0,
    backoff_factor=2.0
)
result = await ExecutionEvent(
    actor=my_actor,
    retry_config=retry_config
)(input_data, "", state)

# Disable retry for specific event
result = await ExecutionEvent(
    actor=my_actor,
    retry_config=None
)(input_data, "", state)
```

### Retry with State Persistence
```python
from rh_agents import ExecutionState

state = ExecutionState()

try:
    result = await ExecutionEvent(actor=my_actor)(input_data, "", state)
except Exception as e:
    # Save state including retry information
    await state.save_checkpoint("checkpoint.json")
    raise

# Later, restore and continue
state = ExecutionState.from_checkpoint("checkpoint.json")
result = await ExecutionEvent(actor=my_actor)(input_data, "", state)
```

### Retry with Interrupt Support
```python
state = ExecutionState()
state.set_timeout(300, "Must complete within 5 minutes")

# Retry mechanism respects timeouts and interrupts
result = await ExecutionEvent(actor=my_actor)(input_data, "", state)
```

## 🔧 Implementation Details

### Retry Configuration Options
- `max_attempts`: Maximum number of retry attempts (default: 3)
- `initial_delay`: Initial delay in seconds (default: 1.0)
- `max_delay`: Maximum delay in seconds (default: 60.0)
- `backoff_factor`: Exponential backoff multiplier (default: 2.0)
- `retry_on_errors`: List of exception types to retry (default: all exceptions)
- `timeout_per_attempt`: Optional timeout for each retry attempt

### New Event Fields
- `retry_config`: Retry configuration for the event
- `retry_attempt`: Current retry attempt number (0 for first attempt)
- `is_retry`: Boolean indicating if this is a retry attempt
- `original_error`: Error from previous attempt (for retry events)
- `retry_delay`: Delay before this retry attempt

### Enhanced ExecutionStatus
- Added `RETRYING` status for retry attempts
- Clear distinction between failed and retrying states

## 📦 What's Included

### New Features
- Complete retry mechanism implementation
- Multi-level retry configuration support
- State persistence and replay integration
- Interrupt integration with responsive cancellation
- Retry display and visualization features
- Comprehensive test coverage

### Demo Scripts
- `demo_retry_simple.py` - Basic retry usage
- `demo_retry_persistence.py` - Retry with state save/restore
- `demo_retry_interrupt.py` - Retry with interrupt handling
- `demo_retry_display.py` - Retry event visualization

### Tests
- `test_retry_config.py` - Retry configuration validation
- `test_retry_integration.py` - End-to-end retry scenarios
- `test_retry_persistence.py` - State persistence with retry data
- `test_retry_interrupt.py` - Interrupt during retry sequences
- `test_retry_multilevel.py` - Multi-level configuration inheritance

## 📋 Migration Notes

### Breaking Changes
None - this is a backward-compatible feature addition.

### Upgrade Recommendations
1. Existing code continues to work without modifications
2. Retry is now enabled by default - to maintain old behavior, set `retry_config=None`
3. Review timeout settings when enabling retry to avoid overly long executions
4. Update state persistence code to take advantage of retry information

## 🐛 Bug Fixes
- Fixed event ordering in HistorySet to be chronological
- Enhanced error handling during retry delays
- Improved cleanup on interrupt during retry sequences

## 📚 Documentation
- Phase 4 completion document: State Persistence & Replay
- Phase 5 completion document: Interrupt Integration
- Comprehensive retry mechanism specification
- Demo scripts with detailed examples

## 🙏 Contributors

Thank you to everyone who contributed to this release!

---

**Full Changelog**: https://github.com/rhodief/rh-agents/compare/0.0.0b11...0.0.0b12
