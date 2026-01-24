# Refactoring Tests

Test suite for architecture refactoring validation.

## Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for component interactions
- `e2e/` - End-to-end tests for complete workflows

## Running Tests

```bash
# All tests
pytest tests_refactory/

# Specific level
pytest tests_refactory/unit/
pytest tests_refactory/integration/
pytest tests_refactory/e2e/

# With coverage
pytest tests_refactory/ --cov=rh_agents
```

## Test Phases

- **Phase 1**: Verify public API exports work correctly
- **Phase 2**: Validate simplified architecture changes
- **Phase 3**: Ensure deprecated code removed, migration works
