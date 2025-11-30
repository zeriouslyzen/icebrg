# ICEBURG Integration Tests

This directory contains integration tests for ICEBURG.

## Test Files

- **`test_all_features.py`**: Tests all enabled features
- **`test_api_parallel.py`**: Tests parallel execution through API
- **`test_api_real.py`**: Tests real API queries
- **`test_full_system_agi.py`**: Tests full AGI system
- **`test_modes_agents.py`**: Tests modes and agents
- **`test_new_features.py`**: Tests new features
- **`test_parallel_execution.py`**: Tests parallel execution infrastructure
- **`test_reliability.py`**: Tests system reliability
- **`test_ux_complete.py`**: Tests UX completeness
- **`test_websocket_diagnostic.py`**: Tests WebSocket connections

## Running Tests

```bash
# Run all integration tests
python3 tests/integration/test_all_features.py

# Run specific test
python3 tests/integration/test_parallel_execution.py
```

