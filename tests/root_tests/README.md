# Root-Level Test Files

This directory contains test files that were previously in the project root.

## Test Files

- `test_always_on_architecture.py` - Tests for always-on architecture
- `test_bottleneck_detection.py` - Bottleneck detection tests
- `test_continuous_conversation.py` - Conversation continuity tests
- `test_conversations.py` - General conversation tests
- `test_deep_knowledge.py` - Deep knowledge system tests
- `test_fast_mode.py` - Fast mode functionality tests
- `test_fine_tuning_collection.py` - Fine-tuning data collection tests
- `test_gnosis_detailed.py` - Detailed gnosis system tests
- `test_gnosis_system.py` - Gnosis system integration tests
- `test_llm_speed.py` - LLM performance tests
- `test_self_healing_integration.py` - Self-healing system tests
- `test_server_comprehensive.py` - Comprehensive server tests
- `test_websocket_consistency.py` - WebSocket consistency tests
- `test_websocket_diagnostics.py` - WebSocket diagnostic tests

## HTML Test Files

- `test_carousel.html` - Carousel component tests
- `test_simple_query.html` - Simple query interface tests
- `test_websocket_connection.html` - WebSocket connection tests

## Running Tests

From the project root:
```bash
python -m pytest tests/root_tests/
```

Or run individual tests:
```bash
python tests/root_tests/test_fast_mode.py
```

