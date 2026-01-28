# ICEBURG Engineering Best Practices

## 🎯 Purpose
This document outlines engineering practices to prevent conflicts, bugs, and maintainability issues like the ones we encountered (mode handling, fast path logic, etc.).

---

## 🔴 **Critical Issues We Encountered**

### Issue 1: Mode Handling Conflicts
- **Problem**: Mode conversion happened AFTER fast path checks
- **Root Cause**: No centralized mode normalization
- **Impact**: "hi" queries didn't work in "fast" mode

### Issue 2: Scattered Logic
- **Problem**: Fast path check was in wrong location (after thinking message)
- **Root Cause**: Logic flow not following early-exit pattern
- **Impact**: Unnecessary processing for simple queries

### Issue 3: Inconsistent State
- **Problem**: Mode checked in multiple places with different conditions
- **Root Cause**: No single source of truth
- **Impact**: Bugs when conditions don't match

---

## ✅ **Engineering Practices to Follow**

### 1. **Early Validation & Normalization (EARN Principle)**

**Rule**: Validate and normalize ALL inputs at the entry point, BEFORE any processing.

```python
# ❌ BAD: Normalization scattered throughout code
def process_query(query, mode):
    # ... processing ...
    if mode == "fast":
        mode = "chat"  # Too late!
    # ... more processing ...

# ✅ GOOD: Normalize at entry point
def process_query(query, mode):
    # Normalize IMMEDIATELY
    mode = normalize_mode(mode)
    query = normalize_query(query)
    
    # Now all downstream code uses normalized values
    # ... processing ...
```

**Implementation**:
```python
# Create: src/iceburg/api/request_normalizer.py
class RequestNormalizer:
    """Centralized request normalization"""
    
    VALID_MODES = {"chat", "fast", "research", "device", "truth", "swarm", 
                   "prediction_lab", "astrophysiology", "gnosis", "civilization"}
    MODE_ALIASES = {
        "fast": "chat",  # fast is alias for chat with degradation_mode=False
    }
    
    @staticmethod
    def normalize_mode(mode: str, default: str = "chat") -> str:
        """Normalize mode to canonical form"""
        if not mode:
            return default
        
        mode = mode.lower().strip()
        
        # Check aliases first
        if mode in RequestNormalizer.MODE_ALIASES:
            return RequestNormalizer.MODE_ALIASES[mode]
        
        # Validate
        if mode not in RequestNormalizer.VALID_MODES:
            logger.warning(f"Invalid mode '{mode}', defaulting to '{default}'")
            return default
        
        return mode
    
    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize query string"""
        if not query:
            return ""
        return query.strip()
```

---

### 2. **Single Source of Truth (SSOT)**

**Rule**: Define constants, enums, or configs in ONE place. Never duplicate definitions.

```python
# ❌ BAD: Definitions scattered
# In server.py:
simple_queries = ["hi", "hello", "hey"]

# In another file:
GREETINGS = ["hi", "hello", "hey", "hey there"]  # Different!

# ✅ GOOD: Single source
# In: src/iceburg/constants.py
class QueryConstants:
    SIMPLE_QUERIES = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
    FAST_PATH_THRESHOLD = 0.3
    MAX_QUERY_LENGTH = 10000
```

**Implementation**:
```python
# Create: src/iceburg/constants.py
from enum import Enum
from typing import List

class ProcessingMode(str, Enum):
    """Canonical processing modes"""
    CHAT = "chat"
    FAST = "fast"  # Alias for chat with fast settings
    RESEARCH = "research"
    DEVICE = "device"
    TRUTH = "truth"
    SWARM = "swarm"
    PREDICTION_LAB = "prediction_lab"
    ASTROPHYSIOLOGY = "astrophysiology"
    GNOSIS = "gnosis"
    CIVILIZATION = "civilization"
    
    @classmethod
    def is_valid(cls, mode: str) -> bool:
        """Check if mode is valid"""
        try:
            cls(mode)
            return True
        except ValueError:
            return False

class QueryConstants:
    """Query-related constants"""
    SIMPLE_QUERIES: List[str] = [
        "hi", "hello", "hey", 
        "thanks", "thank you", 
        "bye", "goodbye"
    ]
    FAST_PATH_COMPLEXITY_THRESHOLD: float = 0.3
    MAX_QUERY_LENGTH: int = 10000
    MIN_QUERY_LENGTH: int = 1
```

---

### 3. **Early Exit Pattern (Guard Clauses)**

**Rule**: Handle special cases FIRST, then proceed with normal flow.

```python
# ❌ BAD: Special case buried in middle
def process_query(query, mode):
    # ... setup ...
    # ... thinking message ...
    # ... prompt interpreter ...
    
    # Fast path check (too late!)
    if query in simple_queries:
        return quick_response()
    
    # ... more processing ...

# ✅ GOOD: Early exit for special cases
def process_query(query, mode):
    # Normalize inputs FIRST
    mode = normalize_mode(mode)
    query = normalize_query(query)
    
    # Validate inputs
    if not is_valid_query(query):
        return error_response("Invalid query")
    
    # Early exit: Simple queries
    if is_simple_query(query) and is_fast_mode(mode):
        return handle_simple_query(query)
    
    # Early exit: Empty queries
    if not query:
        return empty_response()
    
    # Now proceed with normal flow
    # ... thinking message ...
    # ... full processing ...
```

**Implementation**:
```python
# In server.py - restructure with early exits
async def process_websocket_query(websocket, message):
    # 1. Normalize inputs (FIRST)
    query = RequestNormalizer.normalize_query(message.get("query", ""))
    mode = RequestNormalizer.normalize_mode(message.get("mode", "chat"))
    
    # 2. Early exit: Empty query
    if not query:
        logger.debug("Empty query, ignoring")
        return
    
    # 3. Early exit: Invalid query
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        await send_error(websocket, error_msg)
        return
    
    # 4. Early exit: Simple queries (BEFORE thinking message)
    if is_simple_query(query) and is_fast_mode(mode):
        await handle_simple_query(websocket, query)
        return
    
    # 5. Now proceed with normal flow
    await send_thinking(websocket, mode)
    # ... rest of processing ...
```

---

### 4. **Separation of Concerns**

**Rule**: Each function/class should have ONE responsibility.

```python
# ❌ BAD: One function does everything
def websocket_endpoint(websocket):
    # Parse message
    # Normalize mode
    # Validate query
    # Check fast path
    # Send thinking
    # Process query
    # Stream response
    # ... 500 lines ...

# ✅ GOOD: Separate concerns
def websocket_endpoint(websocket):
    message = await receive_message(websocket)
    request = RequestNormalizer.normalize(message)
    
    if request.is_simple_query():
        return await SimpleQueryHandler.handle(websocket, request)
    
    return await FullQueryHandler.handle(websocket, request)

class SimpleQueryHandler:
    @staticmethod
    async def handle(websocket, request):
        response = get_simple_response(request.query)
        await stream_response(websocket, response)

class FullQueryHandler:
    @staticmethod
    async def handle(websocket, request):
        await send_thinking(websocket, request.mode)
        result = await process_full_protocol(request)
        await stream_response(websocket, result)
```

---

### 5. **Consistent Error Handling**

**Rule**: Use consistent error handling patterns throughout.

```python
# ❌ BAD: Inconsistent error handling
try:
    result = process()
except Exception as e:
    logger.error(e)  # Sometimes
    return None  # Sometimes
    raise  # Sometimes

# ✅ GOOD: Consistent error handling
class QueryProcessor:
    async def process(self, request):
        try:
            return await self._process_internal(request)
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            return ErrorResponse("Invalid input", status=400)
        except ProcessingError as e:
            logger.error(f"Processing error: {e}")
            return ErrorResponse("Processing failed", status=500)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return ErrorResponse("Internal error", status=500)
```

---

### 6. **Type Safety & Validation**

**Rule**: Use type hints and validate types at boundaries.

```python
# ❌ BAD: No type hints, implicit conversions
def process_query(query, mode):
    if mode == "fast":  # What if mode is None? Or int?
        mode = "chat"

# ✅ GOOD: Type hints and validation
from typing import Optional
from pydantic import BaseModel, validator

class QueryRequest(BaseModel):
    query: str
    mode: ProcessingMode
    agent: Optional[str] = "auto"
    degradation_mode: bool = False
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > QueryConstants.MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long (max {QueryConstants.MAX_QUERY_LENGTH})")
        return v.strip()
    
    @validator('mode', pre=True)
    def normalize_mode(cls, v):
        return RequestNormalizer.normalize_mode(v)
```

---

### 7. **Configuration Management**

**Rule**: Centralize configuration, avoid magic numbers/strings.

```python
# ❌ BAD: Magic numbers/strings everywhere
if complexity < 0.3:  # What is 0.3?
    chunk_delay = 0.001  # Why 0.001?

# ✅ GOOD: Centralized config
class StreamingConfig:
    FAST_PATH_COMPLEXITY_THRESHOLD = 0.3
    CHUNK_DELAY_FAST = 0.0001  # GPT-5 speed
    CHUNK_DELAY_SLOW = 0.02
    CHUNK_SIZE_FAST = 1  # Character-by-character
    CHUNK_SIZE_SLOW = 20

if complexity < StreamingConfig.FAST_PATH_COMPLEXITY_THRESHOLD:
    chunk_delay = StreamingConfig.CHUNK_DELAY_FAST
```

---

### 8. **Testing Strategy**

**Rule**: Test edge cases and integration points.

```python
# ✅ GOOD: Comprehensive tests
class TestModeNormalization:
    def test_fast_mode_converts_to_chat(self):
        assert RequestNormalizer.normalize_mode("fast") == "chat"
    
    def test_invalid_mode_defaults(self):
        assert RequestNormalizer.normalize_mode("invalid") == "chat"
    
    def test_case_insensitive(self):
        assert RequestNormalizer.normalize_mode("FAST") == "chat"
        assert RequestNormalizer.normalize_mode("Chat") == "chat"

class TestSimpleQueryHandler:
    def test_hi_returns_instant_response(self):
        request = QueryRequest(query="hi", mode=ProcessingMode.CHAT)
        response = SimpleQueryHandler.handle(websocket, request)
        assert response.content == "Hello! How can I help you today?"
        assert response.response_time < 0.05  # <50ms
```

---

### 9. **Documentation & Comments**

**Rule**: Document WHY, not WHAT. Code should be self-documenting.

```python
# ❌ BAD: Comments explain what code does
# Convert fast mode to chat
if mode == "fast":
    mode = "chat"

# ✅ GOOD: Comments explain WHY
# Fast mode is an alias for chat mode with degradation_mode=False
# We normalize it early so all downstream code can assume canonical modes
if mode == "fast":
    mode = "chat"
    degradation_mode = False
```

---

### 10. **Code Organization**

**Rule**: Follow consistent file/function organization.

```
src/iceburg/
├── api/
│   ├── server.py          # WebSocket/HTTP endpoints (thin layer)
│   ├── request_normalizer.py  # Input normalization
│   └── handlers/
│       ├── simple_query_handler.py
│       └── full_query_handler.py
├── constants.py           # All constants, enums
├── config/
│   └── streaming_config.py
└── utils/
    └── validators.py
```

---

## 📋 **Checklist for New Features**

Before adding new code, ensure:

- [ ] Inputs normalized at entry point
- [ ] Constants defined in `constants.py`
- [ ] Early exits for special cases
- [ ] Type hints added
- [ ] Error handling consistent
- [ ] No magic numbers/strings
- [ ] Tests written for edge cases
- [ ] Documentation updated
- [ ] Code follows existing patterns

---

## 🚀 **Refactoring Plan**

To apply these practices to ICEBURG:

1. **Phase 1**: Create `RequestNormalizer` class
2. **Phase 2**: Create `constants.py` with all constants
3. **Phase 3**: Refactor `server.py` to use early exits
4. **Phase 4**: Extract handlers into separate classes
5. **Phase 5**: Add comprehensive tests
6. **Phase 6**: Update documentation

---

## 📚 **References**

- **SOLID Principles**: Single Responsibility, Open/Closed, etc.
- **Clean Code**: Robert C. Martin
- **Design Patterns**: Strategy, Factory, etc.
- **Python Best Practices**: PEP 8, type hints, etc.

---

**Last Updated**: November 17, 2025
**Status**: Active Guidelines

