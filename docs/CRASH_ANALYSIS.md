# Crash Analysis: Research Mode Model Selection

## What's Currently Running

1. **Backend (Uvicorn)**: Process 6067 (main), Process 15533 (worker) - Running on port 8000
2. **Frontend (Vite)**: Process 6135 - Running on port 3000  
3. **Ollama Server**: Process 1127 (serve), Process 15445 (runner) - Running on port 11434

All services are healthy and responding.

## What Crashed

**Error**: Ollama model runner crashed when trying to load `llama3.1:70b`

```
ERROR: Ollama HTTP Error 500: Internal Server Error
Request model: llama3.1:70b
Error: "model runner has unexpectedly stopped, this may be due to resource limitations or an internal error"
```

**Root Cause**: Research mode was using the default config which specifies `llama3:70b-instruct` for Synthesist and Oracle agents. When the system tried to resolve this, it attempted to use `llama3.1:70b`, which is too large for available system resources (likely OOM - Out of Memory).

## Where the 70b Model Comes From

1. **Default Config** (`src/iceburg/config.py` line 37-38):
   - `synthesist_model: "llama3:70b-instruct"`
   - `oracle_model: "llama3:70b-instruct"`

2. **Model Router** (`src/iceburg/investigations/model_router.py` line 43):
   - Has `llama3.1:70b` as a fallback option

3. **Unified LLM Interface** (`src/iceburg/unified_llm_interface.py` line 99):
   - Uses `llama3.1:70b` for "deep" research mode

## Fix Applied

**File**: `src/iceburg/api/server.py` (lines 3673-3700)

**Changes**:
1. Research mode now respects frontend `primaryModel` selection
2. Overrides any large models (>8B) from default config with the frontend-selected model
3. Logs the model being used: `🔬 Research mode using model: {effective_model}`

**Code Flow**:
- Frontend sends `settings.primaryModel` (e.g., `"llama3.1:8b"`)
- Backend extracts `primary_model` from settings
- Research mode uses `effective_model = primary_model or "llama3.1:8b"`
- Any agent configs with large models are replaced with `effective_model`

## Verification

To verify the fix is working:

1. **Check logs** for the new log message:
   ```bash
   tail -f logs/api_server.log | grep "Research mode using model"
   ```

2. **Test in browser**:
   - Open http://localhost:3000
   - Select Fast mode, ask a research question
   - Switch to Research mode
   - Check browser console and backend logs - should see `🔬 Research mode using model: llama3.1:8b`

3. **Expected behavior**:
   - Research mode should use the frontend-selected model (default: `llama3.1:8b`)
   - No more `llama3.1:70b` errors
   - Research pipeline completes successfully

## Backend Status

- **Last modification**: `server.py` modified at 22:43:35
- **Backend reloaded**: Yes (process 15533 started after modification)
- **Fix active**: Code is in place, will be used on next research mode query

## Next Steps

1. Test research mode in the browser to verify fix
2. If issues persist, check:
   - Is `primary_model` being passed correctly from frontend?
   - Are there other code paths using the 70b model?
   - Check `load_config_with_model()` behavior with `use_small_models=False`
