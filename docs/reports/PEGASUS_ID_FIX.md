# PEGASUS Entity ID Format Fix

## Issue

Entity IDs in the database have an `osanc_` prefix (e.g., `osanc_NK-223CQDBzp8MRkdJMDiqXn3`), but the frontend and some API calls use IDs without the prefix (e.g., `NK-223CQDBzp8MRkdJMDiqXn3`).

This causes network queries to fail with: "Entity NK-223CQDBzp8MRkdJMDiqXn3 not found in database"

## Solution

Updated `MatrixStore.get_entity()` to handle ID format variations:
1. Tries exact match first
2. If ID starts with `NK-`, tries with `osanc_` prefix
3. If ID starts with `osanc_`, tries without prefix

Also updated `get_network()` to use the actual entity ID from the database after normalization.

## Changes Made

**File:** `src/iceburg/colossus/matrix_store.py`

1. Enhanced `get_entity()` method with ID normalization
2. Updated `get_network()` to use normalized entity ID throughout

## Testing

After restarting the server, test with:

```bash
# Test with ID without prefix
curl -X POST http://localhost:8000/api/colossus/network \
  -H "Content-Type: application/json" \
  -d '{"entity_id":"NK-223CQDBzp8MRkdJMDiqXn3","depth":2}'

# Should return network data instead of error
```

## Next Steps

1. **Restart the API server** to apply changes
2. Test network queries with both ID formats
3. Consider normalizing IDs at the API boundary for consistency
