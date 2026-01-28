# PEGASUS Troubleshooting Guide

## Quick Diagnostics

### Check Server Status
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/colossus/status
```

### Check Frontend Access
```bash
curl http://localhost:8000/pegasus.html | head -20
```

## Common Errors

### Error: "Network Error" or "Failed to fetch"
**Cause:** API server not running or CORS issue  
**Fix:**
1. Verify server is running: `curl http://localhost:8000/health`
2. Check browser console for CORS errors
3. Restart server if needed

### Error: "404 Not Found" on API endpoints
**Cause:** Colossus router not registered  
**Fix:**
1. Check server logs for: "COLOSSUS Intelligence Platform API routes registered"
2. Verify `src/iceburg/colossus/api.py` exists
3. Restart server

### Error: "No entities found" or empty graph
**Cause:** Graph is empty (NetworkX backend)  
**Fix:**
1. Click "Launch Visualization" - should auto-ingest
2. Or manually trigger: `curl -X POST "http://localhost:8000/api/colossus/ingest?limit=2000"`
3. Wait for ingestion to complete

### Error: "Matrix database not found"
**Cause:** SQLite database missing  
**Fix:**
1. Check if `matrix.db` exists in:
   - `~/Documents/iceburg_matrix/matrix.db`
   - `~/Desktop/Projects/iceburg/matrix.db`
2. If missing, data ingestion will still work but search may be limited

### Error: JavaScript console errors
**Common Issues:**
- `d3 is not defined` - D3.js library not loaded
- `fetch failed` - API server not accessible
- `Cannot read property 'id' of undefined` - Data structure mismatch

**Fix:**
1. Open browser DevTools (F12)
2. Check Console tab for errors
3. Check Network tab for failed requests
4. Verify API_BASE is correct: `/api/colossus`

## Browser-Specific Issues

### Chrome/Edge
- Check CORS in Network tab
- Verify no extensions blocking requests
- Try incognito mode

### Firefox
- Check console for CSP errors
- Verify mixed content settings

### Safari
- Check "Disable Cross-Origin Restrictions" in Develop menu (if available)
- Verify JavaScript is enabled

## API Endpoint Testing

### Test Search
```bash
curl -X POST http://localhost:8000/api/colossus/entities/search \
  -H "Content-Type: application/json" \
  -d '{"query":"test","limit":5}'
```

### Test Network Query
```bash
curl -X POST http://localhost:8000/api/colossus/network \
  -H "Content-Type: application/json" \
  -d '{"entity_id":"osanc_NK-3ezGEQvENCSjmNka3JdMtv","depth":2}'
```

### Test Status
```bash
curl http://localhost:8000/api/colossus/status
```

## Debugging Steps

1. **Check Server Logs**
   - Look for error messages in terminal
   - Check for import errors
   - Verify all dependencies installed

2. **Check Browser Console**
   - Open DevTools (F12)
   - Look for JavaScript errors
   - Check Network tab for failed requests

3. **Verify API Endpoints**
   - Test each endpoint individually
   - Check response format matches frontend expectations

4. **Check Data**
   - Verify Matrix database exists
   - Check entity/relationship counts
   - Test data quality endpoint

## Quick Fixes

### Restart Server
```bash
# Kill existing server
pkill -f "run_server"

# Start fresh
cd /Users/jackdanger/Desktop/Projects/iceburg
python3 -m src.iceburg.api.run_server
```

### Clear Browser Cache
- Hard refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)
- Or clear cache in browser settings

### Check Dependencies
```bash
pip3 list | grep -E "fastapi|neo4j|networkx"
```

## Still Having Issues?

1. Check server terminal output for errors
2. Check browser console for JavaScript errors
3. Verify all files exist:
   - `frontend/pegasus.html`
   - `src/iceburg/colossus/api.py`
   - `src/iceburg/colossus/matrix_store.py`
4. Test API endpoints directly with curl
5. Check network connectivity
