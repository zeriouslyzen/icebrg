# PEGASUS Browser Test Guide

## Current Status

**Direct MatrixStore Tests (Working):**
- Donald Trump: 30 nodes, 29 edges ✅
- Salman bin Abdulaziz Al Saud: 50 nodes, 49 edges ✅
- Center entity always included ✅

**API Endpoint (Needs Restart):**
- Returns empty (server using old code)
- **Solution**: Restart API server

## Browser Testing Steps

### 1. Restart API Server First

```bash
# Stop current server (Ctrl+C if running in terminal)
# Then restart:
python3 -m src.iceburg.api.run_server
```

### 2. Open Pegasus UI

Navigate to: `http://localhost:8000/pegasus.html`

### 3. Test Scenarios

#### Test A: Search and Load Network

1. **Search for "Salman"** (best test - has 49 valid relationships)
   - Type "Salman" in search box
   - Wait for results (300ms debounce)
   - Click "Salman bin Abdulaziz Al Saud"
   - **Expected**: Graph shows ~50 nodes with ~49 edges connected

2. **Search for "Trump"**
   - Type "trump" in search box
   - Click "Donald Trump" or "Tiffany Trump"
   - **Expected**: Graph shows nodes with family relationships

#### Test B: Launch Visualization

1. Click "Launch Visualization" button
2. **Expected**: 
   - Welcome overlay hides
   - Graph appears with central entities
   - Nodes are color-coded (blue=Person, green=Company, red=Organization)

#### Test C: Graph Interactions

1. **Click a node**
   - **Expected**: 
     - Node highlighted (yellow)
     - Details panel shows entity info
     - Network expands (new nodes merge in)

2. **Use zoom controls** (top-right)
   - Click "+" to zoom in
   - Click "−" to zoom out
   - Click "⌂" to reset view
   - Click "Aa" to toggle labels
   - **Expected**: All controls work smoothly

3. **Drag nodes**
   - Click and drag a node
   - **Expected**: Node moves, edges follow

#### Test D: Data Quality Indicator

1. Check left panel stats section
2. Look for "Data Quality" percentage
3. **Expected**:
   - Shows percentage (currently ~2.1% before cleanup)
   - Color-coded (red if <10%, yellow if <50%, green if >90%)
   - "Clean Data" button visible if quality < 50%

#### Test E: Network Expansion

1. Start with one entity (e.g., "Salman")
2. Click on a connected node
3. **Expected**:
   - New nodes merge into graph (don't replace)
   - Graph grows incrementally
   - No duplicate nodes
   - All edges valid

### 4. Check Browser Console

Open DevTools (F12) and check:

**Console Tab:**
- Look for diagnostic logs:
  - "Network query stats: ..."
  - "Merged X new nodes into graph"
  - Any warnings about filtered edges

**Network Tab:**
- Check API calls:
  - `/api/colossus/network` - Should return nodes and edges
  - `/api/colossus/entities/search` - Should return search results
  - `/api/colossus/data-quality` - Should show quality percentage

**Expected API Response:**
```json
{
  "nodes": [...],  // Should have nodes
  "edges": [...],  // Should have edges if entity has relationships
  "center": "osanc_Q367825",
  "query_stats": {
    "relationships_found": 49,
    "nodes_found": 50,
    "edges_filtered": 0
  }
}
```

## Troubleshooting

### If Graph Shows No Edges

1. **Check console for errors**
2. **Verify API response** in Network tab
3. **Check query_stats** - if `edges_filtered` > 0, data needs cleanup
4. **Try different entity** - some entities are isolated

### If Search Doesn't Work

1. Check Network tab for failed requests
2. Verify API server is running
3. Check console for JavaScript errors

### If Nodes Don't Appear

1. Check if API returns nodes in response
2. Verify D3.js loaded (check console)
3. Check if welcome overlay is blocking view

## Best Test Entities

**High Relationship Count:**
- "Salman bin Abdulaziz Al Saud" - 49 relationships
- "Donald Trump" - 39 relationships  
- "Muqrin bin Abdul-Aziz Al Saud" - 35 relationships

**Search Terms:**
- "salman" → Salman bin Abdulaziz Al Saud
- "trump" → Donald Trump, Tiffany Trump
- "putin" → Various entities

## Expected Results After Cleanup

Once data is cleaned (removes 97.9% invalid relationships):

**Before Cleanup:**
- Data Quality: 2.1%
- Many edges filtered out
- Some entities show no edges

**After Cleanup:**
- Data Quality: 100%
- All edges valid
- Full networks visible
- Faster queries

## Quick Test Commands

```bash
# Test API directly
curl -X POST http://localhost:8000/api/colossus/network \
  -H "Content-Type: application/json" \
  -d '{"entity_id":"osanc_Q367825","depth":1,"limit":50}'

# Check data quality
curl http://localhost:8000/api/colossus/data-quality

# Search entities
curl -X POST http://localhost:8000/api/colossus/entities/search \
  -H "Content-Type: application/json" \
  -d '{"query":"salman","limit":5}'
```
