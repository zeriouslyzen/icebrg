# PEGASUS Browser Testing Instructions

## Quick Test Steps

1. **Ensure API Server is Running**
   ```bash
   # Check if server is running
   curl http://localhost:8000/health
   
   # If not running, start it:
   python3 -m src.iceburg.api.run_server
   ```

2. **Open Pegasus UI in Browser**
   - Navigate to: `http://localhost:8000/pegasus.html`
   - You should see the editorial-style interface with three panels

3. **Test Basic Functionality**

   **a. Search Test:**
   - Type "putin" or "trump" in the search box
   - Wait for search results to appear (300ms debounce)
   - Click on a result to load its network

   **b. Graph Visualization:**
   - Click "Launch Visualization" button
   - Should see network graph with nodes and connections
   - Nodes should be color-coded by type:
     - Blue = Person
     - Green = Company
     - Red = Organization
   - Yellow highlight = Selected node

   **c. Graph Controls:**
   - Use zoom buttons (+ / −) in top-right corner
   - Click "⌂" to reset view
   - Click "Aa" to toggle labels
   - Drag nodes to reposition
   - Click nodes to expand network

   **d. Relationship Display:**
   - Select a node to see details in right panel
   - Check "Connections" section for actual relationships
   - Click on relationships to navigate to connected entities

   **e. Network Expansion:**
   - Click on a node to expand its network
   - New nodes should merge with existing graph (not replace)
   - Graph should grow incrementally

4. **Test Edge Cases**

   **Empty Graph:**
   - If graph has no relationships, should show message:
     "No connections found. X isolated entities."
   - Nodes should still render in a grid layout

   **No Data:**
   - If no entities found, welcome overlay should show error message

5. **Check Diagnostics**
   - Click "Diagnostics" button in left panel stats section
   - Should show:
     - Backend type
     - Entity/relationship counts
     - Relationship ratio
     - Matrix DB status
     - Sample relationships

## Expected Behavior

### ✅ Working Features:
- Search returns results from Matrix DB
- Network endpoint uses MatrixStore directly (1.4M relationships available)
- Graph renders with proper node/link binding
- Zoom and pan controls work
- Relationship details show actual connections
- Network expansion merges nodes incrementally
- Empty states show helpful messages

### ⚠️ Known Issues to Watch For:
- If diagnostics endpoint returns 404, server may need restart
- Large networks (1000+ nodes) may be slow to render
- Some entities may have no relationships (expected)

## Troubleshooting

**If graph shows no connections:**
1. Check diagnostics to see relationship count
2. Try different entities (some may be isolated)
3. Verify Matrix DB has relationships table populated

**If search doesn't work:**
1. Check browser console for errors
2. Verify API server is running
3. Check network tab for failed requests

**If zoom/pan doesn't work:**
1. Ensure D3.js loaded correctly
2. Check browser console for JavaScript errors
3. Try refreshing the page

## Test Data

Good test entities with likely connections:
- "Vladimir Putin" (person, likely has many connections)
- "Donald Trump" (person, high-profile)
- Any sanctioned company or organization

## Performance Notes

- Direct MatrixStore query is faster than graph traversal
- Network queries should be sub-second for depth=2, limit=50
- Large networks may take 2-5 seconds to render
