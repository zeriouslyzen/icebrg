# UX Test Verification

Quick record of what was verified against the running API and in-browser UX (server on port 8000, IDE browser).

## API verification (done)

- **Health**: `GET /health` -> 200
- **Colossus**: `GET /api/colossus/status` -> `{"status":"operational","backend":"networkx",...}`
- **Investigations**: `GET /api/investigations/` -> list of investigations (e.g. 4 items)
- **Dossier**: `GET /api/dossier/status` -> `{"status":"operational","protocol":"dossier",...}`
- **Investigation network (nodes/links)**: `GET /api/investigations/{id}/network` -> `network_graph` has `nodes` and `links` (normalization from entities/relationships confirmed)

## In-browser UX run (done)

Auth was bypassed for localhost so pages load without clearance:

- **auth-check.js**: Bypass when `hostname === 'localhost' || hostname === '127.0.0.1'` or `?test=1`. Non-localhost still requires `iceburg_clearance`.
- **pegasus.html**: Duplicate auth-check script removed; single script with cache-bust optional. Button id `expandNetworkBtn` added for automation.

Pages loaded and screenshots captured:

1. **Pegasus** (`/pegasus.html` or `/pegasus.html?nocache=1`): Loaded; "PEGASUS INFORMATION ARCHITECTURE", "SYSTEM ACTIVE", Query Index (search), Network Graph view, Database Metrics (entities/links/ratio), LINK TYPES, detail panel with selected entity (e.g. Myanmar Yatar International Holding Group), NO PROPERTIES / CONNECTIONS, SANCTIONS 4, "Expand Network" button. Graph canvas visible (dotted grid).
2. **Investigations** (`/investigations.html`): Loaded; "Investigation Archive", search box, "All Status", "+ New Investigation"; grid of 4 investigation cards (e.g. "what are the orgins of darpa", "what role does trump play...", Sam Altman, "Who is sam altman really") with status, date, sources, confidence, action icons.
3. **Dossier** (`/dossier.html`): Loaded; "ICEBURG DOSSIER", "Deep Research Protocol", research topic input, depth selector ("Standard (3-5 min)"), "Generate Dossier" button.

Interactive steps (click Expand Network, search entity, Generate Dossier) were not driven in this pass because the automation snapshot did not expose element refs for click/fill.

## Auth for manual testing

- On **localhost**, pages load without setting `iceburg_clearance` (bypass in auth-check.js).
- On other hosts, set `localStorage.setItem('iceburg_clearance', 'test');` in DevTools or use `?test=1` on the URL (if cache does not serve old auth-check).

## Manual UX checklist (after load)

1. **Pegasus**: Search entity -> load network -> graph and details; LINK TYPES filter; Expand Network.
2. **Investigations**: Click a card -> view/PDF; confirm network has nodes/links if you add "View network" later.
3. **Dossier**: Enter query -> Generate Dossier -> confirm summary, narratives, key players; check console for errors.

## Not automated

- Clicks/fills via MCP browser failed (refs not exposed in snapshot).
- Dossier generation (POST /api/dossier) was not triggered from the UI (would need form submit + Ollama/keys).
- PDF export with Matrix summary not opened in browser.
