# UX to Backend: How It All Fits Together

This document maps frontend entry points to backend APIs and the dossier/Colossus pipeline, and notes where corpus/silence and investigation network connect (or do not yet).

---

## 1. Frontend entry points and APIs

| UX | Entry | APIs used | Backend |
|----|--------|-----------|---------|
| **Main chat** | `app.html` + `main.js` | WebSocket `/ws` (query, mode), `POST /api/export/generate`, `POST /api/upload` | `server.py` WebSocket handler, routes |
| **Dossier (standalone)** | `dossier.html` | `POST /api/dossier` (query, depth, format) | `dossier_routes.py` -> DossierSynthesizer |
| **Investigations archive** | `investigations.html` | `GET /api/investigations/`, `GET /api/investigations/{id}`, `GET /api/investigations/{id}/network`, `GET /api/investigations/{id}/pdf`, `DELETE ...` | `investigation_routes.py`, storage |
| **Pegasus (network graph)** | `pegasus.html` | `GET /api/colossus/status`, `POST /api/colossus/network` (entity_id, depth, limit), search/central/query/* | `colossus/api.py`, graph, MatrixStore |
| **Colossus hub** | `colossus/index.html`, `graph.html`, `entity.html` | Same Colossus APIs; graph view = same network by entity | Same as Pegasus |

So: **chat** and **dossier page** drive the dossier pipeline; **Pegasus/Colossus** drive the Colossus graph; **investigations** list/view/PDF/network come from the investigation store.

---

## 2. Dossier pipeline (Gatherer → Decoder → Mapper → Synthesizer)

When a dossier is generated:

1. **Dossier page**: User submits query on `dossier.html` → `POST /api/dossier` → `DossierSynthesizer.generate_dossier()`.
2. **Chat (dossier mode)**: User sends message with `mode: "dossier"` over WebSocket → same `DossierSynthesizer.generate_dossier()` in `server.py`, then investigation is saved and `investigation_id` is returned in the “done” metadata.

Pipeline steps (both entry points):

- **Gatherer** – Web search (surface / alternative / academic / historical / deep), builds `IntelligencePackage`.
- **Decoder** – Symbol/pattern analysis on gathered content → `DecoderReport` (includes linguistic markers).
- **Mapper** – Key players, hidden connections, network map with entities/relationships (roles, domains, themes, relationship_type, domain); uses decoder report for linguistic flags.
- **Synthesizer** – Builds `IcebergDossier` (executive summary, narratives, key_players, network_map, etc.) and returns it.

**Corpus ingest and silence tracker** are not in the UX yet:

- **Corpus ingest**: `GathererAgent.ingest_corpus(path)` (and optional hook) + `load_corpus_from_path` / `ingest_corpus_for_dossier` in `corpus_ingest.py`. No UI calls these; they are backend-only. A future “Upload corpus” or “Ingest folder” could call an API that runs this and optionally feeds the dossier pipeline.
- **Silence/mention tracker**: `track_silence_mentions()` and `entities_silent_in_corpus()` in `silence_mention_tracker.py`. Same: backend-only; no UX. Could be exposed as an API and used from a “Corpus analysis” or “Mention report” view.

---

## 3. Where the dossier result goes

- **Dossier page**: Response is rendered in the same page (summary, narratives, key players, etc.). No “Save to archive” or “View in Pegasus” on this page.
- **Chat (dossier mode)**: Server builds an `Investigation` from the dossier (including `network_graph = dossier.network_map`), saves it via `InvestigationStore.save()`, and sends back `investigation_id`. The app can open `app.html?investigation={id}` to resume context; `window.activeInvestigationId` is set from the URL.

So the **investigation archive** is populated when the user runs dossier **from chat**, not from the standalone dossier page.

---

## 4. Investigation store and network format

- **Stored**: `Investigation.network_graph` = dossier `network_map` = `{ "entities": [...], "relationships": [...] }` (from Mapper).
- **Served**: `GET /api/investigations/{id}/network` normalizes this to a Pegasus-friendly shape via `_network_graph_to_nodes_links()`: if the stored graph has `entities`/`relationships` but no `nodes`/`links`, the API adds `nodes` and `links` (and `edges`) so the frontend always gets a graph it can render.

So the **investigation network** is connected to the UX only in the sense that the API is ready for a viewer that uses it. No frontend currently loads an investigation’s graph into Pegasus.

---

## 5. Pegasus and Colossus (graph UX)

- **Pegasus** (`pegasus.html`):  
  - Calls `GET /api/colossus/status` and `POST /api/colossus/network` with an `entity_id` (and depth/limit).  
  - Renders `nodes` and `links` from that response (with relationship-type colors and filters).  
  - If the Colossus graph is empty, it can trigger ingest or load a static fallback (`/pegasus_network.json`).  
  - It does **not** call `GET /api/investigations/{id}/network`. So the graph shown is **Colossus (or Matrix) data**, not a specific saved investigation.

- **Colossus** (`colossus/index.html`, `graph.html`, `entity.html`): Same Colossus APIs; graph view is “load network for this entity” from the same backend.

So today:

- **Pegasus/Colossus** = one global Colossus graph (in-memory or Neo4j) + MatrixStore fallback.
- **Investigation network** = per-investigation graph stored with the dossier; API supports nodes/links, but no UI loads it into the graph view.

---

## 6. Colossus graph population

- **Bulk ingest**: `POST /api/colossus/ingest` (OpenSanctions JSON or Matrix SQLite).  
- **Dossier ingest**: `POST /api/colossus/ingest/dossier` with a dossier payload; adds that dossier’s entities/relationships into the Colossus graph (and runs the bridge detector).  
- Neither is triggered automatically when a dossier is generated from the dossier page or chat. So the graph in Pegasus is only updated when something explicitly calls ingest (e.g. a script or a future “Push to Colossus” button).

---

## 7. End-to-end flow (current vs potential)

**Current:**

- User runs **dossier** (chat or dossier page) → dossier pipeline runs → result shown (and from chat, investigation is saved with `network_graph`).
- User opens **Pegasus** → sees Colossus graph (or fallback); graph is **not** the last dossier’s network unless that dossier was explicitly ingested into Colossus.
- User opens **Investigations** → sees list; can view investigation (→ app with `?investigation=`), download PDF; **no** “View network in Pegasus” that loads `GET /api/investigations/{id}/network`.

**To tie it together in UX:**

1. **Investigations → Pegasus**: On the investigation card or detail view, add “View network” that opens Pegasus with a query param (e.g. `?investigation={id}`). Pegasus, when that param is set, calls `GET /api/investigations/{id}/network` and renders that graph instead of (or merged with) Colossus.
2. **Dossier page → Archive**: Add “Save as investigation” on the dossier result that POSTs to an endpoint (or reuses existing save flow) so the dossier page also writes to the investigation store.
3. **Dossier/Colossus**: After saving an investigation (from chat or dossier page), optionally call `POST /api/colossus/ingest/dossier` so Pegasus/Colossus shows that dossier’s network.
4. **Corpus/silence in UX**: Add an “Ingest corpus” (folder/zip) that calls an API using `ingest_corpus_for_dossier`, and a “Mention report” (or “Silent entities”) that calls an API using `track_silence_mentions` / `entities_silent_in_corpus` and displays the result.

---

## 8. Summary diagram (ASCII)

```
[User]
  |
  +-- app.html (chat) ----mode=dossier----> WebSocket --> DossierSynthesizer --> save Investigation
  |                                                                                    |
  |                                                                                    v
  |                                                                           InvestigationStore (network_graph = dossier.network_map)
  |
  +-- dossier.html -------POST /api/dossier---------> DossierSynthesizer --> render in page (no save/Colossus)
  |
  +-- investigations.html --> GET /api/investigations/, /{id}, /{id}/network, /{id}/pdf
  |                              |
  |                              v
  |                         _network_graph_to_nodes_links() so response has nodes/links
  |
  +-- pegasus.html --------> GET /api/colossus/status, POST /api/colossus/network (entity_id)
  |                              |
  |                              v
  |                         ColossusGraph (or MatrixStore fallback)  [not investigation network]
  |
  +-- colossus/*.html -----> same Colossus APIs

Corpus ingest / silence tracker: backend only (Gatherer.ingest_corpus, track_silence_mentions); no API/UX yet.
```

---

## 9. Files to look at

- **UX**: `frontend/app.html`, `main.js`, `dossier.html`, `investigations.html`, `pegasus.html`, `colossus/*.html`
- **API**: `api/server.py` (WebSocket, mount), `api/dossier_routes.py`, `api/investigation_routes.py`, `api/colossus/api.py`
- **Dossier pipeline**: `protocols/dossier/synthesizer.py`, `gatherer.py`, `decoder.py`, `mapper.py`, `corpus_ingest.py`, `silence_mention_tracker.py`
- **Investigation**: `investigations/storage.py` (from_dossier, network_graph), `investigations/pdf_export.py` (Matrix summary)
- **Graph**: `colossus/core/graph.py`, `colossus/matrix_store.py`

This is how the UX and backend connect today and how they can be wired further (investigation network in Pegasus, corpus/silence in UI, dossier→archive and→Colossus).
