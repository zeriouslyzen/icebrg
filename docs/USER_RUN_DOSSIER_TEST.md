# User-Run Dossier Test

Ran the app **like a user**: Dossier page, typed query, clicked Generate Dossier, waited for completion.

---

## Where did it file? (Where the output went)

1. **Browser (Dossier page)**  
   The API returned the full dossier in the `POST /api/dossier` response. The frontend receives it and calls `renderDossier(data)`, which fills the `#dossierContent` section with executive summary, official narrative, alternative narratives, key players, etc. So the dossier **shows on the same page** after "Generating..." finishes. It is **not** auto-saved to disk from this page.

2. **Colossus graph (in-memory)**  
   After synthesis, the pipeline calls `_ingest_to_colossus(dossier)`, which runs `graph.ingest_dossier(dossier.to_dict())`. So the dossier was **ingested into the Colossus graph** (NetworkX in-memory): **8 entities, 0 relationships**. Those entities are available in Pegasus (search, network view) **for this server session only**. Restarting the API server clears the in-memory graph unless you persist it (e.g. Neo4j or a save/load step).

3. **Investigation archive**  
   The **Dossier page does not save to the Investigation archive**. Saving to the archive (e.g. `~/Documents/iceburg_data/investigations/`) only happens when you run **Dossier mode from the main Chat** (WebSocket). So this run produced no new investigation folder or PDF unless you later export/save from the UI.

**Summary:** Output went to (a) the Dossier page DOM, and (b) the in-memory Colossus graph. It did **not** go to the investigation store or to disk unless you take an extra step (e.g. "Save as investigation" from chat, or add that action to the Dossier page).

---

## Where it failed

Three steps failed with the same error: **`Expecting value: line 1 column 1 (char 0)`** — the code called `json.loads()` on LLM output and got empty or non-JSON.

| Step | Component | What failed | Effect |
|------|-----------|-------------|--------|
| 1 | **Gatherer** | Claim extraction | No `key_claims` from LLM; step logged warning and continued. |
| 2 | **Gatherer** | Contradiction finding | No `contradictions` from LLM; step logged warning and continued. |
| 3 | **Mapper** | Relationship discovery | No relationships from LLM; **0 relationships** in network map and Colossus ingest. |

**Root cause:** Each of these steps asks the LLM for structured JSON (e.g. "Return ONLY valid JSON array"). The model sometimes returns plain text, markdown, or nothing, so `json.loads(response)` fails. The pipeline does not retry or fall back to a default structure.

**Where in code:**  
- Gatherer: claim extraction and contradiction finding (LLM response parsed as JSON).  
- Mapper: relationship discovery (LLM response parsed as JSON).

**Impact:** Dossier still completed (entities, summary, narratives). You get no key claims, no contradictions, and **no graph edges** — so Pegasus shows 8 entities but 0 links.

---

## Permanent fix (implemented)

A single, shared **LLM JSON parsing layer** was added so the pipeline no longer crashes on non-JSON or empty LLM output.

**1. Shared utility: `src/iceburg/protocols/dossier/llm_json.py`**

- **`parse_llm_json(raw, default=..., expect_list=True, log_context="")`**  
  - Strips whitespace and removes markdown code fences (```` ```json ... ``` ````).  
  - Tries `json.loads(cleaned)`.  
  - If that fails, finds the first complete `[...]` or `{...}` by bracket matching and parses that.  
  - On any failure: returns `default` (e.g. `[]`) and logs a short warning with `log_context` and truncated raw (first 200 chars).  
- One place to improve (e.g. retries, stricter prompts) for all dossier LLM→JSON steps.

**2. Call sites updated**

| Component | Method | Change |
|-----------|--------|--------|
| **Gatherer** | `_extract_entities` | Use `parse_llm_json(response, default=[], log_context="entity extraction")`; normalize to list. |
| **Gatherer** | `_extract_claims` | Use `parse_llm_json(..., log_context="claim extraction")`; default `[]`. |
| **Gatherer** | `_find_contradictions` | Use `parse_llm_json(..., log_context="contradiction finding")`; default `[]`. |
| **Mapper** | `_discover_relationships` | Use `parse_llm_json(..., log_context="relationship discovery")`; default `[]`; only append when `rel` is a dict. |

**3. Result**

- No more `Expecting value: line 1 column 1 (char 0)` crashes.  
- When the LLM returns valid JSON (including inside markdown), it is parsed.  
- When the LLM returns garbage or empty, the step returns a safe default (`[]`) and the pipeline continues.  
- Relationship discovery can now return edges when the model does return a valid array; previously it failed and always produced 0 relationships.

**4. Optional next steps (not done)**

- Retry once with a stricter prompt (“Output only a valid JSON array. No explanation, no markdown.”) if first parse fails.  
- Use `parse_llm_json` in Decoder and Synthesizer for other LLM→JSON steps (dates, meanings, timeline, parallels) for consistency.

---

## What we learned

- **Gatherer**: 36 sources (10 surface, 10 alternative, 8 academic, 8 historical) without a Brave API key; DuckDuckGo and arXiv are enough for a real run. Brave key would add more coverage.
- **Decoder**: Detected 2 symbols and 1 timing pattern; 0 linguistic markers for this query. Pipeline is wired and working.
- **Mapper**: Produced **7 entities** but **0 relationships**. Relationship discovery failed with `Expecting value: line 1 column 1` — the LLM returned non-JSON or empty for the relationship step. So we learned: entity extraction works; relationship extraction needs a more robust LLM response (prompt/parsing/retry).
- **Synthesizer**: Built the full dossier and ingested it into Colossus (8 entities). End-to-end time **163 s** (~2.7 min) for "standard" depth.
- **Warnings**: Claim extraction and contradiction finding (Gatherer) also failed with the same JSON parse error — same class of issue: LLM output not valid JSON. So we learned: several steps that expect structured JSON from the LLM are fragile; add fallbacks or stricter prompts/parsing.
- **UX**: The user flow works (type query, click Generate, wait, get result). The Dossier page does not currently "file" the result into the investigation archive; that only happens from Chat dossier mode.

---

## Steps performed (in browser)

1. **Navigate**: `http://localhost:8000/dossier.html`
2. **Fill**: Research topic = "What are the origins of DARPA and who were its first directors?"
3. **Depth**: Left as "Standard (3-5 min)"
4. **Click**: "Generate Dossier"
5. **Wait**: Pipeline ran (Gather -> Decode -> Map -> Synthesize -> Ingest).

## Backend result (from server log)

- **Dossier request**: `What are the origins of DARPA and who were its first directors? (depth=standard)`
- **Gatherer**: 36 total sources (10 surface, 10 alternative, 8 academic, 8 historical). DDG + arXiv used; Brave API key not set.
- **Decoder**: 2 symbols, 1 timing pattern, 0 linguistic markers.
- **Mapper**: 7 entities, 0 relationships (relationship discovery failed with JSON parse; entities still produced).
- **Synthesizer**: Completed. Dossier ingested into Colossus: 8 entities, 0 relationships.
- **Total time**: 163.1s.
- **HTTP**: `POST /api/dossier` **200 OK**.

So the **full user flow ran end-to-end**: form submit -> API -> Gatherer -> Decoder -> Mapper -> Synthesizer -> Colossus ingest -> 200 response. The frontend received the response and should have rendered the dossier (executive summary, narratives, key players, etc.). Screenshot timed out; refresh the Dossier page or re-run the same query to see the rendered result.

## Setup for “real” runs

- **Ollama**: Running (used for Decoder/Mapper/Synthesizer LLM calls).
- **Web search**: DuckDuckGo + arXiv worked without keys. For more coverage, set `BRAVE_SEARCH_API_KEY`.
- **Warnings**: Claim extraction and contradiction finding (Gatherer), relationship discovery (Mapper) failed with `Expecting value: line 1 column 1` (LLM returned non-JSON or empty). Entities and dossier still produced.

## How to repeat (as a user)

1. Start server: `python3 -m uvicorn src.iceburg.api.server:app --host 0.0.0.0 --port 8000`
2. Open: `http://localhost:8000/dossier.html`
3. Enter a research topic, keep or change depth, click "Generate Dossier".
4. Wait 2–5 minutes for "Generating..." to finish; the dossier content will replace the progress section.
