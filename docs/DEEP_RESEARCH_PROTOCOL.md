# Deep Research Protocol (ICEBURG Way)

**Purpose:** Document investigation prompts, Pegasus/Colossus connection, external tools (paid/free/open/city data), entity-tracing and vulnerability exposure (public data only), and the phased deep research protocol. Includes the Epstein deep research use case.

**Last Updated:** February 2026  
**Status:** Reference implementation

---

## 1. Investigation Prompts and Pegasus Connection

### How to trigger investigations (what to prompt)

**Dossier (new investigation):**

- **API:** `POST /api/dossier` with body:
  - `query`: string (topic or entity)
  - `depth`: `"quick"` | `"standard"` | `"deep"`
  - `format`: `"full"` | `"markdown"` | `"json"`
- **WebSocket/UI:** Send message with `mode: "dossier"` and `query: "<topic>"`. For a new investigation, `action: "full"` runs the full pipeline (Gatherer, Decoder, Mapper, Synthesizer). Depth is configurable (e.g. "deep" for new investigations in server code).
- **CLI:** `iceburg investigate "<topic>" -o report.md` (see `src/iceburg/cli.py`).

**Suggested prompts for investigations:**

| Type | Example prompt |
|------|----------------|
| Entity/topic | "Company X ownership structure and key executives" |
| Contradiction check | "Official narrative vs evidence on [topic]" |
| Network | "Key players and hidden connections in [sector/event]" |
| Civic | "[City/county] [issue]: zoning, developer, contract, budget" |
| Due diligence | "Due diligence dossier on [person/company] including sanctions, PEPs, offshore links" |

**Follow-up (chat on existing investigation):**

- Send `mode: "dossier"`, `investigation_id: "<id>"`, and `query: "<question>"`. The server loads the saved investigation and answers from the dossier context (no new gather).

### Pegasus connection (yes, it is connected)

- **Dossier to Colossus:** After synthesis, `DossierSynthesizer._ingest_to_colossus` calls `graph.ingest_dossier(dossier.to_dict())`. Colossus `Graph.ingest_dossier` creates/updates: (1) main topic entity, (2) key_players as entities with INVOLVED_IN to topic, (3) hidden_connections as HIDDEN_CONNECTION edges.
- **Pegasus:** Frontend at `frontend/pegasus.html` visualizes the Colossus graph (entities and relationships). Flow: **dossier to Colossus to Pegasus**. Same-entity search and traversal use Colossus/Matrix Store; Pegasus is the visualization layer.

---

## 2. External Tools Catalog (In-Repo vs To-Add)

### Already in ICEBURG (implemented or wired)

| Tool / Source | Purpose | Access | Location / Env |
|---------------|---------|--------|----------------|
| Brave Search | Web search | API key | `src/iceburg/search/web_search.py`; `BRAVE_SEARCH_API_KEY` |
| DuckDuckGo | Web search (no key) | Free | Same aggregator |
| arXiv | Academic papers | Free API | Same; `tools/deep_web_search.py` also uses arXiv |
| OpenCorporates | Companies, officers, jurisdictions | API (free tier / paid) | `src/iceburg/tools/osint/apis/opencorporates.py`; `OPENCORPORATES_API_TOKEN` |
| OpenSecrets | Political donations, lobbying | API key | `src/iceburg/tools/osint/apis/opensecrets.py`; `OPENSECRETS_API_KEY` |
| Wikidata | Entities, structured facts | Free | `src/iceburg/tools/osint/apis/wikidata.py` |
| OpenSanctions | Sanctions, PEPs, crime | Bulk JSON (free) | `src/iceburg/matrix/scrapers/opensanctions_scraper.py`; datasets: sanctions, peps, crime |
| ICIJ Offshore Leaks | Panama/Paradise/Pandora, offshore entities | CSV download (free) | `src/iceburg/matrix/scrapers/icij_scraper.py` |
| FEC | US federal campaign finance | Scraper/API | `src/iceburg/matrix/scrapers/fec_scraper.py` |
| Deep web search | arXiv, Mojeek, Qwant | Env-gated | `src/iceburg/tools/deep_web_search.py`; `ICEBURG_ENABLE_WEB` |

Gatherer uses web_search (Brave/DDG) and depth-driven layers: surface, alternative, academic, historical, deep. The deep layer is "corporate, legal"; it can be extended to call OSINT APIs (see Section 6).

### External tools to add (recommended list)

**Free / open:**

- **SEC EDGAR** – Filings, insiders, ownership (10-K, 10-Q, ownership reports).
- **City/county open data** – Council agendas, permits, contracts, parcels (e.g. Socrata, CKAN, local CSV/API).
- **Court records** – PACER (US federal, paid), state court portals (varies), RECAP; or aggregators (e.g. CourtListener API).
- **Lobbying** – Senate LDA, House LD-2 (public).
- **Land/parcels** – County assessors, state GIS (varies).
- **Nonprofits** – IRS 990 (IRS TEO, ProPublica Nonprofit Explorer).

**Paid / subscription (request or pay for):**

- **LexisNexis / Westlaw** – Legal and news (APIs or export).
- **D&B / Bureau van Dijk** – Company financials and ownership (API).
- **Bloomberg / Refinitiv** – Ownership, filings, news (API).
- **Local newspapers** – Archive access (subscription or one-off).
- **Document cloud / PDF archives** – For ingested reports and city docs.

**Same-entity and ownership tracing (public):**

- OpenCorporates (parent/subsidiary, officers) + ICIJ (offshore links) + OpenSanctions (PEPs/sanctions) + SEC EDGAR (insiders, 13F) + FEC/OpenSecrets (donations). Matrix Store / Colossus hold resolved entities and relationships; Pegasus visualizes. Entity resolution and "follow shareholders to root" = traverse graph from one entity by ownership/officer/donation links.

### Environment variables and API keys (config list)

| Variable | Purpose | Required for |
|----------|---------|--------------|
| `BRAVE_SEARCH_API_KEY` | Brave Search API | Web search (Gatherer surface/alternative layers) |
| `OPENCORPORATES_API_TOKEN` | OpenCorporates API | Company/officer lookup (optional deep layer) |
| `OPENSECRETS_API_KEY` | OpenSecrets API | Political donations, lobbying (optional deep layer) |
| `ICEBURG_ENABLE_WEB` | Set to `1` or `true` to enable web | Deep web search (arXiv, etc.) |

Optional keys for future connectors: SEC EDGAR (no key for public access), PACER credentials for court docs, city open data API keys as needed.

---

## 3. Entity Tracing and Vulnerability Exposure (Public Data Only)

**Goal:** Same-entity search, follow money/shareholders to root, expose methods/flaws/vulnerabilities that are documentable from public sources (filings, court docs, donations, sanctions, offshore leaks).

**Existing building blocks:**

- `src/iceburg/protocols/dossier/mapper.py`: Builds network of entities and relationships (member_of, funds, owns, connected_to, etc.); key_players, hidden_connections.
- `src/iceburg/colossus/core/graph.py`: In-memory graph; ingest_dossier adds topic + key_players + hidden_connections.
- `src/iceburg/matrix/entity_resolver.py`, `src/iceburg/matrix/entity_extractor.py`: Entity resolution and extraction.
- OSINT: OpenCorporates (officers, parents), OpenSecrets (donations, lobbying), ICIJ (offshore), OpenSanctions (PEPs/sanctions).

**Protocol (conceptual):**

1. **Seed:** Company or person name (from user prompt or dossier).
2. **Resolve:** Same-entity search across Matrix/Colossus + OpenCorporates/Wikidata to canonical ID and aliases.
3. **Expand:** Pull officers (OpenCorporates), donors/lobbying (OpenSecrets), offshore links (ICIJ), sanctions/PEPs (OpenSanctions), filings (SEC when added). Ingest into graph as entities and edges.
4. **Traverse:** From seed, follow ownership/officer/donation/offshore edges to parent companies, ultimate beneficial owners, related persons.
5. **Contradictions / vulnerabilities:** Use Dissident + Synthesist on "official narrative vs public record" (e.g. stated ownership vs filings, stated independence vs donations). Surface methods (e.g. offshore structures, lobbying patterns) and flaws (e.g. contradictions, sanctions exposure) as dossier sections.

**Deliverable:** A "deep entity" sub-protocol or dossier depth that: (a) takes an entity name, (b) calls external APIs/scrapers above, (c) resolves and merges into Colossus/Matrix, (d) runs mapper + synthesizer with "ownership chain and vulnerabilities" focus, (e) outputs dossier + graph for Pegasus. Implemented: `src/iceburg/tools/osint/deep_entity.py` provides `gather_entity_osint(entity_name)` returning companies, raw_sources, and entities_found for use by the dossier pipeline or Colossus. The Gatherer deep layer calls OpenCorporates when depth=deep and the query looks like an entity (see `src/iceburg/protocols/dossier/gatherer.py`).

---

## 4. Deep Research Protocol "The ICEBURG Way" (Over the Net)

**Principle:** Multi-source gathering + deliberation (Surveyor/Dissident/Synthesist/Oracle) + entity/network mapping + Colossus/Pegasus + export. All over the net = external data + web search + optional paid docs.

**Phases:**

1. **Define scope and seed**
   - Input: Research question + optional seed entities (company, person, city, topic).
   - Output: Structured query(ies) and depth (quick / standard / deep).

2. **External data pull (human or automated)**
   - Free: Brave/DDG, arXiv, OpenCorporates, OpenSecrets, Wikidata, OpenSanctions, ICIJ, FEC, SEC EDGAR (when added), city open data (when added).
   - Paid/subscription: User obtains docs (court, Lexis, D&B, etc.); upload or paste into ICEBURG (e.g. as context or ingested into vector store).
   - Same-entity: Resolve seed entities; pull ownership, officers, donations, offshore, sanctions; push into Matrix/Colossus.

3. **Gather (existing)**
   - GathererAgent.gather: Surface, alternative, academic, historical, deep layers via web_search. Deep layer can be extended to call OSINT/APIs and add to IntelligencePackage.

4. **Decode + Map (existing)**
   - Decoder: Symbol/pattern analysis on combined content.
   - Mapper: Build network from entities_found + content; key_players, hidden_connections. Optionally merge in Graph from Colossus/Matrix (entity resolution).

5. **Synthesize (existing)**
   - DossierSynthesizer: Executive summary, official/alternative narratives, contradictions, timeline, confidence, follow-up.

6. **Deliberation (optional, ICEBURG way)**
   - After dossier, run Surveyor/Dissident/Synthesist/Oracle on the dossier output: "Surveyor: summarize evidence. Dissident: challenge narrative and expose gaps. Synthesist: integrate. Oracle: principles and testable claims." Append to dossier or produce a separate "assessment" section.

7. **Ingest and visualize**
   - Ingest dossier into Colossus (already done in Phase 5 of synthesizer). Pegasus shows graph. Same-entity search = Colossus/Matrix queries.

8. **Export and persist**
   - Save as Investigation (dossier.md, metadata.json, network_graph.json, sources). PDF export via `src/iceburg/investigations/pdf_export.py`. Optional: webhook or API callback for downstream systems.

---

## 5. Epstein Deep Research Use Case

**Purpose:** Named deep-research thread: Epstein as a documented case of intel-gathering methods, elite finance and trade patterns, and reported beliefs/frameworks — and how that research maps onto ICEBURG's protocol and the Matrix (Colossus/Pegasus).

### Why this belongs in the protocol

- **Methods:** His methods (how he gathered intel, cultivated access, used finance and entities) are increasingly documented in court filings, victim testimony, flight logs, and investigations. Researching those methods = understanding patterns of elite access, leverage-adjacent structures, and financial/entity structures — all from public or released material.
- **Elite trade and finance:** How elites trade influence, use offshore entities, and move money is what ICIJ, OpenSanctions, OpenCorporates, and FEC/OpenSecrets expose. The same entity-graph and "follow the money" protocol (Section 3) applies: seed person/entity, resolve, expand (officers, donations, offshore links), traverse, contradictions/vulnerabilities.
- **Structured intel (Palantir-style):** Structured intel = entity graphs, relationship mapping, pattern detection over networks. ICEBURG's Matrix (Colossus + Pegasus) does exactly that: entities, relationships, hidden_connections, visualization. Deep research on this case = one concrete application (people, orgs, funds, flights, properties) so patterns become visible.
- **Metaphysical patterns and ideas:** His stated or reported beliefs (e.g. about science, evolution, influence, "the mind") are part of the record (interviews, books, depositions, secondary sources). Research = extract those claims from available material, analyze how they relate to his methods (e.g. framing of influence, networks, persuasion) and to what ICEBURG does (multi-perspective reasoning, dissent, network mapping). "Uncovering metaphysical patterns" in a research sense = pattern-in-beliefs and pattern-in-methods, from public sources only.

### Data sources (all public or released)

- **Court:** SDNY filings, plea agreements, victim statements (summaries/public portions), Florida state court records.
- **Investigative:** ICIJ (offshore entities linked to him or associates), flight logs (released), property and entity records.
- **Financial:** Entities and trusts (public registries), donations and political links (OpenSecrets/FEC where applicable).
- **Secondary:** Books (e.g. Whitney Webb, others), long-form journalism, depositions (released portions), documentaries and interviews — for narratives and for stated beliefs to extract and compare.
- **No illegal access:** No hacked or non-public material; only court filings, released docs, or published journalism/books.

### How ICEBURG and the Matrix apply

- **Dossier protocol:** One or more dossiers, e.g. "Epstein: intel-gathering methods and entity network," "Epstein: elite finance and trade patterns," "Epstein: stated beliefs and how they relate to his methods." Each runs Gatherer (multi-source), Decoder (symbols/patterns in text), Mapper (entities, key_players, hidden_connections), Synthesizer (narratives, contradictions, confidence).
- **Entity and the Matrix:** Seed = Epstein + named associates, entities, properties. Resolve and expand via OpenCorporates, ICIJ, OpenSecrets, court docs (when ingested). Colossus holds entities and relationships; Pegasus visualizes. "How the Matrix works" = same-entity search + traversal (who funded, who flew, who owned what) so elite trade and finance show up as graph structure.
- **Deliberation (ICEBURG way):** Surveyor summarizes evidence; Dissident challenges official narrative and exposes gaps; Synthesist integrates; Oracle states principles and testable claims. Apply to dossier output so "methods," "elite finance," and "metaphysical patterns" are stress-tested, not taken at face value.
- **Metaphysical patterns doc:** In this subsection: (a) prompt templates for methods / elite finance / stated beliefs, (b) data sources list (court, ICIJ, flight logs, books), (c) how his reported ideas relate to his methods and to ICEBURG (network mapping, influence, dissent). All framed as research questions over public material.

### Suggested prompts for Epstein deep research

- "Epstein: intel-gathering methods and how they relate to elite access and leverage — multi-source dossier with entity network."
- "Epstein: elite trade and finance — entities, offshore links, donations, and key players; map to Colossus/Pegasus."
- "Epstein: his stated or reported metaphysical/scientific ideas and how they relate to his methods and to network-based intel (Palantir-style / ICEBURG Matrix)."

### Deliverable

- **In this doc:** This subsection: scope (methods, elite finance, metaphysical patterns), data sources (court, ICIJ, flight logs, books), relation to ICEBURG (dossier, entity graph, deliberation) and how the Matrix works (Colossus/Pegasus, same-entity, traversal), plus the three prompt templates above.
- **Optional:** A single "Epstein" seed config or prompt preset that triggers depth=deep and entity expansion (Epstein + associates + entities) so the first dossier run pulls in OSINT/ICIJ/OpenSanctions where applicable.

---

## 6. Summary

| Item | Status / Action |
|------|-----------------|
| Investigation prompts | Use `mode: "dossier"` + `query` (and optional `depth`); API: `POST /api/dossier`. Prompt templates in this doc. |
| Pegasus connection | Yes: dossier to Colossus.ingest_dossier; Pegasus visualizes Colossus graph. |
| External tools list | In-repo: Brave, DDG, arXiv, OpenCorporates, OpenSecrets, Wikidata, OpenSanctions, ICIJ, FEC, deep_web_search. To-add: SEC EDGAR, city/open data, court/Lexis (paid), D&B/Bloomberg (paid). |
| Same-entity / shareholder / vulnerabilities | Use existing mapper + Colossus + OSINT; add entity-resolution and "deep entity" flow that pulls ownership/donations/offshore/sanctions and runs dossier + deliberation. |
| Deep research protocol | 8-phase protocol (scope, external pull, gather, decode, map, synthesize, optional deliberation, ingest/Pegasus, export); document and optionally implement "deep entity" and Gatherer deep-layer OSINT. |
| Epstein deep research use case | Subsection in this doc: methods, elite finance, metaphysical patterns (from public sources); data sources; relation to ICEBURG and the Matrix; three prompt templates; optional Epstein seed preset. |

All methods stay within **public records and legal OSINT** (filings, court dockets, donations, sanctions, offshore leaks, open data). This protocol does not include or endorse illegal access; it focuses on structured use of what ICEBURG has plus recommended external sources so deep research is repeatable and auditable.
