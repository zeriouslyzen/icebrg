# ICEBURG Core Data Schema (KI)

This Knowledge Item documents the unified database structure used by the ICEBURG agent swarm.

## Database: `data/iceburg_unified.db`

The primary persistent storage for ICEBURG. It uses a multi-domain schema to support research, agent memory, and system state.

### 1. Research & Discovery
- **`research_outputs`**: Stores the final results of research protocols.
  - `output_id` (PK), `research_type`, `title`, `content`, `domains`, `quality_score`, `validation_status`, `created_at`, `updated_at`, `metadata`.
- **`breakthrough_discoveries`**: High-confidence insights or major findings.
- **`cross_domain_synthesis`**: Results of synthesizing information across different fields.
- **`curiosity_queries`**: Internal agent-generated queries for proactive research.
- **`emergence_patterns`**: Documented patterns found in data streams.

### 2. Knowledge & Memory
- **`memory_entries`**: Short and long-term agent memory.
  - `memory_id` (PK), `memory_type`, `content`, `domain`, `importance`, `created_at`, `last_accessed`, `metadata`.
- **`knowledge_concepts`**: Atomic units of knowledge.
- **`knowledge_relationships`**: Semantic mapping between concepts.
- **`knowledge_gaps`**: Identified areas where information is lacking.
- **`latent_space_reasoning`**: Traces of LLM reasoning logic and embeddings.

### 3. Core Logic & Principles
- **`principles`**: Core values and foundational logic (the "ICEBURG Constitution").
  - `principle_id` (PK), `principle_name`, `version`, `summary`, `framing`, `domains`, `concepts`, `confidence`, `validation_status`, `is_current`.
- **`principle_evolution`**: Tracks changes to the core principles over time.
- **`multi_agent_sessions`**: Logs of collaborative sessions between agents.
- **`verification_records`**: Audit logs for agent validation and fact-checking.

### 4. Astro-Physiology (Health & Telemetry)
- **`astro_physiology_analyses`**: Detailed analysis of physical/biological metrics.
- **`astro_physiology_health_tracking`**: Time-series health data.
- **`astro_physiology_monitoring`**: Real-time monitoring logs.
- **`astro_physiology_feedback`**: User feedback on bio-telemetry.

### 5. System & Users
- **`users`**: Core user profiles.
  - `user_id` (PK), `email`, `name`, `created_at`, `updated_at`, `metadata`.
- **`teams` / `team_members`**: Organization and collaboration structure.
- **`system_config`**: Dynamic configuration overrides.
- **`system_metrics`**: Performance tracking (token usage, latency).

---

## Database: `matrix.db` (Experimental)
- **Status**: Currently initialized as an empty database. 
- **Purpose**: Intended for real-time high-throughput telemetry or transient state tracking that doesn't belong in the unified permanent record.

## Memory Strategy
- **Short-term**: Handled via `user_sessions` and fast-path caching.
- **Long-term**: Persisted in `memory_entries` with periodic consolidation by the Synthesist agent.
- **Vector Search**: While the relational schema is in SQLite, semantic search is augmented by ChromaDB (currently in Mock mode on M4 systems).
