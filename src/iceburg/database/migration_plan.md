# PostgreSQL Migration Plan
## ICEBURG Database Migration Strategy

**Created:** February 2026  
**Status:** Design Phase  
**Target:** Migrate from SQLite + ChromaDB to PostgreSQL + pgvector

---

## Current State

- **SQLite**: Matrix Store (1.5M entities), conversations, metadata
- **ChromaDB**: Vector embeddings (currently mocked due to Rust binding issues)
- **JSONL**: Telemetry logs

## Target State

- **PostgreSQL + pgvector**: Unified database for all structured and vector data
- **PostgreSQL tables**: Conversations, telemetry (structured)
- **pgvector**: Vector embeddings (replacing ChromaDB)

---

## Migration Strategy

### Phase 1: Dual-Write Layer (Week 9)

**Goal:** Write to both SQLite and PostgreSQL simultaneously

**Implementation:**
- Create `unified_db.py` abstraction layer
- Feature flag: `ICEBURG_USE_POSTGRES` (default: false)
- When enabled: Write to PostgreSQL, also write to SQLite (dual-write)
- Read from PostgreSQL when enabled, SQLite when disabled
- Validate data consistency between both databases

**Files:**
- `src/iceburg/database/unified_db.py` - Abstraction layer
- `src/iceburg/database/sqlite_adapter.py` - SQLite operations
- `src/iceburg/database/postgres_adapter.py` - PostgreSQL operations

### Phase 2: Data Migration (Week 10)

**Steps:**
1. Export Matrix Store to PostgreSQL (batched, with progress)
2. Migrate conversations table
3. Export ChromaDB embeddings → pgvector
4. Convert JSONL telemetry to PostgreSQL tables
5. Verify data integrity (compare counts, sample records)

**Migration Scripts:**
- `scripts/migrate_to_postgresql.py` - Main migration script
- `scripts/verify_migration.py` - Validation script
- `scripts/rollback_to_sqlite.py` - Rollback script (if needed)

### Phase 3: Cutover (Week 11)

**Steps:**
1. Enable `ICEBURG_USE_POSTGRES=true` in production
2. Monitor for 24-48 hours
3. If issues: Rollback to SQLite via feature flag
4. If stable: Remove dual-write, remove SQLite code

**Risk Mitigation:**
- Keep SQLite code for 2 weeks after cutover
- Monitor error rates, performance
- Have rollback script ready

---

## Database Schema

### PostgreSQL Tables

```sql
-- Conversations
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    query TEXT NOT NULL,
    response TEXT,
    mode VARCHAR(50),
    agent VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Matrix Store (entities)
CREATE TABLE matrix_entities (
    id UUID PRIMARY KEY,
    entity_type VARCHAR(100),
    entity_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Vector embeddings (pgvector)
CREATE TABLE embeddings (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(1536), -- Adjust dimension as needed
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops);

-- Telemetry
CREATE TABLE telemetry (
    id UUID PRIMARY KEY,
    event_type VARCHAR(100),
    event_data JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

---

## Rollback Strategy

1. **Feature Flag Rollback**: Set `ICEBURG_USE_POSTGRES=false` to revert to SQLite
2. **Code Rollback**: Keep SQLite code in git, can revert commits
3. **Data Rollback**: Export from PostgreSQL back to SQLite if needed (script provided)

---

## Performance Considerations

- **Batch Size**: 10,000 entities per batch for Matrix Store migration
- **Connection Pooling**: Use connection pool for PostgreSQL
- **Indexes**: Create indexes after migration, not during
- **Monitoring**: Track migration progress and performance

---

## Success Criteria

- All data migrated successfully
- Counts match between SQLite and PostgreSQL
- Sample queries return identical results
- Performance acceptable (< 10% slower than SQLite)
- No data loss

---

**This plan prioritizes safety and rollback capability over speed.**
