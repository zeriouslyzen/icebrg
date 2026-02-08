# ICEBURG Unification Plan
## From Multi-Service Research Tool → Single Consumer Product

**Created:** February 7, 2026  
**Status:** DRAFT - DO NOT EXECUTE  
**Author:** Architecture Review  

---

## Executive Summary

Transform ICEBURG from a complex multi-service research platform into a unified consumer product that works seamlessly on iPhone, Desktop, and Web—while preserving the unique multi-agent research protocol as the core differentiator.

---

## Current State Analysis

### What We Have Now
```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Vite)         → localhost:3000                   │
│  Backend (FastAPI)       → localhost:8000                   │
│  Ollama                  → localhost:11434                  │
│  ChromaDB                → In-process (currently mocked)    │
│  SQLite                  → File-based (multiple DBs)        │
│  WebSocket + HTTP SSE    → Dual transport (fragile)        │
├─────────────────────────────────────────────────────────────┤
│  Total Services: 3+ independent processes                   │
│  Total Ports: 3+                                            │
│  Mobile Ready: ❌                                           │
│  Consumer Ready: ❌                                         │
└─────────────────────────────────────────────────────────────┘
```

### Problems
1. **Too Many Moving Parts** - Users need Ollama installed, multiple ports open
2. **WebSocket Fragility** - Race conditions, connection timeouts
3. **No Mobile Path** - Architecture requires local services
4. **Slow First Response** - 3-15 seconds for multi-agent orchestration
5. **ChromaDB Issues** - Rust binding problems, currently mocked

### What Works Well (KEEP)
1. **Multi-Agent Research Protocol** - Unique IP, generates genuine insights
2. **37+ Specialized Agents** - Deep domain expertise
3. **Matrix Store** - 1.5M+ entity database
4. **Celestial Encyclopedia** - 302 entries of curated knowledge
5. **Emergence Detection** - Tracks novel discoveries
6. **Hallucination Prevention** - Grounding layer middleware

---

## Target State

### Vision: "One ICEBURG"
```
┌─────────────────────────────────────────────────────────────┐
│                    TARGET ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              ICEBURG UNIFIED CLIENT                  │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐  │   │
│   │  │  Web    │  │ iPhone  │  │   Desktop (Electron) │  │   │
│   │  │  App    │  │  App    │  │   or Native MacOS   │  │   │
│   │  └────┬────┘  └────┬────┘  └──────────┬──────────┘  │   │
│   │       │            │                   │            │   │
│   │       └────────────┼───────────────────┘            │   │
│   │                    │                                │   │
│   │                    ▼                                │   │
│   │         ┌──────────────────────┐                    │   │
│   │         │   SINGLE API LAYER   │                    │   │
│   │         │   api.iceburg.app    │                    │   │
│   │         └──────────┬───────────┘                    │   │
│   └──────────────────────────────────────────────────────   │
│                        │                                │   │
│                        ▼                                │   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              ICEBURG CLOUD BACKEND                   │   │
│   │  ┌────────────────┐  ┌────────────────────────────┐ │   │
│   │  │  FAST MODE     │  │    RESEARCH MODE           │ │   │
│   │  │  (Frontier API)│  │    (Multi-Agent Protocol)  │ │   │
│   │  │  Claude/Gemini │  │    Surveyor→Dissident→     │ │   │
│   │  │  50-200ms      │  │    Synthesist→Oracle       │ │   │
│   │  └────────────────┘  └────────────────────────────┘ │   │
│   │                                                     │   │
│   │  ┌────────────────────────────────────────────────┐ │   │
│   │  │  KNOWLEDGE LAYER (PostgreSQL + pgvector)       │ │   │
│   │  │  Matrix Store | Encyclopedia | User Memory     │ │   │
│   │  └────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Backend Unification (Week 1-2)

### 1.1 Create Unified API Layer

**Goal:** Single endpoint that handles all modes

**New File:** `src/iceburg/api/unified_endpoint.py`

```python
# Architecture sketch (DO NOT IMPLEMENT YET)

@app.post("/v2/query")
async def unified_query(request: QueryRequest):
    """
    Single endpoint for all ICEBURG interactions.
    
    Modes:
    - "fast": Route to frontier API (Claude/Gemini)
    - "research": Trigger multi-agent protocol
    - "encyclopedia": Query knowledge base
    """
    
    if request.mode == "fast":
        return await fast_mode_handler(request)
    elif request.mode == "research":
        return await research_mode_handler(request)
    elif request.mode == "encyclopedia":
        return await encyclopedia_handler(request)
```

**Changes Required:**
- [ ] Create `/v2/query` endpoint
- [ ] Abstract LLM provider (Ollama vs Cloud APIs)
- [ ] Unified response format
- [ ] Single SSE stream for all modes

### 1.2 LLM Provider Abstraction

**Goal:** Seamlessly switch between local Ollama and cloud APIs

**New File:** `src/iceburg/providers/unified_provider.py`

```python
# Architecture sketch

class UnifiedLLMProvider:
    """
    Single interface for all LLM backends.
    
    Backends:
    - ollama (local development, power users)
    - anthropic (Claude - production fast mode)
    - google (Gemini - backup/cost optimization)
    - openai (GPT-4 - compatibility)
    """
    
    async def complete(self, prompt: str, mode: str = "fast"):
        if mode == "fast":
            return await self._cloud_complete(prompt)
        elif mode == "research":
            return await self._research_complete(prompt)
```

**Changes Required:**
- [ ] Create provider abstraction layer
- [ ] Add API key management (env vars or config)
- [ ] Implement fallback chain (primary → backup)
- [ ] Add cost tracking per provider

### 1.3 Database Migration

**Current State:**
- SQLite (iceburg_unified.db, matrix.db, token_usage.db)
- ChromaDB (mocked)
- JSONL files (telemetry, emergence)

**Target State:**
- PostgreSQL + pgvector (single database)
- Supabase or self-hosted

**Migration Plan:**
- [ ] Export Matrix Store entities to PostgreSQL
- [ ] Migrate conversations table
- [ ] Replace ChromaDB with pgvector
- [ ] Convert JSONL telemetry to structured tables

---

## Phase 2: Frontend Simplification (Week 2-3)

### 2.1 Connection Simplification

**Goal:** Remove WebSocket complexity, use SSE-only

**Current:** WebSocket + HTTP SSE fallback (causes race conditions)  
**Target:** SSE-only with reconnection logic

**Changes to `frontend/main.js`:**
- [ ] Remove WebSocket initialization (lines 186-521)
- [ ] Use fetch + ReadableStream for SSE
- [ ] Simplify connection status management
- [ ] Remove `useFallback` logic

### 2.2 Unified Configuration

**Goal:** Single configuration source

**New File:** `frontend/iceburg-config.js`

```javascript
// Architecture sketch

export const ICEBURG_CONFIG = {
    apiEndpoint: import.meta.env.VITE_API_URL || 'https://api.iceburg.app',
    
    modes: {
        fast: {
            name: 'Fast',
            description: 'Instant responses from frontier AI',
            icon: '⚡',
            estimatedLatency: '< 1s'
        },
        research: {
            name: 'Research',
            description: 'Deep multi-agent analysis',
            icon: '🔬',
            estimatedLatency: '2-5 min'
        }
    }
};
```

### 2.3 Reduce main.js Complexity

**Current:** 9,525 lines  
**Target:** < 2,000 lines (core) + modular components

**Refactoring Plan:**
- [ ] Extract WebSocket handling → DELETE
- [ ] Extract streaming logic → `streaming.js`
- [ ] Extract UI components → `components/`
- [ ] Extract markdown rendering → `rendering.js`
- [ ] Keep only core orchestration in main.js

---

## Phase 3: Mobile-Ready Packaging (Week 3-4)

### 3.1 Progressive Web App (PWA) Optimization

**Goal:** Install on iPhone home screen

**Changes:**
- [ ] Update `manifest.json` with icons, splash screens
- [ ] Optimize service worker for offline caching
- [ ] Add iOS-specific meta tags
- [ ] Test Safari iOS compatibility

### 3.2 Capacitor Wrapper (Optional)

**Goal:** Native iOS/Android apps from web codebase

**New Files:**
```
/mobile/
├── capacitor.config.ts
├── ios/
│   └── App/
└── android/
    └── app/
```

**Implementation:**
- [ ] Initialize Capacitor project
- [ ] Configure iOS build
- [ ] Add native plugins (haptics, push notifications)
- [ ] Submit to App Store

### 3.3 Electron Desktop App (Optional)

**Goal:** Native macOS/Windows app

**New Files:**
```
/desktop/
├── electron-main.js
├── electron-preload.js
└── package.json
```

---

## Phase 4: Research Protocol Optimization (Week 4-5)

### 4.1 Keep the IP, Optimize the Speed

**Goal:** Reduce research mode from 2-5 min to 30-60 seconds

**Optimizations:**
- [ ] Parallelize Surveyor + Archaeologist
- [ ] Reduce deliberation overhead (currently instant, but can be removed)
- [ ] Cache frequent research patterns
- [ ] Pre-compute common synthesis paths

### 4.2 Research Credits System

**Goal:** Monetize the unique research capability

**Implementation:**
- [ ] Track research queries per user
- [ ] Free tier: 5 research queries/month
- [ ] Pro tier: Unlimited research
- [ ] API access for developers

### 4.3 Preserve Agent Specializations

**Keep These Agents Active:**
1. **Secretary** - Chat orchestrator (refactor to work with cloud APIs)
2. **Surveyor** - Information gathering
3. **Dissident** - Challenge assumptions
4. **Synthesist** - Integrate findings
5. **Oracle** - Pattern detection
6. **Archaeologist** - Historical research
7. **Molecular Synthesis** - Domain expert
8. **Corporate Network Analyzer** - Investigation tool

**Consider Deprecating:**
- Agents that duplicate cloud model capabilities
- Experimental/unused agents
- Field-aware consciousness interface (experimental)

---

## Phase 5: Deployment Architecture (Week 5-6)

### 5.1 Cloud Infrastructure

**Recommended Stack:**
```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION INFRASTRUCTURE                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CDN (Cloudflare/Vercel)                                   │
│       │                                                     │
│       ▼                                                     │
│  Frontend (Vercel/Netlify)                                 │
│       │                                                     │
│       ▼                                                     │
│  API Gateway (Vercel Edge / Cloudflare Workers)            │
│       │                                                     │
│       ▼                                                     │
│  Backend (Railway / Fly.io / Render)                       │
│       │                                                     │
│       ├── Cloud LLM APIs (Claude, Gemini)                  │
│       │                                                     │
│       ├── PostgreSQL + pgvector (Supabase / Neon)          │
│       │                                                     │
│       └── Redis (Upstash) - Session/Cache                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Environment Configuration

**Environment Variables (Production):**
```bash
# API URLs
ICEBURG_API_URL=https://api.iceburg.app

# LLM Providers
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_AI_KEY=...
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=postgresql://...

# Feature Flags
ICEBURG_FAST_MODE_PROVIDER=anthropic  # or 'google', 'openai'
ICEBURG_RESEARCH_ENABLED=true
ICEBURG_LOCAL_OLLAMA_FALLBACK=false

# Security
ICEBURG_API_KEY=... (for server-to-server)
JWT_SECRET=...
```

### 5.3 Local Development Mode

**Keep Ollama support for:**
- Development/testing
- Privacy-focused users
- Enterprise on-premise deployments
- Cost-conscious power users

```bash
# Local development with Ollama
ICEBURG_FAST_MODE_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
```

---

## Migration Timeline

```
Week 1: Backend Unification
├── Day 1-2: Create unified API endpoint
├── Day 3-4: Implement LLM provider abstraction
└── Day 5:   Test cloud API integration

Week 2: Database & Frontend Start
├── Day 1-2: PostgreSQL migration
├── Day 3-4: Start frontend simplification
└── Day 5:   Remove WebSocket, SSE-only

Week 3: Frontend Completion
├── Day 1-2: Refactor main.js
├── Day 3-4: PWA optimization
└── Day 5:   iOS Safari testing

Week 4: Research Protocol
├── Day 1-2: Optimize agent parallelization
├── Day 3-4: Implement research credits
└── Day 5:   Performance testing

Week 5: Packaging
├── Day 1-2: Capacitor iOS build
├── Day 3-4: Electron desktop build
└── Day 5:   App Store preparation

Week 6: Deployment
├── Day 1-2: Infrastructure setup
├── Day 3-4: Production deployment
└── Day 5:   Monitoring & launch
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Cloud API costs | High | Implement rate limiting, caching, cost caps |
| Research mode speed | Medium | Pre-compute, parallelize, cache patterns |
| User data privacy | High | Clear privacy policy, EU data residency option |
| Ollama users upset | Medium | Keep local mode as option, enterprise tier |
| App Store rejection | Medium | Follow guidelines, no crypto/gambling |
| WebSocket removal breaks features | Low | SSE handles all current use cases |

---

## Success Metrics

**Launch Goals:**
- [ ] First token latency < 500ms (fast mode)
- [ ] Research mode < 90 seconds
- [ ] iOS app installable and functional
- [ ] 99.9% uptime
- [ ] < 3 clicks to first query

**User Experience:**
- [ ] No mention of "Ollama" or "localhost" in consumer UI
- [ ] Seamless mobile/desktop experience
- [ ] Research mode clearly differentiated as premium

---

## What We Keep vs. Remove

### ✅ KEEP (Core IP)
- Multi-agent research protocol
- Surveyor → Dissident → Synthesist → Oracle flow
- Matrix Store entity database
- Celestial Encyclopedia
- Emergence detection
- Hallucination prevention middleware
- Agent specializations (Corporate Network Analyzer, etc.)

### ❌ REMOVE or SIMPLIFY
- WebSocket complexity (use SSE only)
- ChromaDB (replace with pgvector)
- Local-only dependencies
- 9500-line main.js (refactor to ~2000)
- Experimental field-aware consciousness interface
- Complex visualization panels (simplify for mobile)

### 🔄 REFACTOR
- LLM provider layer (abstract Ollama vs Cloud)
- Configuration management (single source of truth)
- Frontend build (Vite → potentially Next.js for SSR)
- Database layer (SQLite → PostgreSQL)

---

## Next Steps (When Ready to Execute)

1. **Review this plan** with stakeholders
2. **Decide on infrastructure provider** (Vercel, Railway, etc.)
3. **Set up cloud LLM API accounts** (Anthropic, Google)
4. **Create feature branch** for v2 development
5. **Begin Phase 1** - Backend unification

---

## Appendix: Files to Modify

**Backend (Priority Order):**
1. `src/iceburg/api/server.py` - Add unified endpoint
2. `src/iceburg/providers/` - Create unified provider
3. `src/iceburg/config.py` - Add cloud API config
4. `src/iceburg/agents/secretary.py` - Abstract LLM calls

**Frontend (Priority Order):**
1. `frontend/main.js` - Remove WebSocket, simplify
2. `frontend/vite.config.js` - Update for PWA
3. `frontend/manifest.json` - iOS optimization
4. `frontend/styles.css` - Mobile-first refactor

**New Files:**
1. `src/iceburg/api/unified_endpoint.py`
2. `src/iceburg/providers/unified_provider.py`
3. `frontend/src/streaming.js`
4. `mobile/capacitor.config.ts`
5. `desktop/electron-main.js`

---

**Document Status:** DRAFT - Awaiting approval before execution
