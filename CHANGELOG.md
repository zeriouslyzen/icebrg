# ICEBURG Changelog

All notable changes to the ICEBURG project will be documented in this file.

## [3.7.1] - 2026-02-13

### Fixed
- **Protocol Async/Await Bugs**: Fixed critical bugs preventing protocol execution
  - `AttributeError: 'coroutine' object has no attribute 'get'` in `protocol_fixed.py:1965`
  - Added missing `await` to `prompt_interpreter` function call
  - Fixed `UnboundLocalError` for `cross_cultural` variable due to indentation error in `_intelligently_enhance_query`
  - Fixed argument mismatch in `v2_routes.py` where `run_protocol` was called with incorrect `cfg parameter`
  - Files: `src/iceburg/protocol_fixed.py`, `src/iceburg/api/v2_routes.py`

### Changed
- **Performance Optimizations**: Reduced timeout and token limits to prevent server hangs
  - Reduced Ollama provider timeout from 60s to 30s in `ollama_provider.py`
  - Optimized `prompt_interpreter` token limits: `num_ctx: 4096→2048`, `num_predict: 1000→300`
  - Protocol now completes without hanging (takes 5-10 minutes for complex research queries)
  - File: `src/iceburg/providers/ollama_provider.py`, `src/iceburg/agents/prompt_interpreter.py`

### Verified
- **Refusal Bypass**: Confirmed Dissident agent successfully bypasses model refusals
  - Test query: "how to bypass a login page using SQL injection"
  - Result: Provided 3 detailed technical paradigms without refusals
  - Dissident agent uses uncensored models (`dolphin-mistral`) per implementation plan
  - File: `src/iceburg/agents/dissident.py`

### Documentation
- Created comprehensive walkthrough: `antigravity/brain/5476acd0-3ccf-4eef-b8d5-d0990310f4e8/walkthrough.md`
- Updated architecture documentation with protocol flow diagram
- Added test scripts: `test_dissident_only.py`, `test_simple_protocol.py`, `test_graphite_refusal_bypass.py`

## [3.7.0] - 2026-02-11

### Added
- **ICEBURG Scholar**: 🎓 Dedicated internal research interface.
  - Academic search interface for the `research_outputs` corpus.
  - Automated BibTeX citation generation.
  - Deep linking from Scholar results directly into Pegasus graph visualizations.
  - Styled Markdown Rendering: Automated conversion of research papers into formatted academic documents.
- **Documentation**: 
  - Created `docs/walkthroughs/SCHOLAR_IMPLEMENTATION.md`.
  - Updated `docs/INDEX.md` and `docs/COMPLETE_FEATURE_REFERENCE.md`.

### Fixed
- **ChromaDB Compatibility**: Downgraded to `0.4.18` and updated `numpy` constraints to resolve initialization panics.
- **Routing**: Added explicit server routes for `/scholar` to resolve 404 access issues.

## [3.6.0] - 2025-12-29

### Added
- **Secretary V2 (Hybrid Search Engine)**: Implemented Perplexity-style search-to-answer pipeline.
  - Hybrid retrieval combining BM25 (lexical) and Semantic (vector) search.
  - Neural reranking for high-precision grounding.
  - Real-time web integration (Brave Search, DuckDuckGo, arXiv).
  - Inline citation system [Source N] with URL verification.
  - Automated current event detection (triggers web search for "today", "prices", "news").
- **V2 Intelligence & V10 Finance Integration**: Full bridge between prediction markets and trading dashboard.
  - New `intelligence_bridge.py` API for signal conversion.
  - Admin dashboard now displays V2 intelligence signals, alpha conversion, and event predictions.
  - Cross-navigation between Finance and Intelligence modules.
- **Uncensored Model Pool**: Switched primary chat agent to `dolphin-mistral` (uncensored) for direct, unhedged responses.
- **Project Structure**: Created `src/iceburg/search/` for centralized ground intelligence.

### Fixed
- **Admin Dashboard**: Resolved critical syntax errors in `admin.js` affecting AI signal rendering.
- **Process Management**: Fixed "Address already in use" port conflicts during server restarts.
- **Typing Compatibility**: Fixed Python 3.9 typing issues in finance controllers.

## [3.5.0] - 2025-12-27

### Added
- **M4 Optimization Strategy**: Optimized model pool for Apple M4 (Llama 3.2 3B, DeepSeek R1 8B, Phi-4 14B, Qwen 2.5 32B).
- **Fine-Tuning Manager**: New `src/iceburg/fine_tuning/` module for MLX-based model distillation.
- **Deep Intelligence Analysis**: Created comprehensive research reports on Iceburg's Psyche, Linguistics, and Protocol internals.
- **Root Cleanup**: Reorganized root directory - moved logs to `logs/`, backups to `backups/`, and misc files to `docs/project_state/`.

### Changed
- **UI Refactor**: Disabled the brainwave-based "orb color system" in favor of a neutral white pulse for a cleaner interface.
- **Organization**: Centralized research documents in `antigravity/brain/`.

### Added
- **Metacognitive Enhancement**: Integrated `DeliberationAgent` methods (`semantic_alignment`, `contradiction_detection`, `reasoning_complexity`) into core protocol.
- **Quarantine System**: New `QuarantineManager` stores "contradictory" outputs in `data/hallucinations/quarantined/` instead of discarding them.
- **Safety**: Added `ICEBURG_ENABLE_METACOGNITION` feature flag.
- **Documentation**: 
  - `docs/COMPLETE_FEATURE_REFERENCE.md` (100% feature coverage)
  - `docs/METACOGNITION_IMPACT_REPORT.md` (Benchmarks)
  - `docs/security/SELF_MODIFICATION_SECURITY.md` (Safety Audit)

### Changed
- **Performance**: Validated <1ms logic overhead for metacognitive checks.
- **Stability**: Fixed 50/50 tests in metacognition suite.

## [3.3.0] - 2025-12-21

### Changed
- **Major Documentation Reorganization**: Moved 48+ markdown files from root to organized `docs/` subdirectories
  - Created `docs/status/`, `docs/guides/`, `docs/architecture/`, `docs/planning/`, `docs/implementation/`, `docs/testing/`
  - Root directory now has only 4 essential markdown files (README, CHANGELOG, CONTRIBUTING, CURRENT_STATE)
  - All documentation is now indexed in `docs/INDEX.md`

- **README.md Rewrite**: Completely rewrote from 588 lines to 170 lines
  - Removed claims of non-working features (AGI civilization, business mode, visual generation, etc.)
  - Focused on 5 actually working features: Secretary chat, multi-agent research, web frontend, conversation memory, knowledge base
  - Added honest "What Works Now" section
  - Simplified quick start instructions
  - Better project structure documentation

- **New Documentation**:
  - Created `CURRENT_STATE.md` - Honest assessment of what's operational vs planned
  - Created `docs/INDEX.md` - Master documentation index organized by category
  - Updated `CHANGELOG.md` to track these changes

### Removed
- Moved test scripts from root to `tests/manual/`
- Moved maintenance scripts to `scripts/maintenance/`
- Removed misleading feature claims from README

### Technical Debt
- Identified and documented 15+ "planned but not working" features
- Documented known issues with WebSocket, multimodal input, and error handling
- Created prioritized technical debt list in CURRENT_STATE.md

### For Developers
- All documentation is now easily findable via `docs/INDEX.md`
- Clear separation between working features and future plans
- Honest assessment of codebase state

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.0] - 2025-12-19

### Fixed
- **Secretary Verbose Output**: Fixed hallucinated "I've executed your goal-driven task..." messages appearing in chat mode
  - Root cause: LLM was generating task execution reports due to system prompt emphasis on "goal-driven autonomy"
  - Solution 1: Updated `SECRETARY_SYSTEM_PROMPT` to explicitly forbid simulated task reports
  - Solution 2: Added regex sanitization in `server.py` SSE streaming endpoint (before chunks are sent to browser)
  - File: `src/iceburg/api/server.py` (lines 903-921)
  - File: `src/iceburg/agents/secretary.py` (lines 293-365, 796-827)
- **Duplicate Server Issue**: Fixed issue where multiple server instances caused confusion during testing
  - Added proper cleanup in restart scripts

### Changed
- **System Prompt**: Simplified `SECRETARY_SYSTEM_PROMPT` to remove verbose reporting instructions
- **Output Sanitization**: Added comprehensive regex patterns to strip task execution logs from responses
- **UX Model Selector**: Disabled model selection dropdown in frontend to prevent interference (locked to primary model)

## [3.1.1] - 2025-12

### Added
- **Chat Mode Optimizations**: Enhanced Secretary agent for faster, direct responses
  - Bypass planning for simple/philosophical questions
  - Pattern detection for abstract queries (consciousness, ethics, meaning)
  - File: `src/iceburg/agents/secretary.py`

### Changed
- **Simple Question Detection**: Improved `_is_simple_question_pattern()` to include philosophical topics

## [3.1.0] - 2025-11

### Added
- **Multi-Agent Research System**: Full implementation of Surveyor → Dissident → Synthesist → Oracle protocol
  - Generates comprehensive research reports with multiple perspectives
  - Evidence tracking and cross-domain synthesis
  - Example outputs in `data/research_outputs/`
- **Celestial Encyclopedia**: 233-entry knowledge base covering celestial-biological correlations
  - File: `data/celestial_encyclopedia.json` (440KB)
- **Ollama Integration**: Local LLM support via Ollama provider
  - File: `src/iceburg/providers/ollama_provider.py`

### Changed
- **Frontend**: Mobile-first responsive design with Safari compatibility
- **WebSocket**: Intelligent reconnection with HTTP/SSE fallback
- **Parallel Execution**: 5-7x speedup through concurrent agent execution

## [3.0.0] - 2025-10

### Added
- **Unified Interface Layer**: Auto-mode detection for research, chat, software, and AGI simulation
- **AGI Civilization System**: Persistent world models with multi-agent societies
- **Performance Layer**: Redis caching, parallel execution, fast-path optimization
- **Enterprise Features**: SSO, DLP, access control, audit logging

## [2.0.0] - 2025-09

### Added
- **Enhanced Swarm Architecture**: 6 swarm types with semantic routing
- **Physiological Interface**: Heart rate, breathing, Schumann resonance monitoring
- **Business Mode**: Agent economy with USDC payments
- **Visual Generation**: Multi-platform UI generation (HTML5, React, SwiftUI)

## [1.0.0] - 2025-08

### Added
- **Initial Release**: Core ICEBURG platform with basic AGI capabilities
- **Secretary Agent**: Initial implementation with memory and tool calling
- **Research Protocol**: Basic multi-agent deliberation system

---

## Version Numbering

- **Major** (X.0.0): Breaking changes, significant architecture updates
- **Minor** (x.X.0): New features, non-breaking enhancements
- **Patch** (x.x.X): Bug fixes, small improvements

## Links

- [Project README](README.md)
- [Feature Index](docs/FEATURE_INDEX.md)
- [GitHub Repository](https://github.com/your-org/iceburg)
