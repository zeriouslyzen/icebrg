# ICEBURG Changelog

All notable changes to the ICEBURG project will be documented in this file.

## [3.4.0] - 2025-12-23

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
