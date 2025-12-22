# ⚠️ CODEBASE NOTICE — READ BEFORE MODIFYING

> **DO NOT delete, move, or rename files without understanding the system architecture.**  
> This codebase contains interconnected systems, generated agents, and persistent data that can break if restructured incorrectly.

---

## What Is ICEBURG?

ICEBURG is a **multi-agent cognitive research architecture** — not a simple chatbot. It includes:

- **61 specialized AI agents** that work together
- **Self-modification capabilities** (agents can spawn new agents)
- **Persistent memory systems** (emotional, emergence, conversation)
- **Research outputs** that have taken hours to generate
- **Quarantine systems** that isolate potentially bad data

---

## Critical Directories — DO NOT DELETE

| Directory | Purpose | Risk If Deleted |
|-----------|---------|-----------------|
| `data/` | All persistent data, research outputs, knowledge bases | **CATASTROPHIC** — Lose all generated knowledge |
| `src/iceburg/agents/` | 61 agent implementations | **SEVERE** — Core functionality lost |
| `src/iceburg/civilization/` | Agent society, persistent agents, world model | **SEVERE** — Civilization system breaks |
| `data/celestial_encyclopedia.json` | 440KB knowledge base (233 entries) | **HIGH** — Unique proprietary data |
| `data/research_outputs/` | 50+ generated research reports | **HIGH** — Hours of computation lost |
| `data/self_improvement/` | Autonomous goal/learning artifacts | **HIGH** — Self-improvement data lost |
| `_quarantine/` | Isolated potentially-contaminated data | **MEDIUM** — May reintroduce bad data |
| `backups/` | State checkpoints | **MEDIUM** — Lose recovery options |

---

## Directories Safe to Modify

| Directory | Notes |
|-----------|-------|
| `docs/` | Documentation can be reorganized |
| `tests/` | Test files can be modified |
| `scripts/` | Utility scripts |
| `frontend/` | UI code (won't break backend) |
| `config/` | Configuration files |

---

## Before Making Changes

1. **Read `CURRENT_STATE.md`** — Understand what actually works
2. **Check `docs/INDEX.md`** — Find existing documentation
3. **Search before creating** — Many features already exist
4. **Test before deleting** — Code may be used unexpectedly

---

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `CURRENT_STATE.md` | Honest feature status |
| `CHANGELOG.md` | Version history |
| `docs/INDEX.md` | Documentation navigation |
| `docs/ROADMAP.md` | Feature implementation plan |
| `docs/FEATURE_HIGHLIGHTS.md` | Showcase of unique capabilities |
| `docs/research/ICEBURG_DISCOVERY_DOCUMENT.md` | Deep-dive technical analysis |

---

## Architecture Quick Reference

```
ICEBURG/
├── src/iceburg/           # Core Python package
│   ├── agents/            # 61 specialized agents
│   ├── api/               # FastAPI server
│   ├── civilization/      # Agent society system
│   ├── memory/            # Persistence layers
│   └── trading/           # Financial systems
├── data/                  # All persistent data
│   ├── research_outputs/  # Generated research
│   ├── celestial_encyclopedia.json
│   └── *.db               # SQLite databases
├── frontend/              # Web UI
├── docs/                  # Documentation
└── config/                # Configuration
```

---

## Contact

If unsure about changes, check git history or consult project maintainers.

---

**Last Updated:** December 22, 2025
