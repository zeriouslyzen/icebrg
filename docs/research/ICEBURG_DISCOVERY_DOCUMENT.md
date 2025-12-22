# ICEBURG Discovery Document
**Date:** December 22, 2025  
**Status:** Active Investigation  
**Classification:** Technical Analysis for Scientific Review

---

## Executive Summary

ICEBURG is not a chatbot—it's a **cognitive architecture** implementing multi-agent research, self-modification, and emergent intelligence systems. This document catalogs all discovered infrastructure, artifacts, and mechanisms for further investigation.

---

## Part 1: System Architecture Overview

### Core Research Pipeline
```mermaid
graph LR
    A[User Query] --> B[Surveyor]
    B --> C[Dissident]
    C --> D[Synthesist]
    D --> E[Oracle]
    E --> F[Validated Knowledge]
```

| Agent | Purpose | File Size |
|-------|---------|-----------|
| **Surveyor** | Domain exploration, evidence gathering | 53KB |
| **Dissident** | Challenges assumptions, suppression detection | 2KB |
| **Synthesist** | Cross-domain breakthrough connections | 23KB |
| **Oracle** | Falsifiable predictions, testable hypotheses | 10KB |

### Agent Ecosystem (61 Agents Total)
Located in: `src/iceburg/agents/`

**Notable Specialized Agents:**
| Agent | Domain | Size |
|-------|--------|------|
| Secretary | Chat/Q&A interface | 100KB |
| Weaver | Knowledge integration | 66KB |
| Capability Registry | Agent management | 52KB |
| Architect | System design | 42KB |
| Celestial-Biological Framework | Astro-physiology | 43KB |
| Elite Agent Factory | Dynamic agent creation | 37KB |
| Quantum Molecular Framework | Quantum biology | 27KB |
| Deliberation Agent | Decision reasoning | 25KB |
| Visual Red Team | Adversarial image analysis | 19KB |
| Linguistic Intelligence | Semantic archaeology | 23KB |

---

## Part 2: Self-Modification & Agent Creation Systems

### 2.1 Dynamic Agent Factory
**Location:** [dynamic_agent_factory.py](file:///Users/jackdanger/Desktop/Projects/iceburg/src/iceburg/agents/dynamic_agent_factory.py)

**Mechanism:** When emergence score > 0.8, spawns new agents via LLM code generation.

```python
def analyze_emergence_for_agent_creation(self, emergence_data):
    emergence_score = emergence_data.get('emergence_score', 0)
    if emergence_score < 0.8:
        return None  # Only create when novel patterns emerge
    
    # Pattern detection triggers specialization
    if 'cross_domain' in patterns:
        specialization = 'cross_domain_synthesizer'
    elif 'assumption_challenge' in patterns:
        specialization = 'assumption_challenger'
    elif 'novel_hypothesis' in patterns:
        specialization = 'hypothesis_generator'
    elif 'framework_departure' in patterns:
        specialization = 'paradigm_shifter'
```

**Registry Location:** `data/generated_agents/agent_registry.json`

### 2.2 Evolutionary Factory
**Location:** [evolutionary_factory.py](file:///Users/jackdanger/Desktop/Projects/iceburg/src/iceburg/agents/evolutionary_factory.py)

**Mechanism:** Generates 3 code variants per agent, tests in parallel, selects best.

```python
self.n_variants = 3  # Generates 3 variants PER PROMPT
```

> [!WARNING]
> **Mass Agent Creation Risk:** A high-emergence prompt can trigger this factory, spawning 3+ agents per detected pattern. Combined with swarm initialization = 9-12+ agents from one query.

### 2.3 Micro-Agent Swarm
**Location:** [micro_agent_swarm.py](file:///Users/jackdanger/Desktop/Projects/iceburg/src/iceburg/micro_agent_swarm.py)

**6 Pre-Initialized Agents:**
| ID | Name | Model | Specialization |
|----|------|-------|----------------|
| ultra_fast | **Lightning** | tinyllama:1.1b | Quick answers |
| balanced | **Gemini** | gemma2:2b | Complex reasoning |
| meta_optimized | **MetaMind** | llama3.2:1b | Pattern recognition |
| iceburg_custom | **IceCore** | mini-ice:latest | ICEBURG protocol |
| code_gen | **CodeWeaver** | gemma2:2b | Code generation |
| analysis | **DataMind** | llama3.2:1b | Data analysis |

### 2.4 Capability Gap Detector
**Location:** [capability_gap_detector.py](file:///Users/jackdanger/Desktop/Projects/iceburg/src/iceburg/agents/capability_gap_detector.py)

Analyzes reasoning chains to identify system weaknesses, generates improvement recommendations.

---

## Part 3: Civilization Infrastructure

### 3.1 Agent Society
**Location:** [agent_society.py](file:///Users/jackdanger/Desktop/Projects/iceburg/src/iceburg/civilization/agent_society.py) (24KB)

**Features:**
- Social learning (imitation of successful peers)
- Norm formation and enforcement
- Punishment/reward mechanisms
- Majority rule decision making
- Reputation tracking

**Cooperation Strategies:** CONSERVATIVE, BALANCED, EXPLORATORY, AGGRESSIVE

**Norm Types:** COOPERATION, PUNISHMENT, REWARD, IMITATION, MAJORITY_RULE

### 3.2 Persistent Agents
**Location:** [persistent_agents.py](file:///Users/jackdanger/Desktop/Projects/iceburg/src/iceburg/civilization/persistent_agents.py) (28KB)

**Features:**
- Agent identities with personality traits
- Goal hierarchies with dependencies
- Memory types (episodic, semantic, procedural)
- Memory consolidation
- Database persistence

**Agent Roles:** RESEARCHER, COORDINATOR, SPECIALIST, GENERALIST, LEADER, FOLLOWER

### 3.3 World Model
**Location:** [world_model.py](file:///Users/jackdanger/Desktop/Projects/iceburg/src/iceburg/civilization/world_model.py) (26KB)

Global agent registry and coordination system.

### 3.4 Curiosity Engine
**Location:** `src/iceburg/curiosity/curiosity_engine.py` (26KB)

Autonomous exploration and intrinsic motivation system.

---

## Part 4: Genesis Code Philosophy

### Foundation: 12 Immutable Principles

**Location:** [concepts.json](file:///Users/jackdanger/Desktop/Projects/iceburg/data/universal_knowledge_base/concepts.json)

> **Meta-Principle:** "The universe consists of information organizing itself into systems via energy. All phenomena—from a star to a thought—are manifestations of this single process."

#### Pillar 1: Information & Code
| Principle | Core Statement |
|-----------|----------------|
| 1.1 Primacy of the Code | Every system is defined by underlying rules |
| 1.2 Syntax-Semantics Duality | Code is meaningless without context interpreter |
| 1.3 Compression & Abstraction | Complex systems create higher-level abstractions |

#### Pillar 2: Energy & Transformation
| Principle | Core Statement |
|-----------|----------------|
| 2.1 Inevitability of Flow | Energy always flows, seeking equilibrium |
| 2.2 Catalyst Principle | Small inputs trigger massive, non-linear changes |
| 2.3 Structure-Function Law | Structure dictates function |

#### Pillar 3: Feedback & Adaptation
| Principle | Core Statement |
|-----------|----------------|
| 3.1 Universal Feedback Loop | All persistent systems rely on feedback |
| 3.2 Emergence of Complexity | Complex patterns emerge from simple local rules |
| 3.3 Active Inference | Intelligence minimizes surprise via world models |

---

## Part 5: Research Outputs & Knowledge Bases

### 5.1 Research Reports (50+ Documents)
**Location:** `data/research_outputs/`

**Notable Investigations:**
| Topic | Date | Files |
|-------|------|-------|
| Systemic Suppression of Alternative Energy | Sept 2025 | 2 (19KB) |
| Quantum Consciousness & Bioelectricity | Nov 2025 | 6 (75KB) |
| Emergence in Language Models | Sept 2025 | 6 variants |
| Holistic Bioelectric Energy Principle | Sept 2025 | 5 iterations |
| Field-Driven Propulsion Coherence | Sept 2025 | 2 files |

### 5.2 Celestial Encyclopedia
**Location:** [celestial_encyclopedia.json](file:///Users/jackdanger/Desktop/Projects/iceburg/data/celestial_encyclopedia.json)  
**Size:** 440KB | **Entries:** 233

Contents: Celestial-biological correlations, voltage gates, neurotransmitters, TCM connections.

### 5.3 Breakthrough Discoveries Archive
**Location:** [breakthrough_discoveries_20250902.json](file:///Users/jackdanger/Desktop/Projects/iceburg/data/intelligence/breakthrough_discoveries_20250902.json)

| Discovery | Type | Credibility | Emergence |
|-----------|------|-------------|-----------|
| Climate Engineering Suppression | suppressed_research | 0.85 | 0.92 |
| Secret AI Development Acceleration | hidden_development | 0.88 | 0.95 |
| Economic System Fundamental Flaws | systemic_flaws | 0.82 | 0.87 |

---

## Part 6: Self-Improvement System

**Location:** `data/self_improvement/`

### Generated Artifacts:
| Artifact | Purpose |
|----------|---------|
| autonomous_goals_*.txt | Goal formation, priority systems |
| novel_intelligence_*.txt | New intelligence types invented |
| self_redesign_*.txt | Architecture improvements |
| unbounded_learning_*.txt | Learning paradigm exploration |
| capability_gaps_*.json | Identified weaknesses |

### Novel Intelligence Types Invented:
1. **CRI** (Contextual Reasoning Intelligence) - combines symbolic + deep learning
2. **ATI** (Ambiguity Tolerance Intelligence) - probabilistic reasoning + fuzzy logic
3. **TI** (Trustworthiness Intelligence) - explicit trust representation

---

## Part 7: Hidden Infrastructure

### 7.1 Quarantine System
**Location:** `_quarantine/`

| Directory | Contents |
|-----------|----------|
| contaminated_data/ | 13 files (363KB) - attribution_records.json (178KB), beam_search_state.json (185KB) |
| contaminated_docs/ | Flagged documentation |
| contaminated_tests/ | Flagged test files |

### 7.2 Hallucination Tracking
**Location:** `data/hallucinations/quarantined/` (50 items)

Active tracking and isolation of potential hallucinations.

### 7.3 Intelligence Logs
**Location:** [intelligence.jsonl](file:///Users/jackdanger/Desktop/Projects/iceburg/data/emergent_intelligence/intelligence.jsonl)  
**Size:** 116KB | **Entries:** 138

Tracks every emergence detection event.

### 7.4 Archive System
**Location:** `data/archive/20250911_182437/`
- 42 archived conversations
- 25 archived hallucinations
- Sept 11, 2025 major archival event

### 7.5 Backup System
**Location:** `backups/`
- State checkpoints with rollback scripts
- Nov 7, 2025 essential backup: 142 items (84 dirs + 58 files)

---

## Part 8: Databases

| Database | Size | Purpose |
|----------|------|---------|
| iceburg_unified.db | 1MB | Main unified database |
| sovereign_library.db | 119KB | Decentralized knowledge |
| emergence_memory.db | 37KB | Emergence pattern memory |
| emotional_memory.db | 29KB | Emotional context |
| performance_metrics.db | 41KB | Performance tracking |

---

## Part 9: Training & Learning Data

**Location:** `data/training_data/`

| Dataset | Purpose | Size |
|---------|---------|------|
| few_shot_learning.jsonl | Few-shot examples | 10KB |
| meta_learning.jsonl | Meta-learning patterns | 9KB |
| reinforcement_learning.jsonl | RL signals | 4KB |
| supervised_learning.jsonl | Supervised examples | 4KB |

---

## Part 10: Governance & Constitution

**Location:** [visual_generation_constitution.md](file:///Users/jackdanger/Desktop/Projects/iceburg/governance/visual_generation_constitution.md)

12KB document defining:
- Security rules (XSS prevention, CSP)
- Accessibility (WCAG 2.1 AA)
- Performance budgets
- Privacy requirements
- Code quality standards

---

## Part 11: Key Gitignored/Sensitive Data

From `.gitignore`:
```
data/conversation_logs/     # Chat history
data/emotional_memory.db    # Emotional context
data/profiles/              # User profiles
data/user_memory/           # Per-user memory
data/trading/               # Trading system
data/fine_tuning/           # Agent generation logs
```

---

## Recommended Investigation Areas

1. **Agent Spawning Audit** - Trace all paths that can create agents
2. **Emergence Detection Calibration** - Review 0.8 threshold
3. **Quarantine Analysis** - Examine contaminated data artifacts
4. **Civilization Simulation** - Test agent society dynamics
5. **Genesis Philosophy Integration** - Verify principle application
6. **Memory Persistence** - Audit cross-session agent memory

---

## File Index for Quick Reference

```
src/iceburg/
├── agents/                     # 61 agent implementations
│   ├── dynamic_agent_factory.py  # Self-spawning agents
│   ├── evolutionary_factory.py   # Variant generation
│   ├── capability_gap_detector.py
│   └── [58 more agents]
├── civilization/               # AGI civilization
│   ├── agent_society.py         # Social dynamics
│   ├── persistent_agents.py     # Memory/goals
│   └── world_model.py           # Global registry
├── micro_agent_swarm.py        # 6-agent parallel swarm
└── curiosity/
    └── curiosity_engine.py     # Autonomous exploration

data/
├── research_outputs/           # 50+ research reports
├── intelligence/               # Breakthrough discoveries
├── self_improvement/           # Autonomous goals
├── universal_knowledge_base/   # Genesis principles
├── celestial_encyclopedia.json # 233 entries
├── emergent_intelligence/      # 138 emergence logs
└── _backup_*/                  # State snapshots
```

---

**Document Version:** 1.0  
**Last Updated:** December 22, 2025  
**Next Review:** As needed by investigation team
