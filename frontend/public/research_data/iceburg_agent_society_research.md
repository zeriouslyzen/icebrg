# Iceburg Deep Dive: Agent Society & Core Systems

## Overview
Investigation into high-priority autonomous systems within Iceburg, specifically focusing on the "Agent Society," "Micro-Agent Swarms," and the underlying execution protocols.

## 1. Agent Society (`src/iceburg/civilization/agent_society.py`)
*Status: Analyzed*

This module implements a sophisticated **Multi-Agent Social Dynamic System** designed to mimic civilization-level interactions.

### Key Components:
- **SocialLearningSystem**: The core engine managing interactions, reputation, and learning.
- **SocialNorm**: Represents emergent rules (e.g., "cooperation", "majority_rule") with strength and enforcement rates.
- **SocialInteraction**: Records interactions (cooperation, punishment, reward) between agents.
- **Mechanisms**:
    - **Imitation**: Agents copy successful peers (`get_peer_success_models`).
    - **Reputation**: Tracks agent reliability based on past behavior.
    - **Majority Rule**: Implements democratic decision-making among peer groups.
    - **Punishment/Reward**: Agents can enforce norms through social pressure.
    - **Norm Formation**: New norms can be created dynamically based on interaction patterns.

## 2. Micro-Agent Swarm (`src/iceburg/micro_agent_swarm.py`)
*Status: Analyzed*

A distributed system for parallelizing tasks across specialized, lightweight models.

### Key Components:
- **MicroAgent**: Lightweight agent specialized in specific domains (e.g., "Python Expert", "Data Analyst").
- **SwarmCoordinator**: Assigns tasks based on agent capability scores (`_calculate_agent_score`).
- **Optimization**:
    - **Pre-warming**: Loads models into memory before execution (`warmup`).
    - **Parallel Execution**: Thread-pooled execution of multiple tasks.
    - **Dynamic Metrics**: Tracks processing time and success rates to optimize future assignments.
- **Usage**: Enables high-throughput processing by breaking complex queries into sub-tasks handled by specialized 3B/8B models.

## 3. Protocol Execution Layer (`src/iceburg/protocol/execution/`)
*Status: Analyzed*

The execution backbone of Iceburg, managing the lifecycle of 23 specialized agents.

### Key Components:
- **Runner (`runner.py`)**: orchestrates agent task execution, supporting both sequential fallback and parallel execution modes.
- **Agents (`agents/`)**: Contains **23 specialized agents**, including:
    - **Core**: `Oracle`, `Synthesist`, `Surveyor`, `Dissident`
    - **Scientific**: `MolecularSynthesis`, `BioelectricIntegration`, `HypothesisTestingLaboratory`
    - **Governance**: `DecentralizedPeerReview`, `BlockchainVerification`, `ConsumerProtection`
    - **Evolution**: `UnboundedLearningEngine`, `SelfRedesignEngine`, `NovelIntelligenceCreator`

## 4. Middleware & Core Infrastructure
*Status: Analyzed*

### Global Middleware (`middleware/global_agent_middleware.py`)
A wrapper layer that intercepts all agent outputs to ensure safety and system awareness.
- **Hallucination Detection**: Automatically verifies agent claims against known facts.
- **Emergence Detection**: Scores output for emergent properties or self-awareness signs.
- **Sync/Async Support**: Handles both execution modes transparently.

### Parallel Execution (`parallel_execution.py`)
Enhanced engine for efficiency.
- **Dependency Graphs**: Builds execution DAGs to run independent agents simultaneously.
- **Early Termination**: optimized specifically for simple queries (terminate if complexity < 0.3).
- **Intelligent Timeouts**: Dynamically adjusts agent timeouts based on task complexity.

### Vectorstore (`vectorstore.py`)
Robust knowledge management.
- **Graceful Degradation**: "Mock Mode" allows system to function even if ChromaDB fails.
- **Persistence**: Singleton client management for stable database connections.
- **Semantic Search**: Wraps embedding generation and nearest-neighbor search.

## 5. IIR Core (Intermediate Internal Representation)
*Status: Analyzed*

The intermediate representation layer that enables safe, verifiable, and cross-platform agent execution.

### Contract Language (`iir/contract_language.py`)
A sophisticated specification language for ensuring reliability.
- **Temporal Logic**: Supports LTL/CTL operators (Globally, Finally, Next, Until) to define behavioral guarantees over time.
- **Quantifiers**: `forall` and `exists` logic for validating collections.
- **Invariants**: enforces mathematical consistencies throughout execution.

### Visual Contracts (`iir/visual_contracts.py`)
Specialized validation for UI generation.
- **Security**: checks for `no_external_scripts` and `no_inline_eval`.
- **Accessibility**: Enforces `wcag_aa_compliant` and `keyboard_navigable` standards.
- **Performance**: Validates `small_bundle_size` and `no_layout_shift`.

### Interpreter (`iir/interpreter.py`)
The runtime engine for IIR.
- **Operations**: Supports high-level primitives like `Map`, `Reduce`, `MatMul`, `Convolution`.
- **Sandboxing**: Executes code in isolated environments with trace hashing for reproducibility.
- **Math Kernel**: Includes elemental implementations of `sin`, `cos`, `exp`, `log` for neural operations.
