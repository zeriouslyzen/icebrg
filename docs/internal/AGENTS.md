# ICEBURG Agent Swarm: Chain of Command & Architecture

## 🎯 Architecture Overview
ICEBURG uses a multi-layered, specialized agent swarm architecture. The system is designed to transition from a single "Secretary" entry point into a hierarchical "Research Protocol" that deliberate across multiple domains.

---

## 👑 The Orchestrator (Top Level)

### **Secretary Agent** (`secretary.py`)
- **Role**: The "Face" of ICEBURG.
- **Function**: Handles WebSocket/HTTP incoming queries, manages conversation memory, and determines if a query needs **Chat Mode** (fast) or **Research Mode** (deep).
- **Expertise**: Routing, intent detection, and final response formatting.

---

## 🔬 Core Research Protocol (The Deliberation Chain)

When a query enters **Research Mode**, it follows this standard chain of command:

1.  **Surveyor** (`surveyor.py`)
    - **Goal**: Breadth.
    - **Actions**: Gathers initial info, explores domains, and collects evidence from the VectorStore (ChromaDB) and Web (Brave).
    
2.  **Dissident** (`dissident.py`)
    - **Goal**: Contrast.
    - **Actions**: Challenges the Surveyor’s assumptions. Hunts for suppressed information or alternative perspectives to ensure no bias.
    
3.  **Synthesist** (`synthesist.py`)
    - **Goal**: Integration.
    - **Actions**: Merges the Surveyor's data and Dissident's challenges into a coherent set of integrated insights.

4.  **Oracle** (`oracle.py`)
    - **Goal**: Intelligence.
    - **Actions**: Takes synthesized data and distills it into **Core Principles** and predictions. This is the "Aha!" moment of the research.

---

## 🛠 Support & specialized Agents

### **Coordination & Speed**
- **Prompt Interpreter**: Analyzes linguistic intent and etymology before the chain starts.
- **Reflex Agent**: Used in "Fast Chat" mode to compress verbose responses and provide instant bullet points.

### **Validation & Quality**
- **Scrutineer**: Reviews the Synthesist’s output for contradictions or logical gaps.
- **Supervisor**: Acts as the final quality gate before the user sees the output.
- **Hallucination Detector**: (Middleware) Cross-references facts against the internal knowledge base.

### **Implementation (The Doers)**
- **Weaver**: Translates Oracle principles into code or executable logic.
- **Scribe**: Formats research into academic reports and structured documentation.
- **IDE Agent**: Safely executes commands or edits files in the project workspace.

### **Intelligence Expansion**
- **Archaeologist**: Specialized in deep historical research and "buried" evidence recovery.
- **Deliberation Agent**: Adds "thinking cycles" to the swarm for meta-analysis.
- **Capability Gap Detector**: Monitors the swarm's performance and identifies when a new specialized agent needs to be created.

---

## 🏗 Architect Agents (System Evolution)
These agents aren't just for research; they are for **Self-Improvement**:
- **Swarm Architect**: Manages parallel execution of micro-agents.
- **Pyramid DAG Architect**: Hierarchical task decomposition for extremely complex engineering.
- **Visual Architect**: Generates UI and diagrams (powering the frontend visuals).

---

## 🔄 Data Flow Summary
`User` → `Secretary` → `Prompt Interpreter` → `[Protocol Swarm: Surveyor -> Dissident -> Synthesist -> Oracle]` → `Scrutineer` → `Reflex Agent` → `User`
