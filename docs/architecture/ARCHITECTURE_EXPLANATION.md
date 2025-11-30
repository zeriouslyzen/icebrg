# ICEBURG Architecture: How It Actually Works

## The Problem You Identified

Hardcoding forbidden phrases is a losing battle - the AI will just use different vague language. The real issue is architectural.

## How ICEBURG Is Supposed To Work

### Full Protocol (Research Mode)
```
1. Surveyor → gathers info, initial analysis
2. Deliberation Pause → reflects on Surveyor's output
3. Dissident → challenges assumptions, finds contradictions
4. Deliberation Pause → reflects on Dissident's output
5. Archaeologist → deep research, suppressed info
6. Synthesist → combines all perspectives, checks consistency
7. Oracle → extracts principles, validates truth
```

**Key Point**: Each agent checks the previous ones. Dissident catches Surveyor's bullshit. Synthesist catches contradictions. Oracle validates truth.

### Chat Mode (Current Problem)
```
1. Surveyor → runs alone, no cross-checking
2. Done
```

**Problem**: Surveyor runs in isolation with no other agents to catch pseudo-profound patterns.

## The Real Solution

### Option 1: Use Full Protocol Even in Chat Mode (Recommended)
- Run Surveyor → Dissident → Synthesist (fast path)
- Dissident will catch Surveyor's vague connections
- Synthesist will enforce truth consistency
- Takes 10-15s instead of 3-5s, but quality is much better

### Option 2: Make Surveyor Self-Critical
- Add a self-review step in Surveyor that checks its own output
- Use a simple LLM call: "Does this response contain vague pseudo-profound language?"
- If yes, rewrite it

### Option 3: Use Scrutineer Agent
- ICEBURG has a `scrutineer.py` agent designed to catch hallucinations
- Run Surveyor → Scrutineer → return Scrutineer's corrected version

## Current Code Flow

**Chat Mode** (`server.py` line 1597):
- If agent == "surveyor": calls `surveyor.run()` directly
- No other agents run
- No cross-checking

**Research Mode** (`system_integrator.py` line 116):
- Calls `process_query_with_full_integration()`
- Runs all agents in sequence with deliberation pauses
- Each agent validates the previous ones

## Recommendation

**Don't hardcode forbidden phrases.** Instead:

1. **In Chat Mode**: Run a minimal protocol: Surveyor → Dissident → Synthesist
   - Dissident will catch vague connections
   - Synthesist will enforce truth consistency
   - Still fast (10-15s) but much higher quality

2. **Or**: Add Scrutineer as a post-processing step
   - Surveyor generates response
   - Scrutineer reviews it for hallucinations/pseudo-profound language
   - Return Scrutineer's corrected version

This is how ICEBURG is designed to work - multiple agents checking each other, not hardcoded rules.

