# Research Mode vs Chat: When Does the Pipeline Run?

## Is what the AI said true?

The AI (Secretary) said it would "have the Surveyor provide an overview, the Dissident offer critical perspectives, the Synthesist integrate findings, the Oracle validate evidence" and "guide you through this process with real-time streaming."

- **Capability:** Yes. The system can do exactly that when the **Research** pipeline runs.
- **What actually happened:** You were in **Fast (Chat)** mode (the default). In that mode a **single agent (Secretary)** answers. It described what Research mode would do; it did **not** run Surveyor, Dissident, Synthesist, or Oracle. So the reply was a **description** of the system, not the result of running the multi-agent pipeline.

## How will you know, and how do you get the research?

You only get real multi-agent research when the **mode** sent with your message is **Research** or **Dossier**, not Fast/Chat.

| Mode (dropdown) | What runs | What you see |
|-----------------|-----------|---------------|
| **Fast (Chat)** (default) | Secretary only (one LLM). No Surveyor/Dissident/Synthesist/Oracle. | One conversational answer. The AI may *describe* the Research pipeline without running it. |
| **Research** | Full pipeline: Surveyor → Dissident → Synthesist → Oracle → Scrutineer → Scribe → Weaver → Supervisor. Progress streams as "thinking_stream" / "engines" / "algorithms". | Final response is the **synthesized research** (e.g. Supervisor or Synthesist output). You see progress messages during the run. |
| **Dossier (Deep Intel)** | Dossier pipeline: Gatherer (web/academic search) → Decoder → Mapper → Synthesizer. Uses Brave/DuckDuckGo/arXiv. | Streamed dossier text, then saved investigation with sources. Progress: "Gathering...", "Deep investigation active: Synthesis...", etc. |

**How to know you got research:**

1. **Before sending:** In the main app, set the mode dropdown to **Research** or **Dossier (Deep Intel)** instead of Fast (Chat).
2. **During the run:** You should see progress (e.g. "Initializing research protocol...", "Surveyor...", "Dissident...", or "Gathering multi-source intelligence...", "Deep investigation active: Synthesis...").
3. **After:** For Research, the final message is the integrated analysis. For Dossier, you get a long markdown report and (from Chat) a saved investigation ID.

If you leave the dropdown on **Fast (Chat)** and just type "explore connections between astrology and organs", the system runs **chat only** and the answer is the Secretary describing what *would* happen in Research mode, not the output of that pipeline.

## Is there a pipeline for "background research"?

Yes. There are two:

1. **Research mode (multi-agent deliberation)**  
   - **Backend:** `system_integrator.process_query_with_full_integration()` runs Surveyor, then Dissident (with Surveyor output), then Synthesist (Surveyor + Dissident), then Oracle, Scrutineer, Scribe, Weaver, Supervisor.  
   - **Trigger:** WebSocket message with `mode: "research"` (or `"deep_research"`).  
   - **Output:** `result.results.content` (or supervisor/synthesist/oracle text) streamed as chunks; `result.results.agent_results` has per-agent text.  
   - **UX:** Select **Research** in the dropdown, then send your prompt.

2. **Dossier mode (deep intel with web sources)**  
   - **Backend:** Gatherer (Brave/DuckDuckGo/arXiv) → Decoder → Mapper → Synthesizer; optional deep path via `RecursiveDossierPipeline`.  
   - **Trigger:** WebSocket message with `mode: "dossier"` or POST `/api/dossier`.  
   - **Output:** Dossier markdown streamed; investigation saved when run from Chat.  
   - **UX:** Select **Dossier (Deep Intel)** in the dropdown, then send; or use the Dossier page (`dossier.html`) and click Generate Dossier.

So "background research" exists: use **Research** for multi-agent deliberation (Surveyor/Dissident/Synthesist/Oracle), or **Dossier** for sourced, web-grounded intel. The pipeline does not run when the mode is Fast (Chat); only the Secretary responds.

## Quick reference

- **Want the answer the AI described (Surveyor + Dissident + Synthesist + Oracle)?**  
  Select **Research**, then send the same prompt again.

- **Want web-sourced intel and a saved investigation?**  
  Select **Dossier (Deep Intel)**, then send the prompt.

- **Default (Fast/Chat):** Single Secretary reply only; no multi-agent or dossier pipeline.

---

## Secretary prompt and Fast Chat + research

- **Secretary prompt:** Tuned for maximum effect and hyper-vigilant awareness. When **LAST RESEARCH** is in context, the Secretary is instructed to build on it: synthesize, surface emergent patterns, and co-explore with the user in Fast Chat.
- **Research progress:** During Research mode the frontend shows a **research status strip**: stage (Surveyor, Dissident, Synthesist, Oracle, etc.) and elapsed time. When research finishes it shows **Research complete (Xs)** and the mode dropdown is set back to **Fast** so the next message stays in Fast Chat.
- **Last research in Fast Chat:** The backend stores the last Research result per WebSocket connection. When you send a message in **Fast (Chat)** after a Research run, the Secretary receives that result as context and can reference it, synthesize, and build on it in conversation.
