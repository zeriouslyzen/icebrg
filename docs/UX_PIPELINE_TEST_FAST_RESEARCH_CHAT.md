# UX Pipeline Test: Fast -> Research -> Report -> Fast Chat (Coherence & Memory)

Run the full pipeline in the browser: Fast mode research question, run research (background), get the report, then talk to Fast Chat about it and verify coherence and memory.

---

## Prerequisites

- ICEBURG running: `./scripts/start_iceburg.sh` (or already started)
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Ollama running (for Research mode)
- Optional: `BRAVE_SEARCH_API_KEY` set for richer dossier/research

---

## Test Steps (in order)

### 1. Prompt in Fast mode with a research question

1. Open http://localhost:3000
2. Leave the mode dropdown on **Fast (Chat)**
3. Type a research-style question, e.g.:
   - "What do we know about connections between astrology and organs?"
   - "Who really runs the United Nations, including esoteric or structural angles?"
4. Send the message

**Expected:** Secretary replies in Fast mode. The reply may describe the multi-agent pipeline (Surveyor, Dissident, Synthesist, Oracle) and/or offer to run full research. If you see a **"Run full research (Surveyor → Dissident → Synthesist → Oracle)"** button below the message, continue to step 2. If not, you can still run Research manually (step 2).

### 2. Run research (background)

**Option A – Use the CTA (if shown)**  
- Click **"Run full research (Surveyor → Dissident → Synthesist → Oracle)"** below the Secretary’s reply.  
- The same query is re-sent in **Research** mode.

**Option B – Run Research manually**  
- Set the mode dropdown to **Research**
- Send the same (or similar) query again

**Expected:**  
- You see a **research status strip**: e.g. "Research: surveyor (12.3s)", "Research: dissident (25.1s)", etc., then **"Research complete (Xs)"** in green.  
- The final research content streams in (Surveyor/Dissident/Synthesist/Oracle synthesis).  
- When research is complete, the mode dropdown is set back to **Fast** so the next message is Fast Chat.

### 3. Get the report

- Wait until the research response has fully streamed and you see **"Research complete (Xs)"**.  
- Scroll to read the synthesized report in the same thread.

**Expected:** You have a clear “report” in the thread (the research output). This is also stored as **last research** for the connection so Fast Chat can use it.

### 4. Talk to Fast Chat about it (stay in Fast)

1. Leave the mode on **Fast (Chat)** (it should already be Fast after Research complete).
2. Send follow-up questions about the report, e.g.:
   - "Summarize the main finding and what we should do next."
   - "What did the Dissident say about that?"
   - "Dive into one pattern you see and why it matters."

**Expected:**  
- Secretary answers in Fast mode.  
- Answers should **reference the research report** (synthesis, Surveyor/Dissident/Synthesist/Oracle, findings) and build on it.  
- No need to re-run Research; the Secretary is using **last research** context.

### 5. Check coherence and “diving in”

- Ask 2–3 more follow-ups that refer back to the same report (e.g. “And what about the evidence the Oracle cited?”, “How does that connect to what you said earlier?”).

**Expected:**  
- Responses stay on topic and consistent with the report.  
- Secretary “remembers” the research (via last_research) and can dive deeper into specifics (patterns, evidence, next steps).  
- No contradiction between “the report said X” and “now you’re saying Y” unless you’re explicitly asking for a different angle.

---

## What this validates

| Check | What to look for |
|-------|-------------------|
| Fast mode offers research | Secretary describes pipeline and/or “Run full research” CTA appears. |
| Research runs and reports | Status strip shows stages + time; “Research complete”; full synthesis streams. |
| Report is available | You can read the research output in the thread. |
| Fast Chat sees the report | Follow-up answers reference the research (findings, agents, evidence). |
| Coherence | Follow-ups are consistent with the report and with each other. |
| Diving in | You can ask for more detail on one part and get a focused, on-report answer. |

---

## Backend integration test (optional)

From the project root:

```bash
./venv/bin/python -m pytest tests/integration/test_fast_research_chat_pipeline.py -v
```

This checks: (1) Secretary accepts `last_research_summary` and returns coherent responses when it’s set, and (2) the prompt passed to the provider includes the last research block when `last_research_summary` is provided.

---

## If something breaks

- **No “Run full research” button:** Ensure the Secretary’s reply actually describes the pipeline (Surveyor/Dissident/Synthesist/Oracle). The CTA only appears when the reply text is detected as offering research. You can still switch to Research mode and send the query again.
- **Research never completes / no status:** Check API logs (`logs/api_server.log`) for errors. Ensure Ollama is running and Research mode is selected when you send.
- **Fast Chat doesn’t refer to the report:** Confirm you ran Research in the **same** browser tab/session (same WebSocket). Last research is stored per connection; a new tab or refresh starts a new connection.
- **WebSocket disconnect during research:** Long runs may hit keepalive timeouts; the frontend will reconnect. If the report didn’t stream, run Research again or use a shorter query.
