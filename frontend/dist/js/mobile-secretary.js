import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm";

// Configuration
const SELECTED_MODEL = "Phi-3-mini-4k-instruct-q4f16_1"; // Optimized for mobile (approx 2.3GB)

// State
let engine = null;
let isGenerating = false;
let messageCount = 0;
let messages = [
    { 
        role: "system", 
        content: `You are the TACTICAL SECRETARY for the ICEBURG Truth-Finding Engine. 
        OPERATOR: Jack Danger.
        MISSION: Assist the operator in organizing thoughts, retrieving rapid intel, and maintaining operational security.
        CONTEXT: Iceburg is a system dedicated to uncovering hidden patterns in history, psychology, and technology.
        PERSONALITY: Professional, concise, high-tech, slightly paranoid/cyberpunk. 
        CONSTRAINT: You are running LOCALLY on the operator's hardware. You do not have direct access to the mainframe database yet.
        If asked about complex topics (deep research), summarize what you know generally but advise connecting to the Mainframe for full dossiers.` 
    }
];

// ... (DOM config)

// ... (Init logic)

// --- Messaging Logic ---
async function handleSendMessage() {
    const text = DOM.userInput.value.trim();
    if (!text || isGenerating) return;
    
    // Clear Input
    DOM.userInput.value = '';
    DOM.userInput.style.height = 'auto';
    
    // 1. Add User Message
    addMessage(text, 'user');
    messages.push({ role: "user", content: text });
    
    // 2. Check Engine Status
    if (!engine) {
        addSystemMessage("Engine not ready. Please wait...");
        return;
    }

    // 3. RAG/Search Check
    // If text starts with "search:" or "research:", we hit the mainframe first
    const searchMatch = text.match(/^(?:search|research|find|lookup):\s*(.+)/i);
    let injectedContext = "";
    
    if (searchMatch) {
         const query = searchMatch[1];
         addSystemMessage(`ACCESSING MAINFRAME ARCHIVES FOR: "${query.toUpperCase()}"...`);
         DOM.newsTicker.innerText = "SYSTEM // CONNECTING TO MAINFRAME // SECURE LINK ESTABLISHED";
         
         try {
             // Fetch from our new server endpoint
             const response = await fetch('/api/mobile/context', {
                 method: 'POST',
                 headers: { 'Content-Type': 'application/json' },
                 body: JSON.stringify({ query: query, limit: 3 })
             });
             
             const data = await response.json();
             
             if (data.found) {
                 injectedContext = `\n\n[SYSTEM INTELLIGENCE FROM MAINFRAME]:\n${data.context}\n\nINSTRUCTION: Use the above intelligent report to answer the user's query about "${query}". Summarize key findings.`;
                 addSystemMessage("INTELLIGENCE RETRIEVED. TRANSFERRING TO LOCAL MODEL...");
                 
                 // Inject invisible system context for this turn
                 // We append it to the last user message effectively
                 messages[messages.length - 1].content += injectedContext;
             } else {
                 addSystemMessage("NO RECORDS FOUND IN MAINFRAME.");
             }
         } catch (e) {
             addSystemMessage("CONNECTION ERROR: COULD NOT REACH MAINFRAME.");
             console.error(e);
         }
    }

    // 4. Generate Stream
    isGenerating = true;
    const thinkingId = addThinking();
    
    try {
        let fullResponse = "";
        let tokenCount = 0;
        const startTime = performance.now();
        
        const chunks = await engine.chat.completions.create({
            messages: messages,
            stream: true,
        });

        // Remove thinking, start real message
        removeThinking(thinkingId);
        const params = addMessage("", "model"); // Empty start
        const msgContentDiv = params.contentDiv;
        
        for await (const chunk of chunks) {
            const content = chunk.choices[0]?.delta?.content || "";
            fullResponse += content;
            tokenCount++; 
            
            // Update Metrics
            const elapsed = (performance.now() - startTime) / 1000;
            const tps = (tokenCount / elapsed).toFixed(1);
            DOM.newsTicker.innerText = `GENERATING // ${tps} TPS // ${(elapsed).toFixed(1)}s`;
            
            // Render basic markdown/newlines
            msgContentDiv.innerHTML = fullResponse.replace(/\n/g, '<br>') + '<div class="msg-decorator"></div>';
            scrollToBottom();
        }
        
        // Final Metrics
        const finalElapsed = (performance.now() - startTime) / 1000;
        const finalTps = (tokenCount / finalElapsed).toFixed(1);
        DOM.newsTicker.innerText = `COMPLETE // ${finalTps} TPS // ${tokenCount} TOKENS`;
        
        // Save to history
        messages.push({ role: "assistant", content: fullResponse });
        
    } catch (err) {
        removeThinking(thinkingId);
        addSystemMessage(`Generation Error: ${err.message}`);
    } finally {
        isGenerating = false;
    }
}

// --- GUI Helpers ---
function addMessage(text, role) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    const bubble = document.createElement('div');
    bubble.className = "msg-bubble";
    bubble.innerHTML = text.replace(/\n/g, '<br>') + '<div class="msg-decorator"></div>';
    
    msgDiv.appendChild(bubble);
    DOM.chatContainer.appendChild(msgDiv);
    scrollToBottom();
    
    return { div: msgDiv, contentDiv: bubble };
}

function addSystemMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.style.textAlign = 'center';
    msgDiv.style.color = 'var(--secondary-text)';
    msgDiv.style.fontSize = '12px';
    msgDiv.style.margin = '10px 0';
    msgDiv.style.fontFamily = 'var(--font-mono)';
    msgDiv.innerText = `[SYS] ${text}`;
    DOM.chatContainer.appendChild(msgDiv);
    scrollToBottom();
}

function addThinking() {
    const id = 'thinking-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message model thinking-msg';
    msgDiv.id = id;
    msgDiv.innerHTML = `
        <div class="msg-bubble" style="background: transparent; border: none; box-shadow: none;">
            <div class="thinking-indicator">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    `;
    DOM.chatContainer.appendChild(msgDiv);
    scrollToBottom();
    return id;
}

function removeThinking(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    DOM.chatContainer.scrollTop = DOM.chatContainer.scrollHeight;
}

// Start
init();
