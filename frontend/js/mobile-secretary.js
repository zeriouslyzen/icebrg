import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm";

// Configuration
const SELECTED_MODEL = "Phi-3-mini-4k-instruct-q4f16_1"; // Optimized for mobile (approx 2.3GB)

// State
let engine = null;
let isGenerating = false;
let messages = [
    { role: "system", content: "You are Iceburg's Mobile Secretary. You are running locally on the user's device via WebGPU. Be concise, helpful, and professional. Keep answers short (under 3 sentences) unless asked for more details." }
];

// DOM Elements
const DOM = {
    chatContainer: document.getElementById('chatContainer'),
    userInput: document.getElementById('userInput'),
    sendBtn: document.getElementById('sendBtn'),
    gpuStatus: document.getElementById('gpuStatus'),
    newsTicker: document.getElementById('newsTicker'),
};

// --- Initialization ---
async function init() {
    // Check WebGPU support
    if (!navigator.gpu) {
        DOM.gpuStatus.textContent = "UNSUPPORTED";
        DOM.gpuStatus.style.color = "#ff3333";
        addSystemMessage("CRITICAL: WebGPU not supported on this device. Local inference unavailable.");
        return;
    }

    DOM.gpuStatus.textContent = "DETECTED";
    DOM.gpuStatus.style.color = "#ffff00"; // Yellow for detected but not loaded
    
    // Auto-connect on load (or could be button triggered)
    // For now, we wait for first user message to trigger load, or load immediately?
    // Let's load immediately to be "ready" as requested.
    loadEngine();
    
    // Event Listeners
    DOM.sendBtn.addEventListener('click', handleSendMessage);
    DOM.userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Auto-resize textarea
    DOM.userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if(this.value === '') this.style.height = 'auto';
    });
}

// --- Engine Loading ---
async function loadEngine() {
    addSystemMessage("Initializing Local Inference Engine (Phi-3)...");
    DOM.gpuStatus.textContent = "LOADING...";
    
    const initProgressCallback = (report) => {
        console.log(report.text);
        DOM.newsTicker.innerText = `SYSTEM // ${report.text.toUpperCase()}`;
    };

    // Explicitly define the model config to avoid registry lookup errors
    const appConfig = {
        model_list: [
            {
                "model": "https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC",
                "model_id": "Phi-3-mini-4k-instruct-q4f16_1",
                "model_lib": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_80/Phi-3-mini-4k-instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
                "vram_required_MB": 3800,
                "low_resource_required": false,
            }
        ]
    };

    try {
        engine = await CreateMLCEngine(
            SELECTED_MODEL,
            { 
                appConfig: appConfig,
                initProgressCallback: initProgressCallback 
            }
        );
        
        DOM.gpuStatus.textContent = "ONLINE";
        DOM.gpuStatus.style.color = "#00ffaa";
        DOM.newsTicker.innerText = "SYSTEM // ENGINE ONLINE // READY FOR INPUT";
        addSystemMessage("Engine loaded. Encrypted local channel established.");
        
    } catch (err) {
        console.error("Engine Init Failed:", err);
        DOM.gpuStatus.textContent = "ERROR";
        DOM.gpuStatus.style.color = "#ff3333";
        addSystemMessage(`Error loading engine: ${err.message}`);
    }
}

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
        // Fallback or wait
        addSystemMessage("Engine not ready. Please wait...");
        return;
    }

    // 3. Generate Stream
    isGenerating = true;
    const thinkingId = addThinking();
    
    try {
        let fullResponse = "";
        const chunks = await engine.chat.completion({
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
            msgContentDiv.innerHTML = fullResponse.replace(/\n/g, '<br>') + '<div class="msg-decorator"></div>';
            scrollToBottom();
        }
        
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
