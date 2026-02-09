// ICEBURG 2.0 - Mobile-First Frontend
// Modern, Futuristic UI with Morphing Animations

// Dependencies loaded from CDN - available as global variables:
// - marked (UMD)
// - hljs (highlight.js)
// - katex
import { ClientProcessor } from './client-processor.js';

// Immediate test - this should run immediately when the script loads
// Debug mode (set DEBUG=true in localStorage to enable)
const DEBUG_MODE = localStorage.getItem('ICEBURG_DEBUG') === 'true';
if (DEBUG_MODE) {
    console.log('🔥 ICEBURG Debug Mode Enabled');
    console.log('📅 Timestamp:', new Date().toISOString());
    console.log('🌐 Location:', window.location.href);
}

// Check if CDN dependencies are loaded (only in debug mode)
if (DEBUG_MODE) {
    console.log('📦 CDN Dependencies Check:', {
        marked: typeof marked !== 'undefined',
        hljs: typeof hljs !== 'undefined',
        katex: typeof katex !== 'undefined',
        window_marked: typeof window.marked !== 'undefined',
        window_hljs: typeof window.hljs !== 'undefined',
        window_katex: typeof window.katex !== 'undefined'
    });
}

// Simple test function (only in debug mode)
window.testICEBURG = function () {
    if (DEBUG_MODE) {
        console.log('🧪 ICEBURG TEST FUNCTION CALLED');
    }
    alert('ICEBURG Test: JavaScript is working!\n\nDependencies loaded:\n- marked: ' + (typeof marked !== 'undefined') + '\n- hljs: ' + (typeof hljs !== 'undefined') + '\n- katex: ' + (typeof katex !== 'undefined'));
};

// Load ICEBURG server configuration from embedded script tag
// Load ICEBURG server configuration from global config
let iceburgConfig = window.ICEBURG_CONFIG || null;

// Listen for config updates (e.g. from async fetch)
window.addEventListener('iceburg-config-updated', (event) => {
    iceburgConfig = event.detail;
    if (DEBUG_MODE) console.log('🔄 ICEBURG Config updated via event');
});

function loadICEBURGConfig() {
    // Just return the global config, potentially logging debug info
    if (iceburgConfig) {
        if (DEBUG_MODE) {
            console.log('✅ ICEBURG Config loaded:', {
                version: iceburgConfig?.serverConfig?.iceburg_version,
                branding: iceburgConfig?.serverConfig?.iceburg_branding,
                agents: iceburgConfig?.serverConfig?.agents?.agentPresets?.length,
                models: iceburgConfig?.serverConfig?.models?.length
            });
        }
        return iceburgConfig;
    }
    return null;
}

// Configuration from environment variables or defaults
// Detect if running on mobile/network access (iPhone, etc.)
const isNetworkAccess = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
const networkIP = isNetworkAccess ? window.location.hostname : 'localhost';

// Use environment variables or detect network IP
// Handle both Vite (import.meta.env) and static file serving (no import.meta.env)
let env = {};
try {
    // Try to access import.meta.env (available in Vite)
    env = import.meta.env || {};
} catch (e) {
    // import.meta not available (static file serving) - use empty env
    env = {};
}
const API_URL = env.VITE_API_URL || (isNetworkAccess ? `http://${networkIP}:8000/api/query` : 'http://localhost:8000/api/query');
const WS_URL = env.VITE_WS_URL || (isNetworkAccess ? `ws://${networkIP}:8000/ws` : 'ws://localhost:8000/ws');

// Always use HTTP/WS in development (never force HTTPS)
const FINAL_API_URL = API_URL;
const FINAL_WS_URL = WS_URL;

let ws = null;
let messageId = 0;
let isConnected = false;
let conversations = [];
let currentConversationId = 'current';
let attachedFiles = [];
let conversationContext = {
    topics: [],
    previousPoints: [],
    userPreferences: {},
    lastUserMessage: null,
    lastAssistantResponse: null
};
let settings = {
    primaryModel: 'gemini-2.0-flash-exp',  // Default to Gemini Flash (fast, cheap)
    temperature: 0.7,
    maxTokens: 2000
};

// Chunk ordering and buffering for consistent text display
let chunkBuffers = new Map(); // messageId -> {chunks: [], lastRender: 0}
let renderDebounceDelay = 50; // ms to wait before rendering after last chunk
let searchQuery = '';
let isRecording = false;
let recognition = null;
let synthesis = null;
let enableWebSearch = false;
let enableImageGeneration = false;

// Initialize client-side processor (leverages user's device)
let clientProcessor = null;

// Advanced visualization components
let astroPhysiologyViz = null;
let predictionLab = null;

// Configure marked for markdown rendering
marked.setOptions({
    highlight: function (code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try {
                return hljs.highlight(code, { language: lang }).value;
            } catch (err) {
                return hljs.highlightAuto(code).value;
            }
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
    gfm: true
});

// Custom renderer for charts
const renderer = new marked.Renderer();
const originalCode = renderer.code.bind(renderer);
renderer.code = function (code, lang, escaped) {
    if (lang === 'chart' || lang === 'chartjs') {
        try {
            JSON.parse(code); // Validate JSON
            const chartId = 'chart-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
            return `<div class="chart-container"><canvas id="${chartId}" data-chart='${code.replace(/'/g, "&apos;")}'></canvas></div>`;
        } catch (e) {
            return originalCode(code, lang, escaped);
        }
    }
    return originalCode(code, lang, escaped);
};
marked.setOptions({ renderer });

// Render math equations
function renderMath(content) {
    // Render inline math: $...$
    content = content.replace(/\$([^$]+)\$/g, (match, formula) => {
        try {
            return katex.renderToString(formula, { throwOnError: false, displayMode: false });
        } catch (e) {
            return match;
        }
    });
    // Render block math: $$...$$
    content = content.replace(/\$\$([^$]+)\$\$/g, (match, formula) => {
        try {
            return katex.renderToString(formula, { throwOnError: false, displayMode: true });
        } catch (e) {
            return match;
        }
    });
    return content;
}

// Connection state managed by connection-bridge.js
// isConnected is declared at top of file
// useFallback is declared at top if needed, or we can just ignore it as we are always in SSE mode


// Initialize connection status listener
if (window.ICEBURG_CONNECTION) {
    window.ICEBURG_CONNECTION.onConnectionChange((connected) => {
        isConnected = connected;
        updateConnectionStatus(connected);
    });
}


// WebSocket initialization removed - using SSE-only connection
// initWebSocket stub removed



// Enable HTTP fallback mode
// Fallback logic removed - always using SSE
function enableFallback() {
    console.log('enableFallback called - already in SSE mode');
    return; 
}


// Try to reconnect WebSocket (for manual retry)
// Retry logic removed
function retryWebSocket() {
    console.log('retryWebSocket called - functionality replaced by SSE connection');
    if (window.ICEBURG_CONNECTION) {
        window.ICEBURG_CONNECTION.checkHealth().then(health => {
             console.log('Connection check:', health);
        });
    }
}


// Update connection status (visual indicator)
// Update always-on AI status indicator
function updateAlwaysOnStatus(enabled) {
    let statusIndicator = document.getElementById('alwaysOnStatus');
    if (!statusIndicator) {
        // Create status indicator if it doesn't exist
        statusIndicator = document.createElement('div');
        statusIndicator.id = 'alwaysOnStatus';
        statusIndicator.style.cssText = 'position: fixed; top: 1rem; right: 1rem; padding: 0.5rem 1rem; background: rgba(0, 255, 255, 0.2); border: 1px solid rgba(0, 255, 255, 0.5); border-radius: 8px; font-size: 0.75rem; color: #00ffff; z-index: 1000; display: flex; align-items: center; gap: 0.5rem;';

        const header = document.querySelector('header') || document.body;
        header.appendChild(statusIndicator);
    }

    if (enabled) {
        statusIndicator.innerHTML = '⚡ <span style="font-weight: bold;">Always-On AI</span> <span style="color: #00ff00;">Active</span>';
        statusIndicator.style.display = 'flex';
    } else {
        statusIndicator.style.display = 'none';
    }
}

function updateConnectionStatus(connected) {
    const header = document.querySelector('.app-header');
    if (header) {
        if (connected) {
            header.classList.add('connected');
        } else {
            header.classList.remove('connected');
        }
    }
}

// Helper functions for message handling
function getOrCreateLastMessage() {
    const chatContainer = document.getElementById('chatContainer');
    if (!chatContainer) return null;

    // Remove welcome message if present
    const welcomeMessage = chatContainer.querySelector('.welcome-message');
    if (welcomeMessage) welcomeMessage.remove();

    // Get or create last assistant message
    let lastMessage = chatContainer.querySelector('.message.assistant:last-child');
    if (!lastMessage) {
        lastMessage = addMessage('', 'assistant');
    }
    return lastMessage;
}

function removeLoadingIndicators() {
    const chatContainer = document.getElementById('chatContainer');
    if (!chatContainer) return;

    // Remove loading messages
    const loadingMessages = chatContainer.querySelectorAll('.message.loading');
    loadingMessages.forEach(msg => msg.remove());

    // Remove loading indicators
    const loadingIndicators = chatContainer.querySelectorAll('.loading-indicator');
    loadingIndicators.forEach(indicator => indicator.remove());
}

function scrollToBottom(smooth = true) {
    const chatContainer = document.getElementById('chatContainer');
    if (!chatContainer) return;

    requestAnimationFrame(() => {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: smooth ? 'smooth' : 'auto'
        });
    });
}

// Handle streaming messages
function handleStreamingMessage(data) {
    console.log('🔵 handleStreamingMessage:', data.type, data);
    if (data.type === 'chunk') {
        console.log('📝 CHUNK received:', data.content?.substring(0, 50), '...');
    }

    const chatContainer = document.getElementById('chatContainer');
    if (!chatContainer) {
        console.error('❌ chatContainer not found!');
        return;
    }

    // Get or create last message
    const lastMessage = getOrCreateLastMessage();
    if (!lastMessage) return;

    // Remove loading indicators for most message types (but NOT thinking_stream - it needs to stay visible)
    const shouldRemoveLoading = ['action', 'thinking', 'algorithm_step', 'word_breakdown', 'chunk', 'agent_thinking', 'done', 'error'].includes(data.type);
    if (shouldRemoveLoading) {
        removeLoadingIndicators();
    }

    if (data.type === 'action') {
        addToStatusCarousel('action', data, lastMessage);
        scrollToBottom();
    } else if (data.type === 'thinking') {
        // Skip old thinking status completely - we use thinking_stream instead
        // Do NOT show the old "Thinking: Preparing response..." status
        scrollToBottom();
    } else if (data.type === 'thinking_stream') {
        // ICEBURG real-time thinking stream
        console.log('💭 Received thinking_stream:', data.content);
        console.log('💭 lastMessage element:', lastMessage);
        console.log('💭 lastMessage classes:', lastMessage?.className);

        if (!lastMessage) {
            console.error('❌ No lastMessage for thinking_stream! Creating one...');
            lastMessage = getOrCreateLastMessage();
            if (!lastMessage) {
                console.error('❌ Failed to create lastMessage!');
                return;
            }
        }

        // Ensure message element is visible
        if (lastMessage) {
            lastMessage.style.display = 'flex';
            lastMessage.style.visibility = 'visible';
        }

        // Update astro loading state if it exists
        const astroLoading = lastMessage.querySelector('.astro-loading-state');
        if (astroLoading) {
            const content = data.content || '';
            // Update step text based on content
            const steps = astroLoading.querySelectorAll('.astro-step');
            steps.forEach((step, index) => {
                const stepText = step.querySelector('.astro-step-text');
                if (stepText && content.toLowerCase().includes(stepText.textContent.toLowerCase().split('...')[0].toLowerCase())) {
                    step.style.color = '#00D4FF';
                    const icon = step.querySelector('.astro-step-icon');
                    if (icon) icon.textContent = '✓';
                }
            });

            // Also update the main spinner text
            const spinnerText = astroLoading.querySelector('h3');
            if (spinnerText && content) {
                spinnerText.textContent = '🔬 ' + content;
            }
        }

        handleThinkingStream(data, lastMessage);
        scrollToBottom();
    } else if (data.type === 'research_status') {
        // Research progress: stage + elapsed time; "complete" when done
        let strip = lastMessage.querySelector('.research-status-strip');
        if (!strip) {
            strip = document.createElement('div');
            strip.className = 'research-status-strip';
            strip.style.cssText = 'margin: 8px 0; padding: 8px 12px; background: rgba(0,0,0,0.04); border-radius: 6px; font-size: 12px; color: #444;';
            lastMessage.querySelector('.message-content')?.appendChild(strip);
        }
        const stage = data.stage || 'processing';
        const elapsed = data.elapsed_seconds != null ? data.elapsed_seconds : 0;
        const stageLabel = stage === 'complete' ? 'Research complete' : `Research: ${stage}`;
        strip.textContent = `${stageLabel} (${elapsed}s)`;
        strip.style.color = stage === 'complete' ? '#0a0' : '#444';
        if (stage === 'complete') {
            const modeSelectEl = document.getElementById('modeSelect');
            if (modeSelectEl) modeSelectEl.value = 'fast';
            if (typeof updateAgentOptionsForMode === 'function') updateAgentOptionsForMode('fast');
        }
        scrollToBottom();
    } else if (data.type === 'informatics') {
        addToStatusCarousel('informatics', data, lastMessage);
        scrollToBottom();
    } else if (data.type === 'conclusion') {
        addConclusionItem(data.content, lastMessage);
        scrollToBottom();
    } else if (data.type === 'step_complete') {
        handleStepComplete(data, lastMessage);
        scrollToBottom();
    } else if (data.type === 'algorithm_step' || data.type === 'word_breakdown') {
        // Get or create word breakdown display (shared logic)
        let wordBreakdownDisplay = lastMessage.querySelector('.word-breakdown-display');
        if (!wordBreakdownDisplay) {
            const lastUserMessage = chatContainer.querySelector('.message.user:last-child');
            const originalQuery = lastUserMessage ? lastUserMessage.querySelector('.message-content')?.textContent || '' : '';
            wordBreakdownDisplay = createWordBreakdownDisplay(lastMessage, originalQuery);
        }

        ensureWordBreakdownVisible(wordBreakdownDisplay);

        // Handle based on data structure
        if (data.type === 'algorithm_step' || data.step) {
            addWordBreakdown({
                type: 'algorithm_step',
                step: data.step,
                status: data.status,
                processing_time: data.processing_time
            }, lastMessage);
        } else if (data.word) {
            addWordBreakdown({
                type: 'word_breakdown',
                word: data.word,
                morphological: data.morphological,
                etymology: data.etymology,
                semantic: data.semantic,
                compression_hints: data.compression_hints
            }, lastMessage);

            // Collect linguistic origins after words are added
            setTimeout(() => {
                const flowingText = wordBreakdownDisplay.querySelector('.word-flowing-text');
                const originsDiv = wordBreakdownDisplay.querySelector('.word-linguistic-origins');
                if (flowingText && originsDiv) {
                    const wordContainers = flowingText.querySelectorAll('.word-neon-container');
                    const origins = [];
                    wordContainers.forEach(container => {
                        const origin = container.getAttribute('data-origin');
                        const roots = container.getAttribute('data-roots');
                        const wordText = container.textContent.trim();
                        if (origin && origin !== 'unknown' && wordText) {
                            origins.push(`${wordText}: ${origin}${roots ? ` (${roots})` : ''}`);
                        }
                    });
                    if (origins.length > 0 && originsDiv.textContent === '') {
                        originsDiv.textContent = origins.join(' • ');
                        originsDiv.style.display = 'block';
                        setTimeout(() => {
                            if (originsDiv.parentNode) {
                                originsDiv.style.opacity = '0';
                                originsDiv.style.transform = 'translateY(-5px)';
                                setTimeout(() => {
                                    if (originsDiv.parentNode) originsDiv.style.display = 'none';
                                }, 300);
                            }
                        }, 3000);
                    }
                }
            }, 1000);
        }

        scrollToBottom();
    } else if (data.type === 'reflex_preview') {
        // Display 3-bullet preview from Reflex Agent (FIRST - before chunks)
        if (data.bullets) {
            addReflexPreview(data.bullets, lastMessage, data.compression_ratio);
            scrollToBottom();
        }
    } else if (data.type === 'chunk') {
        // Ensure we have content
        if (data.content) {
            // Hide/fade thinking messages when chunks start (smooth transition)
            const statusCarousel = lastMessage.querySelector('.status-carousel');
            if (statusCarousel) {
                const thinkingItems = statusCarousel.querySelectorAll('.status-item.thinking');
                thinkingItems.forEach(item => {
                    // Fade out thinking items smoothly
                    item.style.transition = 'opacity 0.3s ease-out';
                    item.style.opacity = '0.3';
                    // Mark as complete so they stop cycling
                    item.classList.add('complete');
                });
            }

            // GPT-5-style instant rendering - no animation delays
            // Use requestAnimationFrame for smooth, instant updates
            requestAnimationFrame(() => {
                const contentDiv = lastMessage.querySelector('.message-content');
                if (contentDiv) {
                    // Instant streaming indicator (no delay)
                    contentDiv.classList.add('streaming-instant');
                    // Remove after very short time (50ms for visual feedback only)
                    setTimeout(() => contentDiv.classList.remove('streaming-instant'), 50);
                }

                // Instant append with requestAnimationFrame for smooth rendering
                appendToLastMessage(data.content, lastMessage);

                // Smooth scroll (only if near bottom)
                const chatContainer = document.getElementById('chatContainer');
                if (chatContainer) {
                    const isNearBottom = chatContainer.scrollHeight - chatContainer.scrollTop < chatContainer.clientHeight + 100;
                    if (isNearBottom) {
                        requestAnimationFrame(() => {
                            scrollToBottom();
                        });
                    }
                }
            });
        }
    } else if (data.type === 'sources') {
        addSourcesDisplay(data.sources, lastMessage);
        scrollToBottom();
    } else if (data.type === 'total_knowledge' || data.total_knowledge) {
        // Display deep knowledge decoding results
        addTotalKnowledgeDisplay(data.total_knowledge || data, lastMessage);
        scrollToBottom();
    } else if (data.type === 'agent_thinking') {
        // Add agent thinking to status carousel with user-friendly name
        // The backend already sends user-friendly content, so use it directly
        addToStatusCarousel('thinking', {
            content: data.content, // Already formatted with user-friendly name
            agent: data.agent
        }, lastMessage);
        scrollToBottom();
    } else if (data.type === 'engines') {
        addToStatusCarousel('engines', data, lastMessage);
        scrollToBottom();
    } else if (data.type === 'algorithms') {
        addToStatusCarousel('algorithms', data, lastMessage);
        scrollToBottom();
    } else if (data.type === 'agent_start') {
        // Phase 4: Agent Workflow Visualization
        console.log('🚀 Agent started:', data.agent, `(${data.index + 1}/${data.total})`);
        updateWorkflowPill(data.agent, 'active', data.index, data.total);
        showWorkflowPill();
    } else if (data.type === 'agent_complete') {
        // Phase 4: Agent completed
        console.log('✅ Agent completed:', data.agent, `(${data.duration_ms}ms)`);
        updateWorkflowPill(data.agent, 'completed');
    } else if (data.type === 'metacog_result') {
        // Phase 4: Metacognition results
        console.log('🧠 Metacog result:', data);
        updateMetacogIndicators({
            alignment: data.alignment_score,
            contradictions: data.contradiction_count,
            complexity: data.complexity,
            quarantined: data.quarantined || 0
        });
    } else if (data.type === 'workflow_complete') {
        // Phase 4: Entire workflow finished
        console.log('🏁 Workflow complete:', data.mode, `(${data.total_duration_ms}ms)`);
        setTimeout(() => hideWorkflowPill(), 2000);
    } else if (data.type === 'done') {
        console.log('✅ Done message received:', data);
        console.log('✅ Done message mode:', data.mode);
        console.log('✅ Done message results:', data.results);

        // Remove glitch animation when response is complete
        if (lastMessage) {
            const thoughtsContainer = lastMessage.querySelector('.thoughts-stream-container');
            if (thoughtsContainer) {
                const currentDisplay = thoughtsContainer.querySelector('.thoughts-stream-current');
                if (currentDisplay) {
                    currentDisplay.classList.remove('glitch-active');
                }
            }

            // Hide the mode loading state when response is complete
            hideModeLoadingState(lastMessage);

            const statusCarousel = lastMessage.querySelector('.status-carousel');
            if (statusCarousel) {
                const thinkingItems = statusCarousel.querySelectorAll('.status-item.thinking');
                thinkingItems.forEach(item => {
                    // Fully fade out and mark as complete
                    item.style.transition = 'opacity 0.5s ease-out';
                    item.style.opacity = '0';
                    item.classList.add('complete');
                    // Hide after transition
                    setTimeout(() => {
                        item.style.display = 'none';
                    }, 500);
                });
            }

            // Force final render of accumulated text when stream completes
            const visibleText = lastMessage.querySelector('.visible-text-content');
            if (visibleText && visibleText.dataset.accumulatedText) {
                // Clear any pending debounced render
                if (visibleText._renderTimeout) {
                    clearTimeout(visibleText._renderTimeout);
                }

                // Force immediate final render
                const finalText = visibleText.dataset.accumulatedText || '';
                if (finalText.trim()) {
                    let cleanedText = formatLLMResponse(finalText);
                    let html = marked.parse(cleanedText);
                    html = renderMath(html);
                    visibleText.innerHTML = html;

                    // Update conversation context with assistant response
                    const userQuery = conversationContext.lastUserMessage;
                    if (userQuery) {
                        updateConversationContext(null, finalText);
                    }

                    // Re-highlight code blocks
                    requestAnimationFrame(() => {
                        visibleText.querySelectorAll('pre code').forEach((block) => {
                            try {
                                hljs.highlightElement(block);
                            } catch (e) {
                                // Ignore highlighting errors
                            }
                        });
                        renderCharts(visibleText);
                    });

                    // If Fast/Chat response offered multi-agent research, show option to proceed
                    const doneMode = data.mode || (data.metadata && data.metadata.mode);
                    const isChatOrFast = (doneMode === 'fast' || doneMode === 'chat');
                    if (isChatOrFast && finalText && textSuggestsResearchOffer(finalText)) {
                        const msgContent = lastMessage.querySelector('.message-content');
                        if (msgContent && !lastMessage.querySelector('.run-full-research-cta')) {
                            const cta = document.createElement('div');
                            cta.className = 'run-full-research-cta';
                            cta.style.cssText = 'margin-top: 12px; padding: 10px 14px; background: rgba(0,0,0,0.04); border-radius: 8px; border: 1px solid rgba(0,0,0,0.08);';
                            const btn = document.createElement('button');
                            btn.type = 'button';
                            btn.textContent = 'Run full research (Surveyor → Dissident → Synthesist → Oracle)';
                            btn.style.cssText = 'padding: 8px 14px; border-radius: 6px; border: 1px solid rgba(0,0,0,0.12); background: #fff; cursor: pointer; font-size: 13px;';
                            btn.addEventListener('click', () => {
                                const modeSelectEl = document.getElementById('modeSelect');
                                const inputEl = document.getElementById('queryInput');
                                if (modeSelectEl) modeSelectEl.value = 'research';
                                const query = conversationContext.lastUserMessage || (() => {
                                    const m = document.querySelector('#chatContainer .message.user:last-child .message-content');
                                    return m ? m.textContent.trim() : '';
                                })();
                                if (query && inputEl) {
                                    inputEl.value = query;
                                    if (typeof updateAgentOptionsForMode === 'function') updateAgentOptionsForMode('research');
                                    sendQuery();
                                }
                            });
                            cta.appendChild(btn);
                            msgContent.appendChild(cta);
                        }
                    }
                }
            }

            // Handle astro-physiology results if present
            if (data.mode === 'astrophysiology') {
                console.log('Astro-physiology done message received:', data);
                console.log('Results data:', data.results);
                console.log('Error data:', data.error);

                // Remove loading state
                const loadingState = lastMessage.querySelector('.astro-loading-state');
                if (loadingState) {
                    console.log('Removing loading state');
                    loadingState.remove();
                }

                // Check for errors first
                if (data.error || (data.results && data.results.error)) {
                    const errorMsg = data.results?.content || data.results?.message || 'An error occurred during calculation';
                    console.error('Astro-physiology error:', errorMsg);

                    // Show error message
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'error-message';
                    errorDiv.style.cssText = 'color: #ff6b6b; padding: 15px; background: rgba(255, 107, 107, 0.1); border-radius: 6px; margin: 15px 0;';
                    errorDiv.textContent = `Error: ${errorMsg}`;
                    lastMessage.querySelector('.message-content')?.appendChild(errorDiv);

                    // If missing birth data, show form
                    if (data.error === 'missing_birth_data' || data.results?.error === 'missing_birth_data') {
                        astro_showBirthDataForm(lastMessage);
                    }
                } else if (data.results) {
                    // Display astro-physiology results
                    console.log('Creating astro-physiology card...');
                    console.log('Results keys:', Object.keys(data.results));
                    console.log('Has expert_consultations:', !!data.results.expert_consultations);
                    console.log('Has interventions:', !!data.results.interventions);
                    console.log('Expert consultations:', data.results.expert_consultations);
                    console.log('Interventions:', data.results.interventions);

                    // Store algorithmic_data for follow-up questions (with timestamp)
                    if (data.results.algorithmic_data) {
                        const algoDataWithTimestamp = {
                            ...data.results.algorithmic_data,
                            timestamp: Date.now(),
                            birth_datetime: data.results.algorithmic_data?.molecular_imprint?.birth_datetime || null
                        };
                        localStorage.setItem('iceburg_astro_algorithmic_data', JSON.stringify(algoDataWithTimestamp));
                        console.log('Stored algorithmic_data for follow-up questions');
                    }

                    astro_createAstroPhysiologyCard(data.results, lastMessage);

                    // Store expert registry in message element for access by expert display functions
                    if (data.results.expert_registry) {
                        lastMessage.dataset.expertRegistry = JSON.stringify(data.results.expert_registry);
                    }

                    // Display expert suggestions if available
                    if (data.results.available_experts && data.results.expert_switching_enabled) {
                        astro_showExpertSuggestions(data.results.available_experts, lastMessage);
                    }

                    // Display switch back button if in expert mode
                    if (data.results.current_expert && data.results.can_switch_back) {
                        astro_addSwitchBackButton(lastMessage, data.results.current_expert);
                    }

                    // Display expert consultations (Phase 6)
                    if (data.results.expert_consultations) {
                        console.log('Displaying expert consultations...');
                        astro_displayExpertConsultations(data.results.expert_consultations, lastMessage);
                    } else {
                        console.warn('No expert_consultations in results');
                    }

                    // Display interventions (Phase 7)
                    if (data.results.interventions) {
                        console.log('Displaying interventions...');
                        astro_displayInterventions(data.results.interventions, lastMessage);
                    } else {
                        console.warn('No interventions in results');
                    }

                    // V2: Display testable hypotheses (Phase 8: Research Tool Mode)
                    if (data.results.hypotheses && data.results.hypotheses.length > 0) {
                        console.log('Displaying testable hypotheses...');
                        astro_displayHypotheses(data.results.hypotheses, lastMessage);
                    }

                    // Legacy display removed - astro_createAstroPhysiologyCard handles all display
                    // No need for duplicate displayMolecularBlueprint, displayPatternTruth, or addTruthInsight calls
                }
            }
        }

        // Display portal metadata if available (always-on architecture)
        if (data.metadata) {
            const metadata = data.metadata;
            if (metadata.source || metadata.layer || metadata.response_time) {
                addPortalMetadata(metadata, lastMessage);
            }
        }

        // Display monitoring status if available (self-healing systems)
        if (data.monitoring) {
            addMonitoringStatus(data.monitoring, lastMessage);
        }

        // Display truth-finding data for astro-physiology mode
        // NOTE: This is handled in the "done" message handler to prevent duplicates
        // Only process here if it's a streaming update, not final results
        if (data.results && data.mode === 'astrophysiology' && data.type !== 'done') {
            // This is a streaming update - don't create full card here
            // Full card creation happens in "done" handler only
            console.log('Streaming update for astro-physiology (skipping full card creation)');
        }

        // Mark all action items as complete
        if (lastMessage) {
            const actionItems = lastMessage.querySelectorAll('.action-item.processing');
            actionItems.forEach(item => {
                item.classList.remove('processing');
                item.classList.add('complete');
                const statusEl = item.querySelector('.action-status');
                if (statusEl) statusEl.textContent = 'complete';
            });

            // Remove duplicate content - check all possible duplicate sources
            const reflexPreview = lastMessage.querySelector('.reflex-preview');
            const visibleText = lastMessage.querySelector('.visible-text-content');
            const streamingText = lastMessage.querySelector('.streaming-text-container');
            const statusCarousel = lastMessage.querySelector('.status-carousel');

            // Check if status carousel content matches main content and prevent duplication
            // Only truncate if there's a very close match (not just common words)
            if (statusCarousel && visibleText) {
                const statusText = statusCarousel.textContent || '';
                const mainText = visibleText.textContent || '';
                // Only check if both have substantial content
                if (mainText.trim().length > 50 && statusText.trim().length > 100) {
                    // Check if status carousel contains a substantial portion of main content
                    const mainTextStart = mainText.trim().substring(0, 200);
                    if (mainTextStart.length > 50 && statusText.includes(mainTextStart)) {
                        // Calculate similarity to avoid false positives
                        const similarity = calculateSimilarity(statusText, mainTextStart);
                        if (similarity > 0.7) {
                            // Status carousel is showing duplicate content - truncate status items
                            const statusItems = statusCarousel.querySelectorAll('.status-item');
                            statusItems.forEach(item => {
                                const itemContent = item.querySelector('.status-item-content');
                                if (itemContent) {
                                    const itemText = itemContent.textContent.trim();
                                    // Only truncate if item text is very similar to main content
                                    if (itemText.length > 50 && calculateSimilarity(itemText, mainTextStart) > 0.7) {
                                        itemContent.textContent = itemText.substring(0, 50) + '...';
                                    }
                                }
                            });
                        }
                    }
                }
            }

            // Remove duplicate from visible text if reflex preview exists
            if (reflexPreview && visibleText) {
                const previewText = reflexPreview.textContent || '';
                const mainText = visibleText.textContent || '';
                if (mainText.trim().startsWith(previewText.trim())) {
                    const cleanedText = mainText.substring(previewText.length).trim();
                    if (cleanedText) {
                        let html = marked.parse(cleanedText);
                        html = renderMath(html);
                        visibleText.innerHTML = html;
                    } else {
                        visibleText.style.display = 'none';
                    }
                }
            }

            // Remove duplicate streaming text if visible text exists
            if (visibleText && streamingText) {
                const visibleTextContent = visibleText.textContent || '';
                const streamingTextContent = streamingText.textContent || '';
                if (visibleTextContent.trim() === streamingTextContent.trim()) {
                    streamingText.style.display = 'none';
                }
            }

            // Convert streaming text to visible text on done
            if (streamingText && !visibleText) {
                const accumulatedText = streamingText.textContent || '';
                if (accumulatedText.trim()) {
                    let html = marked.parse(accumulatedText);
                    html = renderMath(html);
                    streamingText.innerHTML = html;
                    streamingText.className = 'visible-text-content';
                }
            }

            markMessageComplete(lastMessage);
            addThoughtsTrigger(lastMessage);

            // Add export options for substantial responses
            const messageContent = lastMessage.querySelector('.message-content');
            if (messageContent && messageContent.textContent.trim().length > 50) {
                addExportOptions(lastMessage, data);
            }
        }

        // Re-enable input
        reenableInput();
        scrollToBottom();
    } else if (data.type === 'error') {
        // Ignore validation errors
        if (data.message && data.message.includes("Query must be a non-empty string")) {
            return;
        }

        // Handle user-friendly error format with recovery suggestions
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.cssText = 'color: #ff4444; padding: 1rem; background: rgba(255, 68, 68, 0.1); border-left: 3px solid #ff4444; margin-top: 0.5rem; border-radius: 6px;';

        // Main error message
        const errorTitle = document.createElement('div');
        errorTitle.style.cssText = 'font-weight: bold; margin-bottom: 0.5rem; font-size: 1rem;';
        errorTitle.textContent = data.message || 'An error occurred';
        errorDiv.appendChild(errorTitle);

        // Recovery suggestions if available
        if (data.recovery_suggestions && Array.isArray(data.recovery_suggestions) && data.recovery_suggestions.length > 0) {
            const suggestionsTitle = document.createElement('div');
            suggestionsTitle.style.cssText = 'margin-top: 0.75rem; margin-bottom: 0.5rem; font-size: 0.9rem; color: #ffaa44; font-weight: 600;';
            suggestionsTitle.textContent = 'What you can try:';
            errorDiv.appendChild(suggestionsTitle);

            const suggestionsList = document.createElement('ul');
            suggestionsList.style.cssText = 'margin: 0; padding-left: 1.5rem; list-style-type: disc;';
            suggestionsList.style.color = '#ffaa44';

            data.recovery_suggestions.forEach(suggestion => {
                const li = document.createElement('li');
                li.style.cssText = 'margin: 0.25rem 0; font-size: 0.9rem;';
                li.textContent = suggestion;
                suggestionsList.appendChild(li);
            });

            errorDiv.appendChild(suggestionsList);
        }

        // Technical details (collapsible, for debugging)
        if (data.technical_details && DEBUG_MODE) {
            const detailsToggle = document.createElement('button');
            detailsToggle.textContent = 'Show technical details';
            detailsToggle.style.cssText = 'margin-top: 0.5rem; padding: 0.25rem 0.5rem; background: rgba(255, 68, 68, 0.2); border: 1px solid #ff4444; border-radius: 4px; color: #ff4444; cursor: pointer; font-size: 0.85rem;';
            detailsToggle.onclick = () => {
                const details = errorDiv.querySelector('.technical-details');
                if (details) {
                    details.style.display = details.style.display === 'none' ? 'block' : 'none';
                    detailsToggle.textContent = details.style.display === 'none' ? 'Show technical details' : 'Hide technical details';
                }
            };
            errorDiv.appendChild(detailsToggle);

            const details = document.createElement('div');
            details.className = 'technical-details';
            details.style.cssText = 'display: none; margin-top: 0.5rem; padding: 0.5rem; background: rgba(0, 0, 0, 0.3); border-radius: 4px; font-family: monospace; font-size: 0.8rem; color: #aaa;';
            details.textContent = data.technical_details;
            errorDiv.appendChild(details);
        }

        const messageContent = lastMessage.querySelector('.message-content');
        if (messageContent) messageContent.appendChild(errorDiv);

        reenableInput();
        showToast(data.message || 'An error occurred', 'error');
        scrollToBottom();
    }
}

// Helper: Create word breakdown display
function createWordBreakdownDisplay(messageElement, originalQuery) {
    const wordBreakdownDisplay = document.createElement('div');
    wordBreakdownDisplay.className = 'word-breakdown-display';
    wordBreakdownDisplay.style.setProperty('display', 'block', 'important');
    wordBreakdownDisplay.style.setProperty('visibility', 'visible', 'important');
    wordBreakdownDisplay.style.setProperty('opacity', '1', 'important');
    wordBreakdownDisplay.innerHTML = '<div class="word-breakdown-header"><strong>Analyzing Prompt:</strong><span class="word-breakdown-collapse">▼</span></div><div class="word-breakdown-content"></div><div class="word-linguistic-origins"></div>';
    wordBreakdownDisplay.setAttribute('data-original-query', originalQuery);

    const messageContent = messageElement.querySelector('.message-content');
    if (messageContent) {
        messageContent.insertBefore(wordBreakdownDisplay, messageContent.firstChild);
    } else {
        messageElement.appendChild(wordBreakdownDisplay);
    }

    // Add click handler
    const header = wordBreakdownDisplay.querySelector('.word-breakdown-header');
    header.addEventListener('click', () => toggleThoughtsSlideUp(messageElement));

    return wordBreakdownDisplay;
}

// Helper: Ensure word breakdown is visible
function ensureWordBreakdownVisible(wordBreakdownDisplay) {
    if (!wordBreakdownDisplay) return;

    wordBreakdownDisplay.classList.remove('collapsed');
    wordBreakdownDisplay.style.setProperty('display', 'block', 'important');
    wordBreakdownDisplay.style.setProperty('visibility', 'visible', 'important');
    wordBreakdownDisplay.style.setProperty('opacity', '1', 'important');

    const contentDiv = wordBreakdownDisplay.querySelector('.word-breakdown-content');
    if (contentDiv) {
        contentDiv.style.setProperty('display', 'block', 'important');
        contentDiv.style.setProperty('visibility', 'visible', 'important');
        contentDiv.style.setProperty('opacity', '1', 'important');
    }
}

// Helper: Re-enable input and button
function reenableInput() {
    const input = document.getElementById('queryInput');
    const sendButton = document.getElementById('sendButton');
    if (input) input.disabled = false;
    if (sendButton) sendButton.disabled = false;
    if (input) input.focus();
}

// Helper: Add Reflex Agent 3-bullet preview
function addReflexPreview(bullets, messageElement, compressionRatio) {
    if (!messageElement || !bullets) return;

    // Check if preview already exists
    let previewDiv = messageElement.querySelector('.reflex-preview');
    if (!previewDiv) {
        previewDiv = document.createElement('div');
        previewDiv.className = 'reflex-preview';
        previewDiv.style.cssText = `
            padding: 0.25rem 0.5rem;
            margin: 0.125rem 0;
            background: transparent;
            border: none;
            border-radius: 0;
            font-size: 0.6875rem;
            line-height: 1.2;
        `;

        // Insert before message content (FIRST - before chunks)
        const messageContent = messageElement.querySelector('.message-content');
        if (messageContent) {
            messageContent.insertBefore(previewDiv, messageContent.firstChild);
        } else {
            messageElement.appendChild(previewDiv);
        }
    }

    // Remove markdown bold formatting and clean text
    const cleanText = (text) => {
        if (!text) return '';
        return text.replace(/\*\*/g, '').replace(/\*/g, '').trim();
    };

    // Build preview HTML - compact, no spacing, small font, inline
    const previewHTML = `
        <div class="reflex-preview-header" style="display: inline; font-size: 0.625rem; color: var(--text-tertiary); margin: 0; padding: 0; line-height: 1.2;">
            <span>Quick Preview</span>
            ${compressionRatio ? `<span style="opacity: 0.6;">(${Math.round((1 - compressionRatio) * 100)}% compressed)</span>` : ''}
        </div>
        <div class="reflex-preview-content" style="display: inline; font-size: 0.6875rem; line-height: 1.2; color: var(--text-primary); margin: 0; padding: 0;">
            ${bullets.core_insight ? `<span style="display: inline;">${cleanText(bullets.core_insight)}</span>` : ''}
        </div>
    `;

    previewDiv.innerHTML = previewHTML;

    // Handle actionable guidance and key context as modular items (separate from preview)
    if (bullets.actionable_guidance || bullets.key_context) {
        let modularContainer = messageElement.querySelector('.actionable-modular');
        if (!modularContainer) {
            modularContainer = document.createElement('div');
            modularContainer.className = 'actionable-modular';
            modularContainer.style.cssText = `
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin: 0.25rem 0;
                padding: 0;
                font-size: 0.6875rem;
                line-height: 1.2;
            `;

            const messageContent = messageElement.querySelector('.message-content');
            if (messageContent) {
                messageContent.appendChild(modularContainer);
            } else {
                messageElement.appendChild(modularContainer);
            }
        }

        // Add actionable guidance as modular item
        if (bullets.actionable_guidance) {
            const item = document.createElement('div');
            item.className = 'actionable-modular-item';
            item.style.cssText = `
                display: inline-block;
                padding: 0.25rem 0.5rem;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                font-size: 0.6875rem;
                line-height: 1.2;
                color: var(--text-secondary);
                max-width: 300px;
                word-wrap: break-word;
            `;
            item.textContent = cleanText(bullets.actionable_guidance);
            modularContainer.appendChild(item);
        }

        // Add key context as modular item
        if (bullets.key_context) {
            const item = document.createElement('div');
            item.className = 'actionable-modular-item';
            item.style.cssText = `
                display: inline-block;
                padding: 0.25rem 0.5rem;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                font-size: 0.6875rem;
                line-height: 1.2;
                color: var(--text-secondary);
                max-width: 300px;
                word-wrap: break-word;
            `;
            item.textContent = cleanText(bullets.key_context);
            modularContainer.appendChild(item);
        }
    }
}

// Add message to chat
function addMessage(content, type = 'assistant') {
    const chatContainer = document.getElementById('chatContainer');
    const welcomeMessage = chatContainer.querySelector('.welcome-message');

    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.id = `message-${messageId++}`;

    const header = document.createElement('div');
    header.className = 'message-header';
    header.innerHTML = `
        <span>${type === 'user' ? 'You' : 'ICEBURG'}</span>
        <span>${new Date().toLocaleTimeString()}</span>
    `;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Render markdown for assistant messages, plain text for user
    if (type === 'assistant' && content) {
        let html = marked.parse(content);
        html = renderMath(html);
        contentDiv.innerHTML = html;
        // Highlight code blocks after rendering
        contentDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
        // Render charts after a short delay to ensure DOM is ready
        setTimeout(() => {
            renderCharts(contentDiv);
        }, 100);
    } else {
        contentDiv.textContent = content;
    }

    // Add message actions for assistant messages
    let actionsDiv = null;
    if (type === 'assistant') {
        actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        actionsDiv.innerHTML = `
            <button class="action-btn copy-btn" aria-label="Copy message" title="Copy">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M5.5 4.5V1.5C5.5 1.22386 5.72386 1 6 1H10.5C10.7761 1 11 1.22386 11 1.5V4.5M5.5 4.5H3.5C3.22386 4.5 3 4.72386 3 5V13.5C3 13.7761 3.22386 14 3.5 14H9.5C9.77614 14 10 13.7761 10 13.5V11.5M5.5 4.5C5.5 4.22386 5.72386 4 6 4H10.5C10.7761 4 11 4.22386 11 4.5V11.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                </svg>
            </button>
            <button class="action-btn regenerate-btn" aria-label="Regenerate response" title="Regenerate">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M2 8C2 5.23858 4.23858 3 7 3C8.65685 3 10.1569 3.84315 11 5.2M14 8C14 10.7614 11.7614 13 9 13C7.34315 13 5.84315 12.1569 5 10.8M5 2L3 5L5 8M11 14L13 11L11 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </button>
        `;
        // Don't append actionsDiv yet - will append after contentDiv

        // Copy button handler
        const copyBtn = actionsDiv.querySelector('.copy-btn');
        copyBtn.addEventListener('click', () => {
            const text = contentDiv.innerText || contentDiv.textContent;
            navigator.clipboard.writeText(text).then(() => {
                copyBtn.classList.add('copied');
                setTimeout(() => copyBtn.classList.remove('copied'), 2000);
            });
        });

        // Regenerate button handler
        const regenerateBtn = actionsDiv.querySelector('.regenerate-btn');
        regenerateBtn.addEventListener('click', () => {
            const lastUserMessage = chatContainer.querySelector('.message.user:last-child');
            if (lastUserMessage) {
                const userQuery = lastUserMessage.querySelector('.message-content').textContent;
                messageDiv.remove();
                sendQuery(userQuery);
            }
        });
    }

    messageDiv.appendChild(header);
    messageDiv.appendChild(contentDiv);
    // Append actions AFTER content so buttons appear below the message
    if (actionsDiv) {
        messageDiv.appendChild(actionsDiv);
    }
    chatContainer.appendChild(messageDiv);

    // Scroll to bottom
    setTimeout(() => scrollToBottom(), 100);

    return messageDiv;
}

// Note: Thinking items are now handled by status carousel via addToStatusCarousel('thinking', ...)
// These functions are kept for backward compatibility but are not actively used

// Unified status carousel - cycles through all status items
function addToStatusCarousel(type, data, messageElement) {
    console.log(`🔵 addToStatusCarousel called: type=${type}, messageElement=`, messageElement);

    if (!messageElement) {
        console.error('❌ No messageElement provided to addToStatusCarousel!');
        return;
    }

    // Get or create unified status carousel
    let statusCarousel = messageElement.querySelector('.status-carousel');
    if (!statusCarousel) {
        console.log('🔵 Creating new status carousel...');
        statusCarousel = document.createElement('div');
        statusCarousel.className = 'status-carousel';
        // Force visibility with !important to override any CSS
        statusCarousel.style.setProperty('display', 'block', 'important');
        statusCarousel.style.setProperty('visibility', 'visible', 'important');
        statusCarousel.style.setProperty('opacity', '1', 'important');
        statusCarousel.innerHTML = '<div class="status-carousel-header"><span class="status-title">Status</span><span class="status-indicator">●</span><span class="status-collapse">▼</span></div><div class="status-carousel-content"></div><div class="status-carousel-expanded" style="display: none;"></div>';

        // Insert BEFORE word breakdown display (FIRST) - prompt interpreter appears first
        const wordBreakdown = messageElement.querySelector('.word-breakdown-display');
        const messageContent = messageElement.querySelector('.message-content');

        if (messageContent) {
            if (wordBreakdown) {
                // Insert BEFORE word breakdown (status carousel first)
                messageContent.insertBefore(statusCarousel, wordBreakdown);
            } else {
                messageContent.insertBefore(statusCarousel, messageContent.firstChild);
            }
        } else {
            messageElement.appendChild(statusCarousel);
        }
        console.log('✅ Status carousel created and inserted');

        // Add click handler to expand/collapse
        const header = statusCarousel.querySelector('.status-carousel-header');
        const collapse = statusCarousel.querySelector('.status-collapse');
        header.addEventListener('click', () => {
            const expanded = statusCarousel.querySelector('.status-carousel-expanded');
            const content = statusCarousel.querySelector('.status-carousel-content');
            const isExpanded = expanded.style.display !== 'none';

            if (isExpanded) {
                // Collapse - show cycling view
                expanded.style.display = 'none';
                content.style.display = 'block';
                collapse.textContent = '▼';
                statusCarousel.classList.remove('expanded');
            } else {
                // Expand - show all items
                content.style.display = 'none';
                expanded.style.display = 'block';
                expanded.style.visibility = 'visible';
                expanded.style.opacity = '1';
                collapse.textContent = '▲';
                statusCarousel.classList.add('expanded');

                // Ensure expanded view is visible and not cut off
                expanded.style.position = 'relative';
                expanded.style.zIndex = '10';
                expanded.style.maxHeight = '500px';
                expanded.style.overflowY = 'auto';
                expanded.style.overflowX = 'hidden';
            }
        });

        // Start auto-cycling with transitions
        startStatusCarousel(statusCarousel);
    }

    // Check for duplicates - don't add if same type and similar content
    const items = statusCarousel.dataset.items ? JSON.parse(statusCarousel.dataset.items) : [];
    const isDuplicate = items.some(existing => {
        if (existing.type !== type) return false;
        const existingContent = existing.data.content || existing.data.description || existing.data.action || '';
        const newContent = data.content || data.description || data.action || '';
        // If both are thinking or action with same content, it's a duplicate
        if (type === 'thinking' || type === 'action') {
            return existingContent.trim() === newContent.trim();
        }
        return false;
    });

    if (isDuplicate) {
        console.log(`⏭️ Skipping duplicate ${type} item`);
        return;
    }

    // Add item to carousel
    const item = createStatusItem(type, data);
    items.push({ type, data });
    statusCarousel.dataset.items = JSON.stringify(items);

    // Add to expanded view
    const expanded = statusCarousel.querySelector('.status-carousel-expanded');
    const expandedItem = createStatusItem(type, data);
    expanded.appendChild(expandedItem);

    // Update cycling with transition
    updateStatusCarousel(statusCarousel);
}

// Create status item element
function createStatusItem(type, data) {
    const item = document.createElement('div');
    item.className = `status-item status-item-${type}`;

    switch (type) {
        case 'action':
            // Support bullet points for actions - show up to 3 main points
            const actionContent = data.description || data.action || 'Processing...';
            let actionBullets = [];
            if (Array.isArray(actionContent)) {
                actionBullets = actionContent;
            } else if (typeof actionContent === 'string' && actionContent.includes('\n')) {
                actionBullets = actionContent.split('\n').filter(line => line.trim());
            } else {
                actionBullets = [actionContent];
            }

            // Limit to first 3 bullets for main points
            const mainPoints = actionBullets.slice(0, 3).map(b => b.trim()).filter(b => b);

            // Add loading class if status is processing
            const isProcessing = (data.status || 'processing') === 'processing';
            item.className += isProcessing ? ' loading' : '';

            // Action name
            const actionName = data.action === 'prompt_interpreter' ? 'Prompt Interpreter' :
                data.action ? data.action.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) : 'Processing';

            // Check for cache status
            const isCached = data.is_cached || data.cache_status?.cache_hit;
            const responseTime = data.response_time || data.cache_status?.response_time;
            const cacheIndicator = isCached ? ' ⚡' : '';

            // Build content with exactly 3 bullet points, inline with separators
            let contentHtml = '';
            if (mainPoints.length > 0) {
                const cacheNote = isCached ? ' (instant)' : '';
                // Ensure exactly 3 bullets, truncate if more
                const displayPoints = mainPoints.slice(0, 3);
                // Join with " • " separator for inline display (bullet is in separator, not in each item)
                contentHtml = `<div class="status-item-content">${displayPoints.map(p => p.substring(0, 40) + (p.length > 40 ? '...' : '')).join(' • ')}${cacheNote}</div>`;
            }

            item.innerHTML = `
                <div class="status-item-header">
                    <span class="status-item-icon">🔍${cacheIndicator}</span>
                    <span class="status-item-title">${actionName}</span>
                    <span class="status-item-status">${data.status || 'processing'}${isCached ? ' ⚡' : ''}</span>
                </div>
                ${contentHtml}
            `;
            break;
        case 'thinking':
            const thinkingContent = data.content || 'Thinking...';
            // Support both string and array formats
            let thinkingBullets = [];
            if (Array.isArray(thinkingContent)) {
                thinkingBullets = thinkingContent;
            } else if (typeof thinkingContent === 'string') {
                thinkingBullets = thinkingContent.split('\n').filter(line => line.trim());
            }

            // Limit to first 3 bullets for main points
            const mainThinkingPoints = thinkingBullets.slice(0, 3).map(b => b.trim()).filter(b => b);

            // Add thinking class for pulsing animations
            item.className += ' thinking';

            // Build content with exactly 3 bullet points, inline with separators
            const displayThinkingPoints = mainThinkingPoints.slice(0, 3);
            const thinkingContentHtml = displayThinkingPoints.length > 0
                ? `<div class="status-item-content">${displayThinkingPoints.map(p => p.substring(0, 40) + (p.length > 40 ? '...' : '')).join(' • ')}</div>`
                : '<div class="status-item-content">Thinking...</div>';

            item.innerHTML = `
                <div class="status-item-header">
                    <span class="status-item-icon">💭</span>
                    <span class="status-item-title">Thinking</span>
                </div>
                ${thinkingContentHtml}
            `;
            break;
        case 'engines':
            const engines = data.engines || data;
            const enginesList = Array.isArray(engines) ? engines : [engines];
            item.innerHTML = `
                <div class="status-item-header">
                    <span class="status-item-icon">⚙️</span>
                    <span class="status-item-title">Engines Active</span>
                </div>
                <div class="status-item-content">
                    ${enginesList.map(e => {
                const name = e.name || e.engine || e;
                const algo = e.algorithm || e.description || '';
                return `${name}${algo ? ': ' + algo : ''}`;
            }).join('<br>')}
                </div>
            `;
            break;
        case 'algorithms':
            const algorithms = data.algorithms || data;
            const algorithmsList = Array.isArray(algorithms) ? algorithms : [algorithms];
            item.innerHTML = `
                <div class="status-item-header">
                    <span class="status-item-icon">🔬</span>
                    <span class="status-item-title">Algorithms Used</span>
                </div>
                <div class="status-item-content">
                    ${algorithmsList.map(a => {
                const name = a.name || a.algorithm || a;
                const desc = a.description || '';
                return `${name}${desc ? ': ' + desc : ''}`;
            }).join('<br>')}
                </div>
            `;
            break;
        case 'informatics':
            const info = data.data || data;
            item.innerHTML = `
                <div class="status-item-header">
                    <span class="status-item-icon">📊</span>
                    <span class="status-item-title">Informatics</span>
                </div>
                <div class="status-item-content">
                    ${Object.entries(info).map(([key, val]) => `<strong>${key}:</strong> ${val}`).join('<br>')}
                </div>
            `;
            break;
    }

    return item;
}

// Update status carousel with morphing/fade transitions
let carouselIntervals = new WeakMap();
let carouselCurrentElements = new WeakMap();

function startStatusCarousel(carousel) {
    // Stop existing interval if any
    if (carouselIntervals.has(carousel)) {
        clearInterval(carouselIntervals.get(carousel));
    }

    const content = carousel.querySelector('.status-carousel-content');
    let currentIndex = 0;
    let currentElement = null;

    const morphToNext = () => {
        const items = carousel.dataset.items ? JSON.parse(carousel.dataset.items) : [];
        if (items.length === 0) return;

        // Get next item
        const nextItem = items[currentIndex];
        if (!nextItem) return;

        // Remove ALL existing items first (ensure only one is visible)
        const existingItems = content.querySelectorAll('.status-item');
        existingItems.forEach(item => {
            item.classList.remove('active');
            item.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            item.style.opacity = '0';
            item.style.transform = 'translateY(-4px)';

            // Remove after fade
            setTimeout(() => {
                if (item && item.parentNode) {
                    item.remove();
                }
            }, 300);
        });

        // Create new element
        const nextElement = createStatusItem(nextItem.type, nextItem.data);
        nextElement.classList.add('active');
        nextElement.style.opacity = '0';
        nextElement.style.transform = 'translateY(4px)';

        // Clear content and add new element (ensure only one is visible)
        // Don't clear immediately - wait for fade out
        setTimeout(() => {
            // Clear any remaining items
            const remainingItems = content.querySelectorAll('.status-item');
            remainingItems.forEach(item => item.remove());

            // Add new element
            content.appendChild(nextElement);

            // Fade in next
            setTimeout(() => {
                nextElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                nextElement.style.opacity = '1';
                nextElement.style.transform = 'translateY(0)';
            }, 10);
        }, 50);

        // Update current
        currentElement = nextElement;

        // Move to next item
        currentIndex = (currentIndex + 1) % items.length;
    };

    // Start morphing
    morphToNext(); // Show first item immediately
    // Cycle every 2 seconds for smooth transitions
    const interval = setInterval(morphToNext, 2000);
    carouselIntervals.set(carousel, interval);
    carouselCurrentElements.set(carousel, currentElement);

    // Update status indicator with loading/thinking classes
    const indicator = carousel.querySelector('.status-indicator');
    if (indicator) {
        const items = carousel.dataset.items ? JSON.parse(carousel.dataset.items) : [];
        const hasThinking = items.some(item => item.type === 'thinking');
        const hasLoading = items.some(item => item.type === 'action' && (item.data.status === 'processing' || item.data.status === 'starting'));

        if (hasThinking) {
            indicator.classList.add('thinking');
            indicator.classList.remove('loading');
        } else if (hasLoading) {
            indicator.classList.add('loading');
            indicator.classList.remove('thinking');
        } else {
            indicator.classList.remove('loading', 'thinking');
        }
    }
}

function updateStatusCarousel(carousel) {
    // Restart cycling with updated items
    startStatusCarousel(carousel);
}

// Add thoughts trigger button to message
function addThoughtsTrigger(messageElement) {
    // Check if trigger already exists
    if (messageElement.querySelector('.thoughts-trigger')) {
        return;
    }

    const trigger = document.createElement('button');
    trigger.className = 'thoughts-trigger';
    trigger.innerHTML = `
        <span>Thoughts</span>
        <svg class="thoughts-trigger-icon" width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    `;

    trigger.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        toggleThoughtsSlideUp(messageElement);
    });

    // Ensure trigger is clickable and not covered
    trigger.style.pointerEvents = 'auto';
    trigger.style.zIndex = '1000';

    // Position relative to message element
    const messageContent = messageElement.querySelector('.message-content');
    if (messageContent) {
        messageContent.style.position = 'relative';
        messageContent.appendChild(trigger);
    } else {
        messageElement.style.position = 'relative';
        messageElement.appendChild(trigger);
    }
}

// Toggle thoughts slide-up panel
function toggleThoughtsSlideUp(messageElement) {
    // Check if slide-up already exists
    let slideUp = document.getElementById('thoughts-slide-up');
    const trigger = messageElement ? messageElement.querySelector('.thoughts-trigger') : document.querySelector('.thoughts-trigger');

    if (!slideUp) {
        // Create slide-up panel
        slideUp = document.createElement('div');
        slideUp.id = 'thoughts-slide-up';
        slideUp.className = 'thoughts-slide-up';
        slideUp.innerHTML = `
            <div class="thoughts-slide-up-header">
                <span class="thoughts-slide-up-title">Thoughts & Analysis</span>
                <button class="thoughts-slide-up-close" aria-label="Close">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M12 4L4 12M4 4L12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </button>
            </div>
            <div class="thoughts-slide-up-content"></div>
        `;

        document.body.appendChild(slideUp);

        // Close button handler
        const closeBtn = slideUp.querySelector('.thoughts-slide-up-close');
        closeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            slideUp.classList.remove('open');
            if (trigger) trigger.classList.remove('active');
        });

        // Close on backdrop click
        slideUp.addEventListener('click', (e) => {
            if (e.target === slideUp) {
                slideUp.classList.remove('open');
                if (trigger) trigger.classList.remove('active');
            }
        });
    }

    // Toggle open/closed
    const isOpen = slideUp.classList.contains('open');
    if (isOpen) {
        slideUp.classList.remove('open');
        if (trigger) trigger.classList.remove('active');
        return; // Don't populate content when closing
    } else {
        slideUp.classList.add('open');
        if (trigger) trigger.classList.add('active');
    }

    // Populate content when opening
    const content = slideUp.querySelector('.thoughts-slide-up-content');
    content.innerHTML = ''; // Clear previous content

    // Collect all data from message
    const wordBreakdown = messageElement.querySelector('.word-breakdown-display');
    const statusCarousel = messageElement.querySelector('.status-carousel');
    const sources = messageElement.querySelector('.sources-display');

    // Add word breakdown section
    if (wordBreakdown) {
        const section = document.createElement('div');
        section.className = 'thoughts-slide-up-section';
        section.innerHTML = '<div class="thoughts-slide-up-section-title">Linguistics Breakdown</div>';

        const wordText = wordBreakdown.querySelector('.word-flowing-text');
        if (wordText) {
            const item = document.createElement('div');
            item.className = 'thoughts-slide-up-item';
            item.innerHTML = wordText.innerHTML;
            section.appendChild(item);
        }

        const origins = wordBreakdown.querySelector('.word-linguistic-origins');
        if (origins && origins.textContent.trim()) {
            const item = document.createElement('div');
            item.className = 'thoughts-slide-up-item';
            item.textContent = origins.textContent;
            section.appendChild(item);
        }

        if (wordText || (origins && origins.textContent.trim())) {
            content.appendChild(section);
        }
    }

    // Add status/thinking section
    if (statusCarousel) {
        const section = document.createElement('div');
        section.className = 'thoughts-slide-up-section';
        section.innerHTML = '<div class="thoughts-slide-up-section-title">Thinking & Actions</div>';

        const items = statusCarousel.dataset.items ? JSON.parse(statusCarousel.dataset.items) : [];
        if (items.length > 0) {
            items.forEach(item => {
                if (item.type === 'action') {
                    const content = item.data.description || item.data.action || 'Processing...';
                    // Handle array of bullets
                    if (Array.isArray(content)) {
                        content.forEach(bulletText => {
                            const bullet = document.createElement('div');
                            bullet.className = 'thoughts-slide-up-bullet';
                            bullet.textContent = bulletText;
                            section.appendChild(bullet);
                        });
                    } else {
                        const bullet = document.createElement('div');
                        bullet.className = 'thoughts-slide-up-bullet';
                        bullet.textContent = content;
                        section.appendChild(bullet);
                    }
                } else if (item.type === 'thinking') {
                    const content = item.data.content || 'Thinking...';
                    // Handle array of thinking points
                    if (Array.isArray(content)) {
                        content.forEach(thought => {
                            const bullet = document.createElement('div');
                            bullet.className = 'thoughts-slide-up-bullet';
                            bullet.textContent = thought;
                            section.appendChild(bullet);
                        });
                    } else {
                        const bullet = document.createElement('div');
                        bullet.className = 'thoughts-slide-up-bullet';
                        bullet.textContent = content;
                        section.appendChild(bullet);
                    }
                } else {
                    const bullet = document.createElement('div');
                    bullet.className = 'thoughts-slide-up-bullet';
                    bullet.textContent = `${item.type}: ${JSON.stringify(item.data)}`;
                    section.appendChild(bullet);
                }
            });

            content.appendChild(section);
        }
    }

    // Add sources section
    if (sources) {
        const section = document.createElement('div');
        section.className = 'thoughts-slide-up-section';
        section.innerHTML = '<div class="thoughts-slide-up-section-title">Cited Sources</div>';

        const sourceLinks = sources.querySelectorAll('a');
        sourceLinks.forEach(link => {
            const item = document.createElement('div');
            item.className = 'thoughts-slide-up-item';
            item.innerHTML = `<a href="${link.href}" target="_blank" style="color: var(--accent-primary); text-decoration: none;">${link.textContent}</a>`;
            section.appendChild(item);
        });

        if (sourceLinks.length > 0) {
            content.appendChild(section);
        }
    }
}

// Add export/generation options to final answer (small buttons at bottom)
function addExportOptions(messageElement, data) {
    // Check if export options already exist
    if (messageElement.querySelector('.export-buttons-inline')) {
        return;
    }

    // Create export buttons container at bottom of message (but not covering thoughts trigger)
    const exportButtons = document.createElement('div');
    exportButtons.className = 'export-buttons-inline';
    exportButtons.style.cssText = 'position: relative; z-index: 50;'; // Lower than thoughts trigger
    exportButtons.innerHTML = `
        <button class="export-btn-small pdf-btn" data-format="pdf" title="PDF">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                <path d="M3 2.5C3 2.22386 3.22386 2 3.5 2H8.5L12 5.5V13.5C12 13.7761 11.7761 14 11.5 14H3.5C3.22386 14 3 13.7761 3 13.5V2.5Z" stroke="currentColor" stroke-width="1.5"/>
                <path d="M8 2V5.5H11.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
        </button>
        <button class="export-btn-small chart-btn" data-format="chart" title="Chart">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                <path d="M2 12L5 9L8 11L13 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M13 6H10L8 8V11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </button>
        <button class="export-btn-small code-btn" data-format="code" title="Code">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                <path d="M6 4L2 8L6 12M10 4L14 8L10 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </button>
        <button class="export-btn-small visual-btn" data-format="visual" title="Visual">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                <rect x="2" y="4" width="12" height="8" rx="1" stroke="currentColor" stroke-width="1.5"/>
                <circle cx="6" cy="8" r="1.5" fill="currentColor"/>
                <path d="M10 6L12 8L10 10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </button>
    `;

    // Append to message content (at bottom, before thoughts trigger)
    const messageContent = messageElement.querySelector('.message-content');
    if (messageContent) {
        messageContent.appendChild(exportButtons);
    } else {
        messageElement.appendChild(exportButtons);
    }

    // Add event handlers
    exportButtons.querySelectorAll('.export-btn-small').forEach(btn => {
        btn.addEventListener('click', () => {
            const format = btn.dataset.format;
            generateExport(messageElement, format);
        });
    });
}

// Generate export based on format
async function generateExport(messageElement, format) {
    const messageContent = messageElement.querySelector('.message-content');
    const content = messageContent ? messageContent.textContent : '';

    showToast(`Generating ${format.toUpperCase()}...`, 'info');

    try {
        const response = await fetch('/api/export/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                format: format,
                content: content,
                message_id: messageElement.id
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to generate ${format}`);
        }

        const result = await response.json();

        if (result.download_url) {
            // Download the file
            const link = document.createElement('a');
            link.href = result.download_url;
            link.download = result.filename || `iceburg_export.${format === 'pdf' ? 'pdf' : format === 'chart' ? 'png' : format === 'code' ? 'py' : 'html'}`;
            link.click();
            showToast(`${format.toUpperCase()} generated successfully!`, 'success');
        } else if (result.content) {
            // Display inline (for charts, visuals)
            displayInlineExport(messageElement, format, result.content);
            showToast(`${format.toUpperCase()} generated successfully!`, 'success');
        }
    } catch (error) {
        console.error(`Error generating ${format}:`, error);
        showToast(`Failed to generate ${format}`, 'error');
    }
}

// Display inline export (for charts, visuals)
function displayInlineExport(messageElement, format, content) {
    const exportDisplay = document.createElement('div');
    exportDisplay.className = `export-display export-${format}`;

    if (format === 'chart') {
        // Render chart using Chart.js or similar
        exportDisplay.innerHTML = `<canvas id="chart-${Date.now()}"></canvas>`;
        messageElement.appendChild(exportDisplay);
        // Initialize chart with content data
        // This would use Chart.js or similar library
    } else if (format === 'visual') {
        // Render visual HTML
        exportDisplay.innerHTML = content;
        messageElement.appendChild(exportDisplay);
    } else if (format === 'code') {
        // Display code in code block
        exportDisplay.innerHTML = `<pre><code>${content}</code></pre>`;
        messageElement.appendChild(exportDisplay);
        // Highlight code
        exportDisplay.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    }
}

// Update informatics panel
function updateInformatics(data, messageElement) {
    let informaticsPanel = messageElement.querySelector('.informatics-panel');
    if (!informaticsPanel) {
        informaticsPanel = document.createElement('div');
        informaticsPanel.className = 'informatics-panel';
        informaticsPanel.innerHTML = '<strong>Informatics:</strong>';
        messageElement.appendChild(informaticsPanel);
    }

    for (const [key, value] of Object.entries(data)) {
        const item = document.createElement('div');
        item.className = 'informatics-item';

        let displayValue = value;
        if (key === 'confidence' && typeof value === 'number') {
            const percentage = Math.round(value * 100);
            displayValue = `${percentage}% <span class="confidence-bar"><span class="confidence-fill" style="width: ${percentage}%"></span></span>`;
        }

        item.innerHTML = `
            <span class="informatics-label">${key}:</span>
            <span class="informatics-value">${displayValue}</span>
        `;
        informaticsPanel.appendChild(item);
    }
}

// Add conclusion bullet
function addConclusionItem(conclusion, messageElement) {
    // Remove markdown formatting
    const cleanText = (text) => {
        if (!text) return '';
        return text.replace(/\*\*/g, '').replace(/\*/g, '').trim();
    };

    let conclusionContainer = messageElement.querySelector('.conclusion-modular');
    if (!conclusionContainer) {
        conclusionContainer = document.createElement('div');
        conclusionContainer.className = 'conclusion-modular';
        conclusionContainer.style.cssText = `
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.25rem 0;
            padding: 0;
            font-size: 0.6875rem;
            line-height: 1.2;
        `;

        const messageContent = messageElement.querySelector('.message-content');
        if (messageContent) {
            messageContent.appendChild(conclusionContainer);
        } else {
            messageElement.appendChild(conclusionContainer);
        }
    }

    // Create small modular item
    const conclusionItem = document.createElement('div');
    conclusionItem.className = 'conclusion-modular-item';
    conclusionItem.style.cssText = `
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        font-size: 0.6875rem;
        line-height: 1.2;
        color: var(--text-secondary);
        max-width: 300px;
        word-wrap: break-word;
    `;
    conclusionItem.textContent = cleanText(conclusion);
    conclusionContainer.appendChild(conclusionItem);
}

// Append to last message - simplified for streaming text
function appendToLastMessage(content, messageElement) {
    console.log('📤 appendToLastMessage called with:', content?.substring(0, 50), '...');
    const contentDiv = messageElement.querySelector('.message-content');
    if (!contentDiv) {
        console.error('❌ Message content div not found!', messageElement);
        return;
    }
    console.log('✅ Found contentDiv');

    // Only skip null/undefined, but preserve spaces (they're important for word separation!)
    if (content === null || content === undefined) {
        console.log('⏭️ Skipping null/undefined content');
        return;
    }

    // Allow empty strings and spaces - they're needed for proper text formatting
    // (spaces arrive as separate chunks and must be preserved)

    // Use only one container - visible-text-content (remove streaming-text-container to prevent duplicates)
    let visibleText = contentDiv.querySelector('.visible-text-content');
    if (!visibleText) {
        visibleText = document.createElement('div');
        visibleText.className = 'visible-text-content answer-container';
        // Initialize dataset for accumulated text
        visibleText.dataset.accumulatedText = '';
        // Insert AFTER thinking stream (answer appears below thinking)
        const thinkingStream = contentDiv.querySelector('.thoughts-stream-container');
        const blackboard = contentDiv.querySelector('.thinking-blackboard');
        const wordBreakdown = contentDiv.querySelector('.word-breakdown-display');
        const statusCarousel = contentDiv.querySelector('.status-carousel');
        const insertAfter = thinkingStream || blackboard || statusCarousel || wordBreakdown;

        if (insertAfter && insertAfter.nextSibling) {
            contentDiv.insertBefore(visibleText, insertAfter.nextSibling);
        } else if (insertAfter) {
            contentDiv.appendChild(visibleText);
        } else {
            contentDiv.insertBefore(visibleText, contentDiv.firstChild);
        }
    }

    // Ensure dataset is initialized even if element already exists
    if (!visibleText.dataset.accumulatedText) {
        visibleText.dataset.accumulatedText = visibleText.textContent || '';
    }

    // Append text to the container (accumulate as plain text, render as markdown)
    // Always read from dataset.accumulatedText as source of truth (not textContent which may be out of sync after markdown rendering)
    const currentText = visibleText.dataset.accumulatedText || visibleText.textContent || '';
    const accumulatedText = currentText + content;

    // Check if this content is already in status carousel (prevent duplication)
    // VERY conservative check - only block exact duplicates from thinking messages
    // This prevents thinking messages from appearing as both status and main content
    const statusCarousel = contentDiv.querySelector('.status-carousel');
    if (statusCarousel) {
        // Only check thinking items (not action, engines, etc.)
        const thinkingItems = statusCarousel.querySelectorAll('.status-item.thinking');
        if (thinkingItems.length > 0) {
            // Get all thinking content
            const thinkingTexts = Array.from(thinkingItems).map(item => {
                const text = item.textContent || '';
                return text.trim();
            }).filter(text => text.length > 0);

            // Only block if the chunk content exactly matches a thinking message
            // This prevents thinking messages from duplicating as main content
            const chunkText = content.trim();
            const isExactMatch = thinkingTexts.some(thinkingText => {
                // Check if chunk is a substring of thinking (thinking messages are usually longer)
                // OR if thinking is a substring of chunk (chunk might be the full response)
                // Only block if it's a very close match (>95% similarity) to avoid false positives
                if (chunkText.length > 20 && thinkingText.length > 20) {
                    const longer = chunkText.length > thinkingText.length ? chunkText : thinkingText;
                    const shorter = chunkText.length > thinkingText.length ? thinkingText : chunkText;
                    // Check if shorter is >95% contained in longer
                    if (longer.includes(shorter) && (shorter.length / longer.length) > 0.95) {
                        return true;
                    }
                }
                return false;
            });

            if (isExactMatch) {
                console.log('⏭️ Skipping duplicate content (matches thinking message)');
                return;
            }
        }
    }

    // Remove duplicate markdown bold formatting before rendering
    let cleanedText = accumulatedText.replace(/\*\*([^*]+)\*\*/g, '$1'); // Remove **bold** but keep text
    cleanedText = cleanedText.replace(/\*([^*]+)\*/g, '$1'); // Remove *italic* but keep text

    // Remove duplicate citations (e.g., "[LLM Knowledge]" appearing multiple times)
    // Pattern: [LLM Knowledge], [Training Knowledge], [ICEBURG Research], etc.
    const citationPatterns = [
        /\[LLM Knowledge\]/gi,
        /\[Training Knowledge\]/gi,
        /\[ICEBURG Research\]/gi,
        /\[External Source\]/gi
    ];

    for (const pattern of citationPatterns) {
        // Replace multiple occurrences with a single occurrence
        cleanedText = cleanedText.replace(new RegExp(`(${pattern.source})\\s*\\1+`, 'g'), '$1');
        // Also remove if it appears multiple times in the same sentence/paragraph
        cleanedText = cleanedText.replace(new RegExp(`(${pattern.source})\\s*[^\\]]*\\s*\\1`, 'g'), '$1');
    }

    // Remove duplicate phrases (e.g., "Training Knowledge [LLM Knowledge]" appearing twice)
    const duplicatePhrases = [
        /Training Knowledge\s*\[LLM Knowledge\]/gi,
        /\[LLM Knowledge\]\s*Training Knowledge/gi
    ];

    for (const pattern of duplicatePhrases) {
        // Keep only the first occurrence
        const matches = cleanedText.match(new RegExp(pattern.source, 'gi'));
        if (matches && matches.length > 1) {
            // Replace all but the first occurrence
            let first = true;
            cleanedText = cleanedText.replace(new RegExp(pattern.source, 'gi'), (match) => {
                if (first) {
                    first = false;
                    return match;
                }
                return ''; // Remove duplicate
            });
        }
    }

    // Store accumulated text in dataset for thread-safe access
    visibleText.dataset.accumulatedText = accumulatedText;
    console.log('💾 Stored accumulated text:', accumulatedText.length, 'chars, preview:', accumulatedText.substring(0, 100));

    // GPT-5-style instant rendering: Batch markdown processing for performance
    // FIX: Debounce rendering to prevent race conditions with rapid chunks
    // Clear any pending render timeout
    if (visibleText._renderTimeout) {
        clearTimeout(visibleText._renderTimeout);
    }

    // Schedule render after a short delay (debounce)
    visibleText._renderTimeout = setTimeout(() => {
        // Always read the latest text from data attribute (thread-safe)
        // This ensures we render the most up-to-date content even if more chunks arrived
        const textToRender = visibleText.dataset.accumulatedText || '';
        console.log('🎨 Rendering text:', textToRender.length, 'chars, preview:', textToRender.substring(0, 100));

        // Use requestAnimationFrame to avoid blocking the UI thread
        requestAnimationFrame(() => {
            // Always render the latest accumulated text (don't skip if more chunks arrived)
            // This ensures the UI stays up-to-date even during rapid streaming

            // Post-process markdown: break long paragraphs, ensure proper formatting
            let cleanedText = formatLLMResponse(textToRender);

            // Render as markdown for display (instant, non-blocking)
            let html = marked.parse(cleanedText);
            html = renderMath(html);
            visibleText.innerHTML = html;
            console.log('✅ Rendered HTML, length:', html.length);

            // Re-highlight code blocks (async, non-blocking)
            requestAnimationFrame(() => {
                visibleText.querySelectorAll('pre code').forEach((block) => {
                    try {
                        hljs.highlightElement(block);
                    } catch (e) {
                        // Ignore highlighting errors for speed
                    }
                });

                // Re-render charts (async)
                renderCharts(visibleText);
            });
        });
    }, renderDebounceDelay);
}

// Calculate similarity between two strings (simple Jaccard similarity)
function calculateSimilarity(str1, str2) {
    if (!str1 || !str2) return 0;
    const words1 = new Set(str1.toLowerCase().split(/\s+/));
    const words2 = new Set(str2.toLowerCase().split(/\s+/));
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);
    return intersection.size / union.size;
}

// Format LLM response: break long paragraphs, ensure proper structure
function formatLLMResponse(text) {
    if (!text || typeof text !== 'string') return text;

    // Break long paragraphs (over 500 chars) into shorter ones
    const lines = text.split('\n');
    const formattedLines = [];

    for (const line of lines) {
        if (line.trim().length === 0) {
            formattedLines.push(line);
            continue;
        }

        // If line is a heading, list item, or code block, keep as is
        if (line.match(/^#{1,6}\s/) || line.match(/^[-*+]\s/) || line.match(/^\d+\.\s/) || line.match(/^```/) || line.match(/^`/)) {
            formattedLines.push(line);
            continue;
        }

        // If line is very long (over 500 chars), break it into shorter paragraphs
        if (line.length > 500) {
            const words = line.split(' ');
            let currentParagraph = '';

            for (const word of words) {
                if ((currentParagraph + ' ' + word).length > 300) {
                    if (currentParagraph) {
                        formattedLines.push(currentParagraph.trim());
                        formattedLines.push(''); // Add blank line between paragraphs
                    }
                    currentParagraph = word;
                } else {
                    currentParagraph = currentParagraph ? currentParagraph + ' ' + word : word;
                }
            }

            if (currentParagraph) {
                formattedLines.push(currentParagraph.trim());
            }
        } else {
            formattedLines.push(line);
        }
    }

    return formattedLines.join('\n');
}

// Mark message as complete
function markMessageComplete(messageElement) {
    if (messageElement) {
        messageElement.classList.add('complete');
    }
}

// Add portal metadata display (always-on architecture) - More visible UX
function addMonitoringStatus(monitoring, messageElement) {
    if (!monitoring || !messageElement) return;

    // Only show if monitoring is enabled and has status
    if (!monitoring.enabled || !monitoring.status) return;

    let statusDisplay = messageElement.querySelector('.monitoring-status');
    if (!statusDisplay) {
        statusDisplay = document.createElement('div');
        statusDisplay.className = 'monitoring-status';
        statusDisplay.style.cssText = 'font-size: 0.75rem; color: #00ff88; margin-top: 0.5rem; padding: 0.5rem; background: rgba(0, 255, 136, 0.1); border: 1px solid rgba(0, 255, 136, 0.3); border-radius: 6px; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;';

        const messageContent = messageElement.querySelector('.message-content');
        if (messageContent) {
            messageContent.appendChild(statusDisplay);
        } else {
            messageElement.appendChild(statusDisplay);
        }
    }

    // Clear existing content
    statusDisplay.innerHTML = '';

    const status = monitoring.status;

    // Add header
    const header = document.createElement('span');
    header.style.cssText = 'font-weight: bold; color: #00ff88; margin-right: 0.5rem;';
    header.textContent = '🔧 Self-Healing:';
    statusDisplay.appendChild(header);

    // Add status badges
    const badges = [];

    if (status.monitoring_active) {
        badges.push(`Active`);
    }

    if (status.total_alerts > 0) {
        badges.push(`${status.total_alerts} alerts`);
    }

    if (status.resolved_alerts > 0) {
        badges.push(`${status.resolved_alerts} resolved`);
    }

    if (status.auto_healing_success_rate > 0) {
        badges.push(`${(status.auto_healing_success_rate * 100).toFixed(0)}% success`);
    }

    if (status.llm_analyses > 0) {
        badges.push(`${status.llm_analyses} LLM analyses`);
    }

    if (status.cached_analyses > 0) {
        badges.push(`${status.cached_analyses} cached`);
    }

    // Add badges
    badges.forEach(badge => {
        const badgeEl = document.createElement('span');
        badgeEl.style.cssText = 'padding: 0.25rem 0.5rem; background: rgba(0, 255, 136, 0.2); border-radius: 4px; font-size: 0.7rem;';
        badgeEl.textContent = badge;
        statusDisplay.appendChild(badgeEl);
    });
}

function addPortalMetadata(metadata, messageElement) {
    if (!metadata || !messageElement) return;

    let metadataDisplay = messageElement.querySelector('.portal-metadata');
    if (!metadataDisplay) {
        metadataDisplay = document.createElement('div');
        metadataDisplay.className = 'portal-metadata';
        // More visible styling
        metadataDisplay.style.cssText = 'font-size: 0.8rem; color: #00ffff; margin-top: 0.75rem; padding: 0.5rem; background: rgba(0, 255, 255, 0.1); border: 1px solid rgba(0, 255, 255, 0.3); border-radius: 6px; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;';

        const messageContent = messageElement.querySelector('.message-content');
        if (messageContent) {
            messageContent.appendChild(metadataDisplay);
        } else {
            messageElement.appendChild(metadataDisplay);
        }
    }

    // Clear existing content
    metadataDisplay.innerHTML = '';

    // Add header
    const header = document.createElement('span');
    header.style.cssText = 'font-weight: bold; color: #00ffff; margin-right: 0.5rem;';
    header.textContent = '⚡ Always-On AI:';
    metadataDisplay.appendChild(header);

    // Add metadata badges
    const badges = [];

    if (metadata.source) {
        const sourceBadge = document.createElement('span');
        sourceBadge.style.cssText = 'padding: 0.25rem 0.5rem; background: rgba(0, 255, 255, 0.2); border-radius: 4px; font-size: 0.75rem;';
        sourceBadge.textContent = `Source: ${metadata.source}`;
        badges.push(sourceBadge);
    }

    if (metadata.layer) {
        const layerBadge = document.createElement('span');
        layerBadge.style.cssText = 'padding: 0.25rem 0.5rem; background: rgba(0, 255, 0, 0.2); border-radius: 4px; font-size: 0.75rem;';
        layerBadge.textContent = `Layer: ${metadata.layer}`;
        badges.push(layerBadge);
    }

    if (metadata.response_time !== undefined) {
        const timeBadge = document.createElement('span');
        const timeColor = metadata.response_time < 0.1 ? '#00ff00' : metadata.response_time < 1 ? '#ffff00' : '#ff8800';
        timeBadge.style.cssText = `padding: 0.25rem 0.5rem; background: rgba(${timeColor === '#00ff00' ? '0, 255, 0' : timeColor === '#ffff00' ? '255, 255, 0' : '255, 136, 0'}, 0.2); border-radius: 4px; font-size: 0.75rem; color: ${timeColor}; font-weight: bold;`;
        timeBadge.textContent = `${metadata.response_time.toFixed(3)}s`;
        badges.push(timeBadge);
    }

    if (metadata.cached) {
        const cachedBadge = document.createElement('span');
        cachedBadge.style.cssText = 'padding: 0.25rem 0.5rem; background: rgba(255, 255, 0, 0.3); border-radius: 4px; font-size: 0.75rem; color: #ffff00; font-weight: bold;';
        cachedBadge.textContent = '⚡ Cached';
        badges.push(cachedBadge);
    }

    // Add badges to display
    badges.forEach(badge => metadataDisplay.appendChild(badge));

    if (badges.length > 0) {
        metadataDisplay.style.display = 'flex';
    } else {
        metadataDisplay.style.display = 'none';
    }
}

// Add sources display
// Display deep knowledge decoding results
function addTotalKnowledgeDisplay(totalKnowledge, messageElement) {
    if (!totalKnowledge || !messageElement) return;

    let knowledgeDiv = messageElement.querySelector('.total-knowledge-display');
    if (!knowledgeDiv) {
        knowledgeDiv = document.createElement('div');
        knowledgeDiv.className = 'total-knowledge-display';
        knowledgeDiv.style.cssText = 'margin-top: 1rem; padding: 1rem; background: rgba(0, 255, 255, 0.05); border-left: 3px solid #00ffff; border-radius: 4px;';

        const messageContent = messageElement.querySelector('.message-content');
        if (messageContent) {
            messageContent.appendChild(knowledgeDiv);
        } else {
            messageElement.appendChild(knowledgeDiv);
        }
    }

    // Compact collapsible display - documents pop out instead of cluttering
    const hasContent = (totalKnowledge.etymology_traces && totalKnowledge.etymology_traces.length > 0) ||
        (totalKnowledge.occult_connections && totalKnowledge.occult_connections.length > 0) ||
        (totalKnowledge.secret_society_connections && totalKnowledge.secret_society_connections.length > 0) ||
        (totalKnowledge.suppressed_knowledge && totalKnowledge.suppressed_knowledge.length > 0) ||
        (totalKnowledge.historical_patterns && totalKnowledge.historical_patterns.length > 0);

    if (!hasContent) return;

    let html = `<div style="display: flex; justify-content: space-between; align-items: center; cursor: pointer;" class="knowledge-header">
        <div style="color: #00ffff; font-weight: bold; font-size: 0.9em;">🔍 Deep Knowledge Decoding</div>
        <span class="knowledge-toggle" style="color: #00ffff; font-size: 0.8em;">▼</span>
    </div>
    <div class="knowledge-content" style="display: none; margin-top: 0.5rem; max-height: 300px; overflow-y: auto;">`;

    // Etymology traces - compact
    if (totalKnowledge.etymology_traces && totalKnowledge.etymology_traces.length > 0) {
        html += `<div style="margin: 0.25rem 0; font-size: 0.85em;"><strong style="color: #00ff00;">Etymology:</strong> `;
        html += totalKnowledge.etymology_traces.slice(0, 2).map(t => t.term).join(', ');
        html += ` <button class="view-doc-btn" data-doc='${JSON.stringify({ type: 'knowledge', section: 'etymology', data: totalKnowledge.etymology_traces })}' style="padding: 2px 6px; margin-left: 4px; background: rgba(0,255,255,0.2); border: 1px solid #00ffff; border-radius: 3px; color: #00ffff; cursor: pointer; font-size: 0.75em;">View</button></div>`;
    }

    // Occult connections - compact
    if (totalKnowledge.occult_connections && totalKnowledge.occult_connections.length > 0) {
        html += `<div style="margin: 0.25rem 0; font-size: 0.85em;"><strong style="color: #ff00ff;">Occult:</strong> `;
        html += totalKnowledge.occult_connections.slice(0, 2).map(c => c.term).join(', ');
        html += ` <button class="view-doc-btn" data-doc='${JSON.stringify({ type: 'knowledge', section: 'occult', data: totalKnowledge.occult_connections })}' style="padding: 2px 6px; margin-left: 4px; background: rgba(255,0,255,0.2); border: 1px solid #ff00ff; border-radius: 3px; color: #ff00ff; cursor: pointer; font-size: 0.75em;">View</button></div>`;
    }

    // Secret society connections - compact
    if (totalKnowledge.secret_society_connections && totalKnowledge.secret_society_connections.length > 0) {
        html += `<div style="margin: 0.25rem 0; font-size: 0.85em;"><strong style="color: #ffff00;">Societies:</strong> `;
        html += totalKnowledge.secret_society_connections.slice(0, 2).map(s => s.term).join(', ');
        html += ` <button class="view-doc-btn" data-doc='${JSON.stringify({ type: 'knowledge', section: 'societies', data: totalKnowledge.secret_society_connections })}' style="padding: 2px 6px; margin-left: 4px; background: rgba(255,255,0,0.2); border: 1px solid #ffff00; border-radius: 3px; color: #ffff00; cursor: pointer; font-size: 0.75em;">View</button></div>`;
    }

    // Suppressed knowledge - compact
    if (totalKnowledge.suppressed_knowledge && totalKnowledge.suppressed_knowledge.length > 0) {
        html += `<div style="margin: 0.25rem 0; font-size: 0.85em;"><strong style="color: #ff4444;">Suppressed:</strong> `;
        html += totalKnowledge.suppressed_knowledge.slice(0, 2).map(s => s.term).join(', ');
        html += ` <button class="view-doc-btn" data-doc='${JSON.stringify({ type: 'knowledge', section: 'suppressed', data: totalKnowledge.suppressed_knowledge })}' style="padding: 2px 6px; margin-left: 4px; background: rgba(255,68,68,0.2); border: 1px solid #ff4444; border-radius: 3px; color: #ff4444; cursor: pointer; font-size: 0.75em;">View</button></div>`;
    }

    // Historical patterns - compact
    if (totalKnowledge.historical_patterns && totalKnowledge.historical_patterns.length > 0) {
        html += `<div style="margin: 0.25rem 0; font-size: 0.85em;"><strong style="color: #00ff00;">Patterns:</strong> `;
        html += totalKnowledge.historical_patterns.slice(0, 2).map(p => p.term).join(', ');
        html += ` <button class="view-doc-btn" data-doc='${JSON.stringify({ type: 'knowledge', section: 'patterns', data: totalKnowledge.historical_patterns })}' style="padding: 2px 6px; margin-left: 4px; background: rgba(0,255,0,0.2); border: 1px solid #00ff00; border-radius: 3px; color: #00ff00; cursor: pointer; font-size: 0.75em;">View</button></div>`;
    }

    html += '</div>';

    knowledgeDiv.innerHTML = html;

    // Add toggle functionality
    const header = knowledgeDiv.querySelector('.knowledge-header');
    const content = knowledgeDiv.querySelector('.knowledge-content');
    const toggle = knowledgeDiv.querySelector('.knowledge-toggle');

    if (header && content && toggle) {
        header.addEventListener('click', () => {
            const isHidden = content.style.display === 'none';
            content.style.display = isHidden ? 'block' : 'none';
            toggle.textContent = isHidden ? '▲' : '▼';
        });
    }

    // Add document viewer handlers
    knowledgeDiv.querySelectorAll('.view-doc-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            try {
                const docData = JSON.parse(btn.getAttribute('data-doc'));
                // Format knowledge data for viewer
                const formattedDoc = {
                    title: `Deep Knowledge: ${docData.section}`,
                    content: JSON.stringify(docData.data, null, 2),
                    format: 'json'
                };
                openDocumentViewer(formattedDoc);
            } catch (err) {
                console.error('Error opening knowledge document:', err);
            }
        });
    });
}

function addSourcesDisplay(sources, messageElement) {
    let sourcesDisplay = messageElement.querySelector('.sources-display');
    if (!sourcesDisplay) {
        sourcesDisplay = document.createElement('div');
        sourcesDisplay.className = 'sources-display';
        sourcesDisplay.style.display = 'none'; /* Hide from main message, show in slide-up */
        sourcesDisplay.innerHTML = '<div class="sources-header"><strong>Websites Browsed:</strong><span class="sources-collapse">▼</span></div><div class="sources-content"></div>';
        messageElement.appendChild(sourcesDisplay);

        // Add click handler to collapse/expand
        const header = sourcesDisplay.querySelector('.sources-header');
        const content = sourcesDisplay.querySelector('.sources-content');
        const collapse = sourcesDisplay.querySelector('.sources-collapse');

        header.addEventListener('click', () => {
            const isCollapsed = sourcesDisplay.classList.contains('collapsed');
            if (isCollapsed) {
                content.style.display = 'block';
                collapse.textContent = '▼';
                sourcesDisplay.classList.remove('collapsed');
            } else {
                content.style.display = 'none';
                collapse.textContent = '▶';
                sourcesDisplay.classList.add('collapsed');
            }
        });
    }

    const content = sourcesDisplay.querySelector('.sources-content');

    if (Array.isArray(sources)) {
        sources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';

            const sourceLink = document.createElement('a');
            sourceLink.href = source.url || '#';
            sourceLink.target = '_blank';
            sourceLink.rel = 'noopener noreferrer';
            sourceLink.textContent = source.title || source.url || `Source ${index + 1}`;
            sourceLink.className = 'source-link';

            if (source.source_type) {
                const sourceType = document.createElement('span');
                sourceType.className = 'source-type';
                sourceType.textContent = ` (${source.source_type})`;
                sourceLink.appendChild(sourceType);
            }

            sourceItem.appendChild(sourceLink);
            content.appendChild(sourceItem);
        });
    }
}

// Add agent thinking item
function addAgentThinkingItem(agentName, thought, messageElement) {
    let agentThinkingDisplay = messageElement.querySelector('.agent-thinking-display');
    if (!agentThinkingDisplay) {
        agentThinkingDisplay = document.createElement('div');
        agentThinkingDisplay.className = 'agent-thinking-display';
        messageElement.appendChild(agentThinkingDisplay);
    }

    let agentSection = agentThinkingDisplay.querySelector(`[data-agent="${agentName}"]`);
    if (!agentSection) {
        agentSection = document.createElement('div');
        agentSection.className = 'agent-section';
        agentSection.setAttribute('data-agent', agentName);
        agentSection.innerHTML = `<strong>${agentName} thinking:</strong>`;
        agentThinkingDisplay.appendChild(agentSection);
    }

    const thinkingItem = document.createElement('div');
    thinkingItem.className = 'thinking-item agent-thinking-item';
    thinkingItem.textContent = thought;
    agentSection.appendChild(thinkingItem);

    // Scroll to show new thinking item
    const chatContainer = document.getElementById('chatContainer');
    setTimeout(() => {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    }, 50);
}

// Add action item with bullet animation (flipping through thoughts)
function addActionItemBullet(actionData, messageElement) {
    let actionTracking = messageElement.querySelector('.action-tracking');
    if (!actionTracking) {
        actionTracking = document.createElement('div');
        actionTracking.className = 'action-tracking';
        actionTracking.innerHTML = '<strong>Thoughts & Actions:</strong>';
        messageElement.appendChild(actionTracking);
    }

    const actionItem = document.createElement('div');
    actionItem.className = 'action-item bullet-item';
    if (actionData.status === 'processing' || actionData.status === 'starting') {
        actionItem.classList.add('processing');
    } else if (actionData.status === 'complete') {
        actionItem.classList.add('complete');
    } else if (actionData.status === 'error') {
        actionItem.classList.add('error');
    }
    actionItem.setAttribute('data-action-id', Date.now());

    // Create bullet point with summary
    const bulletContent = document.createElement('div');
    bulletContent.className = 'bullet-content';

    // Summarize action description
    let summary = actionData.description || actionData.action || 'Processing...';
    if (summary.length > 60) {
        summary = summary.substring(0, 60) + '...';
    }

    bulletContent.innerHTML = `
        <span class="bullet">•</span>
        <span class="bullet-text">${summary}</span>
        <span class="bullet-status">${actionData.status || 'processing'}</span>
    `;

    actionItem.appendChild(bulletContent);
    actionTracking.appendChild(actionItem);

    // Animate bullet appearance (flip through effect)
    setTimeout(() => {
        actionItem.style.opacity = '1';
        actionItem.style.transform = 'translateX(0)';
    }, 50);

    // Auto-remove after showing (cycling effect)
    if (actionData.status === 'complete') {
        setTimeout(() => {
            if (actionItem.parentNode) {
                actionItem.style.opacity = '0';
                actionItem.style.transform = 'translateX(-10px)';
                setTimeout(() => actionItem.remove(), 300);
            }
        }, 2000); // Show for 2 seconds then fade out
    }

    // Scroll to show new action item
    const chatContainer = document.getElementById('chatContainer');
    requestAnimationFrame(() => {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    });
}

// Add action tracking item (replaces loading circle)
function addActionItem(actionData, messageElement) {
    let actionTracking = messageElement.querySelector('.action-tracking');
    if (!actionTracking) {
        actionTracking = document.createElement('div');
        actionTracking.className = 'action-tracking';
        actionTracking.innerHTML = '<strong>Actions & Thoughts:</strong>';
        messageElement.appendChild(actionTracking);
    }

    const actionItem = document.createElement('div');
    actionItem.className = 'action-item';
    if (actionData.status === 'processing' || actionData.status === 'starting') {
        actionItem.classList.add('processing');
    } else if (actionData.status === 'complete') {
        actionItem.classList.add('complete');
    } else if (actionData.status === 'error') {
        actionItem.classList.add('error');
    }
    actionItem.setAttribute('data-action-id', Date.now());

    const actionIcon = document.createElement('span');
    actionIcon.className = 'action-icon';

    // Set icon based on action type
    if (actionData.action === 'prompt_interpreter') {
        actionIcon.textContent = '🔍';
    } else if (actionData.action === 'web_search') {
        actionIcon.textContent = '🌐';
    } else if (actionData.action === 'reading_document') {
        actionIcon.textContent = '📄';
    } else {
        actionIcon.textContent = '⚙️';
    }

    const actionContent = document.createElement('div');
    actionContent.className = 'action-content';

    const actionTitle = document.createElement('div');
    actionTitle.className = 'action-title';
    actionTitle.textContent = actionData.description || actionData.action || 'Processing...';

    const actionStatus = document.createElement('div');
    actionStatus.className = 'action-status';
    actionStatus.textContent = actionData.status || 'processing';

    actionContent.appendChild(actionTitle);
    actionContent.appendChild(actionStatus);

    // Add details if available
    if (actionData.intent || actionData.domain || actionData.complexity) {
        const actionDetails = document.createElement('div');
        actionDetails.className = 'action-details';
        actionDetails.style.display = 'none';

        if (actionData.intent) {
            const intentDiv = document.createElement('div');
            intentDiv.innerHTML = `<strong>Intent:</strong> ${actionData.intent} (${Math.round((actionData.confidence || 0.5) * 100)}%)`;
            actionDetails.appendChild(intentDiv);
        }
        if (actionData.domain) {
            const domainDiv = document.createElement('div');
            domainDiv.innerHTML = `<strong>Domain:</strong> ${actionData.domain}`;
            actionDetails.appendChild(domainDiv);
        }
        if (actionData.complexity !== undefined) {
            const complexityDiv = document.createElement('div');
            complexityDiv.innerHTML = `<strong>Complexity:</strong> ${Math.round(actionData.complexity * 100)}%`;
            actionDetails.appendChild(complexityDiv);
        }
        if (actionData.routing) {
            const routingDiv = document.createElement('div');
            routingDiv.innerHTML = `<strong>Routing:</strong> ${actionData.routing}`;
            actionDetails.appendChild(routingDiv);
        }
        if (actionData.websites) {
            const websitesDiv = document.createElement('div');
            websitesDiv.innerHTML = `<strong>Websites:</strong>`;
            const websitesList = document.createElement('ul');
            actionData.websites.forEach(url => {
                const li = document.createElement('li');
                const link = document.createElement('a');
                link.href = url;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                link.textContent = url;
                li.appendChild(link);
                websitesList.appendChild(li);
            });
            websitesDiv.appendChild(websitesList);
            actionDetails.appendChild(websitesDiv);
        }
        if (actionData.thoughts) {
            const thoughtsDiv = document.createElement('div');
            thoughtsDiv.innerHTML = `<strong>Thoughts:</strong>`;
            const thoughtsList = document.createElement('ul');
            actionData.thoughts.forEach(thought => {
                const li = document.createElement('li');
                li.textContent = thought;
                thoughtsList.appendChild(li);
            });
            thoughtsDiv.appendChild(thoughtsList);
            actionDetails.appendChild(thoughtsDiv);
        }
        if (actionData.document) {
            const docDiv = document.createElement('div');
            docDiv.innerHTML = `<strong>Document:</strong> <button class="view-doc-btn" data-doc="${encodeURIComponent(JSON.stringify(actionData.document))}">View Document</button>`;
            actionDetails.appendChild(docDiv);
        }

        actionContent.appendChild(actionDetails);

        // Make action item clickable to toggle details
        actionItem.addEventListener('click', () => {
            const isExpanded = actionDetails.style.display !== 'none';
            actionDetails.style.display = isExpanded ? 'none' : 'block';
            actionItem.classList.toggle('expanded', !isExpanded);
        });

        // Handle document viewer button
        actionItem.querySelectorAll('.view-doc-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const docData = JSON.parse(decodeURIComponent(btn.getAttribute('data-doc')));
                openDocumentViewer(docData);
            });
        });
    }

    actionItem.appendChild(actionIcon);
    actionItem.appendChild(actionContent);

    // Add status indicator
    if (actionData.status === 'complete') {
        actionItem.classList.add('complete');
    } else if (actionData.status === 'error') {
        actionItem.classList.add('error');
    }

    actionTracking.appendChild(actionItem);

    // Scroll to show new action item (smooth and fast)
    const chatContainer = document.getElementById('chatContainer');
    requestAnimationFrame(() => {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    });
}

// Add word breakdown visualization with animated cycling
function addWordBreakdown(data, messageElement) {
    console.log('🎨 addWordBreakdown called with:', data); // Debug log

    let wordBreakdownDisplay = messageElement.querySelector('.word-breakdown-display');
    if (!wordBreakdownDisplay) {
        // This should not happen - word breakdown display should be created in handleStreamingMessage
        console.warn('⚠️ Word breakdown display not found, creating it now');
        wordBreakdownDisplay = document.createElement('div');
        wordBreakdownDisplay.className = 'word-breakdown-display';
        // Show during processing so user can see linguistic analysis - force visibility
        wordBreakdownDisplay.style.setProperty('display', 'block', 'important');
        wordBreakdownDisplay.style.setProperty('visibility', 'visible', 'important');
        wordBreakdownDisplay.style.setProperty('opacity', '1', 'important');
        wordBreakdownDisplay.innerHTML = '<div class="word-breakdown-header"><strong>Analyzing Prompt:</strong><span class="word-breakdown-collapse">▼</span></div><div class="word-breakdown-content"></div>';

        // Insert at the beginning of message content (hidden)
        const messageContent = messageElement.querySelector('.message-content');
        if (messageContent) {
            messageContent.insertBefore(wordBreakdownDisplay, messageContent.firstChild);
        } else {
            messageElement.appendChild(wordBreakdownDisplay);
        }

        // Add click handler to collapse/expand
        const header = wordBreakdownDisplay.querySelector('.word-breakdown-header');
        const content = wordBreakdownDisplay.querySelector('.word-breakdown-content');
        const collapse = wordBreakdownDisplay.querySelector('.word-breakdown-collapse');

        header.addEventListener('click', () => {
            const isCollapsed = wordBreakdownDisplay.classList.contains('collapsed');
            if (isCollapsed) {
                content.style.display = 'block';
                collapse.textContent = '▼';
                wordBreakdownDisplay.classList.remove('collapsed');
            } else {
                content.style.display = 'none';
                collapse.textContent = '▶';
                wordBreakdownDisplay.classList.add('collapsed');
            }
        });
    }

    const content = wordBreakdownDisplay.querySelector('.word-breakdown-content');
    if (!content) {
        console.error('❌ Word breakdown content not found!');
        return;
    }

    if (data.type === 'algorithm_step') {
        // Show algorithm step briefly (quick cycle) - fade in/out
        const stepDiv = document.createElement('div');
        stepDiv.className = 'algorithm-step cycling';
        stepDiv.innerHTML = `
            <div class="step-name">${data.step.replace('_', ' ')}</div>
            <div class="step-status ${data.status}">${data.status}</div>
        `;
        content.appendChild(stepDiv);

        // Auto-remove after animation (cycling effect)
        setTimeout(() => {
            if (stepDiv.parentNode) {
                stepDiv.style.opacity = '0';
                stepDiv.style.transform = 'translateX(-10px)';
                setTimeout(() => stepDiv.remove(), 300);
            }
        }, 1200); // Show for 1.2s then fade out

    } else if (data.type === 'word_breakdown') {
        // Ensure content div is visible
        if (content) {
            content.style.setProperty('display', 'block', 'important');
            content.style.setProperty('visibility', 'visible', 'important');
        }

        // Get or create the flowing text display
        let flowingText = content.querySelector('.word-flowing-text');
        if (!flowingText) {
            flowingText = document.createElement('div');
            flowingText.className = 'word-flowing-text';
            flowingText.style.setProperty('display', 'block', 'important');
            flowingText.style.setProperty('visibility', 'visible', 'important');
            content.appendChild(flowingText);
        } else {
            // Ensure existing flowing text is visible
            flowingText.style.setProperty('display', 'block', 'important');
            flowingText.style.setProperty('visibility', 'visible', 'important');
        }

        const word = data.word;
        const prefix = data.morphological?.prefix || '';
        const root = data.morphological?.root || word;
        const suffix = data.morphological?.suffix || '';
        const etymology = data.etymology || {};
        const likelyOrigin = etymology.likely_origin || etymology.origin || '';
        const linguisticRoots = etymology.linguistic_roots || [];
        const wordOrigins = etymology.word_origins || {};
        const semantic = data.semantic || {};

        // Create word container (inline, normal sentence flow)
        const wordContainer = document.createElement('span');
        wordContainer.className = 'word-neon-container';

        // Main word display (inline, normal sentence flow)
        const wordSpan = document.createElement('span');
        wordSpan.className = 'word-inline-neon';
        wordSpan.setAttribute('data-word', word);
        wordSpan.setAttribute('data-origin', likelyOrigin);

        // Build word with neon color-coded parts (inline, normal spacing) with slow flash
        let wordHTML = '';
        if (prefix && prefix !== word) {
            wordHTML += `<span class="word-part-neon prefix slow-flash" data-part="prefix">${prefix}</span>`;
        }
        if (root && root !== word) {
            wordHTML += `<span class="word-part-neon root slow-flash" data-part="root">${root}</span>`;
        } else {
            wordHTML += `<span class="word-part-neon root slow-flash" data-part="root">${word}</span>`;
        }
        if (suffix && suffix !== word) {
            wordHTML += `<span class="word-part-neon suffix slow-flash" data-part="suffix">${suffix}</span>`;
        }

        // If no prefix/suffix, just show the word
        if (!prefix && !suffix) {
            wordHTML = `<span class="word-part-neon root slow-flash" data-part="root">${word}</span>`;
        }

        wordSpan.innerHTML = wordHTML;
        wordContainer.appendChild(wordSpan);

        // Store etymology data for later display below sentence
        wordContainer.setAttribute('data-origin', likelyOrigin);
        wordContainer.setAttribute('data-roots', linguisticRoots.join(', '));

        flowingText.appendChild(wordContainer);

        // Add space after word (normal sentence spacing)
        const space = document.createTextNode(' ');
        flowingText.appendChild(space);

        // Ensure word breakdown is visible and force display
        if (wordBreakdownDisplay) {
            wordBreakdownDisplay.style.setProperty('display', 'block', 'important');
            wordBreakdownDisplay.style.setProperty('visibility', 'visible', 'important');
            wordBreakdownDisplay.style.setProperty('opacity', '1', 'important');
            wordBreakdownDisplay.classList.remove('collapsed');
        }
        if (content) {
            content.style.setProperty('display', 'block', 'important');
            content.style.setProperty('visibility', 'visible', 'important');
        }

        // Fast machine-like pattern matching animation with neon flash
        requestAnimationFrame(() => {
            const prefixEl = wordSpan.querySelector('.prefix');
            const rootEl = wordSpan.querySelector('.root');
            const suffixEl = wordSpan.querySelector('.suffix');

            // Light up parts simultaneously or sequentially (fast)
            const parts = [prefixEl, rootEl, suffixEl].filter(el => el);

            // Option 1: Light up simultaneously (machine pattern matching)
            parts.forEach((part, index) => {
                setTimeout(() => {
                    part.classList.add('neon-lit');
                    setTimeout(() => {
                        part.classList.remove('neon-lit');
                        // Keep glowing based on etymology (only if not unknown)
                        if (likelyOrigin && likelyOrigin !== 'unknown') {
                            part.classList.add(`origin-${likelyOrigin}`);
                        }
                    }, 150); // Fast flash
                }, index * 50); // Very fast sequence (50ms between parts)
            });

            // Show etymology colors after morphological (only if not unknown)
            if (likelyOrigin && likelyOrigin !== 'unknown') {
                setTimeout(() => {
                    wordSpan.setAttribute('data-origin', likelyOrigin);
                    parts.forEach(part => {
                        part.classList.add(`origin-${likelyOrigin}`);
                    });
                }, 200);
            }
        });
    }
}

// Open document viewer (black background, white text)
function openDocumentViewer(docData) {
    const viewer = document.createElement('div');
    viewer.className = 'document-viewer';

    let bodyContent = '';

    // Handle PDF files
    if (docData.type === 'application/pdf' || docData.url?.endsWith('.pdf') || docData.filename?.endsWith('.pdf')) {
        bodyContent = `
            <iframe 
                src="${docData.url || docData.content || ''}" 
                style="width: 100%; height: 100%; border: none; background: #000000;"
                title="${docData.title || 'PDF Document'}"
            ></iframe>
        `;
    } else if (docData.content || docData.text) {
        // Render markdown or plain text
        let content = docData.content || docData.text;
        if (docData.format === 'markdown' || content.includes('```') || content.includes('##')) {
            content = marked.parse(content);
            content = renderMath(content);
        }
        bodyContent = content;
    } else {
        bodyContent = 'No content available';
    }

    viewer.innerHTML = `
        <div class="document-viewer-content">
            <div class="document-viewer-header">
                <h2>${docData.title || docData.filename || 'Document'}</h2>
                <button class="document-viewer-close" aria-label="Close document viewer">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                        <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </button>
            </div>
            <div class="document-viewer-body">
                ${bodyContent}
            </div>
        </div>
    `;

    document.body.appendChild(viewer);

    // Close button handler
    viewer.querySelector('.document-viewer-close').addEventListener('click', () => {
        viewer.remove();
    });

    // Close on escape key
    const closeHandler = (e) => {
        if (e.key === 'Escape') {
            viewer.remove();
            document.removeEventListener('keydown', closeHandler);
        }
    };
    document.addEventListener('keydown', closeHandler);

    // Close on backdrop click
    viewer.addEventListener('click', (e) => {
        if (e.target === viewer) {
            viewer.remove();
        }
    });

    // Highlight code blocks if markdown was rendered
    if (docData.format === 'markdown' || bodyContent.includes('<pre')) {
        setTimeout(() => {
            viewer.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }, 100);
    }
}

// Add engines display
function addEnginesDisplay(engines, messageElement) {
    let enginesDisplay = messageElement.querySelector('.engines-display');
    if (!enginesDisplay) {
        enginesDisplay = document.createElement('div');
        enginesDisplay.className = 'engines-display';
        enginesDisplay.innerHTML = '<strong>Engines Active:</strong>';
        messageElement.appendChild(enginesDisplay);
    }

    engines.forEach(engine => {
        const engineItem = document.createElement('div');
        engineItem.className = 'engine-item';
        engineItem.innerHTML = `
            <span class="engine-name">${engine.engine}</span>
            <span class="engine-algorithm">${engine.algorithm}</span>
            <span class="engine-description">${engine.description || ''}</span>
        `;
        enginesDisplay.appendChild(engineItem);
    });

    // Scroll to show new engine
    const chatContainer = document.getElementById('chatContainer');
    setTimeout(() => {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    }, 50);
}

// Add algorithms display
function addAlgorithmsDisplay(algorithms, messageElement) {
    let algorithmsDisplay = messageElement.querySelector('.algorithms-display');
    if (!algorithmsDisplay) {
        algorithmsDisplay = document.createElement('div');
        algorithmsDisplay.className = 'algorithms-display';
        algorithmsDisplay.innerHTML = '<strong>Algorithms Used:</strong>';
        messageElement.appendChild(algorithmsDisplay);
    }

    algorithms.forEach(algorithm => {
        const algorithmItem = document.createElement('div');
        algorithmItem.className = 'algorithm-item';
        algorithmItem.innerHTML = `
            <span class="algorithm-name">${algorithm.algorithm}</span>
            <span class="algorithm-method">${algorithm.method || ''}</span>
        `;
        algorithmsDisplay.appendChild(algorithmItem);
    });

    // Scroll to show new algorithm
    const chatContainer = document.getElementById('chatContainer');
    setTimeout(() => {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    }, 50);
}

// Send query
async function sendQuery() {
    const input = document.getElementById('queryInput');
    const sendButton = document.getElementById('sendButton');
    const modeSelect = document.getElementById('modeSelect');
    const agentSelect = document.getElementById('agentSelect');
    const agentDisplay = document.getElementById('agentDisplay');
    const agentDisplayValue = document.getElementById('agentDisplayValue');
    const query = input.value.trim();

    // Strict validation - don't send empty queries
    if (!query || query.length === 0) {
        if (attachedFiles.length === 0) {
            console.warn('⚠️ Empty query, not sending');
            return;
        }
    }

    // Get selected mode and agent
    const mode = modeSelect.value;
    // v5: Handle new search modes (web_research, local_rag, hybrid)
    // In chat mode or v5 search modes, ALWAYS use secretary
    let agent;
    if (mode === 'chat' || mode === 'web_research' || mode === 'local_rag' || mode === 'hybrid') {
        agent = 'secretary'; // ALWAYS secretary in chat/search modes
    } else if (mode === 'astrophysiology') {
        agent = 'celestial_calc'; // Specialized agent for astro-physiology
    } else {
        agent = agentSelect ? agentSelect.value : 'auto';
    }
    const useDegradation = false; // Removed degradation mode

    // Client-side preprocessing (leverages user's device)
    let preprocessedQuery = null;
    if (clientProcessor) {
        try {
            preprocessedQuery = await clientProcessor.preprocessQuery(query);
            // Check cache first
            if (preprocessedQuery.cached) {
                console.log('✅ Using cached response from device');
                const cachedResponse = preprocessedQuery.data;
                const assistantMessage = addMessage(cachedResponse.content || cachedResponse, 'assistant');
                input.disabled = false;
                sendButton.disabled = false;
                input.focus();
                return;
            }
        } catch (e) {
            console.warn('Client-side preprocessing failed:', e);
        }
    }

    // Disable input and button
    input.disabled = true;
    sendButton.disabled = true;

    // Phase 2: Upload files using multipart/form-data instead of Base64
    const filesData = [];
    for (const file of attachedFiles) {
        try {
            // Upload file to /api/upload endpoint
            const formData = new FormData();
            formData.append('file', file);

            const uploadResponse = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                const error = await uploadResponse.json().catch(() => ({ detail: 'Upload failed' }));
                throw new Error(error.detail || 'File upload failed');
            }

            const uploadResult = await uploadResponse.json();

            // Store file_id reference instead of Base64 data
            filesData.push({
                file_id: uploadResult.file_id,
                name: uploadResult.filename,
                type: uploadResult.content_type,
                size: uploadResult.size
            });
        } catch (error) {
            console.error(`Error uploading file ${file.name}:`, error);
            // Show error to user
            const errorMsg = document.createElement('div');
            errorMsg.className = 'error-message';
            errorMsg.textContent = `Failed to upload ${file.name}: ${error.message}`;
            messagesContainer.appendChild(errorMsg);
            // Continue with other files
        }
    }

    // Check for image generation request
    if (detectImageGenerationRequest(query)) {
        const imagePrompt = query.replace(/^(generate|create|draw|show me|image:)\s*(an |a )?(image |of |showing )?/i, '').trim();
        if (imagePrompt) {
            generateImage(imagePrompt);
            return;
        }
    }

    // Update conversation context with user message
    updateConversationContext(query, null);

    // Add user message with mode/agent info
    const userMessage = addMessage(query || `[${attachedFiles.length} file(s) attached]`, 'user');
    if (agent !== 'auto' || mode !== 'chat') {
        const metaInfo = document.createElement('div');
        metaInfo.className = 'message-meta';
        metaInfo.textContent = `${mode} mode${agent !== 'auto' ? ` • ${agent}` : ''}`;
        userMessage.appendChild(metaInfo);
    }

    // Add web search indicator if enabled
    if (enableWebSearch) {
        addWebSearchIndicator(userMessage);
    }

    // Display attached files in message
    if (attachedFiles.length > 0) {
        const filesDisplay = document.createElement('div');
        filesDisplay.className = 'message-files';
        attachedFiles.forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'message-file-item';
            if (file.type.startsWith('image/')) {
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                img.alt = file.name;
                img.className = 'message-file-image';
                fileItem.appendChild(img);
            } else {
                fileItem.textContent = `📄 ${file.name} (${formatFileSize(file.size)})`;
            }
            filesDisplay.appendChild(fileItem);
        });
        userMessage.querySelector('.message-content').appendChild(filesDisplay);
    }

    // Clear input and files
    input.value = '';
    clearAttachedFiles();

    // Create assistant message immediately (ICEBURG instant response)
    const assistantMessage = addMessage('', 'assistant');

    // Make sure the message is visible even if empty
    const messageContent = assistantMessage.querySelector('.message-content');
    if (messageContent && !messageContent.textContent.trim()) {
        // Add a minimal placeholder to ensure visibility
        messageContent.style.minHeight = '20px';
        // DO NOT add loading dots - live event animations will show progress
        // The status carousel will display thinking/processing status

        // Add mode-specific loading for astro-physiology
        if (mode === 'astrophysiology') {
            astro_showLoadingState(messageContent);
        } else {
            // Show rich loading state for all other modes (research, fast, truth, hybrid, etc.)
            showModeLoadingState(messageContent, mode);
        }
    }

    // Immediately add a thinking status to show something is happening
    // This replaces the loading dots with live animations
    setTimeout(() => {
        // Skip old thinking status - we use thinking_stream instead
        // const lastMsg = getOrCreateLastMessage();
        // if (lastMsg && !lastMsg.querySelector('.status-carousel')) {
        //     addToStatusCarousel('thinking', { content: 'Preparing response...' }, lastMsg);
        // }
    }, 100);

    // Show animations IMMEDIATELY (before backend response) - ICEBURG style
    // Add prompt interpreter action immediately (will be added to status carousel)
    // No separate thinking container - thinking goes to status carousel

    // Scroll to show new content immediately
    const chatContainer = document.getElementById('chatContainer');
    requestAnimationFrame(() => {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    });

    try {
        console.log('🚀 Sending query via SSE Connection...', { query: query.substring(0, 50) + '...', mode, agent });
        
        // Collect birth data for astro-physiology mode
        let birthData = null;
        if (mode === 'astrophysiology') {
            // Helper to get birth data logic (extracted from previous complexity)
            // Ideally this should be a separate function, but keeping inline for compatibility with existing scope
            const parsedFromQuery = astro_parseBirthDataFromQuery(query);
            const birthDateInput = document.getElementById('birthDateInput') || document.getElementById('astro_birthDateInput');
            const birthTimeInput = document.getElementById('birthTimeInput') || document.getElementById('astro_birthTimeInput');
            const locationInput = document.getElementById('locationInput') || document.getElementById('astro_locationInput');

            if (birthDateInput && birthDateInput.value) {
                const birthDate = birthDateInput.value;
                const birthTime = birthTimeInput ? (birthTimeInput.value || '12:00') : '12:00';
                const birthDateTime = `${birthDate}T${birthTime}:00Z`;
                let location = null;
                if (locationInput && locationInput.value) {
                     const val = locationInput.value.trim();
                     const match = val.match(/^(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)$/);
                     location = match ? { lat: parseFloat(match[1]), lng: parseFloat(match[2]) } : val;
                }
                birthData = { birth_date: birthDateTime, location: location, timestamp: Date.now() };
                localStorage.setItem('iceburg_astro_birth_data', JSON.stringify(birthData));
                localStorage.removeItem('iceburg_astro_algorithmic_data'); // Clear algo cache on new input
            } else if (parsedFromQuery && parsedFromQuery.birth_date) {
                const birthDateTime = `${parsedFromQuery.birth_date}T${parsedFromQuery.birth_time || '12:00'}:00Z`;
                birthData = { birth_date: birthDateTime, location: parsedFromQuery.location, timestamp: Date.now() };
                localStorage.setItem('iceburg_astro_birth_data', JSON.stringify(birthData));
                localStorage.removeItem('iceburg_astro_algorithmic_data');
            } else {
                 try {
                    const saved = localStorage.getItem('iceburg_astro_birth_data');
                    if (saved) {
                        const parsed = JSON.parse(saved);
                        if (!parsed.timestamp || (Date.now() - parsed.timestamp) < 24 * 60 * 60 * 1000) {
                            birthData = parsed;
                        }
                    }
                 } catch(e) {}
            }
            
            // Should we look for follow-up algo data?
            if (!birthData) {
                try {
                     const algo = localStorage.getItem('iceburg_astro_algorithmic_data');
                     if (algo) {
                         const parsed = JSON.parse(algo);
                         if (!parsed.timestamp || (Date.now() - parsed.timestamp) < 7 * 24 * 60 * 60 * 1000) {
                             birthData = { algorithmic_data: parsed }; // Special case wrapper
                         }
                     }
                } catch(e) {}
            }
        }

        // Use the new SSE connection
        if (window.ICEBURG_CONNECTION) {
            window.ICEBURG_CONNECTION.sendQuery({
                query,
                mode,
                settings: {
                    ...settings,
                    agent, // Pass agent in settings or separate? Connection.js expects flat settings usually, but let's check.
                    // Actually handleStreamingMessage expects 'agent' in the MESSAGE. 
                    // connection.js sends body: { query, mode, ...settings } ?? NO.
                    // connection.js body: { query, mode, conversation_id, stream: true, temperature: settings.temperature, ...data }
                    // It doesn't pass arbitrary settings to body root.
                    // I need to ensure AGENT is passed.
                    // connection.js doesn't explicitly pass 'agent' to body.
                    // I might need to abuse 'data' or 'settings' or modify connection.js.
                    // Let's modify connection.js logic to include extra payload if needed, OR just put it in data.
                    // Actually, let's just cheat and put it in settings if the backend supports it, OR update connection.js to accept 'agent'.
                    // WAIT. connection.js:80 body: JSON.stringify({ query, mode, ... })
                    // It does NOT pass agent.
                    // I should update connection.js to accept agent.
                    // For now, I will treat it as 'data'. 
                },
                // Wait, I should fix connection.js to pass agent. I will do that in next step.
                // For now, assuming connection.js will be updated or I put it in data.
                data: {
                    ...(birthData || {}),
                    agent: agent,
                    files: filesData,
                    client_metadata: preprocessedQuery ? {
                            normalized: preprocessedQuery.normalized,
                            entities: preprocessedQuery.entities,
                            keywords: preprocessedQuery.keywords,
                            complexity: preprocessedQuery.complexity,
                            device: preprocessedQuery.metadata?.device
                    } : null
                },
                onMessage: (data) => {
                     handleStreamingMessage(data);
                },
                onError: (error) => {
                    console.error('❌ SSE Query Error:', error);
                    const loadingIndicator = document.querySelector('.loading-indicator');
                    if (loadingIndicator) loadingIndicator.remove();
                    
                    const lastMsg = getOrCreateLastMessage();
                    if (lastMsg) {
                         const errDiv = document.createElement('div');
                         errDiv.className = 'error-message';
                         errDiv.textContent = 'Error: ' + error.message;
                         lastMsg.querySelector('.message-content').appendChild(errDiv);
                    }
                }
            });
        } else {
            console.error('❌ window.ICEBURG_CONNECTION not found!');
            showToast('Connection module missing. Reload page.', 'error');
        }

    } catch (sendError) {
        console.error('❌ Error initializing query:', sendError);
        showToast('Failed to send query', 'error');
    } finally {
        // Ensure UI is resilient
        const loadingIndicator = document.querySelector('.loading-indicator');
        if (loadingIndicator && !window.ICEBURG_CONNECTION) loadingIndicator.remove();
    }
}


// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// File handling
function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    attachedFiles.push(...files);
    updateAttachedFilesDisplay();
}

function updateAttachedFilesDisplay() {
    const container = document.getElementById('attachedFiles');
    container.innerHTML = '';

    attachedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'attached-file-item';

        if (file.type.startsWith('image/')) {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.alt = file.name;
            img.className = 'attached-file-preview';
            fileItem.appendChild(img);
        } else {
            fileItem.innerHTML = `<span>📄 ${file.name}</span>`;
        }

        const removeBtn = document.createElement('button');
        removeBtn.className = 'attached-file-remove';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = () => {
            attachedFiles.splice(index, 1);
            updateAttachedFilesDisplay();
        };
        fileItem.appendChild(removeBtn);
        container.appendChild(fileItem);
    });
}

function clearAttachedFiles() {
    attachedFiles = [];
    updateAttachedFilesDisplay();
    document.getElementById('fileInput').value = '';
}

// Conversation management
function saveConversation() {
    const messages = Array.from(document.querySelectorAll('.message')).map(msg => ({
        type: msg.classList.contains('user') ? 'user' : 'assistant',
        content: msg.querySelector('.message-content').textContent || msg.querySelector('.message-content').innerText,
        timestamp: msg.querySelector('.message-header span:last-child').textContent
    }));

    const conversation = {
        id: currentConversationId,
        title: messages[0]?.content?.substring(0, 50) || 'New Conversation',
        messages: messages,
        updated: new Date().toISOString()
    };

    conversations = conversations.filter(c => c.id !== currentConversationId);
    conversations.unshift(conversation);
    localStorage.setItem('iceburg_conversations', JSON.stringify(conversations));
    updateConversationsList();
}

function loadConversation(conversationId) {
    const conversation = conversations.find(c => c.id === conversationId);
    if (!conversation) return;

    const chatContainer = document.getElementById('chatContainer');
    chatContainer.innerHTML = '';

    conversation.messages.forEach(msg => {
        addMessage(msg.content, msg.type);
    });

    currentConversationId = conversationId;
    updateConversationsList();
}

function newConversation() {
    currentConversationId = 'conv_' + Date.now();
    window.activeInvestigationId = null; // Clear investigation context
    // Clear URL params without reloading
    window.history.pushState({}, document.title, window.location.pathname);

    const chatContainer = document.getElementById('chatContainer');
    chatContainer.innerHTML = `
        <div class="welcome-message">
            <p>Welcome to ICEBURG 2.0</p>
            <p class="welcome-subtitle">Ask anything to begin</p>
        </div>
    `;
    updateConversationsList();
}

function updateConversationsList() {
    const list = document.getElementById('conversationsList');
    if (!list) return;

    list.innerHTML = `
        <div class="conversation-item ${currentConversationId === 'current' ? 'active' : ''}" data-id="current">
            <span class="conversation-title">Current Conversation</span>
            <button class="conversation-delete" aria-label="Delete conversation">×</button>
        </div>
    `;

    let filteredConversations = conversations;
    if (searchQuery) {
        filteredConversations = conversations.filter(conv =>
            conv.title.toLowerCase().includes(searchQuery) ||
            conv.messages.some(msg => msg.content.toLowerCase().includes(searchQuery))
        );
    }

    filteredConversations.slice(0, 10).forEach(conv => {
        const item = document.createElement('div');
        item.className = `conversation-item ${currentConversationId === conv.id ? 'active' : ''}`;
        item.setAttribute('data-id', conv.id);

        // Highlight search query in title
        let title = conv.title;
        if (searchQuery) {
            const regex = new RegExp(`(${searchQuery})`, 'gi');
            title = title.replace(regex, '<mark>$1</mark>');
        }

        item.innerHTML = `
            <span class="conversation-title">${title}</span>
            <button class="conversation-delete" aria-label="Delete conversation">×</button>
        `;
        item.addEventListener('click', (e) => {
            if (!e.target.classList.contains('conversation-delete')) {
                loadConversation(conv.id);
            }
        });
        item.querySelector('.conversation-delete').addEventListener('click', (e) => {
            e.stopPropagation();
            conversations = conversations.filter(c => c.id !== conv.id);
            localStorage.setItem('iceburg_conversations', JSON.stringify(conversations));
            if (currentConversationId === conv.id) {
                newConversation();
            } else {
                updateConversationsList();
            }
        });
        list.appendChild(item);
    });

    // Show "no results" message if search has no matches
    if (searchQuery && filteredConversations.length === 0) {
        const noResults = document.createElement('div');
        noResults.className = 'conversation-item no-results';
        noResults.textContent = 'No conversations found';
        list.appendChild(noResults);
    }
}

// Settings management
function loadSettings() {
    const saved = localStorage.getItem('iceburg_settings');
    if (saved) {
        try {
            settings = { ...settings, ...JSON.parse(saved) };
        } catch (e) {
            console.error('Error loading settings:', e);
        }
    }
    applySettings();
}

function saveSettings() {
    localStorage.setItem('iceburg_settings', JSON.stringify(settings));
}

function applySettings() {
    const primaryModel = document.getElementById('primaryModel');
    const temperature = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperatureValue');
    const maxTokens = document.getElementById('maxTokens');
    const maxTokensValue = document.getElementById('maxTokensValue');

    if (primaryModel) primaryModel.value = settings.primaryModel;
    if (temperature) {
        temperature.value = settings.temperature;
        if (temperatureValue) temperatureValue.textContent = settings.temperature;
    }
    if (maxTokens) {
        maxTokens.value = settings.maxTokens;
        if (maxTokensValue) maxTokensValue.textContent = settings.maxTokens;
    }
}

function openSettings() {
    const panel = document.getElementById('settingsPanel');
    const overlay = document.getElementById('settingsOverlay');
    if (panel && overlay) {
        panel.classList.add('open');
        overlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

function closeSettings() {
    const panel = document.getElementById('settingsPanel');
    const overlay = document.getElementById('settingsOverlay');
    if (panel && overlay) {
        panel.classList.remove('open');
        overlay.classList.remove('active');
        document.body.style.overflow = '';
    }
}

// Export conversation
function exportConversation() {
    const messages = Array.from(document.querySelectorAll('.message')).map(msg => ({
        type: msg.classList.contains('user') ? 'user' : 'assistant',
        content: msg.querySelector('.message-content').textContent || msg.querySelector('.message-content').innerText,
        timestamp: msg.querySelector('.message-header span:last-child')?.textContent || ''
    }));

    const conversation = {
        title: messages[0]?.content?.substring(0, 50) || 'ICEBURG Conversation',
        exported: new Date().toISOString(),
        messages: messages
    };

    const dataStr = JSON.stringify(conversation, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `iceburg-conversation-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

// Share conversation
function shareConversation() {
    const messages = Array.from(document.querySelectorAll('.message')).map(msg => ({
        type: msg.classList.contains('user') ? 'user' : 'assistant',
        content: msg.querySelector('.message-content').textContent || msg.querySelector('.message-content').innerText
    }));

    const text = messages.map(msg => `${msg.type === 'user' ? 'You' : 'ICEBURG'}: ${msg.content}`).join('\n\n');

    if (navigator.share) {
        navigator.share({
            title: 'ICEBURG Conversation',
            text: text
        }).catch(err => console.error('Error sharing:', err));
    } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(text).then(() => {
            alert('Conversation copied to clipboard!');
        });
    }
}

// Clear all conversations
function clearAllConversations() {
    if (confirm('Are you sure you want to clear all conversations? This cannot be undone.')) {
        conversations = [];
        localStorage.removeItem('iceburg_conversations');
        newConversation();
        updateConversationsList();
    }
}

// Search conversations
function searchConversations(query) {
    searchQuery = query.toLowerCase().trim();
    updateConversationsList();
}

// Detect when assistant (in Fast/Chat) offered to run multi-agent research
function textSuggestsResearchOffer(text) {
    if (!text || typeof text !== 'string') return false;
    const t = text.toLowerCase();
    const hasSurveyor = t.includes('surveyor');
    const hasDissident = t.includes('dissident');
    const hasSynthesist = t.includes('synthesist');
    const hasOracle = t.includes('oracle');
    const hasProtocol = t.includes('research protocol') || t.includes('full pipeline');
    const hasOffer = t.includes("i'll have the surveyor") || t.includes("i'll have the dissident") || t.includes('surveyor provide') || t.includes('dissident offer');
    return (hasSurveyor && (hasDissident || hasSynthesist || hasOracle)) || hasProtocol || hasOffer;
}

// Update conversation context for natural flow
function updateConversationContext(userMessage, assistantResponse) {
    if (userMessage) {
        conversationContext.lastUserMessage = userMessage;
        // Extract key topics (simple keyword extraction)
        const words = userMessage.toLowerCase().split(/\s+/);
        const importantWords = words.filter(w => w.length > 4 && !['what', 'when', 'where', 'which', 'about', 'think', 'would', 'could', 'should'].includes(w));
        conversationContext.topics.push(...importantWords.slice(0, 5));

        // Keep only recent topics (last 20)
        if (conversationContext.topics.length > 20) {
            conversationContext.topics = conversationContext.topics.slice(-20);
        }
    }

    if (assistantResponse) {
        conversationContext.lastAssistantResponse = assistantResponse;
    }

    // Store previous discussion points
    if (userMessage && assistantResponse) {
        conversationContext.previousPoints.push({
            user: userMessage,
            assistant: assistantResponse,
            timestamp: Date.now()
        });

        // Keep only recent context (last 10 exchanges)
        if (conversationContext.previousPoints.length > 10) {
            conversationContext.previousPoints.shift();
        }
    }
}

// Extract topics from message (simple implementation)
function extractTopics(message) {
    const words = message.toLowerCase().split(/\s+/);
    return words.filter(w => w.length > 4 && !['what', 'when', 'where', 'which', 'about', 'think', 'would', 'could', 'should', 'there', 'their', 'these', 'those'].includes(w)).slice(0, 5);
}

// Handle ICEBURG real-time thinking stream (compact single-line with dropdown)
function handleThinkingStream(data, messageElement) {
    if (!messageElement) {
        console.error('❌ handleThinkingStream: messageElement is null!');
        return;
    }

    // Find existing container
    let thoughtsContainer = messageElement.querySelector('.thoughts-stream-container');

    // Remove any duplicate containers if they exist (cleanup/recovery)
    const allContainers = messageElement.querySelectorAll('.thoughts-stream-container');
    if (allContainers.length > 1) {
        console.warn('⚠️ Found duplicate thinking containers, cleaning up...');
        allContainers.forEach((container, index) => {
            if (index > 0) container.remove();
        });
        thoughtsContainer = allContainers[0];
    }

    if (!thoughtsContainer) {
        thoughtsContainer = document.createElement('div');
        thoughtsContainer.className = 'thoughts-stream-container';
        // Simple single-line structure
        thoughtsContainer.innerHTML = `
            <div class="thoughts-stream-header">
                <span class="thoughts-stream-current"></span>
            </div>
        `;

        const messageContent = messageElement.querySelector('.message-content');
        if (messageContent) {
            // ALWAYS insert at the start
            messageContent.insertAdjacentElement('afterbegin', thoughtsContainer);
        } else {
            messageElement.appendChild(thoughtsContainer);
        }

        // Ensure visibility
        thoughtsContainer.style.display = 'block';
        thoughtsContainer.style.visibility = 'visible';
        thoughtsContainer.style.opacity = '1';
    }

    const currentDisplay = thoughtsContainer.querySelector('.thoughts-stream-current');

    // Safety check - if internal structure is broken, fix it
    if (!currentDisplay) {
        thoughtsContainer.innerHTML = `
            <div class="thoughts-stream-header">
                <span class="thoughts-stream-current"></span>
            </div>
        `;
    }

    // Skip "Initializing..." messages ONLY if we already have content
    // Otherwise show it so the user knows something is happening
    const content = data.content || '';
    const shouldShow = content.trim().length > 0;

    if (shouldShow) {
        // Update the single line
        if (currentDisplay) {
            // Use ○ as the indicator
            const text = content + ' ○';
            currentDisplay.textContent = text;
            currentDisplay.dataset.text = text; // For glitch animation

            // Ensure glitch effect is active
            if (!currentDisplay.classList.contains('glitch-active')) {
                currentDisplay.classList.add('glitch-active');
            }
        }
    }
}

// Handle step completion for interactive workflow
function handleStepComplete(data, messageElement) {
    console.log('📊 Step complete:', data);

    const workflowContainer = getOrCreateWorkflowContainer(messageElement);
    const stepCard = createInteractiveStepCard(data);
    workflowContainer.appendChild(stepCard);

    // Auto-advance after 3 seconds if no user action
    const autoAdvanceTimeout = setTimeout(() => {
        const recommendedAction = data.report?.suggested_next?.[0] || 'skip';
        routeToNextStep(recommendedAction, messageElement);
    }, 3000);

    // Store timeout so user can cancel it
    stepCard.dataset.autoAdvanceTimeout = autoAdvanceTimeout;
}

// Get or create workflow container
function getOrCreateWorkflowContainer(messageElement) {
    let container = messageElement.querySelector('.workflow-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'workflow-container';
        const messageContent = messageElement.querySelector('.message-content');
        if (messageContent) {
            messageContent.insertBefore(container, messageContent.firstChild);
        } else {
            messageElement.appendChild(container);
        }
    }
    return container;
}

// Create interactive step card
function createInteractiveStepCard(stepData) {
    const card = document.createElement('div');
    card.className = 'interactive-step-card active';
    card.dataset.step = stepData.step;

    const report = stepData.report || {};
    const findings = report.findings || [];
    const options = stepData.options || [];

    card.innerHTML = `
        <div class="step-header">
            <span class="step-name">${capitalizeFirst(stepData.step)}</span>
            <span class="step-status">✓ Complete</span>
        </div>
        <div class="step-report">
            ${findings.map(f => `<div class="step-finding">• ${f}</div>`).join('')}
            ${report.time_taken ? `<div class="step-time">Time: ${report.time_taken}s</div>` : ''}
        </div>
        <div class="step-options">
            ${options.map(opt => `
                <button class="route-button" data-action="${opt.action}">
                    ${opt.label} ${opt.estimated_time ? `(${opt.estimated_time})` : ''}
                </button>
            `).join('')}
        </div>
        <div class="auto-advance-indicator">
            Auto-advancing in <span class="countdown">3</span>s...
        </div>
    `;

    // Add click handlers
    card.querySelectorAll('.route-button').forEach(btn => {
        btn.addEventListener('click', () => {
            clearTimeout(card.dataset.autoAdvanceTimeout);
            card.querySelector('.auto-advance-indicator').style.display = 'none';
            routeToNextStep(btn.dataset.action, card.closest('.message'));
        });
    });

    // Countdown animation
    let countdown = 3;
    const countdownEl = card.querySelector('.countdown');
    const countdownInterval = setInterval(() => {
        countdown--;
        if (countdownEl) countdownEl.textContent = countdown;
        if (countdown <= 0) {
            clearInterval(countdownInterval);
        }
    }, 1000);

    return card;
}

// Route to next step
function routeToNextStep(action, messageElement) {
    console.log('🔄 Routing to next step:', action);

    // Mark current step as completed
    const activeCard = messageElement?.querySelector('.interactive-step-card.active');
    if (activeCard) {
        activeCard.classList.remove('active');
        activeCard.classList.add('completed');
        activeCard.querySelector('.auto-advance-indicator').style.display = 'none';
    }

    if (action === 'skip') {
        // Skip to answer - no additional steps
        return;
    }

    // For now, just log the action
    // In future, this would trigger the next agent
    // For example: send websocket message to trigger next agent
    // Use sendQuery to trigger next step
    const queryInput = document.getElementById('queryInput');
    if (queryInput) {
        const stepName = activeCard?.dataset.step || 'next step';
        const actionLabel = action === 'next' ? 'Proceeding' : action;
        const query = `[Interactive Step] ${stepName}: ${actionLabel}`;
        
        // Update input for visibility
        queryInput.value = query;
        sendQuery();
    }

}

// Helper function
function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Render charts
function renderCharts(container) {
    if (typeof Chart === 'undefined') return;

    container.querySelectorAll('.chart-container canvas').forEach((canvas) => {
        if (canvas.chart) return; // Already rendered

        const chartData = canvas.getAttribute('data-chart');
        if (!chartData) return;

        try {
            // Decode HTML entities and parse JSON
            const decodedData = chartData.replace(/&apos;/g, "'").replace(/&quot;/g, '"');
            const config = JSON.parse(decodedData);
            const ctx = canvas.getContext('2d');

            // Apply dark theme
            const darkTheme = {
                backgroundColor: '#1a1a1a',
                borderColor: '#333333',
                color: '#ffffff',
                gridColor: '#333333'
            };

            if (config.data && config.data.datasets) {
                config.data.datasets = config.data.datasets.map(dataset => ({
                    ...dataset,
                    backgroundColor: dataset.backgroundColor || darkTheme.backgroundColor,
                    borderColor: dataset.borderColor || darkTheme.borderColor,
                    color: dataset.color || darkTheme.color
                }));
            }

            if (!config.options) config.options = {};
            config.options.plugins = config.options.plugins || {};
            config.options.plugins.legend = {
                ...config.options.plugins.legend,
                labels: {
                    color: darkTheme.color,
                    ...config.options.plugins.legend?.labels
                }
            };
            config.options.scales = config.options.scales || {};
            Object.keys(config.options.scales).forEach(scale => {
                if (!config.options.scales[scale]) config.options.scales[scale] = {};
                config.options.scales[scale].ticks = {
                    ...config.options.scales[scale].ticks,
                    color: darkTheme.color
                };
                config.options.scales[scale].grid = {
                    ...config.options.scales[scale].grid,
                    color: darkTheme.gridColor
                };
            });

            // Set default responsive and maintain aspect ratio
            config.options.responsive = config.options.responsive !== false;
            config.options.maintainAspectRatio = config.options.maintainAspectRatio !== false;

            canvas.chart = new Chart(ctx, config);
        } catch (e) {
            console.error('Error rendering chart:', e);
        }
    });
}

// Voice input/output
function initVoiceRecognition() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            const input = document.getElementById('queryInput');
            input.value = transcript;
            input.focus();
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            stopVoiceRecording();
        };

        recognition.onend = () => {
            stopVoiceRecording();
        };
    }

    if ('speechSynthesis' in window) {
        synthesis = window.speechSynthesis;
    }
}

function startVoiceRecording() {
    if (!recognition) {
        alert('Speech recognition is not supported in your browser.');
        return;
    }

    if (isRecording) {
        stopVoiceRecording();
        return;
    }

    try {
        recognition.start();
        isRecording = true;
        updateVoiceButton(true);
    } catch (e) {
        console.error('Error starting voice recognition:', e);
    }
}

function stopVoiceRecording() {
    if (recognition && isRecording) {
        recognition.stop();
        isRecording = false;
        updateVoiceButton(false);
    }
}

function updateVoiceButton(recording) {
    const voiceButton = document.getElementById('voiceButton');
    if (voiceButton) {
        if (recording) {
            voiceButton.classList.add('recording');
        } else {
            voiceButton.classList.remove('recording');
        }
    }
}

function speakText(text) {
    if (!synthesis) {
        alert('Text-to-speech is not supported in your browser.');
        return;
    }

    // Cancel any ongoing speech
    synthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    synthesis.speak(utterance);
}

// Toast notifications
function showToast(message, type = 'info', duration = 3000, onClick = null) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    // Add click handler if provided
    if (onClick) {
        toast.style.cursor = 'pointer';
        toast.addEventListener('click', () => {
            onClick();
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        });
    }

    container.appendChild(toast);

    // Trigger animation
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);

    // Remove after duration (only if no click handler or longer duration)
    if (!onClick || duration > 0) {
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, duration);
    }
}

// Keyboard shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
        const ctrlKey = isMac ? e.metaKey : e.ctrlKey;

        // Ctrl/Cmd + K: Focus search
        if (ctrlKey && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('sidebarSearchInput');
            if (searchInput) {
                const sidebar = document.getElementById('sidebar');
                if (sidebar) sidebar.classList.add('open');
                searchInput.focus();
            }
        }

        // Ctrl/Cmd + /: Toggle settings
        if (ctrlKey && e.key === '/') {
            e.preventDefault();
            const settingsPanel = document.getElementById('settingsPanel');
            if (settingsPanel) {
                if (settingsPanel.classList.contains('open')) {
                    closeSettings();
                } else {
                    openSettings();
                }
            }
        }

        // Ctrl/Cmd + N: New conversation
        if (ctrlKey && e.key === 'n') {
            e.preventDefault();
            saveConversation();
            newConversation();
            showToast('New conversation started', 'success');
        }

        // Esc: Close panels
        if (e.key === 'Escape') {
            const settingsPanel = document.getElementById('settingsPanel');
            const sidebar = document.getElementById('sidebar');
            if (settingsPanel && settingsPanel.classList.contains('open')) {
                closeSettings();
            } else if (sidebar && sidebar.classList.contains('open')) {
                sidebar.classList.remove('open');
            }
        }
    });
}

// Web search indicator
function addWebSearchIndicator(messageElement) {
    if (!enableWebSearch) return;

    const indicator = document.createElement('div');
    indicator.className = 'web-search-indicator';
    indicator.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M7 12A5 5 0 1 0 7 2a5 5 0 0 0 0 10zM13 13l-3-3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        <span>Web Search Enabled</span>
    `;
    messageElement.appendChild(indicator);
}

// Image generation handler
async function generateImage(prompt) {
    if (!enableImageGeneration) {
        showToast('Image generation is disabled. Enable it in settings.', 'warning');
        return;
    }

    showToast('Generating image...', 'info', 5000);

    try {
        // This would call the backend API when implemented
        const imageAPI = `${FINAL_API_URL.replace('/api/query', '/api/image/generate')}`;
        const response = await fetch(imageAPI, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: prompt })
        });

        if (!response.ok) {
            throw new Error('Image generation failed');
        }

        const data = await response.json();

        // Display generated image
        if (data.image_url || data.image_data) {
            const imageUrl = data.image_url || `data:image/png;base64,${data.image_data}`;
            const messageDiv = addMessage('', 'assistant');
            const contentDiv = messageDiv.querySelector('.message-content');
            const img = document.createElement('img');
            img.src = imageUrl;
            img.alt = prompt;
            img.className = 'generated-image';
            contentDiv.appendChild(img);
            showToast('Image generated successfully', 'success');
        }
    } catch (error) {
        console.error('Image generation error:', error);
        showToast('Image generation failed. Backend not implemented yet.', 'error');
    }
}

// Detect image generation requests
function detectImageGenerationRequest(query) {
    const imagePatterns = [
        /generate (an |a )?image (of |showing )?/i,
        /create (an |a )?image (of |showing )?/i,
        /draw (an |a )?/i,
        /show me (an |a )?image (of |showing )?/i,
        /^image:/i
    ];

    return imagePatterns.some(pattern => pattern.test(query));
}

// Prompt Carousel
const PROMPTS = [
    "What is the nature of consciousness?",
    "Design a device that solves climate change",
    "Analyze the patterns in quantum mechanics",
    "What are the origins of the cosmos?",
    "Create a truth-finding algorithm",
    "How does intelligence emerge?",
    "Map the connections between astrology and organs",
    "What is the purpose of existence?",
    "Design a system that learns infinitely",
    "What patterns exist in the universe?",
    "How can we achieve AGI?",
    "What is the relationship between mind and matter?",
    "Create a device that reads thoughts",
    "What is the meaning of life?",
    "How does the universe compute?",
    "Design a civilization of AI agents",
    "What is the nature of truth?",
    "How can we transcend limitations?",
    "What is the connection between all things?",
    "Create a system that evolves autonomously"
];

let currentPromptIndex = 0; // Track current prompt index
let promptCarouselInterval = null;

function initPromptCarousel() {
    const carouselContent = document.getElementById('promptCarouselContent');
    const prevButton = document.getElementById('promptCarouselPrev');
    const nextButton = document.getElementById('promptCarouselNext');
    const queryInput = document.getElementById('queryInput');
    const carousel = document.getElementById('promptCarousel');

    if (!carouselContent || !carousel) {
        console.error('Prompt carousel elements not found:', {
            carousel: !!carousel,
            content: !!carouselContent
        });
        return;
    }

    // Clear any existing items
    carouselContent.innerHTML = '';

    // Ensure carousel is visible
    carousel.style.display = 'flex';
    carousel.style.visibility = 'visible';
    carousel.style.opacity = '1';

    // Ensure content is visible
    carouselContent.style.display = 'flex';
    carouselContent.style.visibility = 'visible';
    carouselContent.style.opacity = '1';

    // Check if PROMPTS exists
    if (!PROMPTS || PROMPTS.length === 0) {
        console.error('PROMPTS array is empty or undefined');
        return;
    }

    // Create prompt items
    PROMPTS.forEach((prompt, index) => {
        const item = document.createElement('div');
        item.className = 'prompt-item';
        if (index === 0) {
            item.classList.add('active');
        }
        item.textContent = prompt;

        // Set visibility for first item - ensure it's visible
        if (index === 0) {
            item.style.cssText = `
                position: absolute;
                width: 100%;
                text-align: center;
                padding: 0 1rem;
                opacity: 1 !important;
                visibility: visible !important;
                transform: translateX(0);
                pointer-events: auto;
                z-index: 10;
                font-size: 0.875rem;
                color: var(--text-secondary);
                cursor: pointer;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            `;
        } else {
            item.style.cssText = `
                position: absolute;
                width: 100%;
                text-align: center;
                padding: 0 1rem;
                opacity: 0;
                visibility: visible;
                transform: translateX(20px);
                pointer-events: none;
                z-index: 1;
                font-size: 0.875rem;
                color: var(--text-secondary);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            `;
        }

        item.addEventListener('click', () => {
            if (queryInput) {
                queryInput.value = prompt;
                queryInput.focus();
            }
        });
        carouselContent.appendChild(item);
    });

    const itemsCreated = carouselContent.querySelectorAll('.prompt-item').length;
    let firstItem = carouselContent.querySelector('.prompt-item.active');

    console.log('Prompt carousel initialized:', {
        itemsCreated: itemsCreated,
        firstItem: !!firstItem,
        carouselVisible: window.getComputedStyle(carousel).display !== 'none',
        contentVisible: window.getComputedStyle(carouselContent).display !== 'none',
        PROMPTSLength: PROMPTS ? PROMPTS.length : 0,
        PROMPTS: PROMPTS ? PROMPTS.slice(0, 3) : 'undefined'
    });

    // Force show first prompt if carousel is still empty
    if (itemsCreated === 0 && PROMPTS && PROMPTS.length > 0) {
        console.log('Creating fallback prompt item');
        const fallbackItem = document.createElement('div');
        fallbackItem.className = 'prompt-item active';
        fallbackItem.textContent = PROMPTS[0];
        fallbackItem.style.cssText = `
            opacity: 1;
            visibility: visible;
            transform: translateX(0);
            pointer-events: auto;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            cursor: pointer;
            color: white;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        `;
        fallbackItem.addEventListener('click', () => {
            const queryInput = document.getElementById('queryInput');
            if (queryInput) {
                queryInput.value = PROMPTS[0];
                queryInput.focus();
            }
        });
        carouselContent.appendChild(fallbackItem);
        if (DEBUG_MODE) console.log('Fallback prompt item created');
    }

    // If no items created, try creating a test item
    if (itemsCreated === 0) {
        if (DEBUG_MODE) console.error('No items created! Creating test item...');
        const testItem = document.createElement('div');
        testItem.className = 'prompt-item active';
        testItem.textContent = 'TEST PROMPT - If you see this, items can be created';
        testItem.style.cssText = 'position: absolute; width: 100%; text-align: center; opacity: 1; visibility: visible; color: white;';
        carouselContent.appendChild(testItem);
        firstItem = testItem;
    }

    // Ensure first item is visible
    if (!firstItem) {
        firstItem = carouselContent.querySelector('.prompt-item.active');
    }
    if (firstItem) {
        firstItem.style.opacity = '1';
        firstItem.style.visibility = 'visible';
        firstItem.style.transform = 'translateX(0)';
        firstItem.style.pointerEvents = 'auto';
    }

    function showPrompt(index) {
        const items = carouselContent.querySelectorAll('.prompt-item');
        items.forEach((item, i) => {
            if (i === index) {
                item.classList.add('active');
                item.style.opacity = '1';
                item.style.visibility = 'visible';
                item.style.transform = 'translateX(0)';
                item.style.zIndex = '10';
                item.style.pointerEvents = 'auto';
            } else {
                item.classList.remove('active');
                item.style.opacity = '0';
                item.style.transform = 'translateX(20px)';
                item.style.zIndex = '1';
                item.style.pointerEvents = 'none';
            }
        });
        currentPromptIndex = index;
    }

    function nextPrompt() {
        const nextIndex = (currentPromptIndex + 1) % PROMPTS.length;
        showPrompt(nextIndex);
    }

    function prevPrompt() {
        const prevIndex = (currentPromptIndex - 1 + PROMPTS.length) % PROMPTS.length;
        showPrompt(prevIndex);
    }

    // Auto-rotate every 4 seconds
    function startAutoRotate() {
        if (promptCarouselInterval) clearInterval(promptCarouselInterval);
        promptCarouselInterval = setInterval(nextPrompt, 4000);
    }

    function stopAutoRotate() {
        if (promptCarouselInterval) {
            clearInterval(promptCarouselInterval);
            promptCarouselInterval = null;
        }
    }

    // Button handlers
    if (nextButton) {
        nextButton.addEventListener('click', () => {
            nextPrompt();
            stopAutoRotate();
            startAutoRotate();
        });
    }

    if (prevButton) {
        prevButton.addEventListener('click', () => {
            prevPrompt();
            stopAutoRotate();
            startAutoRotate();
        });
    }

    // Pause on hover (carousel already defined at top of function)
    if (carousel) {
        carousel.addEventListener('mouseenter', stopAutoRotate);
        carousel.addEventListener('mouseleave', startAutoRotate);
    }

    // Start auto-rotate
    startAutoRotate();
}

// Auto-retract sidebar
function initAutoRetractSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mainWrapper = document.querySelector('.main-wrapper');

    if (!sidebar || !mainWrapper) return;

    let retractTimeout = null;
    const RETRACT_DELAY = 3000; // 3 seconds on mobile, 5 seconds on desktop
    const isMobile = window.innerWidth <= 768;

    function scheduleRetract() {
        if (retractTimeout) clearTimeout(retractTimeout);

        if (isMobile) {
            // Auto-retract on mobile after delay
            retractTimeout = setTimeout(() => {
                if (sidebar.classList.contains('open')) {
                    sidebar.classList.remove('open');
                }
            }, RETRACT_DELAY);
        }
    }

    function cancelRetract() {
        if (retractTimeout) {
            clearTimeout(retractTimeout);
            retractTimeout = null;
        }
    }

    // Retract when clicking outside on mobile
    if (isMobile) {
        mainWrapper.addEventListener('click', (e) => {
            if (sidebar.classList.contains('open') && !sidebar.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });

        // Retract when interacting with main content
        const chatContainer = document.getElementById('chatContainer');
        const inputContainer = document.querySelector('.input-container');

        [chatContainer, inputContainer].forEach(container => {
            if (container) {
                container.addEventListener('click', () => {
                    if (sidebar.classList.contains('open')) {
                        sidebar.classList.remove('open');
                    }
                });
            }
        });
    }

    // Schedule retract when sidebar opens
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'class') {
                if (sidebar.classList.contains('open')) {
                    scheduleRetract();
                } else {
                    cancelRetract();
                }
            }
        });
    });

    observer.observe(sidebar, { attributes: true });

    // Cancel retract on sidebar interaction
    sidebar.addEventListener('mouseenter', cancelRetract);
    sidebar.addEventListener('mouseleave', scheduleRetract);
    sidebar.addEventListener('click', cancelRetract);
}

// Neural Network Background
class NeuralNetworkBackground {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.nodes = [];
        this.connections = [];
        this.animationId = null;
        this.isActive = false;

        this.setupCanvas();
        this.createNetwork();
        this.animate();

        // Listen for ICEBURG activity
        this.setupActivityListeners();
    }

    setupCanvas() {
        const resize = () => {
            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
        };
        resize();
        window.addEventListener('resize', resize);
    }

    createNetwork() {
        const nodeCount = Math.floor((this.canvas.width * this.canvas.height) / 50000);
        const layers = 4;
        const nodesPerLayer = Math.ceil(nodeCount / layers);

        this.nodes = [];
        this.connections = [];

        // Create nodes in layers
        for (let layer = 0; layer < layers; layer++) {
            const layerNodes = [];
            const x = (this.canvas.width / (layers + 1)) * (layer + 1);

            for (let i = 0; i < nodesPerLayer; i++) {
                const y = (this.canvas.height / (nodesPerLayer + 1)) * (i + 1) +
                    (Math.random() - 0.5) * (this.canvas.height / nodesPerLayer);

                const node = {
                    x: x + (Math.random() - 0.5) * 50,
                    y: y,
                    radius: 2 + Math.random() * 2,
                    baseRadius: 2 + Math.random() * 2,
                    pulse: Math.random() * Math.PI * 2,
                    pulseSpeed: 0.02 + Math.random() * 0.03,
                    intensity: 0,
                    targetIntensity: 0,
                    layer: layer
                };

                layerNodes.push(node);
                this.nodes.push(node);
            }

            // Connect to previous layer
            if (layer > 0) {
                const prevLayerNodes = this.nodes.filter(n => n.layer === layer - 1);
                layerNodes.forEach(node => {
                    // Connect to 2-4 random nodes in previous layer
                    const connections = 2 + Math.floor(Math.random() * 3);
                    const selected = [];
                    for (let i = 0; i < connections; i++) {
                        const randomNode = prevLayerNodes[Math.floor(Math.random() * prevLayerNodes.length)];
                        if (!selected.includes(randomNode)) {
                            selected.push(randomNode);
                            this.connections.push({
                                from: randomNode,
                                to: node,
                                intensity: 0,
                                targetIntensity: 0,
                                pulse: Math.random() * Math.PI * 2,
                                pulseSpeed: 0.01 + Math.random() * 0.02
                            });
                        }
                    }
                });
            }
        }
    }

    setupActivityListeners() {
        // Monitor query input for activity
        const queryInput = document.getElementById('queryInput');
        if (queryInput) {
            queryInput.addEventListener('input', () => {
                this.activate();
            });
        }

        // Monitor send button
        const sendButton = document.getElementById('sendButton');
        if (sendButton) {
            sendButton.addEventListener('click', () => {
                this.activate();
            });
        }

        // Listen for SSE messages via handleStreamingMessage
        // WebSocket removed - all messages come through V2 SSE endpoint
        if (typeof handleStreamingMessage === 'function') {
            // Messages are handled globally via handleStreamingMessage
            // This class can listen for specific message types if needed
        }
    }

    activate() {
        this.isActive = true;
        this.canvas.classList.add('active');

        // Activate random nodes and connections
        const activeNodes = Math.floor(this.nodes.length * 0.3);
        for (let i = 0; i < activeNodes; i++) {
            const node = this.nodes[Math.floor(Math.random() * this.nodes.length)];
            node.targetIntensity = 0.5 + Math.random() * 0.5;
        }

        const activeConnections = Math.floor(this.connections.length * 0.2);
        for (let i = 0; i < activeConnections; i++) {
            const conn = this.connections[Math.floor(Math.random() * this.connections.length)];
            conn.targetIntensity = 0.3 + Math.random() * 0.4;
        }

        // Gradually fade out
        setTimeout(() => {
            this.isActive = false;
            this.canvas.classList.remove('active');
            this.nodes.forEach(node => {
                node.targetIntensity = 0;
            });
            this.connections.forEach(conn => {
                conn.targetIntensity = 0;
            });
        }, 2000);
    }

    animate() {
        // Check if canvas is still valid and visible
        if (!this.canvas || !this.ctx) {
            return;
        }

        try {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

            // Update nodes
            this.nodes.forEach(node => {
                // Pulse animation
                node.pulse += node.pulseSpeed;
                node.radius = node.baseRadius + Math.sin(node.pulse) * 0.5;

                // Intensity interpolation
                node.intensity += (node.targetIntensity - node.intensity) * 0.1;
            });

            // Update connections
            this.connections.forEach(conn => {
                conn.pulse += conn.pulseSpeed;
                conn.intensity += (conn.targetIntensity - conn.intensity) * 0.1;
            });

            // Draw connections
            this.connections.forEach(conn => {
                const alpha = conn.intensity * 0.3;
                if (alpha > 0.01) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(conn.from.x, conn.from.y);
                    this.ctx.lineTo(conn.to.x, conn.to.y);

                    // Glow effect for active connections
                    const glowIntensity = conn.intensity * (0.5 + Math.sin(conn.pulse) * 0.3);
                    this.ctx.strokeStyle = `rgba(255, 255, 255, ${alpha * glowIntensity})`;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.stroke();

                    // Subtle glow
                    if (glowIntensity > 0.2) {
                        this.ctx.shadowBlur = 3;
                        this.ctx.shadowColor = 'rgba(255, 255, 255, 0.3)';
                        this.ctx.stroke();
                        this.ctx.shadowBlur = 0;
                    }
                }
            });

            // Draw nodes
            this.nodes.forEach(node => {
                const alpha = 0.2 + node.intensity * 0.8;
                if (alpha > 0.01) {
                    this.ctx.beginPath();
                    this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);

                    // Fill
                    this.ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
                    this.ctx.fill();

                    // Glow for active nodes
                    if (node.intensity > 0.3) {
                        this.ctx.shadowBlur = node.radius * 2;
                        this.ctx.shadowColor = 'rgba(255, 255, 255, 0.5)';
                        this.ctx.fill();
                        this.ctx.shadowBlur = 0;
                    }
                }
            });

            // Subtle random activation
            if (!this.isActive && Math.random() < 0.01) {
                const randomNode = this.nodes[Math.floor(Math.random() * this.nodes.length)];
                randomNode.targetIntensity = 0.2 + Math.random() * 0.3;
                setTimeout(() => {
                    randomNode.targetIntensity = 0;
                }, 1000);
            }

            this.animationId = requestAnimationFrame(() => this.animate());
        } catch (e) {
            console.error('NeuralNetworkBackground animation error:', e);
            // Stop animation on error to prevent infinite error loop
            if (this.animationId) {
                cancelAnimationFrame(this.animationId);
                this.animationId = null;
            }
        }
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

let neuralNetwork = null;
let meteorShower = null;

// Meteor Shower Background for Chat Container
class MeteorShowerBackground {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.meteors = [];
        this.stars = [];
        this.animationId = null;

        this.setupCanvas();
        this.createStars();
        this.animate();

        // Handle resize
        window.addEventListener('resize', () => this.setupCanvas());
    }

    setupCanvas() {
        const container = this.canvas.parentElement;
        if (!container) return;

        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }

    createStars() {
        const starCount = Math.floor((this.canvas.width * this.canvas.height) / 15000);
        this.stars = [];

        for (let i = 0; i < starCount; i++) {
            this.stars.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                size: Math.random() * 1.5 + 0.5,
                opacity: Math.random() * 0.5 + 0.2,
                twinkle: Math.random() * Math.PI * 2,
                twinkleSpeed: 0.01 + Math.random() * 0.02
            });
        }
    }

    createMeteor() {
        const side = Math.floor(Math.random() * 4); // 0=top, 1=right, 2=bottom, 3=left
        let x, y, vx, vy;

        switch (side) {
            case 0: // Top
                x = Math.random() * this.canvas.width;
                y = -10;
                vx = (Math.random() - 0.5) * 2;
                vy = Math.random() * 3 + 2;
                break;
            case 1: // Right
                x = this.canvas.width + 10;
                y = Math.random() * this.canvas.height;
                vx = -(Math.random() * 3 + 2);
                vy = (Math.random() - 0.5) * 2;
                break;
            case 2: // Bottom
                x = Math.random() * this.canvas.width;
                y = this.canvas.height + 10;
                vx = (Math.random() - 0.5) * 2;
                vy = -(Math.random() * 3 + 2);
                break;
            case 3: // Left
                x = -10;
                y = Math.random() * this.canvas.height;
                vx = Math.random() * 3 + 2;
                vy = (Math.random() - 0.5) * 2;
                break;
        }

        return {
            x: x,
            y: y,
            vx: vx,
            vy: vy,
            length: Math.random() * 30 + 20,
            speed: Math.sqrt(vx * vx + vy * vy),
            opacity: Math.random() * 0.4 + 0.3,
            life: 1.0,
            decay: 0.01 + Math.random() * 0.02
        };
    }

    animate() {
        // Check if canvas is still valid and visible
        if (!this.canvas || !this.ctx || !this.canvas.parentElement) {
            return;
        }

        try {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

            // Update and draw stars
            this.stars.forEach(star => {
                star.twinkle += star.twinkleSpeed;
                const twinkleOpacity = star.opacity + Math.sin(star.twinkle) * 0.2;

                this.ctx.beginPath();
                this.ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
                this.ctx.fillStyle = `rgba(255, 255, 255, ${Math.max(0, Math.min(1, twinkleOpacity))})`;
                this.ctx.fill();
            });

            // Spawn new meteors occasionally
            if (Math.random() < 0.02) {
                this.meteors.push(this.createMeteor());
            }

            // Limit meteors to prevent memory issues
            if (this.meteors.length > 20) {
                this.meteors = this.meteors.slice(-20);
            }

            // Update and draw meteors
            this.meteors = this.meteors.filter(meteor => {
                meteor.x += meteor.vx;
                meteor.y += meteor.vy;
                meteor.life -= meteor.decay;

                // Draw meteor trail
                if (meteor.life > 0 &&
                    meteor.x > -50 && meteor.x < this.canvas.width + 50 &&
                    meteor.y > -50 && meteor.y < this.canvas.height + 50) {

                    const gradient = this.ctx.createLinearGradient(
                        meteor.x - meteor.vx * meteor.length / meteor.speed,
                        meteor.y - meteor.vy * meteor.length / meteor.speed,
                        meteor.x,
                        meteor.y
                    );

                    gradient.addColorStop(0, `rgba(255, 255, 255, 0)`);
                    gradient.addColorStop(0.5, `rgba(255, 255, 255, ${meteor.opacity * meteor.life * 0.3})`);
                    gradient.addColorStop(1, `rgba(255, 255, 255, ${meteor.opacity * meteor.life})`);

                    this.ctx.beginPath();
                    this.ctx.moveTo(
                        meteor.x - meteor.vx * meteor.length / meteor.speed,
                        meteor.y - meteor.vy * meteor.length / meteor.speed
                    );
                    this.ctx.lineTo(meteor.x, meteor.y);
                    this.ctx.strokeStyle = gradient;
                    this.ctx.lineWidth = 1.5;
                    this.ctx.stroke();

                    // Draw meteor head
                    this.ctx.beginPath();
                    this.ctx.arc(meteor.x, meteor.y, 2, 0, Math.PI * 2);
                    this.ctx.fillStyle = `rgba(255, 255, 255, ${meteor.opacity * meteor.life})`;
                    this.ctx.fill();

                    return true;
                }
                return false;
            });

            this.animationId = requestAnimationFrame(() => this.animate());
        } catch (e) {
            console.error('MeteorShowerBackground animation error:', e);
            // Stop animation on error to prevent infinite error loop
            if (this.animationId) {
                cancelAnimationFrame(this.animationId);
                this.animationId = null;
            }
        }
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Load ICEBURG configuration first
    loadICEBURGConfig();

    // Immediate debug check
    console.log('🚀 ICEBURG Frontend initializing...');
    console.log('🔍 DOM elements check:', {
        modeSelect: !!document.getElementById('modeSelect'),
        agentSelect: !!document.getElementById('agentSelect'),
        sidebarToggle: !!document.getElementById('sidebarToggle'),
        visualizationPanels: !!document.getElementById('visualizationPanels'),
        promptCarousel: !!document.getElementById('promptCarousel'),
        iceburgConfig: !!iceburgConfig
    });

    // Initialize client-side processor (leverages user's device)
    try {
        clientProcessor = new ClientProcessor();
        console.log('✅ Client-side processor initialized:', {
            isMobile: clientProcessor.isMobile,
            capabilities: clientProcessor.deviceCapabilities
        });
    } catch (e) {
        console.warn('Failed to initialize client-side processor:', e);
    }

    const sendButton = document.getElementById('sendButton');
    const queryInput = document.getElementById('queryInput');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const newConversationBtn = document.getElementById('newConversationBtn');
    const sidebar = document.getElementById('sidebar');

    // Initialize neural network background
    const canvas = document.getElementById('neuralNetworkCanvas');
    if (canvas) {
        try {
            neuralNetwork = new NeuralNetworkBackground(canvas);

            // Pause animation when page is hidden to save resources
            document.addEventListener('visibilitychange', () => {
                if (document.hidden && neuralNetwork) {
                    if (neuralNetwork.animationId) {
                        cancelAnimationFrame(neuralNetwork.animationId);
                        neuralNetwork.animationId = null;
                    }
                } else if (!document.hidden && neuralNetwork && !neuralNetwork.animationId) {
                    neuralNetwork.animate();
                }
            });
        } catch (e) {
            console.error('Error initializing neural network background:', e);
        }
    }

    // Initialize mode UI if astro-physiology is already selected
    const initialModeSelect = document.getElementById('modeSelect');
    if (initialModeSelect) {
        // Check initial value
        if (initialModeSelect.value === 'astrophysiology') {
            setTimeout(() => {
                console.log('🔵 Initializing astro-physiology UI on page load');
                astro_updateModeUI('astrophysiology');
            }, 200);
        }

        // Also trigger on any mode change to ensure it works
        initialModeSelect.addEventListener('change', () => {
            setTimeout(() => {
                if (initialModeSelect.value === 'astrophysiology') {
                    console.log('🔵 Mode changed to astro-physiology, updating UI');
                    astro_updateModeUI('astrophysiology');
                }
            }, 50);
        });
    }


    // Initialize meteor shower background for chat container
    const meteorCanvas = document.getElementById('meteorShowerCanvas');
    const chatContainer = document.getElementById('chatContainer');
    if (meteorCanvas && chatContainer) {
        // Wait a bit for layout to settle
        setTimeout(() => {
            try {
                meteorShower = new MeteorShowerBackground(meteorCanvas);

                // Update canvas size when container resizes
                const resizeObserver = new ResizeObserver(() => {
                    if (meteorShower && meteorShower.canvas) {
                        try {
                            meteorShower.setupCanvas();
                            meteorShower.createStars();
                        } catch (e) {
                            console.error('Error resizing meteor shower canvas:', e);
                        }
                    }
                });
                resizeObserver.observe(chatContainer);

                // Pause animation when page is hidden to save resources
                document.addEventListener('visibilitychange', () => {
                    if (document.hidden && meteorShower) {
                        if (meteorShower.animationId) {
                            cancelAnimationFrame(meteorShower.animationId);
                            meteorShower.animationId = null;
                        }
                    } else if (!document.hidden && meteorShower && !meteorShower.animationId) {
                        meteorShower.animate();
                    }
                });
            } catch (e) {
                console.error('Error initializing meteor shower background:', e);
            }
        }, 100);
    }
    const attachButton = document.getElementById('attachButton');
    const fileInput = document.getElementById('fileInput');
    const settingsToggle = document.getElementById('settingsToggle');
    const settingsClose = document.getElementById('settingsClose');
    const settingsOverlay = document.getElementById('settingsOverlay');
    const primaryModel = document.getElementById('primaryModel');
    const temperature = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperatureValue');
    const maxTokens = document.getElementById('maxTokens');
    const maxTokensValue = document.getElementById('maxTokensValue');
    const exportBtn = document.getElementById('exportConversation');
    const shareBtn = document.getElementById('shareConversation');
    const clearBtn = document.getElementById('clearConversations');
    const sidebarSearchInput = document.getElementById('sidebarSearchInput');
    const sidebarSearchClear = document.getElementById('sidebarSearchClear');
    const voiceButton = document.getElementById('voiceButton');
    const enableWebSearchCheckbox = document.getElementById('enableWebSearch');
    const enableImageGenerationCheckbox = document.getElementById('enableImageGeneration');

    // Load settings
    loadSettings();

    // Load saved birth data for astro-physiology mode (using new key)
    astro_loadSavedBirthData();

    // Initialize voice recognition
    initVoiceRecognition();

    // Setup keyboard shortcuts
    setupKeyboardShortcuts();

    // Load conversations from localStorage
    const saved = localStorage.getItem('iceburg_conversations');
    if (saved) {
        try {
            conversations = JSON.parse(saved);
        } catch (e) {
            conversations = [];
        }
    }
    updateConversationsList();

    // Initialize prompt carousel immediately
    initPromptCarousel();

    // Also try after a short delay to ensure DOM is ready
    setTimeout(() => {
        initPromptCarousel();
    }, 100);

    // Initialize auto-retract sidebar
    initAutoRetractSidebar();

    // Sidebar toggle
    if (sidebarToggle && sidebar) {
        console.log('🔧 Setting up sidebar toggle:', { sidebarToggle, sidebar });
        sidebarToggle.addEventListener('click', (e) => {
            console.log('🖱️ Sidebar toggle clicked');
            e.preventDefault();
            e.stopPropagation();
            const wasOpen = sidebar.classList.contains('open');
            sidebar.classList.toggle('open');
            const isOpen = sidebar.classList.contains('open');
            console.log('🔄 Sidebar toggle result:', { wasOpen, isOpen, classes: sidebar.className });
        });
        // Also add click handler to the header content area to close sidebar
        const headerContent = document.querySelector('.header-content');
        if (headerContent) {
            headerContent.addEventListener('click', () => {
                if (sidebar.classList.contains('open')) {
                    sidebar.classList.remove('open');
                }
            });
        }
    } else {
        console.error('Sidebar toggle elements not found:', { sidebarToggle: !!sidebarToggle, sidebar: !!sidebar });
    }

    // New conversation button
    if (newConversationBtn) {
        newConversationBtn.addEventListener('click', () => {
            saveConversation();
            newConversation();
        });
    }

    // File attachment
    if (attachButton && fileInput) {
        attachButton.addEventListener('click', () => {
            fileInput.click();
        });
        fileInput.addEventListener('change', handleFileSelect);
    }


    // Mode selection handler for visualization panels and agent dropdown
    const modeSelect = document.getElementById('modeSelect');

    // Check for investigation URL parameter (Resume Context)
    const urlParams = new URLSearchParams(window.location.search);
    const investigationId = urlParams.get('investigation');

    if (investigationId) {
        console.log('📂 Resuming investigation:', investigationId);

        // Auto-switch to dossier mode
        if (modeSelect) {
            modeSelect.value = 'dossier';
            // Update UI manually since change event won't fire yet
            if (typeof updateAgentOptionsForMode === 'function') {
                updateAgentOptionsForMode('dossier');
            }
        }

        // Add system message to chat
        setTimeout(() => {
            const chatContainer = document.getElementById('chatContainer');
            if (chatContainer) {
                const sysMsg = document.createElement('div');
                sysMsg.className = 'message system-message'; // You might need to add this class to CSS
                sysMsg.innerHTML = `
                    <div class="message-content">
                        <strong>📂 Resuming Investigation Context</strong><br>
                        Loading investigation ID: <code>${investigationId}</code><br>
                        <em>You can now ask follow-up questions about this dossier.</em>
                    </div>
                `;
                chatContainer.appendChild(sysMsg);
                // Also add to conversation history if needed
            }

            // Set a global flag or config to send with the first message/connection
            window.activeInvestigationId = investigationId;

            // We need to tell the server about this context. 
            // We can do this by sending a hidden 'context_update' message once WS connects
            // OR simply piggyback on the first user message. 
            // Better: Send a "handshake" message when WS opens if we have this ID.
        }, 1000); // Wait for UI to settle
    }

    const agentSelect = document.getElementById('agentSelect');
    const agentSelectGroup = agentSelect ? agentSelect.closest('.control-group') : null;
    const visualizationPanels = document.getElementById('visualizationPanels');

    // Modes that should show agent dropdown
    const modesWithAgents = ['research', 'chat', 'device', 'truth', 'swarm', 'gnosis', 'civilization', 'astrophysiology'];

    // Initialize agent dropdown visibility on page load
    if (modeSelect && agentSelect) {
        const initialMode = modeSelect.value;
        console.log('🎬 Initial mode:', initialMode);
        const shouldShowInitially = modesWithAgents.includes(initialMode);
        console.log('🎬 Should show agent initially:', shouldShowInitially);

        if (!shouldShowInitially) {
            if (agentSelectGroup) {
                agentSelectGroup.style.display = 'none';
                agentSelectGroup.style.visibility = 'hidden';
                agentSelectGroup.style.opacity = '0';
                console.log('❌ Agent group hidden on init');
            }
            if (agentSelect) {
                agentSelect.style.display = 'none';
                agentSelect.style.visibility = 'hidden';
                agentSelect.style.opacity = '0';
                console.log('❌ Agent select hidden on init');
            }
        } else {
            if (agentSelectGroup) {
                agentSelectGroup.style.display = 'flex';
                agentSelectGroup.style.visibility = 'visible';
                agentSelectGroup.style.opacity = '1';
                console.log('✅ Agent group visible on init');
            }
            if (agentSelect) {
                agentSelect.style.display = 'block';
                agentSelect.style.visibility = 'visible';
                agentSelect.style.opacity = '1';
                console.log('✅ Agent select visible on init');
            }
        }
    }

    if (modeSelect && visualizationPanels) {
        console.log('📋 Setting up mode selection handler');

        // Initialize agent options on page load
        if (modeSelect && typeof updateAgentOptionsForMode === 'function') {
            const initialMode = modeSelect.value || 'chat';
            updateAgentOptionsForMode(initialMode);
            // Initialize agent display for chat mode
            const agentSelect = document.getElementById('agentSelect');
            const agentDisplay = document.getElementById('agentDisplay');
            const agentDisplayValue = document.getElementById('agentDisplayValue');

            if (initialMode === 'chat') {
                if (agentSelect) agentSelect.style.display = 'none';
                if (agentDisplay) {
                    agentDisplay.style.display = 'flex';
                    if (agentDisplayValue) agentDisplayValue.textContent = 'Secretary';
                }
                console.log('👔 Initial chat mode: Showing agent display (Secretary)');
            } else {
                if (agentSelect) agentSelect.style.display = 'block';
                if (agentDisplay) agentDisplay.style.display = 'none';
            }
        }

        modeSelect.addEventListener('change', (e) => {
            const selectedMode = e.target.value;
            console.log('🔄 Mode changed to:', selectedMode);

            const agentSelect = document.getElementById('agentSelect');
            const agentDisplay = document.getElementById('agentDisplay');
            const agentDisplayValue = document.getElementById('agentDisplayValue');

            // In chat mode: Show agent display, hide dropdown
            if (selectedMode === 'chat') {
                if (agentSelect) agentSelect.style.display = 'none';
                if (agentDisplay) {
                    agentDisplay.style.display = 'flex';
                    if (agentDisplayValue) agentDisplayValue.textContent = 'Secretary';
                }
                console.log('👔 Chat mode: Showing agent display (Secretary)');
            } else {
                // Other modes: Show dropdown, hide display
                if (agentSelect) agentSelect.style.display = 'block';
                if (agentDisplay) agentDisplay.style.display = 'none';
                if (agentSelect) {
                    agentSelect.disabled = false;
                    agentSelect.title = '';
                }
                console.log('🔄 Other mode: Showing agent dropdown');
            }

            // V2: Update agent options based on mode (before visibility check)
            if (typeof updateAgentOptionsForMode === 'function') {
                updateAgentOptionsForMode(selectedMode);
            }

            console.log('🎯 Agent dropdown should show for modes:', modesWithAgents);

            // Handle agent dropdown visibility
            const shouldShowAgent = modesWithAgents.includes(selectedMode);
            console.log('🤖 Should show agent dropdown:', shouldShowAgent);

            if (shouldShowAgent) {
                if (agentSelectGroup) {
                    agentSelectGroup.style.display = 'flex';
                    agentSelectGroup.style.visibility = 'visible';
                    agentSelectGroup.style.opacity = '1';
                    console.log('✅ Agent group set to visible (flex)');
                }
                if (agentSelect) {
                    agentSelect.style.display = 'block';
                    agentSelect.style.visibility = 'visible';
                    agentSelect.style.opacity = '1';
                    console.log('✅ Agent select set to visible (block)');
                }
            } else {
                if (agentSelectGroup) {
                    agentSelectGroup.style.display = 'none';
                    agentSelectGroup.style.visibility = 'hidden';
                    agentSelectGroup.style.opacity = '0';
                    console.log('❌ Agent group set to hidden (none)');
                }
                if (agentSelect) {
                    agentSelect.style.display = 'none';
                    agentSelect.style.visibility = 'hidden';
                    agentSelect.style.opacity = '0';
                    console.log('❌ Agent select set to hidden (none)');
                }
            }

            // Handle visualization panels - HIDDEN BY DEFAULT, only show when Dashboard button is clicked
            // Don't auto-show panels when switching modes
            if (selectedMode === 'astrophysiology') {
                // Just load saved birth data, but don't show panels
                astro_loadSavedBirthData();
                // Hide panels by default
                visualizationPanels.style.display = 'none';
                visualizationPanels.classList.remove('show');
            } else if (selectedMode === 'prediction_lab') {
                // Only show for prediction_lab mode if explicitly needed
                visualizationPanels.style.display = 'none';
                visualizationPanels.classList.remove('show');
            } else {
                // Hide panels for all other modes
                visualizationPanels.style.display = 'none';
                visualizationPanels.classList.remove('show');
            }

            // Show mode-specific UI enhancements for astro-physiology
            astro_updateModeUI(selectedMode);
        });
    }

    // Astro-Physiology Panel Event Handlers
    const calculateImprintBtn = document.getElementById('calculateImprintBtn');
    const runPredictionBtn = document.getElementById('runPredictionBtn');
    const testHypothesisBtn = document.getElementById('testHypothesisBtn');
    const generateHypothesisBtn = document.getElementById('generateHypothesisBtn');
    const birthDateInput = document.getElementById('birthDateInput');
    const locationInput = document.getElementById('locationInput');
    const molecularResults = document.getElementById('molecularResults');
    const predictionResults = document.getElementById('predictionResults');
    const hypothesisResults = document.getElementById('hypothesisResults');

    if (calculateImprintBtn) {
        calculateImprintBtn.addEventListener('click', async () => {
            const birthDate = birthDateInput ? birthDateInput.value : '';
            const location = locationInput ? locationInput.value : '';

            if (!birthDate) {
                showToast('Please enter a birth date', 'error');
                return;
            }

            if (molecularResults) {
                molecularResults.innerHTML = '<div style="color: #ccc;">Calculating molecular imprint...</div>';
            }

            // Redirect to chat interface with astro-physiology mode
            const modeSelect = document.getElementById('modeSelect');
            const queryInput = document.getElementById('queryInput');

            if (modeSelect && queryInput) {
                // Set mode to astro-physiology
                modeSelect.value = 'astrophysiology';
                astro_updateModeUI('astrophysiology');

                // Create query with birth data
                const birthTimeInput = document.getElementById('birthTimeInput');
                const birthTime = birthTimeInput ? birthTimeInput.value : '12:00';
                const query = `Calculate my molecular imprint. Birth date: ${birthDate} ${birthTime}, Location: ${location || 'unknown'}`;

                // Set query and send
                queryInput.value = query;

                // Trigger send
                const sendButton = document.getElementById('sendButton');
                if (sendButton) {
                    sendButton.click();
                }

                showToast('Sending query to calculate molecular imprint...', 'info');
            } else {
                showToast('Please use the chat interface to calculate molecular imprint', 'info');
            }
        });
    }

    if (runPredictionBtn) {
        runPredictionBtn.addEventListener('click', async () => {
            const marketQueryInput = document.getElementById('marketQueryInput');
            const physioStateSelect = document.getElementById('physioStateSelect');
            const query = marketQueryInput ? marketQueryInput.value : '';
            const physioState = physioStateSelect ? physioStateSelect.value : 'focused';

            if (!query) {
                showToast('Please enter a prediction query', 'error');
                return;
            }

            if (predictionResults) {
                predictionResults.innerHTML = '<div style="color: #ccc;">Running prediction analysis...</div>';
            }

            try {
                const prediction = await runMarketPrediction(query, physioState);
                if (predictionResults) {
                    predictionResults.innerHTML = formatPredictionResults(prediction);
                }
                showToast('Prediction completed!', 'success');
            } catch (error) {
                console.error('Prediction failed:', error);
                if (predictionResults) {
                    predictionResults.innerHTML = '<div style="color: #ef4444;">Error running prediction</div>';
                }
                showToast('Failed to run prediction', 'error');
            }
        });
    }

    if (testHypothesisBtn) {
        testHypothesisBtn.addEventListener('click', async () => {
            const hypothesisInput = document.getElementById('hypothesisInput');
            const hypothesis = hypothesisInput ? hypothesisInput.value : '';

            if (!hypothesis) {
                showToast('Please enter a hypothesis to test', 'error');
                return;
            }

            if (hypothesisResults) {
                hypothesisResults.innerHTML = '<div style="color: #ccc;">Testing hypothesis...</div>';
            }

            try {
                const results = await testHypothesis(hypothesis);
                if (hypothesisResults) {
                    hypothesisResults.innerHTML = formatHypothesisResults(results);
                }
                showToast('Hypothesis tested!', 'success');
            } catch (error) {
                console.error('Hypothesis testing failed:', error);
                if (hypothesisResults) {
                    hypothesisResults.innerHTML = '<div style="color: #ef4444;">Error testing hypothesis</div>';
                }
                showToast('Failed to test hypothesis', 'error');
            }
        });
    }

    if (generateHypothesisBtn) {
        generateHypothesisBtn.addEventListener('click', async () => {
            if (hypothesisResults) {
                hypothesisResults.innerHTML = '<div style="color: #ccc;">Generating hypothesis...</div>';
            }

            try {
                const hypothesis = await generateHypothesis();
                const hypothesisInput = document.getElementById('hypothesisInput');
                if (hypothesisInput) {
                    hypothesisInput.value = hypothesis;
                }
                showToast('Hypothesis generated!', 'success');
            } catch (error) {
                console.error('Hypothesis generation failed:', error);
                showToast('Failed to generate hypothesis', 'error');
            }
        });
    }

    // Settings panel
    if (settingsToggle) {
        settingsToggle.addEventListener('click', openSettings);
    }
    if (settingsClose) {
        settingsClose.addEventListener('click', closeSettings);
    }
    if (settingsOverlay) {
        settingsOverlay.addEventListener('click', closeSettings);
    }

    // Settings controls
    if (primaryModel) {
        primaryModel.addEventListener('change', (e) => {
            settings.primaryModel = e.target.value;
            saveSettings();
        });
    }
    if (temperature && temperatureValue) {
        temperature.addEventListener('input', (e) => {
            settings.temperature = parseFloat(e.target.value);
            temperatureValue.textContent = settings.temperature;
            saveSettings();
        });
    }
    if (maxTokens && maxTokensValue) {
        maxTokens.addEventListener('input', (e) => {
            settings.maxTokens = parseInt(e.target.value);
            maxTokensValue.textContent = settings.maxTokens;
            saveSettings();
        });
    }

    // Export/Share/Clear
    if (exportBtn) {
        exportBtn.addEventListener('click', exportConversation);
    }
    if (shareBtn) {
        shareBtn.addEventListener('click', shareConversation);
    }
    if (clearBtn) {
        clearBtn.addEventListener('click', clearAllConversations);
    }

    // Search conversations
    if (sidebarSearchInput) {
        sidebarSearchInput.addEventListener('input', (e) => {
            const query = e.target.value;
            searchConversations(query);
            if (query) {
                sidebarSearchClear.style.display = 'block';
            } else {
                sidebarSearchClear.style.display = 'none';
            }
        });
    }
    if (sidebarSearchClear) {
        sidebarSearchClear.addEventListener('click', () => {
            sidebarSearchInput.value = '';
            searchConversations('');
            sidebarSearchClear.style.display = 'none';
        });
    }

    // Voice input
    if (voiceButton) {
        voiceButton.addEventListener('click', startVoiceRecording);
    }

    // Feature toggles
    if (enableWebSearchCheckbox) {
        enableWebSearchCheckbox.checked = enableWebSearch;
        enableWebSearchCheckbox.addEventListener('change', (e) => {
            enableWebSearch = e.target.checked;
            localStorage.setItem('iceburg_enableWebSearch', enableWebSearch);
            showToast(enableWebSearch ? 'Web search enabled' : 'Web search disabled', 'info');
        });
    }
    if (enableImageGenerationCheckbox) {
        enableImageGenerationCheckbox.checked = enableImageGeneration;
        enableImageGenerationCheckbox.addEventListener('change', (e) => {
            enableImageGeneration = e.target.checked;
            localStorage.setItem('iceburg_enableImageGeneration', enableImageGeneration);
            showToast(enableImageGeneration ? 'Image generation enabled' : 'Image generation disabled', 'info');
        });
    }

    // Load feature toggles from localStorage
    const savedWebSearch = localStorage.getItem('iceburg_enableWebSearch');
    const savedImageGen = localStorage.getItem('iceburg_enableImageGeneration');
    if (savedWebSearch !== null) enableWebSearch = savedWebSearch === 'true';
    if (savedImageGen !== null) enableImageGeneration = savedImageGen === 'true';

    // Event listeners
    sendButton.addEventListener('click', () => {
        sendQuery();
        saveConversation();
    });

    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
            saveConversation();
        }
    });

    // Focus input on load
    queryInput.focus();



    // Legacy visibility/focus listeners removed


    // Save conversation periodically
    setInterval(saveConversation, 30000); // Every 30 seconds

    // Final state check after all initialization
    setTimeout(() => {
        console.log('🔍 FINAL STATE CHECK:');
        const modeSelect = document.getElementById('modeSelect');
        const agentSelect = document.getElementById('agentSelect');
        const agentSelectGroup = agentSelect ? agentSelect.closest('.control-group') : null;
        const sidebarToggle = document.getElementById('sidebarToggle');
        const sidebar = document.getElementById('sidebar');
        const visualizationPanels = document.getElementById('visualizationPanels');

        console.log('📊 Element states:', {
            modeSelect: {
                exists: !!modeSelect,
                value: modeSelect ? modeSelect.value : null
            },
            agentSelect: {
                exists: !!agentSelect,
                display: agentSelect ? window.getComputedStyle(agentSelect).display : null
            },
            agentSelectGroup: {
                exists: !!agentSelectGroup,
                display: agentSelectGroup ? window.getComputedStyle(agentSelectGroup).display : null
            },
            sidebarToggle: {
                exists: !!sidebarToggle,
                clickListeners: sidebarToggle ? sidebarToggle.onclick || 'none' : 'none'
            },
            sidebar: {
                exists: !!sidebar,
                classes: sidebar ? sidebar.className : null,
                computedTransform: sidebar ? window.getComputedStyle(sidebar).transform : null
            },
            visualizationPanels: {
                exists: !!visualizationPanels,
                display: visualizationPanels ? window.getComputedStyle(visualizationPanels).display : null,
                classes: visualizationPanels ? visualizationPanels.className : null
            }
        });

        // Test agent visibility logic
        if (modeSelect) {
            const currentMode = modeSelect.value;
            const modesWithAgents = ['research', 'chat', 'device', 'truth', 'swarm', 'gnosis', 'civilization', 'astrophysiology'];
            const shouldShowAgent = modesWithAgents.includes(currentMode);
            console.log('🎯 Agent visibility logic:', {
                currentMode,
                shouldShowAgent,
                modesWithAgents
            });
        }
    }, 1000);
});

// Astro-Physiology Functions
// DEPRECATED: calculateMolecularImprint is now handled by the backend
// Real molecular imprint calculations are done via the chat interface in astro-physiology mode
// The backend returns actual framework results through WebSocket messages

async function runMarketPrediction(query, physioState) {
    // Simulate market prediction with astro-physiology integration
    console.log('Running market prediction:', query, physioState);

    // Base market analysis
    const fundamentals = 0.55 + (Math.random() - 0.5) * 0.2;

    // Physiological influences
    const physioMap = {
        focused: 0.075,
        stressed: -0.025,
        relaxed: 0.035,
        tired: -0.015
    };
    const physioInfluence = physioMap[physioState] || 0.0;

    // Celestial influences (simulated)
    const celestialInfluence = (Math.random() - 0.5) * 0.04;

    // Earth resonance
    const earthInfluence = 0.0425 + (Math.random() - 0.5) * 0.01;

    // TCM timing
    const tcmInfluence = 0.072 + (Math.random() - 0.5) * 0.02;

    // Breakthrough probability
    const breakthroughBonus = 0.1068 + (Math.random() - 0.5) * 0.03;

    const totalScore = fundamentals + physioInfluence + celestialInfluence +
        earthInfluence + tcmInfluence + breakthroughBonus;

    const direction = totalScore > 0.65 ? 'BULLISH' : totalScore < 0.35 ? 'BEARISH' : 'NEUTRAL';
    const confidence = Math.min(0.95, Math.max(0.1, totalScore));

    return {
        direction,
        confidence: Math.round(confidence * 100) / 100,
        query,
        factors: {
            fundamentals: Math.round(fundamentals * 100) / 100,
            physiological: Math.round(physioInfluence * 100) / 100,
            celestial: Math.round(celestialInfluence * 100) / 100,
            earth: Math.round(earthInfluence * 100) / 100,
            tcm: Math.round(tcmInfluence * 100) / 100,
            breakthrough: Math.round(breakthroughBonus * 100) / 100,
            total_score: Math.round(totalScore * 100) / 100
        }
    };
}

function formatPredictionResults(prediction) {
    return `
<div style="color: #fff;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #333;">
        <div style="font-size: 2rem; font-weight: 700; color: #10b981;">
            ${prediction.direction}
        </div>
        <div style="font-size: 1.2rem; color: #ccc;">
            ${(prediction.confidence * 100).toFixed(1)}% Confidence
        </div>
    </div>

    <div style="margin-bottom: 20px;">
        <h4 style="color: #fff; margin-bottom: 8px;">🎯 Analysis for: ${prediction.query}</h4>
    </div>

    <div style="margin-bottom: 20px;">
        <h5 style="color: #ccc; margin-bottom: 10px;">📊 Factor Breakdown:</h5>
        <div style="display: grid; gap: 8px;">
            <div>📈 Fundamentals: <span style="color: #10b981;">+${(prediction.factors.fundamentals * 100).toFixed(1)}%</span></div>
            <div>🧠 Physiological: <span style="color: ${prediction.factors.physiological > 0 ? '#10b981' : '#ef4444'};">${prediction.factors.physiological > 0 ? '+' : ''}${(prediction.factors.physiological * 100).toFixed(1)}%</span></div>
            <div>⭐ Celestial: <span style="color: ${prediction.factors.celestial > 0 ? '#10b981' : '#ef4444'};">${prediction.factors.celestial > 0 ? '+' : ''}${(prediction.factors.celestial * 100).toFixed(1)}%</span></div>
            <div>🌍 Earth Resonance: <span style="color: #10b981;">+${(prediction.factors.earth * 100).toFixed(1)}%</span></div>
            <div>⏰ TCM Timing: <span style="color: #10b981;">+${(prediction.factors.tcm * 100).toFixed(1)}%</span></div>
            <div>🚀 Breakthrough Potential: <span style="color: #10b981;">+${(prediction.factors.breakthrough * 100).toFixed(1)}%</span></div>
        </div>
    </div>

    <div style="background: #1a1a1a; padding: 16px; border-radius: 6px; margin-top: 20px;">
        <h5 style="color: #ccc; margin-bottom: 10px;">💡 Market Insight:</h5>
        <p style="color: #ccc; line-height: 1.6;">
            ${prediction.direction === 'BULLISH'
            ? 'Unified consciousness fields suggest upward momentum. Your physiological state indicates strong market conviction.'
            : prediction.direction === 'BEARISH'
                ? 'Caution advised. Celestial alignments suggest potential downward pressure.'
                : 'Market appears balanced. Monitor physiological and celestial indicators for directional cues.'}
        </p>
    </div>
</div>`;
}

async function testHypothesis(hypothesis) {
    // Simulate hypothesis testing
    console.log('Testing hypothesis:', hypothesis);

    const confidence = 0.65 + (Math.random() - 0.5) * 0.3;
    const supportingEvidence = Math.floor(Math.random() * 5) + 1;
    const contradictingEvidence = Math.floor(Math.random() * 3);

    const results = {
        hypothesis,
        confidence: Math.round(confidence * 100) / 100,
        status: confidence > 0.7 ? 'SUPPORTED' : confidence > 0.4 ? 'PARTIALLY_SUPPORTED' : 'NOT_SUPPORTED',
        evidence: {
            supporting: supportingEvidence,
            contradicting: contradictingEvidence
        },
        implications: [
            "Further experimental validation needed",
            "May require larger sample sizes",
            "Cross-cultural studies recommended"
        ]
    };

    return results;
}

// Truth-Finding UX Components for Astro-Physiology Mode

function addTruthInsight(truthData, messageElement) {
    if (!messageElement || !truthData) return;

    const messageContent = messageElement.querySelector('.message-content');
    if (!messageContent) return;

    const truthPanel = document.createElement('div');
    truthPanel.className = 'truth-insight-panel';

    truthPanel.innerHTML = `
        <div class="truth-header">
            <span class="truth-icon">🔍</span>
            <strong>Truth About This:</strong>
        </div>
        <div class="truth-content">
            <div class="truth-pattern">
                <strong>Pattern Detected:</strong> ${truthData.pattern || 'Unknown pattern'}
            </div>
            ${truthData.evidence ? `<div class="truth-evidence"><strong>Evidence:</strong> ${truthData.evidence}</div>` : ''}
            ${truthData.meaning ? `<div class="truth-meaning"><strong>What This Means:</strong> ${truthData.meaning}</div>` : ''}
            ${truthData.data ? `<div class="truth-data" style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                <details>
                    <summary style="cursor: pointer; color: #ccc;">View Details</summary>
                    <pre style="color: #999; font-size: 0.9em; margin-top: 8px;">${JSON.stringify(truthData.data, null, 2)}</pre>
                </details>
            </div>` : ''}
        </div>
    `;

    messageContent.appendChild(truthPanel);
}

function displayMolecularBlueprint(imprint, messageElement) {
    if (!messageElement || !imprint) return;

    const messageContent = messageElement.querySelector('.message-content');
    if (!messageContent) return;

    const blueprintPanel = document.createElement('div');
    blueprintPanel.className = 'molecular-blueprint-panel';

    const voltageGates = imprint.voltage_gates || {};
    const traits = imprint.trait_amplification_factors || {};

    blueprintPanel.style.cssText = 'margin: 20px 0; padding: 0;';

    blueprintPanel.innerHTML = `
        <h4 style="color: #fff; margin: 0 0 15px 0; padding: 0; display: flex; align-items: center; gap: 8px; font-size: 1.1em;">
            Your Molecular Blueprint
        </h4>
        <div class="blueprint-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 0 0 15px 0; padding: 0;">
            <div class="blueprint-item" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 6px; margin: 0;">
                <strong style="color: #fff; display: block; margin: 0 0 10px 0; padding: 0; font-size: 0.95em;">Voltage Gates:</strong>
                <ul style="list-style: none; padding: 0; margin: 0; color: #ccc;">
                    <li style="margin: 6px 0; padding: 0;">Sodium: <span style="color: #10b981;">${(voltageGates.sodium || 0).toFixed(2)}</span> mV (Action)</li>
                    <li style="margin: 6px 0; padding: 0;">Potassium: <span style="color: #10b981;">${(voltageGates.potassium || 0).toFixed(2)}</span> mV (Control)</li>
                    <li style="margin: 6px 0; padding: 0;">Calcium: <span style="color: #10b981;">${(voltageGates.calcium || 0).toFixed(2)}</span> mV (Emotion)</li>
                    <li style="margin: 6px 0; padding: 0;">Chloride: <span style="color: #10b981;">${(voltageGates.chloride || 0).toFixed(2)}</span> mV (Stability)</li>
                </ul>
            </div>
            <div class="blueprint-item" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 6px; margin: 0;">
                <strong style="color: #fff; display: block; margin: 0 0 10px 0; padding: 0; font-size: 0.95em;">Natural Traits:</strong>
                <ul style="list-style: none; padding: 0; margin: 0; color: #ccc;">
                    ${Object.entries(traits).slice(0, 4).map(([trait, value]) =>
        `<li style="margin: 6px 0; padding: 0;">${trait.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}: <span style="color: #10b981;">${(value * 100).toFixed(0)}%</span></li>`
    ).join('')}
                </ul>
            </div>
        </div>
        <p class="truth-note" style="color: #999; font-size: 0.9em; margin: 15px 0 0 0; padding: 0; font-style: italic;">
            This is your authentic blueprint - not personality, but molecular truth encoded at birth.
        </p>
    `;

    messageContent.appendChild(blueprintPanel);
}

function displayPatternTruth(patternData, messageElement) {
    if (!messageElement || !patternData) return;

    const messageContent = messageElement.querySelector('.message-content');
    if (!messageContent) return;

    const patternPanel = document.createElement('div');
    patternPanel.className = 'pattern-truth-panel';

    const predictions = patternData.predictions || {};
    const sortedPredictions = Object.entries(predictions)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);

    patternPanel.style.cssText = 'margin: 20px 0; padding: 0;';

    patternPanel.innerHTML = `
        <div class="pattern-header" style="display: flex; justify-content: space-between; align-items: center; margin: 0 0 15px 0; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 6px;">
            <span style="color: #fff; font-weight: 600; font-size: 0.95em;">📊 Pattern Detected</span>
            <span class="confidence" style="color: #10b981; font-size: 0.9em; padding: 0; margin: 0;">${((patternData.confidence || 0) * 100).toFixed(0)}% confidence</span>
        </div>
        <div class="pattern-details" style="color: #ccc; margin: 0; padding: 0;">
            <div style="margin: 0 0 10px 0; padding: 0;"><strong style="color: #fff;">Pattern:</strong> ${patternData.pattern || 'Unknown'}</div>
            ${patternData.cycle ? `<div style="margin: 0 0 10px 0; padding: 0;"><strong style="color: #fff;">Cycle:</strong> ${patternData.cycle}</div>` : ''}
            ${patternData.correlation ? `<div style="margin: 0 0 10px 0; padding: 0;"><strong style="color: #fff;">Correlation:</strong> ${patternData.correlation}</div>` : ''}
            ${patternData.truth ? `<div style="margin: 0 0 10px 0; padding: 0;"><strong style="color: #fff;">Truth:</strong> ${patternData.truth}</div>` : ''}
            ${sortedPredictions.length > 0 ? `
                <div style="margin-top: 15px; padding: 0;">
                    <strong style="color: #fff; display: block; margin: 0 0 10px 0; padding: 0;">Top Predictions:</strong>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        ${sortedPredictions.map(([trait, value]) =>
        `<li style="margin: 6px 0; padding: 8px; background: rgba(255,255,255,0.03); border-radius: 4px;">
                                ${trait.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}: 
                                <span style="color: #10b981; font-weight: 600;">${(value * 100).toFixed(0)}%</span>
                            </li>`
    ).join('')}
                    </ul>
                </div>
            ` : ''}
        </div>
    `;

    messageContent.appendChild(patternPanel);
}

// ============================================================================
// ASTRO-PHYSIOLOGY UX ENHANCEMENTS
// ============================================================================

/**
 * Parse birth data from natural language query
 * Supports formats like:
 * - "Dec 26 1991 7:20 AM Dallas"
 * - "December 26, 1991 at 7:20 AM in Dallas, TX"
 * - "Born on 12/26/1991 at 7:20 AM in Dallas"
 */
function astro_parseBirthDataFromQuery(query) {
    if (!query || typeof query !== 'string') return null;

    const result = {
        birth_date: null,
        birth_time: null,
        location: null,
        timezone: null
    };

    // Month name mapping
    const months = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    };

    const queryLower = query.toLowerCase();

    // Pattern 1: "December 26, 1991" or "Dec 26 1991"
    const datePattern1 = /(?:born|birth|on)?\s*(?:the\s+)?(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})/i;
    const dateMatch1 = query.match(datePattern1);

    // Pattern 2: "12/26/1991" or "12-26-1991"
    const datePattern2 = /(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})/;
    const dateMatch2 = query.match(datePattern2);

    let year, month, day;

    if (dateMatch1) {
        const monthName = dateMatch1[1].toLowerCase();
        month = months[monthName];
        day = parseInt(dateMatch1[2]);
        year = parseInt(dateMatch1[3]);
    } else if (dateMatch2) {
        month = parseInt(dateMatch2[1]);
        day = parseInt(dateMatch2[2]);
        year = parseInt(dateMatch2[3]);
    }

    if (year && month && day) {
        // Validate date
        const date = new Date(year, month - 1, day);
        if (date.getFullYear() === year && date.getMonth() === month - 1 && date.getDate() === day) {
            result.birth_date = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
        }
    }

    // Time patterns: "7:20 AM", "07:20", "19:20"
    const timePatterns = [
        /(\d{1,2}):(\d{2})\s*(am|pm)/i,
        /(\d{1,2}):(\d{2})/,
        /at\s+(\d{1,2})\s*(am|pm)/i
    ];

    for (const pattern of timePatterns) {
        const timeMatch = query.match(pattern);
        if (timeMatch) {
            let hours = parseInt(timeMatch[1]);
            const minutes = timeMatch[2] ? parseInt(timeMatch[2]) : 0;
            const ampm = timeMatch[3] ? timeMatch[3].toLowerCase() : null;

            if (ampm === 'pm' && hours < 12) hours += 12;
            if (ampm === 'am' && hours === 12) hours = 0;

            result.birth_time = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
            break;
        }
    }

    // Location patterns: "in Dallas", "Dallas, TX", "at Parkland Hospital, Dallas"
    const locationPatterns = [
        /(?:in|at|location:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:,\s*([A-Z]{2}))?/i,
        /([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})/i,
        /coordinates?:\s*(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)/i
    ];

    for (const pattern of locationPatterns) {
        const locMatch = query.match(pattern);
        if (locMatch) {
            if (locMatch[3] && locMatch[4]) {
                // Coordinates
                result.location = {
                    lat: parseFloat(locMatch[3]),
                    lng: parseFloat(locMatch[4])
                };
            } else {
                // City name
                let city = locMatch[1];
                if (locMatch[2]) city += `, ${locMatch[2]}`;
                result.location = city;
            }
            break;
        }
    }

    // Return null if no data extracted
    if (!result.birth_date && !result.birth_time && !result.location) {
        return null;
    }

    return result;
}

/**
 * Show inline birth data form when mode is astro-physiology and data is missing
 */
function astro_showBirthDataForm(messageElement) {
    if (!messageElement) return;

    const messageContent = messageElement.querySelector('.message-content');
    if (!messageContent) return;

    // Check if form already exists
    if (messageContent.querySelector('.astro-birth-form')) return;

    const formCard = document.createElement('div');
    formCard.className = 'astro-birth-form';
    formCard.style.cssText = `
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
    `;

    formCard.innerHTML = `
        <h4 style="color: #00D4FF; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px;">
            📅 Birth Information Required
        </h4>
        <p style="color: #ccc; margin: 0 0 15px 0; font-size: 0.9em;">
            For celestial-biological analysis, I need your birth information:
        </p>
        <div style="display: grid; gap: 15px;">
            <div>
                <label style="color: #fff; display: block; margin-bottom: 5px; font-size: 0.9em;">Birth Date:</label>
                <input type="date" id="astro_birthDateInput" class="astro-form-input" 
                       style="width: 100%; padding: 8px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff;">
            </div>
            <div>
                <label style="color: #fff; display: block; margin-bottom: 5px; font-size: 0.9em;">Birth Time:</label>
                <input type="time" id="astro_birthTimeInput" class="astro-form-input" 
                       style="width: 100%; padding: 8px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff;">
            </div>
            <div>
                <label style="color: #fff; display: block; margin-bottom: 5px; font-size: 0.9em;">Location (city or coordinates):</label>
                <input type="text" id="astro_locationInput" class="astro-form-input" 
                       placeholder="Dallas, TX or 32.813, -96.8353"
                       style="width: 100%; padding: 8px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff;">
            </div>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <button id="astro_submitBirthData" class="astro-form-button" 
                        style="flex: 1; padding: 10px; background: #00D4FF; color: #000; border: none; border-radius: 4px; cursor: pointer; font-weight: 600;">
                    Analyze
                </button>
                <button id="astro_saveBirthData" class="astro-form-button" 
                        style="flex: 1; padding: 10px; background: rgba(255,255,255,0.1); color: #fff; border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; cursor: pointer;">
                    Save for Later
                </button>
            </div>
            <p style="color: #999; font-size: 0.8em; margin: 10px 0 0 0; font-style: italic;">
                ℹ️ This data is stored locally only and used for analysis
            </p>
        </div>
    `;

    messageContent.appendChild(formCard);

    // Load saved data if available
    astro_loadSavedBirthData();

    // Add event listeners
    const submitBtn = formCard.querySelector('#astro_submitBirthData');
    const saveBtn = formCard.querySelector('#astro_saveBirthData');

    if (submitBtn) {
        submitBtn.addEventListener('click', () => {
            astro_submitBirthDataFromForm();
        });
    }

    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            astro_saveBirthDataToStorage();
            showToast('Birth data saved', 'success');
        });
    }
}

/**
 * Load saved birth data into form fields
 */
function astro_loadSavedBirthData() {
    try {
        const saved = localStorage.getItem('iceburg_astro_birth_data');
        if (saved) {
            const birthData = JSON.parse(saved);

            const dateInput = document.getElementById('astro_birthDateInput') || document.getElementById('birthDateInput');
            const timeInput = document.getElementById('astro_birthTimeInput') || document.getElementById('birthTimeInput');
            const locationInput = document.getElementById('astro_locationInput') || document.getElementById('locationInput');

            if (birthData.birth_date && dateInput) {
                const dt = new Date(birthData.birth_date);
                if (!isNaN(dt.getTime())) {
                    dateInput.value = dt.toISOString().split('T')[0];
                }
            }

            if (birthData.birth_time && timeInput) {
                timeInput.value = birthData.birth_time;
            } else if (birthData.birth_date && timeInput) {
                const dt = new Date(birthData.birth_date);
                if (!isNaN(dt.getTime())) {
                    const hours = String(dt.getUTCHours()).padStart(2, '0');
                    const minutes = String(dt.getUTCMinutes()).padStart(2, '0');
                    timeInput.value = `${hours}:${minutes}`;
                }
            }

            if (birthData.location && locationInput) {
                if (typeof birthData.location === 'object' && birthData.location.lat && birthData.location.lng) {
                    locationInput.value = `${birthData.location.lat},${birthData.location.lng}`;
                } else if (typeof birthData.location === 'string') {
                    locationInput.value = birthData.location;
                }
            }
        }
    } catch (e) {
        console.warn('Could not load saved birth data:', e);
    }
}

/**
 * Save birth data to localStorage
 */
function astro_saveBirthDataToStorage() {
    const dateInput = document.getElementById('astro_birthDateInput') || document.getElementById('birthDateInput');
    const timeInput = document.getElementById('astro_birthTimeInput') || document.getElementById('birthTimeInput');
    const locationInput = document.getElementById('astro_locationInput') || document.getElementById('locationInput');

    if (!dateInput || !dateInput.value) return;

    const birthDate = dateInput.value;
    const birthTime = timeInput ? (timeInput.value || '12:00') : '12:00';
    const locationStr = locationInput ? locationInput.value.trim() : '';

    // Combine date and time
    const birthDateTime = `${birthDate}T${birthTime}:00Z`;

    // Parse location
    let location = null;
    if (locationStr) {
        const coordMatch = locationStr.match(/^(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)$/);
        if (coordMatch) {
            location = {
                lat: parseFloat(coordMatch[1]),
                lng: parseFloat(coordMatch[2])
            };
        } else {
            location = locationStr;
        }
    }

    const birthData = {
        birth_date: birthDateTime,
        birth_time: birthTime,
        location: location
    };

    try {
        localStorage.setItem('iceburg_astro_birth_data', JSON.stringify(birthData));
        return true;
    } catch (e) {
        console.warn('Could not save birth data:', e);
        return false;
    }
}

/**
 * Submit birth data from form and trigger analysis
 */
function astro_submitBirthDataFromForm() {
    if (!astro_saveBirthDataToStorage()) {
        showToast('Please enter a birth date', 'error');
        return;
    }

    const dateInput = document.getElementById('astro_birthDateInput') || document.getElementById('birthDateInput');
    const timeInput = document.getElementById('astro_birthTimeInput') || document.getElementById('birthTimeInput');
    const locationInput = document.getElementById('astro_locationInput') || document.getElementById('locationInput');

    if (!dateInput || !dateInput.value) {
        showToast('Please enter a birth date', 'error');
        return;
    }

    const birthDate = dateInput.value;
    const birthTime = timeInput ? (timeInput.value || '12:00') : '12:00';
    const location = locationInput ? locationInput.value.trim() : 'unknown';

    // Set mode to astro-physiology if not already
    const modeSelect = document.getElementById('modeSelect');
    if (modeSelect) {
        modeSelect.value = 'astrophysiology';
    }

    // Create query
    const queryInput = document.getElementById('queryInput');
    if (queryInput) {
        queryInput.value = `Calculate my molecular imprint. Birth date: ${birthDate} ${birthTime}, Location: ${location}`;

        // Trigger send
        const sendButton = document.getElementById('sendButton');
        if (sendButton) {
            sendButton.click();
        }
    }
}

/**
 * Create rich astro-physiology result card with visualizations
 */
function astro_createAstroPhysiologyCard(resultData, messageElement) {
    if (!messageElement || !resultData) return;

    const messageContent = messageElement.querySelector('.message-content');
    if (!messageContent) return;

    // Check if card already exists - prevent duplicates (more robust check)
    const existingCard = messageContent.querySelector('.astro-card');
    if (existingCard) {
        console.log('Astro card already exists, skipping duplicate creation');
        return;
    }

    // Also check for existing expert consultations and interventions to prevent duplicates
    if (messageContent.querySelector('.astro-expert-consultations') ||
        messageContent.querySelector('.astro-interventions')) {
        console.log('Expert consultations or interventions already exist, skipping duplicate creation');
        return;
    }

    const card = document.createElement('div');
    card.className = 'astro-card';
    card.style.cssText = `
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
    `;

    const molecularImprint = resultData.molecular_imprint || {};
    const behavioralPredictions = resultData.behavioral_predictions || {};
    const tcmPredictions = resultData.tcm_predictions || {};
    const currentConditions = resultData.current_conditions || {};

    // Get key insights
    const traits = molecularImprint.trait_amplification_factors || {};
    const topTraits = Object.entries(traits)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);

    card.innerHTML = `
        <div class="astro-card-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid rgba(0, 212, 255, 0.2);">
            <h3 style="color: #00D4FF; margin: 0; font-size: 0.9em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                Molecular Blueprint
            </h3>
            <span class="astro-card-timestamp" style="color: #666; font-size: 0.7em;">
                ${new Date().toLocaleDateString()}
            </span>
        </div>
        
        <div class="astro-card-level1" id="astro-level1">
            <div class="astro-key-insights" style="margin-bottom: 4px;">
                <h4 style="color: rgba(255,255,255,0.7); margin: 0 0 2px 0; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Key Insights</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 2px;">
                    ${topTraits.map(([trait, value]) => `
                        <div style="background: rgba(0, 212, 255, 0.1); padding: 3px 4px; border-radius: 2px; border: 1px solid rgba(0, 212, 255, 0.2);">
                            <div style="color: rgba(255,255,255,0.6); font-size: 0.7em; margin-bottom: 1px; text-transform: capitalize;">
                                ${trait.replace(/_/g, ' ')}
                            </div>
                            <div style="color: #00D4FF; font-size: 0.85em; font-weight: 600;">
                                ${(value * 100).toFixed(0)}%
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <div class="astro-charts-inline" style="margin-top: 4px;">
                <h4 style="color: rgba(255,255,255,0.7); margin: 0 0 2px 0; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Celestial-Biological Correlations</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2px;">
                    <div class="astro-chart-wrapper-inline" style="background: rgba(255,255,255,0.02); padding: 3px; border-radius: 2px; border: 1px solid rgba(0, 212, 255, 0.1);">
                        <h5 style="color: rgba(255,255,255,0.7); margin: 0 0 2px 0; font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.3px; font-weight: 600;">Behavioral Traits</h5>
                        <canvas id="astro-traits-chart-inline" style="max-height: 180px;"></canvas>
                    </div>
                    <div class="astro-chart-wrapper-inline" style="background: rgba(255,255,255,0.02); padding: 3px; border-radius: 2px; border: 1px solid rgba(0, 212, 255, 0.1);">
                        <h5 style="color: rgba(255,255,255,0.7); margin: 0 0 2px 0; font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.3px; font-weight: 600;">Organ Systems</h5>
                        <canvas id="astro-organs-chart-inline" style="max-height: 180px;"></canvas>
                    </div>
                </div>
            </div>
            
            <div style="display: flex; gap: 2px; margin-top: 4px;">
                <button class="astro-expand-btn" data-level="2" 
                        style="flex: 1; padding: 3px 4px; background: rgba(0, 212, 255, 0.2); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 2px; cursor: pointer; font-size: 0.7em;">
                    More
                </button>
                <button class="astro-expert-btn" data-level="3" 
                        style="flex: 1; padding: 3px 4px; background: rgba(0, 212, 255, 0.2); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 2px; cursor: pointer; font-size: 0.7em;">
                    Data
                </button>
            </div>
        </div>
        
        <div class="astro-card-level2" id="astro-level2" style="display: none; margin-top: 4px;">
            <div class="astro-charts-container" style="display: grid; gap: 4px;">
                <div class="astro-chart-wrapper" style="background: rgba(255,255,255,0.02); padding: 4px; border-radius: 3px; border: 1px solid rgba(0, 212, 255, 0.1);">
                    <h4 style="color: rgba(255,255,255,0.8); margin: 0 0 4px 0; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Behavioral Traits</h4>
                    <canvas id="astro-traits-chart" style="max-height: 200px;"></canvas>
                </div>
                <div class="astro-chart-wrapper" style="background: rgba(255,255,255,0.02); padding: 4px; border-radius: 3px; border: 1px solid rgba(0, 212, 255, 0.1);">
                    <h4 style="color: rgba(255,255,255,0.8); margin: 0 0 4px 0; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Organ Systems</h4>
                    <canvas id="astro-organs-chart" style="max-height: 200px;"></canvas>
                </div>
            </div>
            
            <div style="display: flex; gap: 4px; margin-top: 4px;">
                <button class="astro-collapse-btn" data-level="1" 
                        style="flex: 1; padding: 6px 8px; background: rgba(255,255,255,0.1); color: #fff; border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; cursor: pointer; font-size: 0.85em;">
                    Show Less
                </button>
                <button class="astro-expert-btn" data-level="3" 
                        style="flex: 1; padding: 6px 8px; background: rgba(0, 212, 255, 0.2); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 4px; cursor: pointer; font-size: 0.85em;">
                    Expert View
                </button>
            </div>
        </div>
        
        <div class="astro-card-level3" id="astro-level3" style="display: none; margin-top: 4px;">
            <div class="astro-expert-data" style="background: rgba(255,255,255,0.02); padding: 4px; border-radius: 3px; border: 1px solid rgba(0, 212, 255, 0.1);">
                <h4 style="color: rgba(255,255,255,0.8); margin: 0 0 4px 0; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Raw Data</h4>
                <div style="color: #ccc; font-size: 0.85em; max-height: 400px; overflow-y: auto;">
                    ${Object.keys(molecularImprint).length > 0 ? `
                        <div style="margin-bottom: 8px;">
                            <strong style="color: #00D4FF; font-size: 0.85em;">Molecular Imprint:</strong>
                            <div style="margin-top: 4px; padding-left: 8px;">
                                ${Object.entries(molecularImprint).slice(0, 10).map(([key, value]) => {
        if (typeof value === 'object' && value !== null) {
            return `<div style="margin: 2px 0; font-size: 0.8em;"><strong>${key.replace(/_/g, ' ')}:</strong> ${Object.keys(value).length} items</div>`;
        }
        return `<div style="margin: 2px 0; font-size: 0.8em;"><strong>${key.replace(/_/g, ' ')}:</strong> ${value}</div>`;
    }).join('')}
                            </div>
                        </div>
                    ` : ''}
                    ${Object.keys(behavioralPredictions).length > 0 ? `
                        <div style="margin-bottom: 8px;">
                            <strong style="color: #00D4FF; font-size: 0.85em;">Behavioral Predictions:</strong>
                            <div style="margin-top: 4px; padding-left: 8px;">
                                ${Object.entries(behavioralPredictions).slice(0, 10).map(([key, value]) =>
        `<div style="margin: 2px 0; font-size: 0.8em;"><strong>${key.replace(/_/g, ' ')}:</strong> ${(value * 100).toFixed(1)}%</div>`
    ).join('')}
                            </div>
                        </div>
                    ` : ''}
                    ${Object.keys(tcmPredictions).length > 0 ? `
                        <div style="margin-bottom: 8px;">
                            <strong style="color: #00D4FF; font-size: 0.85em;">TCM Predictions:</strong>
                            <div style="margin-top: 4px; padding-left: 8px;">
                                ${Object.entries(tcmPredictions).slice(0, 10).map(([key, value]) => {
        const org = typeof value === 'object' ? (value.name || key) : key;
        const strength = typeof value === 'object' ? (value.strength || 0) : value;
        return `<div style="margin: 2px 0; font-size: 0.8em;"><strong>${org.replace(/_/g, ' ')}:</strong> ${(strength * 100).toFixed(1)}%</div>`;
    }).join('')}
                            </div>
                        </div>
                    ` : ''}
                    <button class="astro-export-json-btn" style="margin-top: 8px; padding: 6px 10px; background: rgba(0, 212, 255, 0.2); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 4px; cursor: pointer; font-size: 0.8em;">
                        Export JSON
                    </button>
                </div>
            </div>
            
            <div style="display: flex; gap: 6px; margin-top: 8px;">
                <button class="astro-collapse-btn" data-level="2" 
                        style="flex: 1; padding: 6px 8px; background: rgba(255,255,255,0.1); color: #fff; border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; cursor: pointer; font-size: 0.85em;">
                    Back
                </button>
                <button class="astro-export-btn" 
                        style="flex: 1; padding: 6px 8px; background: rgba(0, 212, 255, 0.2); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 4px; cursor: pointer; font-size: 0.85em;">
                    Export
                </button>
            </div>
        </div>
    `;

    messageContent.appendChild(card);

    // V2: Advanced visualizations removed - charts are now inline in main answer format
    // No need for separate dashboard visualizations

    // V2: Display real-time adaptation status if available
    if (resultData.algorithmic_data && resultData.algorithmic_data.adaptation_status) {
        astro_displayAdaptationStatus(resultData.algorithmic_data.adaptation_status, messageContent);
    }

    // Render charts immediately in inline view (main answer format)
    setTimeout(() => {
        const traitsChartInline = card.querySelector('#astro-traits-chart-inline');
        const organsChartInline = card.querySelector('#astro-organs-chart-inline');
        const traitsChart = card.querySelector('#astro-traits-chart');
        const organsChart = card.querySelector('#astro-organs-chart');

        // Render inline charts (main view) - always visible
        if (traitsChartInline && behavioralPredictions && !Chart.getChart(traitsChartInline)) {
            astro_renderBehavioralTraitsChart(behavioralPredictions, traitsChartInline);
        }
        if (organsChartInline && tcmPredictions && !Chart.getChart(organsChartInline)) {
            astro_renderOrganSystemChart(tcmPredictions, organsChartInline);
        }

        // Render level 2 charts if they exist (for expand view)
        if (traitsChart && behavioralPredictions && !Chart.getChart(traitsChart)) {
            astro_renderBehavioralTraitsChart(behavioralPredictions, traitsChart);
        }
        if (organsChart && tcmPredictions && !Chart.getChart(organsChart)) {
            astro_renderOrganSystemChart(tcmPredictions, organsChart);
        }
    }, 100);

    // Add event listeners
    const expandBtn = card.querySelector('.astro-expand-btn');
    const collapseBtn = card.querySelectorAll('.astro-collapse-btn');
    const expertBtn = card.querySelector('.astro-expert-btn');
    const exportBtn = card.querySelector('.astro-export-btn');

    if (expandBtn) {
        expandBtn.addEventListener('click', () => {
            astro_expandAstroCard(card, 2);
        });
    }

    collapseBtn.forEach(btn => {
        btn.addEventListener('click', () => {
            const level = parseInt(btn.dataset.level);
            astro_collapseAstroCard(card, level);
        });
    });

    if (expertBtn) {
        expertBtn.addEventListener('click', () => {
            astro_expandAstroCard(card, 3);
        });
    }

    // Dashboard button removed - charts are now inline in main answer

    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            astro_exportData(resultData);
        });
    }

    // Add JSON export button handler
    const exportJsonBtn = card.querySelector('.astro-export-json-btn');
    if (exportJsonBtn) {
        exportJsonBtn.addEventListener('click', () => {
            const jsonData = {
                molecular_imprint: molecularImprint,
                behavioral_predictions: behavioralPredictions,
                tcm_predictions: tcmPredictions
            };
            const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `iceberg-astro-data-${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }
}

/**
 * Expand astro card to specified level
 */
function astro_expandAstroCard(cardElement, level) {
    if (!cardElement) return;

    // Hide all levels
    for (let i = 1; i <= 3; i++) {
        const levelEl = cardElement.querySelector(`#astro-level${i}`);
        if (levelEl) levelEl.style.display = 'none';
    }

    // Show target level and all previous levels
    for (let i = 1; i <= level; i++) {
        const levelEl = cardElement.querySelector(`#astro-level${i}`);
        if (levelEl) levelEl.style.display = 'block';
    }
}

/**
 * Collapse astro card to specified level
 */
function astro_collapseAstroCard(cardElement, level) {
    astro_expandAstroCard(cardElement, level);
}

/**
 * Render behavioral traits radar chart
 */
function astro_renderBehavioralTraitsChart(data, canvasElement) {
    if (!canvasElement || !data || typeof Chart === 'undefined') return;

    const traits = data || {};
    const labels = Object.keys(traits).map(k => k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
    const values = Object.values(traits).map(v => (v * 100));

    // Destroy existing chart if it exists
    const existingChart = Chart.getChart(canvasElement);
    if (existingChart) {
        existingChart.destroy();
    }

    new Chart(canvasElement, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Trait Strength',
                data: values,
                backgroundColor: 'rgba(0, 212, 255, 0.2)',
                borderColor: 'rgba(0, 212, 255, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(0, 212, 255, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(0, 212, 255, 1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 150,
                    ticks: {
                        color: '#ccc',
                        backdropColor: 'transparent'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    pointLabels: {
                        color: '#ccc'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Render organ system bar chart
 */
function astro_renderOrganSystemChart(data, canvasElement) {
    if (!canvasElement || !data || typeof Chart === 'undefined') return;

    const organs = data || {};
    const labels = Object.keys(organs).map(k => {
        const org = organs[k];
        return org.name || k.replace(/_/g, ' ').toUpperCase();
    });
    const strengths = Object.values(organs).map(v => {
        const org = typeof v === 'object' ? (v.strength || 0) : (v || 0);
        return org * 100;
    });

    // Destroy existing chart if it exists
    const existingChart = Chart.getChart(canvasElement);
    if (existingChart) {
        existingChart.destroy();
    }

    new Chart(canvasElement, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Organ Strength',
                data: strengths,
                backgroundColor: 'rgba(0, 212, 255, 0.6)',
                borderColor: 'rgba(0, 212, 255, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#ccc'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#ccc'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Render celestial map (2D sky map)
 */
function astro_renderCelestialMap(data, containerElement) {
    if (!containerElement || !data) return;

    const positions = data.celestial_positions || {};

    containerElement.innerHTML = `
        <div style="position: relative; width: 100%; height: 300px; background: rgba(0,0,0,0.3); border-radius: 6px; overflow: hidden;">
            <svg width="100%" height="100%" viewBox="0 0 400 300" style="position: absolute; top: 0; left: 0;">
                ${Object.entries(positions).map(([body, pos]) => {
        const [ra, dec, dist] = Array.isArray(pos) ? pos : [0, 0, 0];
        const x = 200 + (ra * 100);
        const y = 150 - (dec * 100);
        const colors = {
            'sun': '#FFD700',
            'moon': '#C0C0C0',
            'mars': '#FF6B6B',
            'mercury': '#87CEEB',
            'jupiter': '#FFA500',
            'venus': '#FFB6C1',
            'saturn': '#F0E68C'
        };
        const color = colors[body.toLowerCase()] || '#00D4FF';
        return `<circle cx="${x}" cy="${y}" r="8" fill="${color}" stroke="#fff" stroke-width="1">
                        <title>${body}: RA=${ra.toFixed(2)}, Dec=${dec.toFixed(2)}</title>
                    </circle>
                    <text x="${x}" y="${y - 12}" fill="#fff" font-size="10" text-anchor="middle">${body.charAt(0).toUpperCase()}</text>`;
    }).join('')}
            </svg>
        </div>
    `;
}

/**
 * Sync chat results to dashboard visualization panels
 */
function astro_syncChatToDashboard(chatData) {
    console.log('Syncing to dashboard with data:', Object.keys(chatData || {}));

    const visualizationPanels = document.getElementById('visualizationPanels');
    const astroPanel = document.getElementById('astrophysiologyPanel');

    if (!visualizationPanels) {
        console.error('visualizationPanels not found');
        showToast('Dashboard panels not found', 'error');
        return;
    }

    if (!astroPanel) {
        console.error('astrophysiologyPanel not found');
        showToast('Astro-physiology panel not found', 'error');
        return;
    }

    // Show panels
    visualizationPanels.style.display = 'block';
    visualizationPanels.classList.add('show');
    astroPanel.style.display = 'block';

    // Wait a bit for DOM to update, then populate charts
    setTimeout(() => {
        const traitsChart = document.getElementById('voltageGatesChart');
        const organsChart = document.getElementById('tcmCyclesChart');

        console.log('Chart elements:', {
            traitsChart: !!traitsChart,
            organsChart: !!organsChart,
            hasBehavioral: !!chatData?.behavioral_predictions,
            hasTcm: !!chatData?.tcm_predictions
        });

        if (chatData?.behavioral_predictions && traitsChart) {
            try {
                astro_renderBehavioralTraitsChart(chatData.behavioral_predictions, traitsChart);
                console.log('Behavioral traits chart rendered');
            } catch (e) {
                console.error('Error rendering behavioral chart:', e);
            }
        }

        if (chatData?.tcm_predictions && organsChart) {
            try {
                astro_renderOrganSystemChart(chatData.tcm_predictions, organsChart);
                console.log('Organ system chart rendered');
            } catch (e) {
                console.error('Error rendering organ chart:', e);
            }
        }

        showToast('Dashboard opened', 'success');
    }, 100);
}

// Phase 6: Display Expert Consultations
function astro_displayExpertConsultations(expertConsultations, messageElement) {
    console.log('astro_displayExpertConsultations called with:', expertConsultations);
    if (!messageElement || !expertConsultations) {
        console.warn('astro_displayExpertConsultations: Missing messageElement or expertConsultations', { messageElement: !!messageElement, expertConsultations: !!expertConsultations });
        return;
    }

    const messageContent = messageElement.querySelector('.message-content');
    if (!messageContent) {
        console.error('astro_displayExpertConsultations: No .message-content found');
        return;
    }

    const expertSection = document.createElement('div');
    expertSection.className = 'astro-expert-consultations';

    // Get expert registry from message element (for descriptions)
    let expertRegistry = {};
    try {
        const expertRegistryStr = messageElement.closest('.message')?.dataset?.expertRegistry;
        if (expertRegistryStr) {
            expertRegistry = JSON.parse(expertRegistryStr);
        }
    } catch (e) {
        console.debug('Could not parse expert registry:', e);
    }

    const expertEntries = Object.entries(expertConsultations)
        .filter(([expert, data]) => {
            if (data.error) return false;
            if (!data.insights || data.insights.trim() === '' || data.insights === 'No insights available') return false;
            return true;
        });

    if (expertEntries.length === 0) {
        return;
    }

    expertSection.innerHTML = `
        <div class="astro-section-header-compact">
            <h3>Expert Consultations</h3>
            <span class="expert-count-badge">${expertEntries.length}</span>
        </div>
        <div class="astro-expert-grid-modern">
            ${expertEntries.map(([expert, data], idx) => {
        const expertId = `expert-${expert}-${idx}`;
        const insightsPreview = data.insights.substring(0, 120) + (data.insights.length > 120 ? '...' : '');
        const expertInfo = expertRegistry[expert] || {};
        const expertName = expertInfo.name || expert.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        const expertDescription = expertInfo.description || '';
        const expertFocus = expertInfo.focus || '';
        return `
                    <div class="astro-expert-card-modern" data-expert="${expert}" title="${expertDescription || expertFocus}">
                        <div class="expert-card-header-modern" data-toggle="${expertId}">
                            <div class="expert-title-row">
                                <div style="flex: 1;">
                                    <h4>${expertName}</h4>
                                    ${expertDescription ? `<p style="color: rgba(255,255,255,0.5); font-size: 0.7em; margin: 2px 0 0 0;">${expertDescription}</p>` : ''}
                                </div>
                                <span class="expert-toggle-icon">▼</span>
                            </div>
                        </div>
                        <div class="expert-card-content-modern" id="${expertId}" style="display: none;">
                            <div class="expert-insights-modern">${data.insights}</div>
                            ${data.organ_systems && data.organ_systems.length > 0 ? `
                                <div class="expert-meta-tags">
                                    <span class="meta-label">Systems:</span>
                                    ${data.organ_systems.slice(0, 5).map(sys => `<span class="meta-tag">${sys}</span>`).join('')}
                                    ${data.organ_systems.length > 5 ? `<span class="meta-tag-more">+${data.organ_systems.length - 5}</span>` : ''}
                                </div>
                            ` : ''}
                            ${data.recommended_elements && data.recommended_elements.length > 0 ? `
                                <div class="expert-meta-tags">
                                    <span class="meta-label">Elements:</span>
                                    ${data.recommended_elements.map(el => `<span class="meta-tag">${el}</span>`).join('')}
                                </div>
                            ` : ''}
                            ${data.optimal_times && Object.keys(data.optimal_times).length > 0 ? `
                                <div class="expert-meta-tags">
                                    <span class="meta-label">Times:</span>
                                    ${Object.keys(data.optimal_times).slice(0, 4).map(time => `<span class="meta-tag">${time}</span>`).join('')}
                                    ${Object.keys(data.optimal_times).length > 4 ? `<span class="meta-tag-more">+${Object.keys(data.optimal_times).length - 4}</span>` : ''}
                                </div>
                            ` : ''}
                            <button class="astro-switch-to-expert-btn" data-expert="${expert}" 
                                    style="margin-top: 8px; padding: 6px 12px; background: rgba(0, 212, 255, 0.2); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 4px; cursor: pointer; font-size: 0.8em; width: 100%;">
                                Switch to ${expert.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            </button>
                        </div>
                    </div>
                `;
    }).join('')}
        </div>
    `;

    // Add toggle functionality
    expertSection.querySelectorAll('.expert-card-header-modern').forEach(header => {
        header.addEventListener('click', () => {
            const contentId = header.dataset.toggle;
            const content = document.getElementById(contentId);
            const icon = header.querySelector('.expert-toggle-icon');
            if (content) {
                const isHidden = content.style.display === 'none';
                content.style.display = isHidden ? 'block' : 'none';
                icon.textContent = isHidden ? '▲' : '▼';
            }
        });
    });

    // Add expert switching functionality
    expertSection.querySelectorAll('.astro-switch-to-expert-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const expert = btn.dataset.expert;
            astro_handleExpertSwitch(expert, messageElement);
        });
    });

    // Only append if there are actual expert cards to show
    const expertCards = expertSection.querySelectorAll('.astro-expert-card-modern');
    if (expertCards.length > 0) {
        messageContent.appendChild(expertSection);
        console.log(`Expert consultations displayed: ${expertCards.length} experts with insights`);
    } else {
        console.log('No expert consultations with insights to display');
        expertSection.remove(); // Clean up empty section
    }

    // V2: Initialize expert consultation visualizations if available
    if (typeof renderExpertConsultationCharts !== 'undefined') {
        try {
            renderExpertConsultationCharts(expertConsultations, expertSection);
        } catch (e) {
            console.warn(' Error rendering expert consultation charts:', e);
        }
    }

    // V2: Display real-time adaptation status if available
    if (expertConsultations.adaptation_status) {
        astro_displayAdaptationStatus(expertConsultations.adaptation_status, expertSection);
    }
}

/**
 * Handle expert switching - send query to switch to specific expert
 */
function astro_handleExpertSwitch(expertName, messageElement) {
    console.log('Switching to expert:', expertName);

    // Get the current query input
    const queryInput = document.getElementById('queryInput');
    if (!queryInput) {
        console.error('Query input not found');
        return;
    }

    // Get current mode
    const modeSelect = document.getElementById('modeSelect');
    const currentMode = modeSelect ? modeSelect.value : 'astrophysiology';

    if (currentMode !== 'astrophysiology') {
        console.warn('Not in astrophysiology mode');
        return;
    }

    // Get algorithmic data from localStorage
    const algoDataStr = localStorage.getItem('iceburg_astro_algorithmic_data');
    if (!algoDataStr) {
        console.warn('No algorithmic data found, cannot switch to expert');
        return;
    }

    // Set query to switch to expert
    queryInput.value = `Switch to ${expertName.replace('_', ' ')}`;

    // Create message with expert specified
    const message = {
        query: queryInput.value,
        mode: 'astrophysiology',
        agent: expertName,
        data: {
            agent: expertName,
            expert: expertName,
            algorithmic_data: JSON.parse(algoDataStr)
        }
    };

    // Send message
    // Send message via sendQuery
    sendQuery();

}

/**
 * Add switch back button when in expert mode
 */
function astro_addSwitchBackButton(messageElement, currentExpert) {
    if (!messageElement) return;

    const messageContent = messageElement.querySelector('.message-content');
    if (!messageContent) return;

    // Check if switch back button already exists
    if (messageContent.querySelector('.astro-switch-back-btn')) {
        return;
    }

    const switchBackSection = document.createElement('div');
    switchBackSection.className = 'astro-switch-back-section';
    switchBackSection.style.cssText = `
        margin: 15px 0;
        padding: 12px;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        text-align: center;
    `;

    switchBackSection.innerHTML = `
        <div style="color: rgba(255,255,255,0.7); margin-bottom: 8px; font-size: 0.85em;">
            Currently viewing: <strong style="color: #00D4FF;">${currentExpert.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</strong>
        </div>
        <button class="astro-switch-back-btn" 
                style="padding: 8px 16px; background: rgba(0, 212, 255, 0.2); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 4px; cursor: pointer; font-size: 0.9em;">
            Switch Back to Main Analysis
        </button>
    `;

    messageContent.appendChild(switchBackSection);

    // Add click handler
    const switchBackBtn = switchBackSection.querySelector('.astro-switch-back-btn');
    if (switchBackBtn) {
        switchBackBtn.addEventListener('click', () => {
            const queryInput = document.getElementById('queryInput');
            if (queryInput) {
                queryInput.value = 'switch back';

                // Get algorithmic data
                const algoDataStr = localStorage.getItem('iceburg_astro_algorithmic_data');
                // switch back via sendQuery
                if (algoDataStr) {
                    // pre-fill input is already done
                    sendQuery(); // This uses the input value 'switch back' and handles the rest
                }

            }
        });
    }
}

/**
 * Display expert suggestions in result card
 */
function astro_showExpertSuggestions(suggestions, messageElement) {
    if (!messageElement || !suggestions || suggestions.length === 0) return;

    const messageContent = messageElement.querySelector('.message-content');
    if (!messageContent) return;

    // Check if suggestions already displayed
    if (messageContent.querySelector('.astro-expert-suggestions')) {
        return;
    }

    const suggestionsSection = document.createElement('div');
    suggestionsSection.className = 'astro-expert-suggestions';
    suggestionsSection.style.cssText = `
        margin: 15px 0;
        padding: 12px;
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
    `;

    suggestionsSection.innerHTML = `
        <div class="astro-section-header-compact" style="margin-bottom: 12px;">
            <h3 style="color: #00D4FF; margin: 0; font-size: 0.9em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                Suggested Experts
            </h3>
        </div>
        <div class="astro-suggestions-grid" style="display: grid; gap: 8px;">
            ${suggestions.map(suggestion => `
                <div class="astro-suggestion-card" style="background: rgba(255,255,255,0.02); padding: 10px; border-radius: 6px; border: 1px solid rgba(0, 212, 255, 0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 6px;">
                        <h4 style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.85em; font-weight: 600;">
                            ${suggestion.expert.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </h4>
                    </div>
                    <p style="color: rgba(255,255,255,0.6); margin: 0 0 6px 0; font-size: 0.75em; line-height: 1.4;">
                        ${suggestion.reason}
                    </p>
                    <p style="color: rgba(255,255,255,0.5); margin: 0 0 8px 0; font-size: 0.7em; font-style: italic;">
                        ${suggestion.description}
                    </p>
                    <button class="astro-switch-to-expert-btn" data-expert="${suggestion.expert}" 
                            style="width: 100%; padding: 6px 10px; background: rgba(0, 212, 255, 0.2); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 4px; cursor: pointer; font-size: 0.8em;">
                        Switch to ${suggestion.expert.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </button>
                </div>
            `).join('')}
        </div>
    `;

    messageContent.appendChild(suggestionsSection);

    // Add click handlers for suggestion buttons
    suggestionsSection.querySelectorAll('.astro-switch-to-expert-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const expert = btn.dataset.expert;
            astro_handleExpertSwitch(expert, messageElement);
        });
    });
}

// Phase 7: Display Interventions
function astro_displayInterventions(interventions, messageElement) {
    console.log(' astro_displayInterventions called with:', interventions);
    if (!messageElement || !interventions) {
        console.warn(' astro_displayInterventions: Missing messageElement or interventions', { messageElement: !!messageElement, interventions: !!interventions });
        return;
    }

    const messageContent = messageElement.querySelector('.message-content');
    if (!messageContent) {
        console.error(' astro_displayInterventions: No .message-content found');
        return;
    }

    const interventionSection = document.createElement('div');
    interventionSection.className = 'astro-interventions-modern';

    // Parse text into sections
    const textContent = interventions.text || '';
    const sections = textContent.split(/\*\*([^*]+)\*\*/).filter(s => s.trim());

    // Parse and format intervention text properly
    const formatInterventionText = (text) => {
        if (!text) return '';

        // Remove "Algorithmic Data Summary" and everything after it
        const summaryIndex = text.indexOf('Algorithmic Data Summary');
        if (summaryIndex > -1) {
            text = text.substring(0, summaryIndex).trim();
        }

        // Split by double newlines but keep structure
        let html = '';
        const lines = text.split('\n');
        let inList = false;
        let currentList = [];

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            // Check for Roman numeral sections (I., II., III., etc.)
            if (line.match(/^[IVX]+\.\s+/)) {
                if (currentList.length > 0) {
                    html += `<ul class="intervention-list-compact">${currentList.map(item => `<li>${item}</li>`).join('')}</ul>`;
                    currentList = [];
                    inList = false;
                }
                const sectionTitle = line.replace(/^[IVX]+\.\s+/, '');
                html += `<h4 class="intervention-section-title">${sectionTitle}</h4>`;
            }
            // Check for bold headings
            else if (line.match(/^\*\*.*\*\*$/)) {
                if (currentList.length > 0) {
                    html += `<ul class="intervention-list-compact">${currentList.map(item => `<li>${item}</li>`).join('')}</ul>`;
                    currentList = [];
                    inList = false;
                }
                html += `<h5 class="intervention-subheading">${line.replace(/\*\*/g, '')}</h5>`;
            }
            // Check for list items (starting with *, -, +, or indented)
            else if (line.match(/^[\*\-\+]\s+/) || line.match(/^\d+\.\s+/) || (line.startsWith('  ') && line.trim().length > 0)) {
                if (!inList) {
                    inList = true;
                }
                const listItem = line.replace(/^[\*\-\+]\s+/, '').replace(/^\d+\.\s+/, '').trim();
                currentList.push(listItem);
            }
            // Regular paragraph
            else {
                if (currentList.length > 0) {
                    html += `<ul class="intervention-list-compact">${currentList.map(item => `<li>${item}</li>`).join('')}</ul>`;
                    currentList = [];
                    inList = false;
                }
                html += `<p class="intervention-para-compact">${line}</p>`;
            }
        }

        // Close any remaining list
        if (currentList.length > 0) {
            html += `<ul class="intervention-list-compact">${currentList.map(item => `<li>${item}</li>`).join('')}</ul>`;
        }

        return html;
    };

    interventionSection.innerHTML = `
        <div class="astro-section-header-compact">
            <h3>Interventions</h3>
        </div>
        <div class="astro-intervention-content-modern">
            ${textContent ? `
                <div class="intervention-text-modern-compact">
                    ${formatInterventionText(textContent)}
                </div>
            ` : '<p class="intervention-empty">No interventions available</p>'}
        </div>
    `;

    messageContent.appendChild(interventionSection);

    console.log(' Interventions displayed');

    // V2: Initialize intervention progress visualizations if available
    if (interventions.tracking_metadata && typeof renderInterventionProgressChart !== 'undefined') {
        try {
            const progressContainer = document.createElement('div');
            progressContainer.className = 'astro-intervention-progress';
            progressContainer.style.cssText = 'margin: 20px 0; padding: 15px; background: rgba(0, 212, 255, 0.05); border-radius: 8px;';
            progressContainer.innerHTML = '<h4 style="color: #00D4FF; margin: 0 0 15px 0;">Intervention Progress Tracking</h4><div id="astro-intervention-progress-chart"></div>';
            interventionSection.appendChild(progressContainer);
            setTimeout(() => {
                renderInterventionProgressChart(interventions.tracking_metadata, document.getElementById('astro-intervention-progress-chart'));
            }, 100);
        } catch (e) {
            console.warn(' Error rendering intervention progress chart:', e);
        }
    }

    // V2: Feedback form removed per user request
}

/**
 * V2: Display testable hypotheses (Research Tool Mode)
 */
function astro_displayHypotheses(hypotheses, messageElement) {
    console.log('Astro displayHypotheses called with:', hypotheses);
    if (!messageElement || !hypotheses || hypotheses.length === 0) {
        console.warn('Astro displayHypotheses: Missing messageElement or hypotheses', { messageElement: !!messageElement, hypotheses: !!hypotheses });
        return;
    }

    const messageContent = messageElement.querySelector('.message-content');
    if (!messageContent) {
        console.error('Astro displayHypotheses: No .message-content found');
        return;
    }

    // Check if hypotheses section already exists
    if (messageContent.querySelector('.astro-hypotheses')) {
        console.log('Hypotheses section already exists, skipping duplicate');
        return;
    }

    const hypothesesSection = document.createElement('div');
    hypothesesSection.className = 'astro-hypotheses-compact';
    hypothesesSection.style.cssText = 'margin: 2px 0; padding: 2px; background: transparent; border: none;';

    // Collapsible header
    const headerId = `hypotheses-header-${Date.now()}`;
    const contentId = `hypotheses-content-${Date.now()}`;

    hypothesesSection.innerHTML = `
        <div class="astro-hypotheses-header-compact" id="${headerId}" style="display: flex; justify-content: space-between; align-items: center; cursor: pointer; padding: 2px 0; border-bottom: 1px solid rgba(138, 43, 226, 0.2);">
            <div style="display: flex; align-items: center; gap: 4px;">
                <h3 style="color: #8A2BE2; margin: 0; font-size: 0.75em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Hypotheses</h3>
                <span style="background: rgba(138, 43, 226, 0.2); color: #8A2BE2; padding: 1px 4px; border-radius: 8px; font-size: 0.65em; font-weight: 600;">${hypotheses.length}</span>
            </div>
            <span class="hypotheses-toggle" style="color: #8A2BE2; font-size: 0.7em;">▼</span>
        </div>
        <div class="astro-hypotheses-content-compact" id="${contentId}" style="display: none; margin-top: 2px;">
            <div class="astro-hypotheses-grid-compact">
                ${hypotheses.map((h, idx) => `
                    <div class="astro-hypothesis-card-compact" style="background: rgba(138, 43, 226, 0.1); border-left: 2px solid ${h.priority === 1 ? '#00D4FF' : h.priority === 2 ? '#8A2BE2' : '#666'}; padding: 3px 4px; border-radius: 2px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                            <span style="color: #00D4FF; font-size: 0.7em; font-weight: 600;">H${idx + 1}</span>
                            <span style="padding: 1px 4px; background: ${h.priority === 1 ? 'rgba(0, 212, 255, 0.2)' : h.priority === 2 ? 'rgba(138, 43, 226, 0.2)' : 'rgba(102, 102, 102, 0.2)'}; border-radius: 2px; font-size: 0.65em; color: #ccc;">
                                ${h.priority === 1 ? 'H' : h.priority === 2 ? 'M' : 'L'}
                            </span>
                        </div>
                        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.7em; line-height: 1.3; font-weight: 500;">${h.hypothesis}</p>
                        <details style="margin-top: 2px;">
                            <summary style="color: #8A2BE2; cursor: pointer; font-size: 0.65em;">More</summary>
                            <div style="margin-top: 2px; padding: 2px; background: rgba(0, 0, 0, 0.2); border-radius: 2px; font-size: 0.65em;">
                                <div style="margin: 1px 0;"><strong style="color: #00D4FF;">P:</strong> <span style="color: #ccc;">${h.testable_prediction}</span></div>
                                <div style="margin: 1px 0;"><strong style="color: #00D4FF;">E:</strong> <span style="color: #ccc;">${h.expected_effect_size}</span></div>
                                <div style="margin: 1px 0;"><strong style="color: #00D4FF;">C:</strong> <span style="color: #ccc;">${h.confidence}</span></div>
                            </div>
                        </details>
                    </div>
                `).join('')}
            </div>
        </div>
    `;

    messageContent.appendChild(hypothesesSection);

    // Add toggle functionality
    const header = document.getElementById(headerId);
    const content = document.getElementById(contentId);
    const toggle = header.querySelector('.hypotheses-toggle');

    if (header && content && toggle) {
        header.addEventListener('click', () => {
            const isHidden = content.style.display === 'none';
            content.style.display = isHidden ? 'block' : 'none';
            toggle.textContent = isHidden ? '▲' : '▼';
        });
    }
}

/**
 * V2: Update agent select options based on mode
 */
function updateAgentOptionsForMode(mode) {
    const agentSelect = document.getElementById('agentSelect');
    if (!agentSelect) return;

    // Save current selection
    const currentValue = agentSelect.value;

    if (mode === 'astrophysiology') {
        // Astro-Physiology mode: Show expert agents
        agentSelect.innerHTML = `
            <option value="auto">Auto (Full Protocol)</option>
            <option value="health_expert">Health Expert</option>
            <option value="nutrition_expert">Nutrition Expert</option>
            <option value="movement_expert">Movement Expert</option>
            <option value="chart_reader">Chart Reader</option>
            <option value="sleep_expert">Sleep Expert</option>
            <option value="stress_expert">Stress Expert</option>
            <option value="hormone_expert">Hormone Expert</option>
            <option value="digestive_expert">Digestive Expert</option>
        `;
        console.log(' Updated agent select for astro-physiology mode');
    } else if (mode === 'chat') {
        // Chat mode: Show agent display instead of dropdown
        const agentDisplay = document.getElementById('agentDisplay');
        const agentDisplayValue = document.getElementById('agentDisplayValue');
        if (agentSelect) agentSelect.style.display = 'none';
        if (agentDisplay) {
            agentDisplay.style.display = 'flex';
            if (agentDisplayValue) agentDisplayValue.textContent = 'Secretary';
        }
        console.log('👔 Chat mode: Showing agent display (Secretary)');
    } else {
        // Other modes: Show standard agents
        agentSelect.innerHTML = `
            <option value="auto">Auto (Full Protocol)</option>
            <option value="secretary">Secretary (Chat Assistant)</option>
            <option value="dissident">Dissident</option>
            <option value="synthesist">Synthesist</option>
            <option value="oracle">Oracle</option>
            <option value="archaeologist">Archaeologist</option>
            <option value="supervisor">Supervisor</option>
            <option value="scribe">Scribe</option>
            <option value="weaver">Weaver</option>
            <option value="scrutineer">Scrutineer</option>
            <option value="swarm">Swarm</option>
            <option value="ide">IDE Agent</option>
        `;
        // Unlock for other modes
        agentSelect.disabled = false;
        agentSelect.title = '';
    }

    // Restore selection if still valid, otherwise default to auto
    if (agentSelect.querySelector(`option[value="${currentValue}"]`)) {
        agentSelect.value = currentValue;
    } else {
        agentSelect.value = 'auto';
    }
}

/**
 * V2: Add feedback form for interventions
 * REMOVED - Feedback/suggestions disabled per user request
 */
function astro_addInterventionFeedbackForm(container, interventions) {
    // Feedback form removed - no longer displaying feedback/suggestions
    return;
    feedbackSection.style.cssText = 'margin: 20px 0; padding: 15px; background: rgba(0, 212, 255, 0.05); border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.2);';

    feedbackSection.innerHTML = `
        <h4 style="color: #00D4FF; margin: 0 0 15px 0;">Share Your Progress</h4>
        <p style="color: #ccc; font-size: 0.9em; margin: 0 0 15px 0;">
            Help us improve recommendations by sharing how interventions are working for you.
        </p>
        <form id="astro-feedback-form" style="display: flex; flex-direction: column; gap: 10px;">
            <div>
                <label style="color: #ccc; font-size: 0.9em; display: block; margin-bottom: 5px;">
                    Overall Effectiveness (1-5)
                </label>
                <select name="effectiveness" required style="width: 100%; padding: 8px; background: rgba(0,0,0,0.3); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 4px; color: #fff;">
                    <option value="">Select...</option>
                    <option value="1">1 - Not effective</option>
                    <option value="2">2 - Slightly effective</option>
                    <option value="3">3 - Moderately effective</option>
                    <option value="4">4 - Very effective</option>
                    <option value="5">5 - Extremely effective</option>
                </select>
            </div>
            <div>
                <label style="color: #ccc; font-size: 0.9em; display: block; margin-bottom: 5px;">
                    Adherence Level (1-5)
                </label>
                <select name="adherence" required style="width: 100%; padding: 8px; background: rgba(0,0,0,0.3); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 4px; color: #fff;">
                    <option value="">Select...</option>
                    <option value="1">1 - Not following</option>
                    <option value="2">2 - Rarely following</option>
                    <option value="3">3 - Sometimes following</option>
                    <option value="4">4 - Usually following</option>
                    <option value="5">5 - Consistently following</option>
                </select>
            </div>
            <div>
                <label style="color: #ccc; font-size: 0.9em; display: block; margin-bottom: 5px;">
                    Notes (optional)
                </label>
                <textarea name="notes" rows="3" placeholder="Share any observations, challenges, or improvements..." style="width: 100%; padding: 8px; background: rgba(0,0,0,0.3); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 4px; color: #fff; resize: vertical;"></textarea>
            </div>
            <button type="submit" style="padding: 10px; background: rgba(0, 212, 255, 0.3); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.5); border-radius: 4px; cursor: pointer; font-weight: 600;">
                Submit Feedback
            </button>
        </form>
        <div id="astro-feedback-status" style="margin-top: 10px; display: none;"></div>
    `;

    container.appendChild(feedbackSection);

    // Handle form submission
    const form = feedbackSection.querySelector('#astro-feedback-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            const feedback = {
                effectiveness: parseInt(formData.get('effectiveness')),
                adherence: parseInt(formData.get('adherence')),
                notes: formData.get('notes') || '',
                timestamp: new Date().toISOString(),
                intervention_id: interventions.id || `intervention_${Date.now()}`
            };

            const statusDiv = feedbackSection.querySelector('#astro-feedback-status');
            statusDiv.style.display = 'block';
            statusDiv.style.color = '#00D4FF';
            statusDiv.textContent = 'Submitting feedback...';

            try {
                // Send feedback to backend
                const response = await fetch('/api/astro-physiology/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(feedback)
                });

                if (response.ok) {
                    statusDiv.style.color = '#4CAF50';
                    statusDiv.textContent = '✓ Feedback submitted successfully! Thank you for helping us improve.';
                    form.reset();

                    // Store feedback locally for tracking
                    const storedFeedback = JSON.parse(localStorage.getItem('iceburg_astro_feedback') || '[]');
                    storedFeedback.push(feedback);
                    localStorage.setItem('iceburg_astro_feedback', JSON.stringify(storedFeedback));
                } else {
                    throw new Error('Failed to submit feedback');
                }
            } catch (error) {
                console.error(' Error submitting feedback:', error);
                statusDiv.style.color = '#FF6B6B';
                statusDiv.textContent = '✗ Error submitting feedback. Please try again.';
            }
        });
    }
}

/**
 * V2: Display real-time adaptation status
 */
function astro_displayAdaptationStatus(adaptationData, container) {
    if (!adaptationData || !container) return;

    const adaptationSection = document.createElement('div');
    adaptationSection.className = 'astro-adaptation-status';
    adaptationSection.style.cssText = 'margin: 20px 0; padding: 15px; background: rgba(0, 212, 255, 0.05); border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.2);';

    const status = adaptationData.status || 'monitoring';
    const lastUpdate = adaptationData.last_update ? new Date(adaptationData.last_update).toLocaleString() : 'Never';
    const changes = adaptationData.recent_changes || [];

    adaptationSection.innerHTML = `
        <h4 style="color: #00D4FF; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${status === 'active' ? '#4CAF50' : status === 'pending' ? '#FFA500' : '#999'};"></span>
            Real-Time Adaptation Status
        </h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 15px;">
            <div>
                <div style="color: #999; font-size: 0.85em; margin-bottom: 5px;">Status</div>
                <div style="color: #fff; font-weight: 600; text-transform: capitalize;">${status}</div>
            </div>
            <div>
                <div style="color: #999; font-size: 0.85em; margin-bottom: 5px;">Last Update</div>
                <div style="color: #fff; font-weight: 600;">${lastUpdate}</div>
            </div>
            <div>
                <div style="color: #999; font-size: 0.85em; margin-bottom: 5px;">Adaptations</div>
                <div style="color: #fff; font-weight: 600;">${changes.length} recent</div>
            </div>
        </div>
        ${changes.length > 0 ? `
            <div style="margin-top: 15px;">
                <div style="color: #ccc; font-size: 0.9em; margin-bottom: 10px;">Recent Adaptations:</div>
                <ul style="list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 8px;">
                    ${changes.slice(0, 3).map(change => `
                        <li style="padding: 8px; background: rgba(255,255,255,0.05); border-radius: 4px; color: #ccc; font-size: 0.9em;">
                            <strong style="color: #00D4FF;">${change.type || 'Update'}:</strong> ${change.description || 'No description'}
                            ${change.timestamp ? `<span style="color: #999; margin-left: 10px; font-size: 0.85em;">${new Date(change.timestamp).toLocaleDateString()}</span>` : ''}
                        </li>
                    `).join('')}
                </ul>
            </div>
        ` : ''}
    `;

    container.appendChild(adaptationSection);
}

/**
 * Export astro-physiology data
 */
function astro_exportData(data) {
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `astro-physiology-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
    showToast('Data exported', 'success');
}

/**
 * Update UI based on selected mode
 */
function astro_updateModeUI(mode) {
    const queryInput = document.getElementById('queryInput');
    const modeDescription = document.getElementById('astro-mode-description');
    const quickActions = document.getElementById('astro-quick-actions');

    if (mode === 'astrophysiology') {
        // Show mode description if it doesn't exist
        if (!modeDescription) {
            const controlsPanel = document.getElementById('controlsPanel');
            console.log('🔍 Creating astro-physiology description box...', {
                controlsPanel: !!controlsPanel,
                existingDesc: !!controlsPanel?.querySelector('#astro-mode-description')
            });

            if (controlsPanel && !controlsPanel.querySelector('#astro-mode-description')) {
                const desc = document.createElement('div');
                desc.id = 'astro-mode-description';
                desc.className = 'astro-mode-description';
                // Make it full width in flex container
                desc.style.cssText = `
                    width: 100%;
                    flex-basis: 100%;
                    background: rgba(0, 212, 255, 0.15);
                    border: 1px solid rgba(0, 212, 255, 0.4);
                    border-radius: 8px;
                    padding: 14px;
                    margin: 8px 0 0 0;
                    color: #fff;
                    font-size: 0.9em;
                    display: block !important;
                    visibility: visible !important;
                    opacity: 1 !important;
                    order: 999;
                `;
                desc.innerHTML = `
                    <strong style="color: #00D4FF; font-size: 1em; display: block; margin-bottom: 6px;">Astro-Physiology Mode</strong>
                    <p style="margin: 0; color: #ccc; line-height: 1.5;">
                        Analyze how celestial electromagnetic fields affect biological systems through molecular mechanisms.
                        <a href="#" id="astro-learn-more" style="color: #00D4FF; text-decoration: underline; margin-left: 6px; cursor: pointer; font-weight: 500;">Learn more</a>
                    </p>
                `;
                // Insert at the end of controls panel (after all control groups)
                controlsPanel.appendChild(desc);
                console.log('✅ Description box created and appended to controlsPanel');
                console.log('📍 ControlsPanel children:', controlsPanel.children.length);

                // Add learn more handler with enhanced popup
                const learnMore = desc.querySelector('#astro-learn-more');
                if (learnMore) {
                    learnMore.addEventListener('click', (e) => {
                        e.preventDefault();
                        console.log('📖 Learn more clicked');
                        astro_showLearnMorePopup();
                    });
                    console.log('✅ Learn more handler attached');
                } else {
                    console.warn('⚠️ Learn more link not found');
                }
            } else {
                console.warn('⚠️ Controls panel not found or description already exists');
            }
        } else {
            modeDescription.style.display = 'block';
            modeDescription.style.visibility = 'visible';
            modeDescription.style.opacity = '1';
            console.log('✅ Existing description box shown');
        }

        // Show quick actions above prompt area, outside suggestion border
        if (!quickActions) {
            const inputContainer = document.querySelector('.input-container');
            const promptCarousel = document.getElementById('promptCarousel');

            if (inputContainer && promptCarousel && !inputContainer.querySelector('#astro-quick-actions')) {
                const actions = document.createElement('div');
                actions.id = 'astro-quick-actions';
                actions.className = 'astro-quick-actions';
                actions.style.cssText = `
                    display: flex;
                    gap: 4px;
                    margin-bottom: 8px;
                    width: 100%;
                    justify-content: flex-start;
                    flex-wrap: nowrap;
                `;
                actions.innerHTML = `
                    <button class="astro-quick-action" data-action="today" 
                            style="flex: 1; min-width: 0; padding: 4px 8px; background: rgba(0, 212, 255, 0.15); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 3px; cursor: pointer; font-size: 0.75em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        Today's Correlations
                    </button>
                    <button class="astro-quick-action" data-action="compare" 
                            style="flex: 1; min-width: 0; padding: 4px 8px; background: rgba(0, 212, 255, 0.15); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 3px; cursor: pointer; font-size: 0.75em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        Compare to Average
                    </button>
                    <button class="astro-quick-action" data-action="export" 
                            style="flex: 1; min-width: 0; padding: 4px 8px; background: rgba(0, 212, 255, 0.15); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 3px; cursor: pointer; font-size: 0.75em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        Export Report
                    </button>
                `;
                // Insert before prompt carousel
                inputContainer.insertBefore(actions, promptCarousel);

                // Add event listeners
                actions.querySelectorAll('.astro-quick-action').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const action = btn.dataset.action;
                        astro_handleQuickAction(action);
                    });
                });
            }
        } else if (quickActions) {
            quickActions.style.display = 'flex';
        }

        // Update query input placeholder and show examples
        if (queryInput) {
            const saved = localStorage.getItem('iceburg_astro_birth_data');
            if (saved) {
                queryInput.placeholder = 'Ask about your molecular blueprint, traits, or organ systems...';
            } else {
                queryInput.placeholder = 'Enter birth info: "Dec 26 1991, 7:20 AM, Dallas" or use the form below...';
            }

            // Show example queries on focus
            if (!queryInput.dataset.astroExamplesAdded) {
                queryInput.addEventListener('focus', () => {
                    if (mode === 'astrophysiology' && !queryInput.value) {
                        astro_showExampleQueries(queryInput);
                    }
                });
                queryInput.dataset.astroExamplesAdded = 'true';
            }
        }
    } else {
        // Hide mode description
        if (modeDescription) {
            modeDescription.style.display = 'none';
        }

        // Hide quick actions
        if (quickActions) {
            quickActions.style.display = 'none';
        }

        // Reset query input placeholder
        if (queryInput) {
            queryInput.placeholder = 'Ask anything to begin...';
        }
    }
}

/**
 * Handle quick action buttons
 */
function astro_handleQuickAction(action) {
    const modeSelect = document.getElementById('modeSelect');
    if (modeSelect && modeSelect.value !== 'astrophysiology') {
        modeSelect.value = 'astrophysiology';
        astro_updateModeUI('astrophysiology');
    }

    const queryInput = document.getElementById('queryInput');
    if (!queryInput) return;

    switch (action) {
        case 'today':
            queryInput.value = "What are today's celestial correlations for me?";
            break;
        case 'compare':
            queryInput.value = "Compare my traits to the population average";
            break;
        case 'export':
            // Find last astro result and export
            const lastMessage = document.querySelector('.message.assistant:last-child');
            if (lastMessage) {
                const card = lastMessage.querySelector('.astro-card');
                if (card) {
                    const exportBtn = card.querySelector('.astro-export-btn');
                    if (exportBtn) exportBtn.click();
                } else {
                    showToast('No analysis data to export. Run an analysis first.', 'warning');
                }
            } else {
                showToast('No analysis data to export. Run an analysis first.', 'warning');
            }
            return;
    }

    // Trigger send if query was set
    if (action !== 'export') {
        const sendButton = document.getElementById('sendButton');
        if (sendButton) {
            sendButton.click();
        }
    }
}

/**
 * Show learn more popup for astro-physiology mode
 */
function astro_showLearnMorePopup() {
    // Remove existing popup if any
    const existingPopup = document.getElementById('astro-learn-more-popup');
    if (existingPopup) {
        existingPopup.remove();
    }

    // Create popup overlay
    const overlay = document.createElement('div');
    overlay.id = 'astro-learn-more-popup';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.7);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
    `;

    // Create popup content
    const popup = document.createElement('div');
    popup.style.cssText = `
        background: linear-gradient(135deg, rgba(0, 20, 40, 0.95), rgba(0, 40, 60, 0.95));
        border: 2px solid rgba(0, 212, 255, 0.5);
        border-radius: 12px;
        padding: 24px;
        max-width: 600px;
        max-height: 80vh;
        overflow-y: auto;
        color: #fff;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.3);
        position: relative;
    `;

    popup.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <h3 style="color: #00D4FF; margin: 0; font-size: 1.3em;">Astro-Physiology Mode</h3>
            <button id="astro-popup-close" style="
                background: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                color: #fff;
                width: 32px;
                height: 32px;
                border-radius: 50%;
                cursor: pointer;
                font-size: 1.2em;
                display: flex;
                align-items: center;
                justify-content: center;
            ">×</button>
        </div>
        <div style="line-height: 1.6; color: #ccc;">
            <p style="margin: 0 0 16px 0;">
                <strong style="color: #00D4FF;">Astro-Physiology</strong> analyzes how celestial electromagnetic fields 
                affect biological systems through molecular mechanisms.
            </p>
            <div style="margin: 16px 0;">
                <h4 style="color: #00D4FF; margin: 0 0 8px 0; font-size: 1.1em;">How It Works:</h4>
                <ul style="margin: 8px 0; padding-left: 20px;">
                    <li style="margin: 6px 0;">Uses quantum chemistry and EM field physics to model correlations</li>
                    <li style="margin: 6px 0;">Analyzes molecular blueprints based on birth data</li>
                    <li style="margin: 6px 0;">Predicts behavioral traits and organ system correlations</li>
                    <li style="margin: 6px 0;">Provides expert consultations and interventions</li>
                    <li style="margin: 6px 0;">Generates testable hypotheses for validation</li>
                </ul>
            </div>
            <div style="margin: 16px 0;">
                <h4 style="color: #00D4FF; margin: 0 0 8px 0; font-size: 1.1em;">What You Need:</h4>
                <p style="margin: 0;">
                    Birth date, time, and location to calculate your molecular imprint and celestial positions.
                </p>
            </div>
            <div style="margin: 16px 0; padding: 12px; background: rgba(0, 212, 255, 0.1); border-radius: 6px; border-left: 3px solid #00D4FF;">
                <p style="margin: 0; font-style: italic; color: #00D4FF;">
                    This mode uses advanced physics modeling to explore the connections between 
                    celestial events and biological systems.
                </p>
            </div>
        </div>
    `;

    overlay.appendChild(popup);
    document.body.appendChild(overlay);

    // Close handlers
    const closeBtn = popup.querySelector('#astro-popup-close');
    const closePopup = () => {
        overlay.remove();
    };

    closeBtn.addEventListener('click', closePopup);
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
            closePopup();
        }
    });

    // ESC key to close
    const escHandler = (e) => {
        if (e.key === 'Escape') {
            closePopup();
            document.removeEventListener('keydown', escHandler);
        }
    };
    document.addEventListener('keydown', escHandler);
}

/**
 * Show loading state for astro-physiology mode
 */
function astro_showLoadingState(messageContent) {
    if (!messageContent) return;

    const loadingCard = document.createElement('div');
    loadingCard.className = 'astro-loading-state';
    loadingCard.style.cssText = `
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
    `;

    loadingCard.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 15px;">
            <div class="astro-loading-spinner" style="width: 24px; height: 24px; border: 3px solid rgba(0, 212, 255, 0.3); border-top-color: #00D4FF; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <h3 style="color: #00D4FF; margin: 0; font-size: 1em;">🔬 Calculating Molecular Blueprint...</h3>
        </div>
        <div class="astro-loading-steps" style="display: grid; gap: 8px;">
            <div class="astro-step" data-step="1" style="display: flex; align-items: center; gap: 8px; color: #ccc; font-size: 0.9em;">
                <span class="astro-step-icon">⏳</span>
                <span class="astro-step-text">Calculating celestial positions...</span>
            </div>
            <div class="astro-step" data-step="2" style="display: flex; align-items: center; gap: 8px; color: #999; font-size: 0.9em;">
                <span class="astro-step-icon">⏳</span>
                <span class="astro-step-text">Analyzing EM environment...</span>
            </div>
            <div class="astro-step" data-step="3" style="display: flex; align-items: center; gap: 8px; color: #999; font-size: 0.9em;">
                <span class="astro-step-icon">⏳</span>
                <span class="astro-step-text">Computing molecular configurations...</span>
            </div>
            <div class="astro-step" data-step="4" style="display: flex; align-items: center; gap: 8px; color: #999; font-size: 0.9em;">
                <span class="astro-step-icon">⏳</span>
                <span class="astro-step-text">Predicting behavioral traits...</span>
            </div>
        </div>
        <style>
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        </style>
    `;

    messageContent.appendChild(loadingCard);

    // Update steps as processing continues
    let currentStep = 0;
    const stepInterval = setInterval(() => {
        currentStep++;
        if (currentStep > 4) {
            clearInterval(stepInterval);
            return;
        }

        const stepEl = loadingCard.querySelector(`.astro-step[data-step="${currentStep}"]`);
        if (stepEl) {
            stepEl.style.color = '#00D4FF';
            const icon = stepEl.querySelector('.astro-step-icon');
            if (icon) icon.textContent = '✓';
        }

        // Hide previous steps
        for (let i = 1; i < currentStep; i++) {
            const prevStep = loadingCard.querySelector(`.astro-step[data-step="${i}"]`);
            if (prevStep) prevStep.style.opacity = '0.5';
        }
    }, 2000);

    // Store reference for cleanup
    loadingCard.dataset.astroLoadingId = Date.now();
}

/**
 * Show rich loading state for any mode (Research, Fast, Truth, Hybrid, etc.)
 * Similar to astrophysiology but with mode-specific processing steps
 */
function showModeLoadingState(messageContent, mode = 'research') {
    if (!messageContent) return null;

    // Mode-specific steps configuration
    const modeSteps = {
        fast: [
            { icon: '1', text: 'Routing to Secretary...' },
            { icon: '2', text: 'Generating response...' }
        ],
        chat: [
            { icon: '1', text: 'Processing query...' },
            { icon: '2', text: 'Generating response...' }
        ],
        research: [
            { icon: '1', text: 'Routing query...' },
            { icon: '2', text: 'Surveyor: Web research...' },
            { icon: '3', text: 'Deliberation: Analyzing sources...' },
            { icon: '4', text: 'Synthesist: Compiling answer...' }
        ],
        deep_research: [
            { icon: '1', text: 'Query interpretation...' },
            { icon: '2', text: 'Surveyor: Deep web research...' },
            { icon: '3', text: 'Dissident: Alternative perspectives...' },
            { icon: '4', text: 'Deliberation: Cross-referencing...' },
            { icon: '5', text: 'Synthesist: Final synthesis...' }
        ],
        truth: [
            { icon: '1', text: 'Query analysis...' },
            { icon: '2', text: 'Suppression detection...' },
            { icon: '3', text: 'Source verification...' },
            { icon: '4', text: 'Truth synthesis...' }
        ],
        hybrid: [
            { icon: '1', text: 'Routing decision...' },
            { icon: '2', text: 'Web search...' },
            { icon: '3', text: 'Local RAG search...' },
            { icon: '4', text: 'Merging results...' }
        ],
        web_research: [
            { icon: '1', text: 'Query parsing...' },
            { icon: '2', text: 'Web search...' },
            { icon: '3', text: 'Result synthesis...' }
        ],
        local_rag: [
            { icon: '1', text: 'Query embedding...' },
            { icon: '2', text: 'Vector search...' },
            { icon: '3', text: 'Context compilation...' }
        ]
    };
    
    // Add missing modes to steps
    Object.assign(modeSteps, {
        protocol: [
            { icon: '1', text: 'Direct agent connection...' },
            { icon: '2', text: 'Processing request...' }
        ],
        code: [
            { icon: '1', text: 'Analysis: Reading codebase...' },
            { icon: '2', text: 'Weaver: Implementing changes...' },
            { icon: '3', text: 'IDE: Verifying syntax...' }
        ],
        finance: [
            { icon: '1', text: 'Market data retrieval...' },
            { icon: '2', text: 'Analyst: Technical analysis...' },
            { icon: '3', text: 'Synthesist: Risk assessment...' }
        ],
        civilization: [
            { icon: '1', text: 'Initializing simulation parameters...' },
            { icon: '2', text: 'SocialNormSystem: Calculating dynamics...' },
            { icon: '3', text: 'Emergence: Detecting patterns...' }
        ],
        astrophysiology: [
            { icon: '1', text: 'Calculating celestial positions...' },
            { icon: '2', text: 'Computing molecular imprint...' },
            { icon: '3', text: 'Mapping biological correlations...' }
        ],
        dossier: [
            { icon: '1', text: 'Archive search...' },
            { icon: '2', text: 'Scrutineer: validating records...' },
            { icon: '3', text: 'Profiler: Constructing timeline...' }
        ]
    });

    const steps = modeSteps[mode] || modeSteps.research;
    const modeLabel = mode.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

    const loadingCard = document.createElement('div');
    loadingCard.className = 'mode-loading-state';
    loadingCard.dataset.mode = mode;
    loadingCard.style.cssText = `
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    `;

    const stepsHTML = steps.map((step, i) => `
        <div class="mode-step" data-step="${i + 1}" style="display: flex; align-items: center; gap: 10px; color: ${i === 0 ? 'var(--text-primary)' : 'var(--text-tertiary)'}; font-size: 0.875rem; padding: 4px 0; transition: all 0.3s ease;">
            <span class="mode-step-icon" style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: ${i === 0 ? 'var(--text-primary)' : 'var(--bg-tertiary)'}; color: ${i === 0 ? 'var(--bg-primary)' : 'var(--text-tertiary)'}; font-size: 0.7rem; font-weight: 600;">${step.icon}</span>
            <span class="mode-step-text">${step.text}</span>
        </div>
    `).join('');

    loadingCard.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <div class="mode-loading-spinner" style="width: 18px; height: 18px; border: 2px solid var(--border-primary); border-top-color: var(--text-primary); border-radius: 50%; animation: mode-spin 1s linear infinite;"></div>
            <span class="mode-loading-title" style="color: var(--text-primary); font-size: 0.9rem; font-weight: 500;">${modeLabel} Mode Processing</span>
        </div>
        <div class="mode-loading-steps" style="display: grid; gap: 4px; padding-left: 4px;">
            ${stepsHTML}
        </div>
        <style>
            @keyframes mode-spin {
                to { transform: rotate(360deg); }
            }
        </style>
    `;

    messageContent.appendChild(loadingCard);

    // Auto-advance through steps (simulated progress)
    let currentStep = 0;
    const stepInterval = setInterval(() => {
        currentStep++;
        if (currentStep >= steps.length) {
            clearInterval(stepInterval);
            return;
        }

        // Update current step to active
        const currentStepEl = loadingCard.querySelector(`.mode-step[data-step="${currentStep + 1}"]`);
        if (currentStepEl) {
            currentStepEl.style.color = 'var(--text-primary)';
            const icon = currentStepEl.querySelector('.mode-step-icon');
            if (icon) {
                icon.style.background = 'var(--text-primary)';
                icon.style.color = 'var(--bg-primary)';
            }
        }

        // Fade previous step
        const prevStepEl = loadingCard.querySelector(`.mode-step[data-step="${currentStep}"]`);
        if (prevStepEl) {
            prevStepEl.style.opacity = '0.5';
            const prevIcon = prevStepEl.querySelector('.mode-step-icon');
            if (prevIcon) {
                prevIcon.textContent = '✓';
            }
        }
    }, 2500);

    // Store reference for cleanup
    loadingCard.dataset.loadingId = Date.now();
    loadingCard.dataset.intervalId = stepInterval;

    return loadingCard;
}

/**
 * Update the loading state with a specific step from backend events
 */
function updateModeLoadingState(messageElement, stepText) {
    const loadingCard = messageElement.querySelector('.mode-loading-state');
    if (!loadingCard) return;

    const steps = loadingCard.querySelectorAll('.mode-step');
    steps.forEach(step => {
        const text = step.querySelector('.mode-step-text')?.textContent || '';
        if (stepText.toLowerCase().includes(text.split('...')[0].toLowerCase().trim()) ||
            text.toLowerCase().includes(stepText.split('...')[0].toLowerCase().trim())) {
            step.style.color = 'var(--text-primary)';
            step.style.opacity = '1';
            const icon = step.querySelector('.mode-step-icon');
            if (icon) {
                icon.style.background = 'var(--text-primary)';
                icon.style.color = 'var(--bg-primary)';
            }
        }
    });

    // Update title with current step
    const title = loadingCard.querySelector('.mode-loading-title');
    if (title && stepText) {
        title.textContent = stepText;
    }
}

/**
 * Hide the loading state when processing is complete
 */
function hideModeLoadingState(messageElement) {
    const loadingCard = messageElement.querySelector('.mode-loading-state');
    if (loadingCard) {
        // Clear the interval if still running
        if (loadingCard.dataset.intervalId) {
            clearInterval(parseInt(loadingCard.dataset.intervalId));
        }
        loadingCard.style.opacity = '0';
        loadingCard.style.transition = 'opacity 0.3s ease';
        setTimeout(() => loadingCard.remove(), 300);
    }
}

// Expose functions globally
window.showModeLoadingState = showModeLoadingState;
window.updateModeLoadingState = updateModeLoadingState;
window.hideModeLoadingState = hideModeLoadingState;

/**
 * Show example queries when input is focused
 */
function astro_showExampleQueries(inputElement) {
    // Remove existing examples if any
    const existing = document.getElementById('astro-example-queries');
    if (existing) existing.remove();

    const examples = document.createElement('div');
    examples.id = 'astro-example-queries';
    examples.style.cssText = `
        position: absolute;
        bottom: 115%; /* Above the input */
        left: 0;
        right: 0;
        background: rgba(0, 20, 40, 0.85); /* Deep space glass */
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
        z-index: 1000;
        display: grid;
        gap: 8px;
        box-shadow: 0 -4px 20px rgba(0, 212, 255, 0.15);
        animation: hudSlideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        transform-origin: bottom center;
    `;
    
    // Add animation style if not exists
    if (!document.getElementById('hud-style-anim')) {
        const style = document.createElement('style');
        style.id = 'hud-style-anim';
        style.textContent = `
            @keyframes hudSlideUp {
                from { opacity: 0; transform: translateY(10px) scale(0.95); }
                to { opacity: 1; transform: translateY(0) scale(1); }
            }
        `;
        document.head.appendChild(style);
    }

    examples.innerHTML = `
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px solid rgba(0,212,255,0.2); padding-bottom: 4px;">
            <div style="color: #00D4FF; font-size: 0.75em; letter-spacing: 1px; text-transform: uppercase;">SUGGESTED_VECTORS</div>
            <div style="color: #00D4FF; font-size: 0.75em;">[HUD_ACTIVE]</div>
        </div>
        <button class="astro-example-query" style="text-align: left; padding: 10px; background: rgba(0, 212, 255, 0.05); color: #e0faff; border: 1px solid rgba(0, 212, 255, 0.1); border-radius: 6px; cursor: pointer; font-size: 0.9em; transition: all 0.2s;">
            "What are my strongest behavioral traits?"
        </button>
        <button class="astro-example-query" style="text-align: left; padding: 10px; background: rgba(0, 212, 255, 0.05); color: #e0faff; border: 1px solid rgba(0, 212, 255, 0.1); border-radius: 6px; cursor: pointer; font-size: 0.9em; transition: all 0.2s;">
            "How does Mars affect me today?"
        </button>
        <button class="astro-example-query" style="text-align: left; padding: 10px; background: rgba(0, 212, 255, 0.05); color: #e0faff; border: 1px solid rgba(0, 212, 255, 0.1); border-radius: 6px; cursor: pointer; font-size: 0.9em; transition: all 0.2s;">
            "Show my organ system correlations"
        </button>
    `;

    const inputContainer = inputElement.parentElement;
    if (inputContainer) {
        inputContainer.style.position = 'relative';
        inputContainer.appendChild(examples);

        // Add click handlers
        examples.querySelectorAll('.astro-example-query').forEach(btn => {
            btn.addEventListener('click', () => {
                inputElement.value = btn.textContent.trim().replace(/^"/, '').replace(/"$/, '');
                examples.remove();
                inputElement.focus();
            });
        });

        // Remove on blur
        inputElement.addEventListener('blur', () => {
            setTimeout(() => {
                if (examples && examples.parentElement) {
                    examples.remove();
                }
            }, 200);
        }, { once: true });
    }
}

async function generateHypothesis() {
    const hypotheses = [
        "Mars conjunction amplifies sodium channel sensitivity, increasing risk-taking behavior",
        "Venus transits enhance calcium channel stability, improving emotional regulation",
        "Moon phases modulate liver voltage gates, affecting detoxification timing",
        "Solar activity influences global Schumann resonance, impacting collective consciousness",
        "Jupiter-Saturn conjunctions amplify trait inheritance through molecular imprinting"
    ];

    return hypotheses[Math.floor(Math.random() * hypotheses.length)];
}

function formatHypothesisResults(results) {
    const statusColors = {
        'SUPPORTED': '#10b981',
        'PARTIALLY_SUPPORTED': '#f59e0b',
        'NOT_SUPPORTED': '#ef4444'
    };

    return `
<div style="color: #fff;">
    <div style="margin-bottom: 20px;">
        <h4 style="color: #fff; margin-bottom: 8px;">🧪 Hypothesis: ${results.hypothesis}</h4>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="font-size: 1.2rem; font-weight: 600; color: ${statusColors[results.status]};">
                ${results.status}
            </div>
            <div style="color: #ccc;">
                ${(results.confidence * 100).toFixed(1)}% Confidence
            </div>
        </div>
    </div>

    <div style="margin-bottom: 20px;">
        <h5 style="color: #ccc; margin-bottom: 10px;">📊 Evidence Analysis:</h5>
        <div style="display: grid; gap: 8px;">
            <div>✅ Supporting Evidence: <span style="color: #10b981;">${results.evidence.supporting} studies</span></div>
            <div>❌ Contradicting Evidence: <span style="color: #ef4444;">${results.evidence.contradicting} studies</span></div>
        </div>
    </div>

    <div style="background: #1a1a1a; padding: 16px; border-radius: 6px;">
        <h5 style="color: #ccc; margin-bottom: 10px;">🎯 Research Implications:</h5>
        <ul style="color: #ccc; padding-left: 20px;">
            ${results.implications.map(imp => `<li>${imp}</li>`).join('')}
        </ul>
    </div>
</div>`;
}

// ================================================
// PHASE 4: AGENT WORKFLOW PILL
// Visual display of which agents are running
// ================================================

// Current workflow state
let workflowState = {
    agents: [],
    currentIndex: 0,
    total: 0,
    visible: false
};

// Mode templates for different processing configurations
// Mode templates for different processing configurations
const MODE_TEMPLATES = {
    fast: { icon: '⚡', name: 'Fast Mode', agents: ['secretary'] },
    research: { icon: '🔬', name: 'Research', agents: ['surveyor', 'dissident', 'synthesist'] },
    protocol: { icon: '🧠', name: 'Protocol', agents: ['dissident', 'oracle'] }, // Manual usually runs single, but we show potential
    code: { icon: '💻', name: 'Code Mode', agents: ['weaver', 'ide', 'code_validator'] },
    finance: { icon: '💰', name: 'Finance', agents: ['finance_analyst', 'synthesist'] },
    civilization: { icon: '🌍', name: 'Civilization', agents: ['social_norm_system', 'emergence_detector'] },
    astrophysiology: { icon: '🌌', name: 'Astro-Phys', agents: ['celestial_calc', 'molecular_sim'] },
    dossier: { icon: '📂', name: 'Dossier', agents: ['archaeologist', 'scrutineer', 'scribe'] },
    deep_research: { icon: '🧬', name: 'Deep Research', agents: ['surveyor', 'dissident', 'deliberation', 'archaeologist', 'synthesist', 'oracle'] },
    unbounded: { icon: '♾️', name: 'Unbounded', agents: ['surveyor', 'dissident', 'deliberation', 'archaeologist', 'synthesist', 'oracle', 'self_redesign'] },
    truth: { icon: '👁️', name: 'Truth Finding', agents: ['scrutineer', 'dissident', 'synthesist'] }
};

// Current processing mode
let currentMode = 'research';

/**
 * Initialize the agent workflow pill (HUD Style)
 */
function initWorkflowPill() {
    // 1. Clean up existing pill to prevent duplicates
    const existingPill = document.getElementById('agentWorkflowPill');
    if (existingPill) existingPill.remove();

    // 2. Inject robust HUD styles
    if (!document.getElementById('hud-pill-style')) {
        const style = document.createElement('style');
        style.id = 'hud-pill-style';
        style.textContent = `
            .hud-workflow-pill {
                position: fixed !important;
                top: 80px !important;
                left: 50% !important;
                transform: translateX(-50%) !important;
                z-index: 9999 !important;
                background: rgba(0, 0, 0, 0.6) !important;
                backdrop-filter: blur(8px) !important;
                -webkit-backdrop-filter: blur(8px) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 20px !important;
                padding: 8px 16px !important;
                display: flex !important;
                flex-direction: row !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 8px !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
                width: max-content !important;
                height: auto !important;
                max-width: 90vw !important;
                white-space: nowrap !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            .hud-workflow-pill.hidden {
                opacity: 0 !important;
                pointer-events: none !important;
                transform: translateX(-50%) translateY(-10px) !important;
            }
        `;
        document.head.appendChild(style);
    }

    // 3. Create the new pill
    const pill = document.createElement('div');
    pill.id = 'agentWorkflowPill';
    pill.className = 'agent-workflow-pill hud-workflow-pill hidden';
    pill.innerHTML = '<span class="workflow-loading">Initializing...</span>';
    document.body.appendChild(pill);

    // Create metacog indicators container
    if (!document.getElementById('metacogIndicators')) {
        const indicators = document.createElement('div');
        indicators.id = 'metacogIndicators';
        indicators.className = 'metacog-indicators';
        indicators.innerHTML = `
            <div class="metacog-badge" id="alignmentBadge" style="display:none;">
                <span class="metacog-badge-icon">🧠</span>
                <span class="metacog-badge-label">Alignment:</span>
                <span class="metacog-badge-value" id="alignmentValue">-</span>
            </div>
            <div class="metacog-badge warning" id="contradictionBadge" style="display:none;">
                <span class="metacog-badge-icon">⚠️</span>
                <span class="metacog-badge-label">Contradictions:</span>
                <span class="metacog-badge-value" id="contradictionValue">0</span>
            </div>
        `;
        document.body.appendChild(indicators);
    }

    console.log('🎨 Workflow pill initialized');
}

/**
 * Show the workflow pill with animation
 */
function showWorkflowPill() {
    const pill = document.getElementById('agentWorkflowPill');
    if (pill) {
        pill.classList.remove('hidden');
        pill.classList.add('visible');
        workflowState.visible = true;
    }
}

/**
 * Hide the workflow pill with animation
 */
function hideWorkflowPill() {
    const pill = document.getElementById('agentWorkflowPill');
    if (pill) {
        pill.classList.remove('visible');
        pill.classList.add('hidden');
        workflowState.visible = false;
    }

    // Also hide metacog indicators
    const indicators = document.getElementById('metacogIndicators');
    if (indicators) {
        indicators.classList.remove('visible');
    }
}

/**
 * Update the workflow pill to show current agent state
 * @param {string} agentName - Name of the agent
 * @param {string} status - 'pending' | 'active' | 'completed'
 * @param {number} index - Current agent index
 * @param {number} total - Total number of agents
 */
function updateWorkflowPill(agentName, status, index = null, total = null) {
    const pill = document.getElementById('agentWorkflowPill');
    if (!pill) {
        initWorkflowPill();
        return updateWorkflowPill(agentName, status, index, total);
    }

    // Update state
    if (index !== null) workflowState.currentIndex = index;
    if (total !== null) workflowState.total = total;

    // Get agent display info
    const agentDisplayNames = {
        'secretary': 'Secretary',
        'surveyor': 'Surveyor',
        'dissident': 'Dissident',
        'deliberation': 'Deliberation',
        'synthesist': 'Synthesist',
        'oracle': 'Oracle',
        'archaeologist': 'Archaeologist',
        'self_redesign': 'Self-Redesign',
        'scrutineer': 'Scrutineer'
    };

    // Build the pill HTML
    const template = MODE_TEMPLATES[currentMode] || MODE_TEMPLATES.research;
    const agents = template.agents;

    let pillHTML = '';
    agents.forEach((agent, i) => {
        const displayName = agentDisplayNames[agent] || agent.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        
        let isActive = false;
        let isCompleted = false;

        if (status === 'preview') {
             // Static preview state
             isActive = false;
             isCompleted = false;
        } else {
             isActive = agent.toLowerCase() === (agentName||'').toLowerCase() && status === 'active';
             isCompleted = i < workflowState.currentIndex ||
                (agent.toLowerCase() === (agentName||'').toLowerCase() && status === 'completed');
        }

        const stepClass = isActive ? 'active' : (isCompleted ? 'completed' : '');

        pillHTML += `
            <div class="agent-step ${stepClass}">
                <span class="agent-step-icon"></span>
                <span class="agent-step-name">${displayName}</span>
            </div>
        `;

        if (i < agents.length - 1) {
            pillHTML += '<span class="agent-step-arrow">→</span>';
        }
    });

    pill.innerHTML = pillHTML;
}

/**
 * Update metacognition indicators
 * @param {Object} data - { alignment, contradictions, complexity, quarantined }
 */
function updateMetacogIndicators(data) {
    const indicators = document.getElementById('metacogIndicators');
    if (!indicators) {
        initWorkflowPill();
        return updateMetacogIndicators(data);
    }

    // Show indicators
    indicators.classList.add('visible');

    // Update alignment badge
    const alignmentBadge = document.getElementById('alignmentBadge');
    const alignmentValue = document.getElementById('alignmentValue');
    if (alignmentBadge && alignmentValue && data.alignment !== undefined) {
        alignmentBadge.style.display = 'flex';
        alignmentValue.textContent = (data.alignment * 100).toFixed(0) + '%';

        // Color based on alignment score
        if (data.alignment < 0.3) {
            alignmentBadge.classList.add('warning');
        } else {
            alignmentBadge.classList.remove('warning');
        }
    }

    // Update contradiction badge
    const contradictionBadge = document.getElementById('contradictionBadge');
    const contradictionValue = document.getElementById('contradictionValue');
    if (contradictionBadge && contradictionValue && data.contradictions !== undefined) {
        if (data.contradictions > 0) {
            contradictionBadge.style.display = 'flex';
            contradictionValue.textContent = data.contradictions;
        } else {
            contradictionBadge.style.display = 'none';
        }
    }
}

/**
 * Set the current processing mode
 * @param {string} mode - 'fast' | 'research' | 'deep_research' | 'unbounded'
 */
function setProcessingMode(mode) {
    if (MODE_TEMPLATES[mode]) {
        currentMode = mode;
        console.log(`📋 Processing mode set to: ${MODE_TEMPLATES[mode].name}`);

        // 1. Update Mode Selector UI (Label/Icon)
        const modeLabel = document.querySelector('.mode-selector-label');
        if (modeLabel) modeLabel.textContent = MODE_TEMPLATES[mode].name;
        
        const modeIcon = document.querySelector('.mode-selector-icon');
        if (modeIcon) modeIcon.textContent = MODE_TEMPLATES[mode].icon;

        // Custom Mode Handler Invocation (Restore specific mode UIs)
        const customHandlers = {
            'astrophysiology': (m) => typeof astro_updateModeUI === 'function' && astro_updateModeUI(m)
        };

        if (customHandlers[mode]) {
             // Let the custom handler manage specific UI elements (like placeholders/helpers)
             customHandlers[mode](mode);
        }

        // 2. Update Input Placeholder (Generic Fallback)
        // Only apply generic placeholders if the custom handler didn't likely handle it
        // For uniformity, we can apply context-aware placeholders, but let's check.
        // Astro sets its own complex placeholder logic based on localstorage.
        
        if (mode !== 'astrophysiology') {
            const placeholders = {
                fast: "Ask a quick question...",
                research: "Enter research topic for swarm debate...",
                protocol: "Message the active agent directly...",
                code: "Describe the feature or bug fix...",
                finance: "Enter ticker or market query...",
                civilization: "Describe social simulation parameters...",
                dossier: "Enter target subject name...",
                deep_research: "Enter complex topic for multi-step analysis...",
                unbounded: "Enter high-level AGI directive...",
                truth: "Enter claim to verify against suppressed data..."
            };

            const newPlaceholder = placeholders[mode] || "Enter your query...";
            const inputs = document.querySelectorAll('#queryInput, #mainInput, .chat-input');
            inputs.forEach(input => {
                input.setAttribute('placeholder', newPlaceholder);
                input.style.transition = "transform 0.2s";
                input.style.transform = "scale(0.98)";
                setTimeout(() => input.style.transform = "scale(1)", 200);
            });
        }

        // 3. Preview Agent Pipeline (Workflow Pill)
        // Show what *will* happen - BUT NOT for Astro-Physiology (User Request)
        if (mode !== 'astrophysiology') {
            showWorkflowPill();
            updateWorkflowPill(null, 'preview'); 
        } else {
            hideWorkflowPill();
        }
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    initWorkflowPill();
});

// Export for testing
if (typeof window !== 'undefined') {
    window.ICEBURG_WORKFLOW = {
        showWorkflowPill,
        hideWorkflowPill,
        updateWorkflowPill,
        updateMetacogIndicators,
        setProcessingMode,
        MODE_TEMPLATES
    };
}

// ================================================
// THEME TOGGLE - Light/Dark Mode
// ================================================

/**
 * Toggle between light and dark mode
 * @param {boolean} isLight - true for light mode, false for dark mode
 */
function toggleLightMode(isLight) {
    if (isLight) {
        document.body.classList.add('light-mode');
        localStorage.setItem('iceburg-theme', 'light');
        console.log('Theme: Light mode enabled');
    } else {
        document.body.classList.remove('light-mode');
        localStorage.setItem('iceburg-theme', 'dark');
        console.log('Theme: Dark mode enabled');
    }
}

/**
 * Load saved theme preference on page load
 */
function loadThemePreference() {
    const savedTheme = localStorage.getItem('iceburg-theme');
    const themeToggle = document.getElementById('themeToggle');

    if (savedTheme === 'light') {
        document.body.classList.add('light-mode');
        if (themeToggle) themeToggle.checked = true;
    }
}

// Load theme on page ready
document.addEventListener('DOMContentLoaded', loadThemePreference);

// Expose to window for inline handler
window.toggleLightMode = toggleLightMode;


/* Paywall / Locked Module Logic - Robust Event Delegation */
(function () {
    // We use an IIFE to avoid polluting global scope, but attach listener to document immediately
    // rather than waiting for DOMContentLoaded, so it's active as soon as this script runs.

    function handleLockedClick(e) {
        // Find the closest ancestor (or self) that is a locked module
        const lockedModule = e.target.closest('.locked-module') || e.target.closest('[data-locked="true"]');

        if (lockedModule) {
            // CHECK CLEARANCE
            if (localStorage.getItem('iceburg_clearance') === 'granted') {
                const targetUrl = lockedModule.getAttribute('data-original-href');
                if (targetUrl) {
                    // Allow navigation if it's a real link, or manually navigate
                    // Since href is likely '#', we manually set window.location
                    window.location.href = targetUrl;
                    return; // Allow navigation
                }
                // If no original href, maybe it's just allowed to bubble? 
                // But let's assume we need to navigate.
            }

            // STOP EVERYTHING
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();

            console.log('🔒 Intercepted click on locked module:', lockedModule.getAttribute('href'));

            const paywallModal = document.getElementById('paywallModal');
            if (paywallModal) {
                paywallModal.style.display = 'flex';
                // Trigger reflow force
                void paywallModal.offsetWidth;
                paywallModal.style.opacity = '1';
            } else {
                console.warn('⚠️ Paywall modal element (#paywallModal) not found in DOM!');
            }

            return false;
        }
    }

    // Use 'click' with capture=true to catch it as early as possible
    document.addEventListener('click', handleLockedClick, true);

    // Also handle touch if needed, though click usually fires on touch devices too
    // We'll trust 'click' for links to avoid double-firing or scrolling issues with touchend

    // Setup Modal Close Logic (this can wait for DOM)
    document.addEventListener('DOMContentLoaded', () => {
        const paywallModal = document.getElementById('paywallModal');
        if (paywallModal) {
            // Check for redirect error param
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.get('error') === 'locked') {
                paywallModal.style.display = 'flex';
                // Trigger reflow force
                void paywallModal.offsetWidth;
                paywallModal.style.opacity = '1';
                console.log('🔒 Access denied redirect detected. Showing paywall.');

                // Clean up URL
                const newUrl = window.location.pathname;
                window.history.replaceState({}, document.title, newUrl);
            }

            // Close on overlay click
            paywallModal.addEventListener('click', (e) => {
                if (e.target === paywallModal) {
                    paywallModal.style.opacity = '0';
                    setTimeout(() => {
                        paywallModal.style.display = 'none';
                    }, 300);
                }
            });

            // Close button
            const closeBtn = paywallModal.querySelector('.paywall-close');
            if (closeBtn) {
                closeBtn.addEventListener('click', () => {
                    paywallModal.style.opacity = '0';
                    setTimeout(() => {
                        paywallModal.style.display = 'none';
                    }, 300);
                });
            }

            // --- LOGIN LOGIC ---
            const loginBtn = paywallModal.querySelector('.login-btn');
            if (loginBtn) {
                // Remove inline onclick if possible, or just add listener behavior
                // The inline onclick might still fire, but this will add the real logic.
                // Ideally we'd remove the inline attribute but let's just add the listener first.
                loginBtn.onclick = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const code = prompt("Identity Verification Required", "");
                    if (code && code.trim().toUpperCase() === 'OBSIDIAN') {
                        localStorage.setItem('iceburg_clearance', 'granted');

                        // Silent success - professional fade out
                        paywallModal.style.opacity = '0';
                        setTimeout(() => {
                            paywallModal.style.display = 'none';
                            // Reload to enable links or redirect back if we have a saved return path
                            const returnPath = sessionStorage.getItem('redirect_after_login');
                            if (returnPath) {
                                sessionStorage.removeItem('redirect_after_login');
                                window.location.href = returnPath;
                            } else {
                                window.location.reload();
                            }
                        }, 300);
                    } else if (code) {
                        alert('Authentication Failed');
                    }
                };
            }
        }
    });
})();
