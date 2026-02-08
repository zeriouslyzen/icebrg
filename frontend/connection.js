/**
 * ICEBURG Connection Module - SSE-Only
 * =====================================
 * Simplified connection handling using only Server-Sent Events (SSE).
 * Replaces the complex WebSocket + fallback code with a simple, reliable pattern.
 * 
 * This module is ~80 lines vs the original ~350 lines of WebSocket complexity.
 */

// Connection state
let isConnected = false;
let activeEventSource = null;
let connectionListeners = [];

/**
 * Get connection status
 */
export function getConnectionStatus() {
    return isConnected;
}

/**
 * Register a listener for connection status changes
 * @param {function(boolean)} listener - Called with (isConnected)
 */
export function onConnectionChange(listener) {
    connectionListeners.push(listener);
    // Immediately notify of current state
    listener(isConnected);
    return () => {
        connectionListeners = connectionListeners.filter(l => l !== listener);
    };
}

function notifyConnectionChange(connected) {
    isConnected = connected;
    connectionListeners.forEach(l => l(connected));
}

/**
 * Send a query to the V2 API with SSE streaming
 * @param {Object} options - Query options
 * @param {string} options.query - The user's query
 * @param {string} options.mode - Query mode: 'fast', 'research', 'local'
 * @param {Object} options.settings - Model settings (temperature, etc.)
 * @param {Object} options.data - Additional data (birth data for astro mode)
 * @param {function} options.onChunk - Called for each text chunk
 * @param {function} options.onStatus - Called for status updates
 * @param {function} options.onError - Called on error
 * @param {function} options.onDone - Called when complete
 * @returns {function} Abort function to cancel the request
 */
export async function sendQuery({
    query,
    mode = 'fast',
    settings = {},
    data = null,
    onChunk,
    onStatus,
    onError,
    onDone,
    onMessage,
}) {
    const API_URL = window.ICEBURG_API_URL || 'http://localhost:8000';
    const conversationId = localStorage.getItem('iceburg_conversation_id') || 
        `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('iceburg_conversation_id', conversationId);

    const controller = new AbortController();
    
    try {
        notifyConnectionChange(true);
        onStatus?.('Connecting...');
        
        const response = await fetch(`${API_URL}/v2/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
            },
            body: JSON.stringify({
                query,
                mode,
                conversation_id: conversationId,
                stream: true,
                temperature: settings.temperature ?? 0.7,
                max_tokens: settings.maxTokens ?? 2000,
                data,
            }),
            signal: controller.signal,
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';  // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        // Always notify raw message listener if provided
                        onMessage?.(data);

                        switch (data.type) {
                            case 'chunk':
                                onChunk?.(data.content);
                                break;
                            case 'status':
                                onStatus?.(data.content);
                                break;
                            case 'done':
                                onDone?.(data);
                                break;
                            case 'error':
                                onError?.(new Error(data.content));
                                break;
                        }
                    } catch (e) {
                        console.warn('Failed to parse SSE data:', line);
                    }
                }
            }
        }

        notifyConnectionChange(false);
        
    } catch (error) {
        notifyConnectionChange(false);
        if (error.name !== 'AbortError') {
            console.error('Query error:', error);
            onError?.(error);
        }
    }

    return () => controller.abort();
}

/**
 * Check API health
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
    const API_URL = window.ICEBURG_API_URL || 'http://localhost:8000';
    try {
        const response = await fetch(`${API_URL}/v2/health`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error('Health check failed:', error);
        return { status: 'error', error: error.message };
    }
}

/**
 * Get available providers
 * @returns {Promise<Object>} Provider info
 */
export async function getProviders() {
    const API_URL = window.ICEBURG_API_URL || 'http://localhost:8000';
    try {
        const response = await fetch(`${API_URL}/v2/providers`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error('Get providers failed:', error);
        return { available: [], error: error.message };
    }
}
