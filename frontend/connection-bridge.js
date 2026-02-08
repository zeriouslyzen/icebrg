/**
 * ICEBURG Connection Bridge
 * =========================
 * Bridges the new simplified connection module with the existing main.js patterns.
 * 
 * This allows gradual migration without breaking existing functionality.
 * Once migration is complete, this bridge can be removed.
 */

import { sendQuery, checkHealth, getProviders, getConnectionStatus, onConnectionChange } from './connection.js';

// Expose as globals for backward compatibility with main.js
window.ICEBURG_CONNECTION = {
    sendQuery,
    checkHealth,
    getProviders,
    getConnectionStatus,
    onConnectionChange,
};

// Backward-compatible globals
window.isConnected = false;
window.useFallback = true;  // Always use "fallback" (SSE) mode now

// Register connection listener to update legacy globals
onConnectionChange((connected) => {
    window.isConnected = connected;
    // Update UI status
    if (typeof updateConnectionStatus === 'function') {
        updateConnectionStatus(connected);
    }
});

/**
 * Simplified query sender that integrates with existing handleStreamingMessage
 * 
 * Usage in main.js:
 *   Instead of the complex WebSocket/HTTP branching, use:
 *   
 *   sendQueryV2(query, mode, { settings, data, agent });
 */
window.sendQueryV2 = function(query, mode, options = {}) {
    const { settings = {}, data = null, agent = null, onMessage } = options;
    
    // Get or create assistant message element
    const chatContainer = document.getElementById('chatContainer');
    if (!chatContainer) {
        console.error('chatContainer not found');
        return;
    }
    
    // Add loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant loading-indicator';
    loadingDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Create message element for streaming content
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = '<div class="message-content"></div>';
    
    let contentElement = null;
    let fullContent = '';
    
    return sendQuery({
        query,
        mode: mode || 'fast',
        settings: {
            temperature: settings.temperature ?? 0.7,
            maxTokens: settings.maxTokens ?? 2000,
        },
        data,
        
        // Pass through raw message handler
        onMessage,
        
        onStatus: (status) => {
            console.log('📡 Status:', status);
            // Update loading indicator text
            const loadingIndicator = chatContainer.querySelector('.loading-indicator');
            if (loadingIndicator) {
                loadingIndicator.innerHTML = `<div class="status-text">${status}</div>`;
            }
        },
        
        onChunk: (chunk) => {
            // First chunk: replace loading indicator with message
            if (!contentElement) {
                const loadingIndicator = chatContainer.querySelector('.loading-indicator');
                if (loadingIndicator) {
                    loadingIndicator.remove();
                }
                chatContainer.appendChild(messageDiv);
                contentElement = messageDiv.querySelector('.message-content');
            }
            
            // Append chunk and re-render
            fullContent += chunk;
            if (contentElement && typeof marked !== 'undefined') {
                contentElement.innerHTML = marked.parse(fullContent);
            } else if (contentElement) {
                contentElement.textContent = fullContent;
            }
            
            // Auto-scroll
            chatContainer.scrollTop = chatContainer.scrollHeight;
        },
        
        onError: (error) => {
            console.error('❌ Query error:', error);
            const loadingIndicator = chatContainer.querySelector('.loading-indicator');
            if (loadingIndicator) {
                loadingIndicator.remove();
            }
            
            const errorDiv = document.createElement('div');
            errorDiv.className = 'message system error';
            errorDiv.textContent = `Error: ${error.message}`;
            chatContainer.appendChild(errorDiv);
            
            if (typeof showToast === 'function') {
                showToast(`Error: ${error.message}`, 'error');
            }
        },
        
        onDone: (data) => {
            console.log('✅ Query complete:', data);
            
            // Re-enable input
            const input = document.getElementById('queryInput') || document.getElementById('messageInput');
            const button = document.getElementById('sendButton') || document.getElementById('submitBtn');
            if (input) input.disabled = false;
            if (button) button.disabled = false;
        },
    });
};

// Check connection on load
checkHealth().then(health => {
    console.log('🏥 V2 API Health:', health);
    if (health.status === 'ok') {
        window.isConnected = true;
        if (typeof updateConnectionStatus === 'function') {
            updateConnectionStatus(true);
        }
        if (typeof showToast === 'function') {
            const provider = health.fast_provider || 'connected';
            showToast(`⚡ ${provider} ready`, 'success', 2000);
        }
    }
});

console.log('🔌 ICEBURG Connection Bridge loaded (SSE-only mode)');
