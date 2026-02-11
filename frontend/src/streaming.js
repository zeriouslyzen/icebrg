/**
 * ICEBURG Streaming Module
 * ========================
 * Abstraction layer for streaming queries via V2 SSE endpoint
 * 
 * This module provides a clean API for sending queries and handling streaming responses,
 * abstracting away the details of connection.js and message handling.
 */

import { sendQuery as connectionSendQuery, getConnectionStatus, onConnectionChange as connectionOnConnectionChange } from '../connection.js';

/**
 * Send a query with streaming support
 * @param {Object} options - Query options
 * @param {string} options.query - User query text
 * @param {string} options.mode - Query mode ('fast', 'research', 'local', etc.)
 * @param {Object} options.settings - Model settings (temperature, maxTokens, etc.)
 * @param {Object} options.data - Additional data (agent, files, birthData, etc.)
 * @param {function} options.onChunk - Called for each text chunk: (chunk: string) => void
 * @param {function} options.onMessage - Called for all message types: (message: Object) => void
 * @param {function} options.onStatus - Called for status updates: (status: string) => void
 * @param {function} options.onError - Called on error: (error: Error) => void
 * @param {function} options.onDone - Called when complete: (data: Object) => void
 * @returns {function} Abort function to cancel the request
 */
export function sendQuery(options) {
    const {
        query,
        mode = 'fast',
        settings = {},
        data = null,
        onChunk,
        onMessage,
        onStatus,
        onError,
        onDone
    } = options;

    if (!window.ICEBURG_CONNECTION) {
        const error = new Error('Connection module not available');
        onError?.(error);
        return () => {}; // No-op abort function
    }

    return window.ICEBURG_CONNECTION.sendQuery({
        query,
        mode,
        settings,
        data,
        onChunk,
        onMessage,
        onStatus,
        onError,
        onDone
    });
}

/**
 * Get current connection status
 * @returns {boolean} True if connected to server
 */
export function getConnectionStatus() {
    if (!window.ICEBURG_CONNECTION) {
        return false;
    }
    return window.ICEBURG_CONNECTION.getConnectionStatus();
}

/**
 * Register callback for connection status changes
 * @param {function(boolean)} callback - Called with (isConnected) when status changes
 * @returns {function} Unsubscribe function
 */
export function onConnectionChange(callback) {
    if (!window.ICEBURG_CONNECTION) {
        return () => {}; // No-op unsubscribe
    }
    return window.ICEBURG_CONNECTION.onConnectionChange(callback);
}

/**
 * Check API health
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
    if (!window.ICEBURG_CONNECTION) {
        return { status: 'error', error: 'Connection module not available' };
    }
    return await window.ICEBURG_CONNECTION.checkHealth();
}

/**
 * Get available providers
 * @returns {Promise<Object>} Provider information
 */
export async function getProviders() {
    if (!window.ICEBURG_CONNECTION) {
        return { available: [], error: 'Connection module not available' };
    }
    return await window.ICEBURG_CONNECTION.getProviders();
}
