// ICEBURG Client-Side Processing Module
// Leverages user's device (iPhone, etc.) for better performance

class ClientProcessor {
    constructor() {
        this.isMobile = this.detectMobile();
        this.deviceCapabilities = this.detectCapabilities();
        this.cache = new ClientCache();
        this.preprocessor = new QueryPreprocessor();
        this.offlineQueue = [];
        this.isOnline = navigator.onLine;
        
        this.init();
    }
    
    init() {
        // Initialize client-side processing
        this.setupOfflineSupport();
        this.setupDeviceOptimizations();
        this.setupBackgroundProcessing();
    }
    
    detectMobile() {
        return /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) ||
               (navigator.maxTouchPoints && navigator.maxTouchPoints > 2);
    }
    
    detectCapabilities() {
        return {
            // Hardware capabilities
            hasGPU: this.checkGPU(),
            hasNeuralEngine: this.checkNeuralEngine(), // iOS Neural Engine
            hasCamera: navigator.mediaDevices && navigator.mediaDevices.getUserMedia,
            hasSensors: 'DeviceOrientationEvent' in window || 'DeviceMotionEvent' in window,
            hasWebGL: this.checkWebGL(),
            hasWebWorker: typeof Worker !== 'undefined',
            hasServiceWorker: 'serviceWorker' in navigator,
            hasIndexedDB: 'indexedDB' in window,
            hasLocalStorage: 'localStorage' in window,
            
            // Performance capabilities
            cores: navigator.hardwareConcurrency || 4,
            memory: navigator.deviceMemory || 4, // GB
            connection: navigator.connection ? {
                effectiveType: navigator.connection.effectiveType,
                downlink: navigator.connection.downlink,
                rtt: navigator.connection.rtt
            } : null
        };
    }
    
    checkGPU() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            if (gl) {
                const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                if (debugInfo && typeof debugInfo.getParameter === 'function') {
                    return debugInfo.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || 'Unknown';
                }
                return 'WebGL Available';
            }
            return false;
        } catch (e) {
            // WebGL not available or error accessing it
            return false;
        }
    }
    
    checkNeuralEngine() {
        // iOS Neural Engine detection (indirect)
        // Check for Core ML support or ML capabilities
        return this.isMobile && 
               ('ML' in window || 'webkitSpeechRecognition' in window);
    }
    
    checkWebGL() {
        const canvas = document.createElement('canvas');
        return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
    }
    
    // Client-side query preprocessing
    async preprocessQuery(query) {
        // 1. Text normalization (client-side)
        const normalized = this.preprocessor.normalize(query);
        
        // 2. Check cache first
        const cached = await this.cache.get(normalized);
        if (cached) {
            return { cached: true, data: cached };
        }
        
        // 3. Extract entities and keywords (client-side)
        const entities = this.preprocessor.extractEntities(normalized);
        const keywords = this.preprocessor.extractKeywords(normalized);
        
        // 4. Determine complexity (client-side)
        const complexity = this.preprocessor.analyzeComplexity(normalized);
        
        return {
            cached: false,
            normalized,
            entities,
            keywords,
            complexity,
            metadata: {
                device: this.deviceCapabilities,
                timestamp: Date.now()
            }
        };
    }
    
    // Client-side response processing
    async processResponse(response, query) {
        // 1. Cache response
        await this.cache.set(query, response);
        
        // 2. Extract and cache entities
        if (response.entities) {
            await this.cache.setEntities(response.entities);
        }
        
        // 3. Pre-render markdown (client-side)
        if (response.content) {
            response.rendered = this.preprocessor.preRenderMarkdown(response.content);
        }
        
        return response;
    }
    
    // Background processing with Web Workers
    setupBackgroundProcessing() {
        if (!this.deviceCapabilities.hasWebWorker) return;
        
        // Create worker for heavy processing
        try {
            this.worker = new Worker('/client-worker.js');
            this.worker.onmessage = (e) => {
                this.handleWorkerMessage(e.data);
            };
        } catch (e) {
            console.warn('Web Worker not available:', e);
        }
    }
    
    // Offline support
    setupOfflineSupport() {
        // Monitor online/offline status
        window.addEventListener('online', () => {
            this.isOnline = true;
            this.processOfflineQueue();
        });
        
        window.addEventListener('offline', () => {
            this.isOnline = false;
        });
        
        // Service Worker for offline caching
        if (this.deviceCapabilities.hasServiceWorker) {
            this.registerServiceWorker();
        }
    }
    
    async registerServiceWorker() {
        // Disable service worker in development (Vite dev server)
        const isDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
        if (isDevelopment) {
            console.log('⚠️ Service Worker disabled in development mode (Vite dev server)');
            // Unregister any existing service workers
            if ('serviceWorker' in navigator) {
                try {
                    const registrations = await navigator.serviceWorker.getRegistrations();
                    for (const registration of registrations) {
                        await registration.unregister();
                        console.log('✅ Unregistered existing service worker');
                    }
                } catch (e) {
                    console.warn('Failed to unregister service workers:', e);
                }
            }
            return;
        }
        
        try {
            const registration = await navigator.serviceWorker.register('/service-worker.js');
            console.log('Service Worker registered:', registration);
        } catch (e) {
            console.warn('Service Worker registration failed:', e);
        }
    }
    
    // Device-specific optimizations
    setupDeviceOptimizations() {
        if (this.isMobile) {
            // Mobile-specific optimizations
            this.optimizeForMobile();
        }
        
        // GPU acceleration for rendering
        if (this.deviceCapabilities.hasGPU) {
            this.enableGPUAcceleration();
        }
    }
    
    optimizeForMobile() {
        // Reduce animation complexity on mobile
        document.documentElement.style.setProperty('--animation-duration', '0.2s');
        
        // Optimize touch interactions
        document.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: true });
    }
    
    enableGPUAcceleration() {
        // Enable GPU acceleration for animations
        const style = document.createElement('style');
        style.textContent = `
            .message, .thinking-item, .word-breakdown {
                transform: translateZ(0);
                will-change: transform;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Queue queries when offline
    queueOffline(query) {
        this.offlineQueue.push({
            query,
            timestamp: Date.now()
        });
        
        // Store in IndexedDB
        this.cache.storeOfflineQuery(query);
    }
    
    // Process queued queries when back online
    async processOfflineQueue() {
        if (this.offlineQueue.length === 0) return;
        
        const queries = [...this.offlineQueue];
        this.offlineQueue = [];
        
        for (const item of queries) {
            // Send to server
            // This would be handled by the main WebSocket connection
        }
    }
    
    handleTouchStart(e) {
        // Optimize touch interactions
        // Could add haptic feedback, etc.
    }
    
    handleWorkerMessage(data) {
        // Handle messages from Web Worker
        switch (data.type) {
            case 'processed':
                // Processed data from worker
                break;
        }
    }
    
    // Send task to worker
    sendToWorker(task) {
        if (this.worker) {
            this.worker.postMessage(task);
        }
    }
}

// Client-side cache using IndexedDB
class ClientCache {
    constructor() {
        this.db = null;
        this.init();
    }
    
    async init() {
        if (!('indexedDB' in window)) return;
        
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('iceburg-cache', 1);
            
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };
            
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                
                // Query cache
                if (!db.objectStoreNames.contains('queries')) {
                    const queryStore = db.createObjectStore('queries', { keyPath: 'key' });
                    queryStore.createIndex('timestamp', 'timestamp', { unique: false });
                }
                
                // Response cache
                if (!db.objectStoreNames.contains('responses')) {
                    const responseStore = db.createObjectStore('responses', { keyPath: 'key' });
                    responseStore.createIndex('timestamp', 'timestamp', { unique: false });
                }
                
                // Entity cache
                if (!db.objectStoreNames.contains('entities')) {
                    db.createObjectStore('entities', { keyPath: 'id' });
                }
            };
        });
    }
    
    async get(key) {
        if (!this.db) return null;
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['queries', 'responses'], 'readonly');
            const queryStore = transaction.objectStore('queries');
            const responseStore = transaction.objectStore('responses');
            
            const queryRequest = queryStore.get(key);
            queryRequest.onsuccess = () => {
                if (queryRequest.result) {
                    const responseRequest = responseStore.get(key);
                    responseRequest.onsuccess = () => {
                        if (responseRequest.result) {
                            // Check if cache is still valid (1 hour TTL)
                            const age = Date.now() - responseRequest.result.timestamp;
                            if (age < 3600000) {
                                resolve(responseRequest.result.data);
                            } else {
                                resolve(null);
                            }
                        } else {
                            resolve(null);
                        }
                    };
                } else {
                    resolve(null);
                }
            };
        });
    }
    
    async set(key, data) {
        if (!this.db) return;
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['queries', 'responses'], 'readwrite');
            const queryStore = transaction.objectStore('queries');
            const responseStore = transaction.objectStore('responses');
            
            const queryData = { key, timestamp: Date.now() };
            const responseData = { key, data, timestamp: Date.now() };
            
            queryStore.put(queryData);
            const request = responseStore.put(responseData);
            
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }
    
    async storeOfflineQuery(query) {
        if (!this.db) return;
        
        const transaction = this.db.transaction(['queries'], 'readwrite');
        const store = transaction.objectStore('queries');
        store.put({
            key: `offline-${Date.now()}`,
            query,
            timestamp: Date.now(),
            offline: true
        });
    }
}

// Query preprocessor (client-side)
class QueryPreprocessor {
    normalize(query) {
        // Remove extra whitespace
        return query.trim().replace(/\s+/g, ' ');
    }
    
    extractEntities(query) {
        // Simple entity extraction (could use NLP.js or similar)
        const entities = [];
        
        // Extract URLs
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        const urls = query.match(urlRegex);
        if (urls) {
            entities.push(...urls.map(url => ({ type: 'url', value: url })));
        }
        
        // Extract emails
        const emailRegex = /([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/g;
        const emails = query.match(emailRegex);
        if (emails) {
            entities.push(...emails.map(email => ({ type: 'email', value: email })));
        }
        
        return entities;
    }
    
    extractKeywords(query) {
        // Simple keyword extraction
        const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']);
        const words = query.toLowerCase().split(/\s+/);
        return words.filter(word => word.length > 2 && !stopWords.has(word));
    }
    
    analyzeComplexity(query) {
        // Simple complexity analysis
        const wordCount = query.split(/\s+/).length;
        const hasQuestion = query.includes('?');
        const hasMultipleQuestions = (query.match(/\?/g) || []).length > 1;
        const hasTechnicalTerms = /algorithm|quantum|neural|machine learning|AI|artificial intelligence/i.test(query);
        
        let complexity = 0.3; // Base complexity
        
        if (wordCount > 20) complexity += 0.2;
        if (hasMultipleQuestions) complexity += 0.2;
        if (hasTechnicalTerms) complexity += 0.3;
        
        return Math.min(1.0, complexity);
    }
    
    preRenderMarkdown(content) {
        // Pre-render markdown on client side
        // This reduces server load
        try {
            // Would use marked.js here
            return content; // Placeholder
        } catch (e) {
            return content;
        }
    }
}

// Export for use in main.js
export { ClientProcessor, ClientCache, QueryPreprocessor };

