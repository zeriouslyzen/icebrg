// Service Worker for offline support and caching
// Enables PWA capabilities and offline functionality

const CACHE_NAME = 'iceburg-v1';
const API_CACHE = 'iceburg-api-v1';

// Install event - cache static assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll([
                '/',
                '/index.html',
                '/main.js',
                '/styles.css'
            ]);
        })
    );
    self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME && cacheName !== API_CACHE) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
    self.clients.claim();
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Skip Vite dev server requests (HMR, etc.)
    if (url.pathname.includes('@vite') || 
        url.pathname.includes('?html-proxy') || 
        url.pathname.includes('?t=') ||
        url.hostname === 'localhost' && url.port === '3000' && url.pathname.includes('main.js')) {
        // Let Vite dev server handle these directly
        return;
    }
    
    // Skip WebSocket connections
    if (url.protocol === 'ws:' || url.protocol === 'wss:') {
        return;
    }
    
    // Cache API responses
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(
            caches.open(API_CACHE).then((cache) => {
                return fetch(request).then((response) => {
                    // Cache successful responses
                    if (response.status === 200) {
                        cache.put(request, response.clone());
                    }
                    return response;
                }).catch(() => {
                    // Return cached response if offline
                    return cache.match(request);
                });
            })
        );
    } else {
        // Cache static assets
        event.respondWith(
            caches.match(request).then((response) => {
                return response || fetch(request);
            })
        );
    }
});

// Background sync for offline queries
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-queries') {
        event.waitUntil(syncOfflineQueries());
    }
});

async function syncOfflineQueries() {
    // Sync queued queries when back online
    // This would read from IndexedDB and send to server
}

