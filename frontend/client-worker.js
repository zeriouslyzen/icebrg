// Web Worker for client-side background processing
// Runs on separate thread, doesn't block UI

self.onmessage = function(e) {
    const { type, data } = e.data;
    
    switch (type) {
        case 'preprocess':
            const processed = preprocessQuery(data.query);
            self.postMessage({ type: 'processed', data: processed });
            break;
            
        case 'analyze':
            const analysis = analyzeQuery(data.query);
            self.postMessage({ type: 'analyzed', data: analysis });
            break;
            
        case 'cache':
            cacheResponse(data.key, data.value);
            self.postMessage({ type: 'cached', key: data.key });
            break;
    }
};

function preprocessQuery(query) {
    // Heavy preprocessing in worker thread
    return {
        normalized: query.trim().toLowerCase(),
        wordCount: query.split(/\s+/).length,
        entities: extractEntities(query),
        keywords: extractKeywords(query)
    };
}

function analyzeQuery(query) {
    // Complexity analysis
    const complexity = calculateComplexity(query);
    return {
        complexity,
        estimatedTime: estimateProcessingTime(complexity),
        recommendedMode: recommendMode(complexity)
    };
}

function extractEntities(query) {
    // Entity extraction
    const entities = [];
    
    // URLs
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    const urls = query.match(urlRegex);
    if (urls) {
        entities.push(...urls.map(url => ({ type: 'url', value: url })));
    }
    
    return entities;
}

function extractKeywords(query) {
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but']);
    return query.toLowerCase()
        .split(/\s+/)
        .filter(word => word.length > 2 && !stopWords.has(word));
}

function calculateComplexity(query) {
    let complexity = 0.3;
    const wordCount = query.split(/\s+/).length;
    
    if (wordCount > 20) complexity += 0.2;
    if (query.includes('?')) complexity += 0.1;
    if (/quantum|neural|AI|algorithm/i.test(query)) complexity += 0.3;
    
    return Math.min(1.0, complexity);
}

function estimateProcessingTime(complexity) {
    if (complexity < 0.3) return '1-3 seconds';
    if (complexity < 0.6) return '5-15 seconds';
    return '20-60 seconds';
}

function recommendMode(complexity) {
    if (complexity < 0.3) return 'fast';
    if (complexity < 0.6) return 'balanced';
    return 'deep';
}

function cacheResponse(key, value) {
    // Cache in IndexedDB via worker
    // This would need IndexedDB access in worker
}

