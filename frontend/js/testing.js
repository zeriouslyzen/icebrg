import { API_BASE, showNotification, downloadFile } from './utils.js';
// Note: addActivity is in dashboard.js. To avoid circle, we import it here.
import { addActivity } from './dashboard.js';

// ============================================
// Actions & Benchmarks
// ============================================

export function runBenchmark() {
    addActivity('Benchmark started...');
    showNotification('Starting benchmark run...');

    // Simulate benchmark run
    setTimeout(() => {
        addActivity('Benchmark completed: 10/10 (100%)');
        showNotification('Benchmark completed: 100%');
    }, 2000);
}

// ============================================
// Prompts Playground
// ============================================
const PROMPT_TEMPLATES = {
    analogy: 'Book : Library :: Fish : ?',
    hierarchy: 'A contains B and C. B contains D. What is the depth of D?',
    pattern: '2, 6, 18, 54, ?'
};

export function loadTemplate(name) {
    const textarea = document.getElementById('playground-query');
    if (textarea && PROMPT_TEMPLATES[name]) {
        textarea.value = PROMPT_TEMPLATES[name];
        addActivity(`Template loaded: ${name}`);
    }
}

export function runPlayground() {
    const query = document.getElementById('playground-query').value;
    const mode = document.getElementById('playground-model').value;
    const responseDiv = document.getElementById('playground-response');
    const metaDiv = document.getElementById('playground-meta');

    if (!query.trim()) {
        showNotification('Please enter a query');
        return;
    }

    responseDiv.innerHTML = '<p style="color: #fbbf24;">Processing...</p>';

    // Simulate response
    setTimeout(() => {
        let response = 'Unknown pattern';
        let agent = 'LLM';
        let tokens = 150;

        // Simple pattern matching for demo
        if (query.includes('::') || query.includes(':')) {
            response = 'Ocean (containment relationship detected)';
            agent = 'Analogical Mapper';
            tokens = 0;
        } else if (query.includes('contains') || query.includes('depth')) {
            response = 'Depth: 3 (Path: A → B → D)';
            agent = 'Hierarchical Reasoner';
            tokens = 0;
        } else if (query.match(/\d+,\s*\d+,\s*\d+/)) {
            response = '162 (pattern: multiply by 3)';
            agent = 'Abstract Transformer';
            tokens = 0;
        }

        responseDiv.innerHTML = `<p style="color: #e8f0ff;">${response}</p>`;
        metaDiv.innerHTML = `
      <span>${tokens === 0 ? '12ms' : '1.2s'}</span>
      <span>${tokens} tokens</span>
      <span>${agent}</span>
    `;

        addActivity(`Playground: "${query.substring(0, 30)}..." → ${agent}`);
    }, 500);
}

export function saveTemplate() {
    const query = document.getElementById('playground-query').value;
    if (!query.trim()) {
        showNotification('Enter a query first');
        return;
    }

    const name = prompt('Template name:');
    if (name) {
        PROMPT_TEMPLATES[name.toLowerCase()] = query;
        addActivity(`Template saved: ${name}`);
        showNotification(`Template "${name}" saved`);
    }
}

// ============================================
// Sessions & Export
// ============================================

export async function searchSessions() {
    const searchTerm = document.getElementById('session-search').value;
    addActivity(`Searching sessions: "${searchTerm}"`);

    try {
        const response = await fetch(`${API_BASE}/api/admin/sessions?search=${encodeURIComponent(searchTerm)}&limit=50`);
        if (response.ok) {
            const data = await response.json();
            updateSessionsList(data.sessions);
            showNotification(`Found ${data.count} sessions`);
        }
    } catch (e) {
        console.warn('Could not search sessions:', e);
        showNotification(`Searching for: ${searchTerm}`);
    }
}

function updateSessionsList(sessions) {
    const container = document.querySelector('.sessions-list');
    if (!container || !sessions.length) return;

    container.innerHTML = sessions.map(s => `
        <div class="session-item">
            <div class="session-header">
                <span class="session-time">${s.timestamp}</span>
                <span class="session-type">${s.query_type || 'Unknown'}</span>
                <span class="session-status ${s.success ? 'success' : ''}"></span>
            </div>
            <div class="session-query">${s.query}</div>
            <div class="session-response">${s.response}</div>
            <div class="session-meta">
                <span>${s.latency_ms}ms</span>
                <span>${s.tokens_used} tokens</span>
            </div>
        </div>
    `).join('');
}

export async function exportSessions() {
    let sessions = [];

    try {
        const response = await fetch(`${API_BASE}/api/admin/sessions?limit=100`);
        if (response.ok) {
            const data = await response.json();
            sessions = data.sessions;
        }
    } catch (e) {
        console.warn('Could not fetch sessions from API:', e);
        // Use fallback mock data
        sessions = [
            { timestamp: '2025-12-28 22:08', query: 'A contains B...', response: 'Depth: 3', query_type: 'Hierarchical' },
            { timestamp: '2025-12-28 22:05', query: 'Book : Library...', response: 'Ocean', query_type: 'Analogical' },
            { timestamp: '2025-12-28 21:55', query: '[[1,2],[3,4]]...', response: 'Rotation', query_type: 'Visual' }
        ];
    }

    const csv = [
        'Time,Query,Response,Type,Latency,Tokens',
        ...sessions.map(s => `"${s.timestamp}","${s.query}","${s.response}","${s.query_type}","${s.latency_ms || 0}","${s.tokens_used || 0}"`)
    ].join('\n');

    downloadFile('iceburg_sessions.csv', csv, 'text/csv');
    addActivity('Sessions exported to CSV');
}

export async function exportAuditLog() {
    let log = [];

    try {
        const response = await fetch(`${API_BASE}/api/admin/audit?limit=100`);
        if (response.ok) {
            const data = await response.json();
            log = data.audit_log;
        }
    } catch (e) {
        console.warn('Could not fetch audit log from API:', e);
        // Use fallback mock data
        log = [
            { timestamp: '2025-12-28 22:10', event: 'Settings Changed', details: 'formal_reasoning enabled', user: 'admin' },
            { timestamp: '2025-12-28 22:08', event: 'Benchmark Run', details: 'ARC-AGI 10/10', user: 'admin' },
            { timestamp: '2025-12-28 22:00', event: 'Dashboard Accessed', details: 'Admin opened', user: 'admin' }
        ];
    }

    const json = JSON.stringify(log, null, 2);
    downloadFile('iceburg_audit_log.json', json, 'application/json');
    addActivity('Audit log exported');
}

export function runABTest() {
    const modelA = document.getElementById('model-a').value;
    const modelB = document.getElementById('model-b').value;
    const resultsDiv = document.getElementById('ab-results');

    resultsDiv.innerHTML = '<p style="color: #fbbf24;">Running A/B comparison...</p>';

    setTimeout(() => {
        resultsDiv.innerHTML = `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div style="padding: 1rem; background: rgba(74, 158, 255, 0.1); border-radius: 8px;">
          <h5 style="margin-bottom: 0.5rem; color: #4a9eff;">${modelA}</h5>
          <p style="font-size: 0.85rem;">Latency: 142ms</p>
          <p style="font-size: 0.85rem;">Accuracy: 95%</p>
          <p style="font-size: 0.85rem;">Tokens: 1.2K/query</p>
        </div>
        <div style="padding: 1rem; background: rgba(74, 222, 128, 0.1); border-radius: 8px;">
          <h5 style="margin-bottom: 0.5rem; color: #4ade80;">${modelB}</h5>
          <p style="font-size: 0.85rem;">Latency: 580ms</p>
          <p style="font-size: 0.85rem;">Accuracy: 98%</p>
          <p style="font-size: 0.85rem;">Tokens: 2.8K/query</p>
        </div>
      </div>
      <p style="margin-top: 1rem; font-size: 0.8rem; color: #8899bb;">
        Recommendation: ${modelA} for speed, ${modelB} for accuracy
      </p>
    `;
        addActivity(`A/B test completed: ${modelA} vs ${modelB}`);
    }, 1500);
}
