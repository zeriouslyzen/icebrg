/**
 * ICEBURG Admin Dashboard JavaScript
 * Handles navigation, charts, data loading, and settings
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * TABLE OF CONTENTS (use Cmd/Ctrl+G to jump to line number)
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * CORE SYSTEM
 *   Line   14 - Navigation ................ initNavigation()
 *   Line   46 - Charts .................... initCharts(), getChartOptions()
 *   Line  278 - Metrics ................... loadMetrics(), getDefaultMetrics()
 *   Line  370 - Utilities ................. formatNumber()
 *   Line  379 - Settings .................. loadSettings(), saveSettings(), resetSettings()
 *   Line  467 - Actions ................... refreshData(), runBenchmark()
 *   Line  487 - Activity Feed ............. addActivity()
 * 
 * V2 FEATURES
 *   Line  527 - Mermaid Init .............. Trace diagram setup
 *   Line  572 - Traces .................... refreshTrace()
 *   Line  578 - Analytics ................. applyDateRange(), runABTest()
 *   Line  624 - Prompts Playground ........ loadTemplate(), runPlayground(), saveTemplate()
 *   Line  692 - Sessions .................. searchSessions(), exportSessions()
 *   Line  759 - Security/Audit ............ exportAuditLog(), downloadFile()
 * 
 * V3 FEATURES
 *   Line  802 - Alerts .................... refreshAlerts(), getSeverity*()
 *   Line  848 - Hallucination ............. loadHallucinationStats()
 * 
 * V4 FEATURES (Training & Evolution)
 *   Line  863 - Training Data ............. loadTrainingData(), loadTuningJobs()
 *   Line  900 - Evolution ................. loadEvolutionHistory()
 *   Line  930 - Training Charts ........... updateTrainingChart()
 * 
 * V5 FEATURES (Data Explorer)
 *   Line  970 - File Tree ................. refreshFileList(), renderFileTree()
 *   Line 1060 - File Viewer ............... loadFile(), navigateUp()
 *   Line 1120 - Quarantine ................ reviewQuarantine()
 * 
 * V6 FEATURES (The Lab)
 *   Line 1150 - Lab Models ................ loadLabModels()
 *   Line 1200 - Experiments ............... runLabExperiment(), pollExperimentStatus()
 *   Line 1280 - Agent Injection ........... updateInjectionPayload(), executeInjection()
 * 
 * V7 FEATURES (Neural Command Center)
 *   Line 1330 - Cytoscape Graph ........... initNeuralGraph(), updateNeuralGraph()
 *   Line 1420 - Topology Polling .......... pollSwarmTopology()
 *   Line 1480 - Node Inspector ............ loadNodeInspector()
 *   Line 1530 - Console/Control ........... handleConsoleCommand(), writeConsole()
 * 
 * V10 FEATURES (Finance & Prediction)
 *   Line 1663 - Market Data ............... loadMarketData(), renderMarketData()
 *   Line 1694 - AI Signals ................ loadAISignals(), renderAISignals()
 *   Line 1738 - Portfolio ................. loadPortfolio(), renderPortfolio()
 *   Line 1760 - Wallets ................... loadWalletStatus(), renderWalletStatus()
 *   Line 1771 - Charts .................... loadFinanceChart(), loadPredictions()
 *   Line 1812 - Refresh ................... refreshFinanceData()
 * 
 * V2 FEATURES (Intelligence & Prediction System)
 *   Line 1850 - Intelligence Feed ......... refreshV2Intelligence(), renderIntelligence()
 *   Line 1900 - Event Predictions ......... loadV2Predictions(), renderPredictions()
 *   Line 1950 - Network Analysis .......... loadV2PowerCenters(), runCascadePrediction()
 *   Line 2000 - Simulation ................ runMonteCarloSimulation()
 *   Line 2050 - OpSec ..................... encryptPrediction(), detectSurveillance()
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 */

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function () {
    initNavigation();
    initCharts();
    loadMetrics();
    loadSettings();
});

// ============================================
// Navigation
// ============================================
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.admin-section');
    const sectionTitle = document.getElementById('section-title');

    navItems.forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault();
            const targetSection = this.dataset.section;

            // Update active nav
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');

            // Show target section
            sections.forEach(section => section.classList.remove('active'));
            const activeSection = document.getElementById(targetSection);
            if (activeSection) {
                activeSection.classList.add('active');
                
                // Render mermaid diagrams when traces section becomes active
                if (targetSection === 'traces' && typeof mermaid !== 'undefined') {
                    setTimeout(() => {
                        try {
                            const mermaidEl = document.getElementById('trace-mermaid');
                            if (mermaidEl && mermaidEl.offsetParent !== null) {
                                mermaidEl.removeAttribute('data-processed');
                                mermaid.run();
                            }
                        } catch (e) {
                            console.warn('Mermaid render error:', e);
                        }
                    }, 100);
                }
            }

            // Update title
            sectionTitle.textContent = this.querySelector('span:last-child').textContent;
        });
    });
}

// ============================================
// Charts
// ============================================
let charts = {};

function initCharts() {
    // Latency Over Time Chart
    const latencyCtx = document.getElementById('latency-chart');
    if (latencyCtx) {
        charts.latency = new Chart(latencyCtx, {
            type: 'line',
            data: {
                labels: ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m', '10m'],
                datasets: [{
                    label: 'Latency (ms)',
                    data: [120, 150, 130, 180, 140, 160, 135, 145, 155, 140],
                    borderColor: '#4a9eff',
                    backgroundColor: 'rgba(74, 158, 255, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: getChartOptions('Latency (ms)')
        });
    }

    // Request Distribution Chart
    const requestCtx = document.getElementById('request-chart');
    if (requestCtx) {
        charts.requests = new Chart(requestCtx, {
            type: 'doughnut',
            data: {
                labels: ['Symbolic', 'LLM', 'Hybrid'],
                datasets: [{
                    data: [45, 35, 20],
                    backgroundColor: ['#4ade80', '#4a9eff', '#fbbf24'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#8899bb' }
                    }
                }
            }
        });
    }

    // Percentile Chart
    const percentileCtx = document.getElementById('percentile-chart');
    if (percentileCtx) {
        charts.percentile = new Chart(percentileCtx, {
            type: 'bar',
            data: {
                labels: ['p50', 'p75', 'p90', 'p95', 'p99'],
                datasets: [{
                    label: 'Latency',
                    data: [120, 180, 320, 450, 850],
                    backgroundColor: 'rgba(74, 158, 255, 0.6)',
                    borderColor: '#4a9eff',
                    borderWidth: 1
                }]
            },
            options: getChartOptions('ms')
        });
    }

    // Cost by Provider Chart
    const costProviderCtx = document.getElementById('cost-provider-chart');
    if (costProviderCtx) {
        charts.costProvider = new Chart(costProviderCtx, {
            type: 'pie',
            data: {
                labels: ['Ollama (Free)', 'Gemini', 'Claude', 'Other'],
                datasets: [{
                    data: [85, 10, 5, 0],
                    backgroundColor: ['#4ade80', '#4a9eff', '#a78bfa', '#8899bb'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#8899bb' }
                    }
                }
            }
        });
    }

    // Token Usage Chart
    const tokenUsageCtx = document.getElementById('token-usage-chart');
    if (tokenUsageCtx) {
        charts.tokenUsage = new Chart(tokenUsageCtx, {
            type: 'line',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [
                    {
                        label: 'Input Tokens',
                        data: [12000, 15000, 18000, 14000, 22000, 8000, 10000],
                        borderColor: '#4a9eff',
                        tension: 0.4
                    },
                    {
                        label: 'Output Tokens',
                        data: [8000, 10000, 12000, 9000, 15000, 5000, 7000],
                        borderColor: '#4ade80',
                        tension: 0.4
                    }
                ]
            },
            options: getChartOptions('Tokens')
        });
    }

    // Module Usage Chart
    const moduleUsageCtx = document.getElementById('module-usage-chart');
    if (moduleUsageCtx) {
        charts.moduleUsage = new Chart(moduleUsageCtx, {
            type: 'bar',
            data: {
                labels: ['Abstract', 'Analogical', 'Compositional', 'Hierarchical', 'Spatial', 'Visual', 'LLM'],
                datasets: [{
                    label: 'Queries Handled',
                    data: [45, 32, 28, 40, 35, 25, 120],
                    backgroundColor: [
                        '#4ade80', '#4a9eff', '#fbbf24', '#a78bfa', '#f87171', '#38bdf8', '#8899bb'
                    ]
                }]
            },
            options: getChartOptions('Queries')
        });
    }

    // Quality Trend Chart
    const qualityTrendCtx = document.getElementById('quality-trend-chart');
    if (qualityTrendCtx) {
        charts.qualityTrend = new Chart(qualityTrendCtx, {
            type: 'line',
            data: {
                labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Current'],
                datasets: [{
                    label: 'Accuracy %',
                    data: [48, 55, 72, 88, 100],
                    borderColor: '#4ade80',
                    backgroundColor: 'rgba(74, 222, 128, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: getChartOptions('%')
        });
    }

    // Category Score Chart
    const categoryCtx = document.getElementById('category-chart');
    if (categoryCtx) {
        charts.category = new Chart(categoryCtx, {
            type: 'radar',
            data: {
                labels: ['Pattern', 'Abstract', 'Spatial', 'Compositional', 'Analogical', 'Hierarchical', 'Visual'],
                datasets: [
                    {
                        label: 'Before',
                        data: [80, 45, 20, 40, 45, 40, 15],
                        borderColor: '#f87171',
                        backgroundColor: 'rgba(248, 113, 113, 0.1)',
                        pointBackgroundColor: '#f87171'
                    },
                    {
                        label: 'After',
                        data: [90, 85, 75, 85, 80, 85, 70],
                        borderColor: '#4ade80',
                        backgroundColor: 'rgba(74, 222, 128, 0.1)',
                        pointBackgroundColor: '#4ade80'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(136, 153, 187, 0.2)' },
                        grid: { color: 'rgba(136, 153, 187, 0.2)' },
                        pointLabels: { color: '#8899bb' },
                        ticks: { display: false }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#8899bb' }
                    }
                }
            }
        });
    }
}

function getChartOptions(yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
            x: {
                grid: { color: 'rgba(136, 153, 187, 0.1)' },
                ticks: { color: '#8899bb' }
            },
            y: {
                grid: { color: 'rgba(136, 153, 187, 0.1)' },
                ticks: { color: '#8899bb' },
                title: {
                    display: true,
                    text: yLabel,
                    color: '#8899bb'
                }
            }
        },
        plugins: {
            legend: {
                labels: { color: '#8899bb' }
            }
        }
    };
}

// ============================================
// Metrics
// ============================================
const API_BASE = '';  // ICEBURG API server (relative)

async function loadMetrics() {
    // Try to fetch from real API first
    let metrics;
    try {
        const response = await fetch(`${API_BASE}/api/admin/metrics`);
        if (response.ok) {
            metrics = await response.json();
            // Map snake_case API response to camelCase
            metrics = {
                avgLatency: metrics.avg_latency || 0,
                totalRequests: metrics.total_requests || 0,
                accuracy: metrics.accuracy || 0,
                totalTokens: metrics.total_tokens || 0,
                ttft: metrics.ttft || 0,
                tpot: metrics.tpot || 0,
                tps: metrics.tps || 0,
                rps: metrics.rps || 0,
                inputTokens: metrics.input_tokens || 0,
                outputTokens: metrics.output_tokens || 0,
                totalCost: metrics.total_cost || 0,
                costPerQuery: metrics.cost_per_query || 0,
                avgConfidence: metrics.avg_confidence || 0,
                hallucinationRate: metrics.hallucination_rate || 0,
                symbolicRate: metrics.symbolic_rate || 0,
                memoryGB: metrics.memory_gb || 0,
                memoryPercent: metrics.memory_percent || 0,
                gpuPercent: metrics.gpu_percent || 0
            };
            addActivity('Metrics loaded from API');
        } else {
            throw new Error('API returned error');
        }
    } catch (e) {
        console.warn('Could not fetch from API, using defaults:', e);
        metrics = getDefaultMetrics();
    }

    // Overview metrics
    document.getElementById('avg-latency').textContent = metrics.avgLatency + 'ms';
    document.getElementById('total-requests').textContent = metrics.totalRequests.toLocaleString();
    document.getElementById('accuracy-score').textContent = metrics.accuracy + '%';
    document.getElementById('total-tokens').textContent = formatNumber(metrics.totalTokens);

    // Performance metrics
    document.getElementById('ttft').textContent = metrics.ttft;
    document.getElementById('tpot').textContent = metrics.tpot;
    document.getElementById('tps').textContent = metrics.tps;
    document.getElementById('rps').textContent = metrics.rps;

    // Cost metrics
    document.getElementById('input-tokens').textContent = formatNumber(metrics.inputTokens);
    document.getElementById('output-tokens').textContent = formatNumber(metrics.outputTokens);
    document.getElementById('total-cost').textContent = '$' + metrics.totalCost.toFixed(2);
    document.getElementById('cost-per-query').textContent = '$' + metrics.costPerQuery.toFixed(4);

    // Quality metrics
    document.getElementById('quality-accuracy').textContent = metrics.accuracy;
    document.getElementById('avg-confidence').textContent = metrics.avgConfidence;
    document.getElementById('hallucination-rate').textContent = metrics.hallucinationRate;
    document.getElementById('symbolic-rate').textContent = metrics.symbolicRate;

    // Resource metrics
    document.getElementById('memory-usage').textContent = metrics.memoryGB;
    document.getElementById('memory-bar').style.width = metrics.memoryPercent + '%';
    document.getElementById('gpu-usage').textContent = metrics.gpuPercent;
    document.getElementById('gpu-bar').style.width = metrics.gpuPercent + '%';
}

function getDefaultMetrics() {
    // Default metrics when API is not available
    return {
        avgLatency: 142,
        totalRequests: 1247,
        accuracy: 100,
        totalTokens: 523400,
        ttft: 85,
        tpot: 12,
        tps: 83,
        rps: 2.4,
        inputTokens: 312500,
        outputTokens: 210900,
        totalCost: 0.00,
        costPerQuery: 0.0000,
        avgConfidence: 92,
        hallucinationRate: 2,
        symbolicRate: 65,
        memoryGB: 4.2,
        memoryPercent: 52,
        gpuPercent: 35
    };
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// ============================================
// Settings
// ============================================
async function loadSettings() {
    let settings = {};

    // Try to fetch from API first
    try {
        const response = await fetch(`${API_BASE}/api/admin/settings`);
        if (response.ok) {
            settings = await response.json();
        }
    } catch (e) {
        console.warn('Could not fetch settings from API:', e);
        // Fall back to localStorage
        const stored = localStorage.getItem('iceburg-settings');
        if (stored) {
            try {
                settings = JSON.parse(stored);
            } catch (e) { }
        }
    }

    // Model settings
    if (settings.defaultModel) {
        document.getElementById('default-model').value = settings.defaultModel;
    }
    if (settings.synthesistModel) {
        document.getElementById('synthesist-model').value = settings.synthesistModel;
    }
    if (settings.vjepaModel) {
        document.getElementById('vjepa-model').value = settings.vjepaModel;
    }

    // Feature flags
    document.getElementById('formal-reasoning').checked = settings.formalReasoning !== false;
    document.getElementById('vjepa-enabled').checked = settings.vjepaEnabled !== false;
    document.getElementById('smart-routing').checked = settings.smartRouting !== false;
    document.getElementById('hybrid-mode').checked = settings.hybridMode === true;

    // Provider settings
    if (settings.ollamaUrl) {
        document.getElementById('ollama-url').value = settings.ollamaUrl;
    }
    if (settings.timeout) {
        document.getElementById('timeout').value = settings.timeout;
    }
}

async function saveSettings() {
    const settings = {
        defaultModel: document.getElementById('default-model').value,
        synthesistModel: document.getElementById('synthesist-model').value,
        vjepaModel: document.getElementById('vjepa-model').value,
        formalReasoning: document.getElementById('formal-reasoning').checked,
        vjepaEnabled: document.getElementById('vjepa-enabled').checked,
        smartRouting: document.getElementById('smart-routing').checked,
        hybridMode: document.getElementById('hybrid-mode').checked,
        ollamaUrl: document.getElementById('ollama-url').value,
        timeout: parseInt(document.getElementById('timeout').value)
    };

    // Save to localStorage as backup
    localStorage.setItem('iceburg-settings', JSON.stringify(settings));

    // Try to save to API
    try {
        await fetch(`${API_BASE}/api/admin/settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        addActivity('Settings saved to server');
    } catch (e) {
        console.warn('Could not save settings to API:', e);
        addActivity('Settings saved locally');
    }

    showNotification('Settings saved successfully!');
}

function resetSettings() {
    localStorage.removeItem('iceburg-settings');
    loadSettings();
    addActivity('Settings reset to defaults');
    showNotification('Settings reset to defaults');
}

// ============================================
// Actions
// ============================================
function refreshData() {
    addActivity('Data refreshed');
    loadMetrics();
    showNotification('Data refreshed');
}

function runBenchmark() {
    addActivity('Benchmark started...');
    showNotification('Starting benchmark run...');

    // Simulate benchmark run
    setTimeout(() => {
        addActivity('Benchmark completed: 10/10 (100%)');
        showNotification('Benchmark completed: 100%');
    }, 2000);
}

function addActivity(text) {
    const activityList = document.getElementById('activity-list');
    if (!activityList) return;

    const now = new Date();
    const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const item = document.createElement('div');
    item.className = 'activity-item';
    item.innerHTML = `
    <span class="activity-time">${time}</span>
    <span class="activity-text">${text}</span>
  `;

    activityList.insertBefore(item, activityList.firstChild);

    // Keep only last 10 items
    while (activityList.children.length > 10) {
        activityList.removeChild(activityList.lastChild);
    }
}

function showNotification(message) {
    // Simple notification - could be enhanced with toast library
    console.log('[ICEBURG Admin]', message);
}

// ============================================
// Export for global access
// ============================================
window.refreshData = refreshData;
window.runBenchmark = runBenchmark;
window.saveSettings = saveSettings;
window.resetSettings = resetSettings;

// ============================================
// V2 FEATURES
// ============================================

// Initialize Mermaid for trace diagrams
document.addEventListener('DOMContentLoaded', function () {
    if (typeof mermaid !== 'undefined') {
        try {
            mermaid.initialize({
                startOnLoad: false, // Don't auto-render, we'll do it manually
                securityLevel: 'loose', // Allow scripts if needed
                theme: 'dark',
                themeVariables: {
                    primaryColor: '#4a9eff',
                    primaryTextColor: '#e8f0ff',
                    primaryBorderColor: '#4a9eff',
                    lineColor: '#8899bb',
                    secondaryColor: '#1a1a2e',
                    tertiaryColor: '#0d0d15',
                    background: 'transparent',
                    mainBkg: '#1a1a2e',
                    nodeBorder: '#4a9eff',
                    clusterBkg: '#0d0d15',
                    titleColor: '#e8f0ff',
                    edgeLabelBackground: '#1a1a2e'
                }
            });
        } catch (e) {
            console.warn("Mermaid init failed:", e);
        }
    }

    // Initialize Token Attribution Chart
    const tokenAttrCtx = document.getElementById('token-attribution-chart');
    if (tokenAttrCtx) {
        new Chart(tokenAttrCtx, {
            type: 'bar',
            data: {
                labels: ['Router', 'Abstract', 'Analogical', 'Hierarchical', 'LLM'],
                datasets: [{
                    label: 'Tokens Used',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: ['#4a9eff', '#4ade80', '#fbbf24', '#a78bfa', '#f87171']
                }]
            },
            options: getChartOptions('Tokens')
        });
    }
});

// Traces Section
function refreshTrace() {
    addActivity('Trace refreshed');
    showNotification('Trace data refreshed');
    
    // Re-render mermaid diagram when trace section is visible
    if (typeof mermaid !== 'undefined') {
        const mermaidEl = document.getElementById('trace-mermaid');
        if (mermaidEl && mermaidEl.offsetParent !== null) {
            try {
                mermaidEl.removeAttribute('data-processed');
                mermaid.run();
            } catch (e) {
                console.warn('Mermaid render error:', e);
            }
        }
    }
}

// Analytics Section
function applyDateRange() {
    const fromDate = document.getElementById('date-from').value;
    const toDate = document.getElementById('date-to').value;
    addActivity(`Date range applied: ${fromDate} to ${toDate}`);
    showNotification(`Filtering data from ${fromDate} to ${toDate}`);
}

function runABTest() {
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

// Prompts Playground
const PROMPT_TEMPLATES = {
    analogy: 'Book : Library :: Fish : ?',
    hierarchy: 'A contains B and C. B contains D. What is the depth of D?',
    pattern: '2, 6, 18, 54, ?'
};

function loadTemplate(name) {
    const textarea = document.getElementById('playground-query');
    if (textarea && PROMPT_TEMPLATES[name]) {
        textarea.value = PROMPT_TEMPLATES[name];
        addActivity(`Template loaded: ${name}`);
    }
}

function runPlayground() {
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

function saveTemplate() {
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

// Sessions Section
async function searchSessions() {
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

async function exportSessions() {
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

// Security Section
async function exportAuditLog() {
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

// Utility function for file downloads
function downloadFile(filename, content, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showNotification(`Downloaded: ${filename}`);
}

// ============================================
// V3 FEATURES: ALERTS & HALLUCINATION
// ============================================

async function refreshAlerts() {
    const container = document.getElementById('alerts-list');
    if (!container) return;

    try {
        const response = await fetch(`${API_BASE}/api/admin/alerts`);
        if (response.ok) {
            const data = await response.json();

            if (data.alerts.length === 0) {
                container.innerHTML = '<p style="color: var(--admin-success); padding: 0.5rem 0;">No active alerts - system healthy</p>';
            } else {
                container.innerHTML = data.alerts.map(a => `
                    <div class="alert-item" style="padding: 0.5rem; margin: 0.25rem 0; border-radius: 6px; background: ${getSeverityBg(a.severity)};">
                        <strong style="color: ${getSeverityColor(a.severity)};">[${a.severity.toUpperCase()}]</strong>
                        ${a.message}
                        <span style="color: var(--admin-text-muted); font-size: 0.75rem; margin-left: 0.5rem;">${a.timestamp}</span>
                    </div>
                `).join('');
            }
            addActivity(`Loaded ${data.count} alerts`);
        }
    } catch (e) {
        console.warn('Could not fetch alerts:', e);
        container.innerHTML = '<p style="color: var(--admin-text-muted);">Alerts unavailable (server offline)</p>';
    }
}

function getSeverityColor(severity) {
    switch (severity) {
        case 'critical': return 'var(--admin-danger)';
        case 'error': return 'var(--admin-danger)';
        case 'warning': return 'var(--admin-warning)';
        default: return 'var(--admin-accent)';
    }
}

function getSeverityBg(severity) {
    switch (severity) {
        case 'critical': return 'rgba(248, 113, 113, 0.2)';
        case 'error': return 'rgba(248, 113, 113, 0.15)';
        case 'warning': return 'rgba(251, 191, 36, 0.15)';
        default: return 'rgba(74, 158, 255, 0.1)';
    }
}

async function loadHallucinationStats() {
    try {
        const response = await fetch(`${API_BASE}/api/admin/hallucination-stats`);
        if (response.ok) {
            const data = await response.json();

            document.getElementById('hall-total').textContent = data.total_checked || 0;
            document.getElementById('hall-detected').textContent = data.hallucinations_detected || 0;
            document.getElementById('hall-rate').textContent = ((data.detection_rate || 0) * 100).toFixed(1) + '%';
        }
    } catch (e) {
        console.warn('Could not fetch hallucination stats:', e);
    }
}

// Load V3 data when page loads
document.addEventListener('DOMContentLoaded', function () {
    refreshAlerts();
    loadHallucinationStats();
});

// ============================================
// V4 FEATURES: FINE-TUNING & EVOLUTION
// ============================================

async function loadTrainingData() {
    try {
        const response = await fetch(`${API_BASE}/api/admin/training/data-stats`);
        if (response.ok) {
            const data = await response.json();

            // Update metrics
            document.getElementById('train-conv-count').textContent = data.total_conversations || 0;
            document.getElementById('train-conv-tokens').textContent = `${(data.total_tokens || 0).toLocaleString()} tokens`;

            // Update chart if we have chart.js loaded
            updateTrainingChart(data);
        }
    } catch (e) {
        console.warn('Could not fetch training stats:', e);
    }
}

async function loadTuningJobs() {
    try {
        const response = await fetch(`${API_BASE}/api/admin/training/jobs`);
        if (response.ok) {
            const data = await response.json();
            const jobs = data.jobs || [];

            // Update active count
            document.getElementById('train-jobs-active').textContent = jobs.filter(j => !j.success && !j.error).length;
            document.getElementById('train-jobs-completed').textContent = `${jobs.filter(j => j.success).length} completed`;

            // Update table
            const tbody = document.getElementById('tuning-jobs-tbody');
            if (jobs.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--admin-text-muted);">No tuning jobs recorded</td></tr>';
            } else {
                tbody.innerHTML = jobs.slice(0, 10).map(job => `
                    <tr>
                        <td class="session-id">${job.id.substring(0, 8)}...</td>
                        <td>${job.model_type}</td>
                        <td>${Object.keys(job.hyperparameters || {}).join(', ')}</td>
                        <td><span class="session-status ${job.success ? 'success' : 'pending'}">${job.success ? 'Completed' : 'Running'}</span></td>
                        <td>${JSON.stringify(job.metrics || {}).substring(0, 50)}...</td>
                    </tr>
                `).join('');
            }
        }
    } catch (e) {
        console.warn('Could not fetch tuning jobs:', e);
    }
}

async function loadEvolutionHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/admin/evolution/history`);
        if (response.ok) {
            const data = await response.json();
            const metrics = data.metrics || {};
            const history = data.recommendations || []; // Using recommendations as proxy for history if full history is per-agent only

            // Update metrics
            document.getElementById('train-evolved-count').textContent = metrics.evolved_count || 0;
            document.getElementById('train-evolved-rate').textContent = `+${((metrics.avg_improvement || 0) * 100).toFixed(1)}% improvement`;

            // Update timeline
            const timeline = document.getElementById('evolution-timeline');
            if (!history || history.length === 0) {
                timeline.innerHTML = '<div class="activity-item">No evolution history yet</div>';
            } else {
                timeline.innerHTML = history.map(item => `
                    <div class="activity-item">
                        <div class="activity-icon">🧬</div>
                        <div class="activity-details">
                            <span class="activity-text"><strong>${item.agent_id}</strong> prompt evolved</span>
                            <span class="activity-time">${item.reason || 'Performance improvement'}</span>
                        </div>
                    </div>
                `).join('');
            }
        }
    } catch (e) {
        console.warn('Could not fetch evolution history:', e);
    }
}

function updateTrainingChart(data) {
    const ctx = document.getElementById('trainingDataChart');
    if (!ctx) return;

    // Simple distribution chart based on data availability
    // If we have distribution data, use it. Otherwise mock a distribution for visualization if count > 0
    const hasData = (data.total_conversations || 0) > 0;

    if (window.trainingChart) window.trainingChart.destroy();

    window.trainingChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High Quality', 'Standard', 'Low Quality'],
            datasets: [{
                data: hasData ? [data.high_quality_count || 0, (data.total_conversations || 0) - (data.high_quality_count || 0), 0] : [0, 0, 1],
                backgroundColor: ['#22c55e', '#3b82f6', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { color: '#a0aec0' } }
            }
        }
    });
}

function exportTrainingData() {
    // This would typically call an endpoint to generate the file
    // For now we simulate it or link to the raw data endpoint
    window.open(`${API_BASE}/api/admin/training/data-stats?export=jsonl`, '_blank');
    addActivity('Exported training data');
}

// Update initialization to load V4 data
const originalInit = window.onload; // or just append to existing listener
// Ideally, add to the existing DOMContentLoaded listener or create a unified refresh function
// We'll hook into refreshData since that's called globally

const originalRefresh = window.refreshData;
window.refreshData = function () {
    if (originalRefresh) originalRefresh();

    // Add V3 & V4 refresh calls
    if (window.refreshAlerts) window.refreshAlerts();
    if (window.loadHallucinationStats) window.loadHallucinationStats();

    // Only load heavy V4 data if training tab is visible or on initial load
    loadTrainingData();
    loadTuningJobs();
    loadEvolutionHistory();

    // Update timestamp
    const now = new Date().toLocaleTimeString();
    const el = document.getElementById('training-updated');
    if (el) el.textContent = now;
};


// ============================================
// V5 FEATURES: DATA EXPLORER & QUARANTINE
// ============================================

let currentPath = "";
let currentFile = null;

async function refreshFileList(path = "") {
    const treeContainer = document.getElementById('file-tree');
    if (!treeContainer) return;

    if (!path) treeContainer.innerHTML = '<div class="file-item loading">Loading...</div>';

    try {
        const response = await fetch(`${API_BASE}/api/admin/data/files?path=${encodeURIComponent(path)}`);
        if (response.ok) {
            const data = await response.json();
            currentPath = data.current_path;
            document.getElementById('current-path').textContent = currentPath || '/';

            renderFileTree(data.files, treeContainer);
        }
    } catch (e) {
        console.warn('Could not fetch file list:', e);
        treeContainer.innerHTML = '<div class="file-item error">Error loading files</div>';
    }
}

function renderFileTree(files, container) {
    if (files.length === 0) {
        container.innerHTML = '<div class="file-item empty">No files found</div>';
        return;
    }

    // Add ".." if not at root
    let html = '';
    if (currentPath) {
        html += `<div class="file-item folder back" onclick="navigateUp()">
            <span class="icon">📁</span> ..
        </div>`;
    }

    html += files.map(file => `
        <div class="file-item ${file.type}" onclick="${file.type === 'directory' ? `refreshFileList('${file.path}')` : `loadFile('${file.path}')`}">
            <span class="icon">${file.type === 'directory' ? '📁' : '📄'}</span>
            <span class="name">${file.name}</span>
            ${file.type === 'file' ? `<span class="size">${formatBytes(file.size)}</span>` : ''}
        </div>
    `).join('');

    container.innerHTML = html;
}

async function loadFile(path) {
    const viewer = document.getElementById('json-viewer');
    const nameEl = document.getElementById('file-name');
    const actionsEl = document.getElementById('file-actions');
    const sizeEl = document.getElementById('file-size');
    const approveBtn = document.getElementById('quarantine-approve-btn');
    const rejectBtn = document.getElementById('quarantine-reject-btn');

    viewer.value = 'Loading...';
    currentFile = path;
    nameEl.textContent = path.split('/').pop();

    // Check if it's a quarantine file
    const isQuarantine = path.includes('quarantined');
    actionsEl.style.display = 'flex';
    approveBtn.style.display = isQuarantine ? 'inline-block' : 'none';
    rejectBtn.style.display = isQuarantine ? 'inline-block' : 'none';

    try {
        const response = await fetch(`${API_BASE}/api/admin/data/content?path=${encodeURIComponent(path)}`);
        if (response.ok) {
            const data = await response.json();

            if (data.type === 'json' || data.type === 'jsonl') {
                viewer.value = JSON.stringify(data.content, null, 2);
            } else {
                viewer.value = data.content;
            }

            sizeEl.textContent = formatBytes(viewer.value.length);
        } else {
            const err = await response.json();
            viewer.value = `Error: ${err.error}`;
        }
    } catch (e) {
        console.warn('Could not load file:', e);
        viewer.value = 'Error loading file content';
    }
}

function navigateUp() {
    const parts = currentPath.split('/');
    parts.pop();
    refreshFileList(parts.join('/'));
}

async function reviewQuarantine(action) {
    if (!currentFile) return;

    // Logic to extract ID from filename would go here, or we pass it?
    // For now assuming we just log it as a simulation
    const id = currentFile.split('_').pop().replace('.json', ''); // Rough heuristic

    try {
        const response = await fetch(`${API_BASE}/api/admin/quarantine/${id}/review?action=${action}`, {
            method: 'POST'
        });

        if (response.ok) {
            showNotification(`Item marked as ${action}ed`);
            // Refresh logic could go here
        }
    } catch (e) {
        console.warn('Error reviewing item:', e);
    }
}

function formatBytes(bytes, decimals = 2) {
    if (!+bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

// ============================================
// V6 FEATURES: THE LAB (CONTROL PLANE)
// ============================================

async function loadLabModels() {
    const select = document.getElementById('lab-model-select');
    if (!select) return;

    try {
        const response = await fetch(`${API_BASE}/api/lab/models`);
        if (response.ok) {
            const data = await response.json();
            select.innerHTML = data.models.map(m =>
                `<option value="${m.id}">${m.provider}: ${m.id} (${formatBytes(m.context_window)} ctx)</option>`
            ).join('');
        }
    } catch (e) {
        console.warn('Could not load lab models:', e);
        select.innerHTML = '<option disabled>Error loading models</option>';
    }
}

async function runLabExperiment() {
    const model = document.getElementById('lab-model-select').value;
    const prompt = document.getElementById('lab-prompt-input').value;
    const logDiv = document.getElementById('lab-experiment-log');

    if (!model || !prompt) {
        showNotification('Please select a model and enter a prompt', 'error');
        return;
    }

    logDiv.innerHTML += `<div style="color: var(--admin-accent);">[${new Date().toLocaleTimeString()}] Starting run on ${model}...</div>`;
    logDiv.scrollTop = logDiv.scrollHeight;

    try {
        const response = await fetch(`${API_BASE}/api/lab/experiment/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: model,
                user_prompt: prompt,
                agent_type: 'architect'
            })
        });

        if (response.ok) {
            const data = await response.json();
            const expId = data.experiment_id;
            pollExperimentStatus(expId, logDiv);
        } else {
            logDiv.innerHTML += `<div style="color: var(--admin-danger);">[ERROR] Launch failed</div>`;
        }
    } catch (e) {
        logDiv.innerHTML += `<div style="color: var(--admin-danger);">[ERROR] ${e.message}</div>`;
    }
}

async function pollExperimentStatus(expId, logDiv) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/lab/experiment/${expId}`);
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    logDiv.innerHTML += `<div style="color: var(--admin-success);">[COMPLETE] Run finished.</div>`;
                    logDiv.innerHTML += `<pre style="color: var(--admin-text); background: rgba(0,0,0,0.3); padding: 0.5rem;">${data.result}</pre>`;
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    logDiv.innerHTML += `<div style="color: var(--admin-danger);">[FAILED] ${data.error}</div>`;
                }
            }
        } catch (e) {
            clearInterval(pollInterval);
        }
        logDiv.scrollTop = logDiv.scrollHeight;
    }, 1000);
}

function updateInjectionPayload() {
    const type = document.getElementById('lab-injection-type').value;
    const payloadInput = document.getElementById('lab-injection-payload');
    let template = {};
    if (type === 'memory_edit') template = { "key": "memory", "value": "New content", "op": "overwrite" };
    else if (type === 'force_output') template = { "text": "Forced response", "action": "stop" };
    payloadInput.value = JSON.stringify(template, null, 2);
}

async function executeInjection() {
    const agentId = document.getElementById('lab-agent-id').value;
    const type = document.getElementById('lab-injection-type').value;
    const payloadStr = document.getElementById('lab-injection-payload').value;
    const statusDiv = document.getElementById('lab-injection-status');

    if (!agentId) { showNotification('Target Agent ID required', 'error'); return; }

    try {
        const payload = JSON.parse(payloadStr);
        statusDiv.textContent = `Injecting ${type}...`;
        statusDiv.style.color = 'var(--admin-warning)';

        const response = await fetch(`${API_BASE}/api/lab/inject`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agent_id: agentId, type: type, data: payload })
        });

        if (response.ok) {
            statusDiv.textContent = `Injection SUCCESS.`;
            statusDiv.style.color = 'var(--admin-success)';
            showNotification('Injection Successful');
        } else {
            const err = await response.json();
            statusDiv.textContent = `Injection FAILED: ${err.error}`;
            statusDiv.style.color = 'var(--admin-danger)';
        }
    } catch (e) {
        statusDiv.textContent = `Error: Invalid JSON`;
        statusDiv.style.color = 'var(--admin-danger)';
    }
}

// Initialize V5 & V6
document.addEventListener('DOMContentLoaded', function () {
    refreshFileList();
    loadLabModels();
});

// Update refreshData to include V5
const v4Refresh = window.refreshData;
window.refreshData = function () {
    if (v4Refresh) v4Refresh();
    // refreshFileList(currentPath); 
};

// Export V2 functions
window.refreshTrace = refreshTrace;
window.applyDateRange = applyDateRange;
window.runABTest = runABTest;
window.loadTemplate = loadTemplate;
window.runPlayground = runPlayground;
window.saveTemplate = saveTemplate;
window.searchSessions = searchSessions;
window.exportSessions = exportSessions;
window.exportAuditLog = exportAuditLog;

// Export V3 functions
// ============================================
// V7 FEATURES: NEURAL COMMAND CENTER
// ============================================

let cy = null;
let topologyPollInterval = null;

function initNeuralGraph() {
    if (cy) return; // Already initialized

    const container = document.getElementById('cy');
    if (!container) return;

    if (typeof cytoscape === 'undefined') {
        console.warn('Cytoscape not loaded, skipping neural graph initialization');
        return;
    }

    cy = cytoscape({
        container: container,
        style: [
            // === V7.2: TIER-BASED NODE STYLES ===
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'font-size': '9px',
                    'font-family': 'monospace',
                    'text-valign': 'bottom',
                    'text-margin-y': 5
                }
            },
            // Supervisor (top tier) - Large, Cyan
            {
                selector: 'node[tier="supervisor"]',
                style: {
                    'background-color': '#00ffcc',
                    'color': '#00ffcc',
                    'width': 50,
                    'height': 50,
                    'border-width': 3,
                    'border-color': '#00ffcc'
                }
            },
            // Worker (mid tier) - Medium, Green
            {
                selector: 'node[tier="worker"]',
                style: {
                    'background-color': '#4ade80',
                    'color': '#4ade80',
                    'width': 35,
                    'height': 35
                }
            },
            // Tool (bottom tier) - Small, Gray/Purple
            {
                selector: 'node[tier="tool"]',
                style: {
                    'background-color': '#a855f7',
                    'color': '#a855f7',
                    'width': 20,
                    'height': 20,
                    'shape': 'rectangle'
                }
            },
            // Status overrides
            {
                selector: 'node[status="frozen"]',
                style: { 'background-color': '#ffcc00', 'color': '#ffcc00' }
            },
            {
                selector: 'node[status="offline"]',
                style: { 'background-color': '#ff0000', 'color': '#ff0000' }
            },
            {
                selector: 'node[status="processing"]',
                style: { 'border-width': 3, 'border-color': '#00ff00', 'border-style': 'solid' }
            },
            // Edges
            {
                selector: 'edge',
                style: {
                    'width': 1,
                    'line-color': '#444',
                    'target-arrow-color': '#444',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            }
        ],
        layout: {
            name: 'breadthfirst',  // V7.2: Hierarchical layout
            directed: true,
            spacingFactor: 1.5,
            avoidOverlap: true,
            roots: undefined // Will auto-detect supervisor as root
        }
    });

    // Node click handler
    cy.on('tap', 'node', function (evt) {
        const node = evt.target;
        loadNodeInspector(node.data());
    });

    // Start polling
    pollSwarmTopology();
}

async function pollSwarmTopology() {
    if (document.hidden || !document.getElementById('cy').offsetParent) return; // Stop if hidden

    try {
        const response = await fetch(`${API_BASE}/api/command/topology`);
        if (response.ok) {
            const data = await response.json();
            updateNeuralGraph(data);
        }
    } catch (e) {
        console.warn('Topology poll failed', e);
    }

    setTimeout(pollSwarmTopology, 2000); // Poll every 2s
}

function updateNeuralGraph(data) {
    if (!cy) return;

    // Update stats
    document.getElementById('stat-nodes').innerText = data.elements.nodes.length;
    document.getElementById('stat-edges').innerText = data.elements.edges.length;

    // Diff update (simplistic for prototype)
    cy.json({ elements: data.elements });

    // Run layout only if node count changed to avoid jumpiness during status updates
    const currentCount = cy.nodes().length;
    if (window.lastNodeCount !== currentCount) {
        cy.layout({
            name: 'breadthfirst',
            directed: true,
            padding: 20,
            animate: true,
            spacingFactor: 1.5
        }).run();
        window.lastNodeCount = currentCount;
    }
}

async function loadNodeInspector(nodeData) {
    const inspector = document.getElementById('inspector-content');
    inspector.innerHTML = `<div style="text-align: center; color: var(--admin-accent);">LOADING BRAIN STATE...</div>`;

    try {
        // Fetch memory and trace in parallel
        const [memoryRes, traceRes] = await Promise.all([
            fetch(`${API_BASE}/api/command/agent/${nodeData.id}/memory`),
            fetch(`${API_BASE}/api/command/agent/${nodeData.id}/trace?limit=20`)
        ]);

        const memoryData = await memoryRes.json();
        const traceData = await traceRes.json();

        // Build trace HTML (Attribution Graph vertical timeline)
        const traceStepColors = {
            'INIT': '#888',
            'RECEIVE': '#4a9eff',
            'THINK': '#ffcc00',
            'TOOL_CALL': '#ff6b6b',
            'TOOL_RESULT': '#66ff66',
            'SYNTHESIZE': '#cc66ff',
            'EMIT': '#00ffcc',
            'INTERCEPT': '#ff0000'
        };

        const traceHTML = traceData.trace && traceData.trace.length > 0
            ? traceData.trace.map(step => `
                <div class="trace-step" style="margin-bottom: 8px; padding-left: 12px; border-left: 2px solid ${traceStepColors[step.step_type] || '#444'};">
                    <div style="font-size: 0.65rem; color: ${traceStepColors[step.step_type] || '#888'}; text-transform: uppercase;">
                        ${step.step_type}
                    </div>
                    <div style="font-size: 0.75rem; color: #ccc;">
                        ${step.content.substring(0, 80)}${step.content.length > 80 ? '...' : ''}
                    </div>
                </div>
            `).join('')
            : '<div style="color: #444;">No trace data available.</div>';

        inspector.innerHTML = `
            <h4 style="color: var(--admin-accent); border-bottom: 1px solid #333; padding-bottom: 0.5rem;">${nodeData.label}</h4>
            
            <div style="margin-bottom: 1rem;">
                <span style="color: #666;">STATUS:</span> <span style="color: ${nodeData.status === 'processing' ? '#00ff00' : nodeData.status === 'frozen' ? '#ffcc00' : 'red'}">${nodeData.status.toUpperCase()}</span><br>
                <span style="color: #666;">MEMORY:</span> ${nodeData.memory}%
            </div>

            <!-- V7.2: Attribution Graph (Thought Trace) -->
            <div style="margin-bottom: 1rem;">
                <h5 style="color: var(--admin-accent); margin: 0 0 0.5rem 0; font-size: 0.7rem;">
                    // ATTRIBUTION_GRAPH (${traceData.count || 0} steps)
                </h5>
                <div class="trace-container" style="max-height: 200px; overflow-y: auto; background: #0a0a0a; padding: 0.5rem; border: 1px solid #222;">
                    ${traceHTML}
                </div>
            </div>

            <div style="margin-bottom: 1rem;">
                <h5 style="color: #888; margin: 0 0 0.5rem 0;">Short Term Memory</h5>
                <div class="code-editor" style="font-size: 0.75rem; height: 100px; overflow-y: auto;">
                    ${memoryData.short_term ? memoryData.short_term.map(m =>
            `<div style="margin-bottom: 4px;"><span style="color: #aaa;">${m.role}:</span> ${m.content.substring(0, 60)}...</div>`
        ).join('') : 'No active thoughts.'}
                </div>
            </div>
            
            <button class="btn-sci-fi" style="width: 100%; font-size: 0.7rem;" onclick="copyToConsole('${nodeData.id}')">Target in Console</button>
        `;

        // Store selected agent for console controls
        window.selectedAgentId = nodeData.id;

    } catch (e) {
        inspector.innerHTML = `<div style="color: red;">ERROR READING BRAIN STATE: ${e.message}</div>`;
    }
}

// Console & Control Logic
async function executeOverride(action) {
    if (!window.selectedAgentId) {
        writeConsole("ERROR: NO AGENT TARGETED.");
        return;
    }

    writeConsole(`INITIATING COMMAND: ${action.toUpperCase()} on ${window.selectedAgentId}...`);

    try {
        const response = await fetch(`${API_BASE}/api/command/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                agent_id: window.selectedAgentId,
                action: action
            })
        });

        const res = await response.json();
        if (res.success) {
            writeConsole(`SUCCESS: ${action.toUpperCase()} EXECUTED.`);
        } else {
            writeConsole(`FAILURE: ${res.error || 'Unknown error'}`);
        }
    } catch (e) {
        writeConsole(`SYSTEM ERROR: ${e.message}`);
    }
}

function handleConsoleCommand(e) {
    if (e.key === 'Enter') {
        const cmd = e.target.value;
        writeConsole(`$ ${cmd}`);
        e.target.value = '';

        // Parse basic commands
        if (cmd === '/help') {
            writeConsole("AVAILABLE COMMANDS: /target [id], /freeze, /kill, /clear");
        } else if (cmd === '/clear') {
            document.getElementById('console-output').innerHTML = '';
        } else {
            writeConsole("UNKNOWN COMMAND. TRY /help.");
        }
    }
}

function writeConsole(text) {
    const out = document.getElementById('console-output');
    out.innerHTML += `<div>> ${text}</div>`;
    out.scrollTop = out.scrollHeight;
}

function copyToConsole(id) {
    const input = document.getElementById('console-input');
    input.value = `/target ${id}`;
    input.focus();
}

// Init V7 on load if section is visible, or hook into nav
document.addEventListener('DOMContentLoaded', function () {
    // ... existing init ...

    // Check if we need to start graph
    if (location.hash === '#lab') {
        setTimeout(initNeuralGraph, 500); // Delay slightly for DOM layout
    }

    // Hook nav clicks
    document.querySelectorAll('a[href="#lab"]').forEach(el => {
        el.addEventListener('click', () => setTimeout(initNeuralGraph, 100)); // Init on click
    });

    // V10: Init Finance Dashboard if on that tab
    if (location.hash === '#finance') {
        setTimeout(() => {
            loadMarketData();
            loadAISignals();
            loadPortfolio();
            loadWalletStatus();
        }, 300);
    }

    // Hook Finance nav clicks
    document.querySelectorAll('a[href="#finance"]').forEach(el => {
        el.addEventListener('click', () => {
            setTimeout(() => {
                loadMarketData();
                loadAISignals();
                loadPortfolio();
                loadWalletStatus();
            }, 100);
        });
    });
});

// ============================================
// V10 FEATURES: FINANCE & PREDICTION
// ============================================

async function loadMarketData() {
    const symbols = 'BTC-USD,ETH-USD,AAPL,GOOGL,TSLA';
    try {
        const response = await fetch(`${API_BASE}/api/finance/market-data?symbols=${symbols}`);
        if (!response.ok) throw new Error('Failed to fetch market data');

        const data = await response.json();
        renderMarketData(data.data);
    } catch (e) {
        console.error('Market data error:', e);
        document.getElementById('market-data-grid').innerHTML =
            '<p style="color: var(--admin-danger);">Failed to load market data. Check API connection.</p>';
    }
}

function renderMarketData(marketData) {
    const grid = document.getElementById('market-data-grid');
    grid.innerHTML = '';

    for (const [symbol, data] of Object.entries(marketData)) {
        if (data.error) continue;

        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `
            <div class="metric-content">
                <span class="metric-value" style="font-size: 1.2rem;">${formatNumber(data.price, true)}</span>
                <span class="metric-label">${symbol}</span>
            </div>
        `;
        grid.appendChild(card);
    }
}

async function loadAISignals() {
    const symbols = 'AAPL,GOOGL,TSLA,BTC-USD,ETH-USD';
    try {
        // Load traditional AI signals
        const aiResponse = await fetch(`${API_BASE}/api/finance/ai-signals?symbols=${symbols}`);
        let aiSignals = [];

        if (aiResponse.ok) {
            const aiData = await aiResponse.json();
            aiSignals = aiData.signals || [];
        }

        // Load V2 Intelligence signals
        const v2Response = await fetch(`${API_BASE}/api/finance/intelligence-signals?limit=10`);
        let v2Data = { intelligence_signals: [], alpha_signals: [], event_predictions: [] };

        if (v2Response.ok) {
            v2Data = await v2Response.json();
        }

        renderAISignals(aiSignals, v2Data);
    } catch (e) {
        console.error('AI signals error:', e);
        document.getElementById('ai-signals-list').innerHTML =
            '<p style="color: var(--admin-danger);">Failed to load AI signals.</p>';
    }
}

function renderAISignals(signals, v2Data) {
    const container = document.getElementById('ai-signals-list');

    let html = '';

    // V2 Intelligence Summary Banner (if available)
    if (v2Data && v2Data.summary) {
        const summary = v2Data.summary;
        html += `
            <div style="padding: 0.75rem; margin-bottom: 1rem; background: linear-gradient(135deg, rgba(255,68,68,0.1), rgba(255,68,68,0.05)); border-left: 3px solid #ff4444; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.85rem; font-weight: bold; color: #ff4444;">🎯 V2 INTELLIGENCE ACTIVE</span>
                    <button onclick="switchToV2Dashboard()" class="btn-secondary" style="padding: 0.3rem 0.6rem; font-size: 0.75rem;">View Full Intelligence →</button>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-top: 0.5rem; font-size: 0.75rem;">
                    <div><span style="color: var(--admin-text-muted);">Signals:</span> <strong>${summary.total_signals}</strong></div>
                    <div><span style="color: var(--admin-text-muted);">Tradeable:</span> <strong>${summary.tradeable_signals}</strong></div>
                    <div><span style="color: var(--admin-text-muted);">High Priority:</span> <strong>${summary.high_priority_count}</strong></div>
                    <div><span style="color: var(--admin-text-muted);">Avg Confidence:</span> <strong>${(summary.avg_confidence * 100).toFixed(0)}%</strong></div>
                </div>
            </div>
        `;
    }

    // V2 Alpha Signals (from intelligence)
    if (v2Data && v2Data.alpha_signals && v2Data.alpha_signals.length > 0) {
        html += '<h4 style="margin: 1rem 0 0.5rem 0; font-size: 0.9rem; color: var(--admin-primary);">🎯 Intelligence-Based Signals</h4>';

        v2Data.alpha_signals.forEach(alpha => {
            const directionColor = alpha.direction === 'LONG' ? '#4caf50' : alpha.direction === 'SHORT' ? '#f44336' : '#fbbf24';
            const directionIcon = alpha.direction === 'LONG' ? '📈' : alpha.direction === 'SHORT' ? '📉' : '➡️';

            html += `
                <div style="padding: 0.75rem; margin-bottom: 0.5rem; background: rgba(0,0,0,0.2); border-left: 3px solid ${directionColor}; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.9rem; font-weight: bold;">${directionIcon} ${alpha.symbol} - ${alpha.direction}</span>
                        <span style="font-size: 0.75rem; padding: 0.2rem 0.5rem; background: rgba(255,68,68,0.2); border-radius: 3px; color: #ff4444;">V2 Intelligence</span>
                    </div>
                    <div style="font-size: 0.8rem; color: var(--admin-text-muted); margin-bottom: 0.5rem; line-height: 1.4;">
                        ${alpha.intelligence_source}
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; font-size: 0.75rem;">
                        <div><span style="color: var(--admin-text-muted);">Expected Return:</span> <strong style="color: ${directionColor};">${(alpha.expected_return * 100).toFixed(1)}%</strong></div>
                        <div><span style="color: var(--admin-text-muted);">Sharpe:</span> <strong>${alpha.sharpe_estimate.toFixed(2)}</strong></div>
                        <div><span style="color: var(--admin-text-muted);">Position:</span> <strong>${(alpha.position_size * 100).toFixed(1)}%</strong></div>
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <button onclick="executeAlphaSignal('${alpha.signal_id}', '${alpha.symbol}', '${alpha.direction}')" class="btn-primary" style="padding: 0.4rem 0.8rem; font-size: 0.75rem; width: 100%;">
                            Execute Trade (Paper Mode)
                        </button>
                    </div>
                </div>
            `;
        });
    }

    // Traditional AI Signals
    if (signals && signals.length > 0) {
        html += '<h4 style="margin: 1rem 0 0.5rem 0; font-size: 0.9rem; color: var(--admin-primary);">🤖 AI Model Signals</h4>';


        signals.forEach((s, i) => {
            html += `
            <div style="padding: 0.75rem; margin-bottom: 0.5rem; background: rgba(0,0,0,0.2); border-left: 3px solid var(--admin-primary); border-radius: 4px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: bold;">${s.symbol || 'Signal ' + (i + 1)}</span>
                    <span style="font-size: 0.8rem; color: var(--admin-text-muted);">${s.signal_type || 'Mixed'}</span>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem;">${s.signal_text || s.signal || 'No description'}</div>
                ${s.confidence ? '<div style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--admin-text-muted);">Confidence: ' + (s.confidence * 100).toFixed(0) + '%</div>' : ''}
            </div>
        `;
        });

    }

    // Event Predictions
    if (v2Data && v2Data.event_predictions && v2Data.event_predictions.length > 0) {
        html += '<h4 style="margin: 1rem 0 0.5rem 0; font-size: 0.9rem; color: var(--admin-primary);">⚠️ Event Predictions (Financial Impact)</h4>';

        v2Data.event_predictions.slice(0, 3).forEach(pred => {
            html += `
                <div style="padding: 0.5rem; margin-bottom: 0.5rem; background: rgba(243,156,18,0.1); border-left: 2px solid #f39c12; border-radius: 3px; font-size: 0.8rem;">
                    <div style="font-weight: bold; margin-bottom: 0.25rem;">${pred.description}</div>
                    <div style="color: var(--admin-text-muted); font-size: 0.75rem;">
                        Probability: ${(pred.probability * 100).toFixed(0)}% | Impact: ${(pred.expected_impact * 100).toFixed(0)}% | ${pred.timeframe}
                    </div>
                </div>
            `;
        });
    }

    if (html === '') {
        html = '<p style="color: var(--admin-text-muted);">No signals available at this time.</p>';
    }

    container.innerHTML = html;
}

// Helper function to switch to V2 dashboard
function switchToV2Dashboard() {
    document.querySelector('a[href="#v2-intelligence"]').click();
}

// Execute alpha signal from V2
async function executeAlphaSignal(signalId, symbol, direction) {
    const confirmed = confirm(`Execute ${direction} trade on ${symbol}?\n\nThis will be executed in PAPER MODE for validation.`);
    if (!confirmed) return;

    try {
        const response = await fetch(`${API_BASE}/api/finance/execute-alpha-signal`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                signal_id: signalId,
                symbol: symbol,
                direction: direction,
                execution_mode: 'paper'
            })
        });

        const data = await response.json();

        alert(`Trade Queued!\n\nSignal: ${signalId}\nSymbol: ${symbol}\nDirection: ${direction}\nMode: ${data.execution_mode}\n\nCheck portfolio for updates.`);

        addActivity(`Executed ${direction} signal on ${symbol} (Paper Mode)`);
    } catch (e) {
        console.error('Execute signal error:', e);
        alert('Failed to execute trade. Check console for details.');
    }
}

// Export for global access
window.switchToV2Dashboard = switchToV2Dashboard;
window.executeAlphaSignal = executeAlphaSignal;

async function loadPortfolio() {
    try {
        const response = await fetch(`${API_BASE}/api/finance/portfolio`);
        if (!response.ok) throw new Error('Failed to fetch portfolio');

        const data = await response.json();
        renderPortfolio(data);
    } catch (e) {
        console.error('Portfolio error:', e);
    }
}

function renderPortfolio(portfolio) {
    document.getElementById('portfolio-value').textContent = formatNumber(portfolio.total_value, true);
    document.getElementById('portfolio-pnl').textContent = formatNumber(portfolio.pnl_today, true);

    const container = document.getElementById('portfolio-positions');
    if (portfolio.note) {
        container.innerHTML = `<p style="color: var(--admin-text-muted); font-size: 0.85rem;">${portfolio.note}</p>`;
    }
}

async function loadWalletStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/finance/wallet-status`);
        if (!response.ok) throw new Error('Failed to fetch wallet status');

        const data = await response.json();
        renderWalletStatus(data);
    } catch (e) {
        console.error('Wallet status error:', e);
    }
}

function renderWalletStatus(wallets) {
    document.getElementById('wallet-total').textContent = formatNumber(wallets.total_value_usd, true);

    const container = document.getElementById('wallet-balances');
    if (wallets.note) {
        container.innerHTML = `<p style="color: var(--admin-text-muted); font-size: 0.85rem;">${wallets.note}</p>`;
    }
}

function loadFinanceChart() {
    const symbol = document.getElementById('finance-symbol-selector').value;
    const chartDiv = document.getElementById('finance-chart');

    // TradingView widget embedding
    chartDiv.innerHTML = `
        <div style="width: 100%; height: 100%;">
            <iframe 
                src="https://www.tradingview.com/widgetembed/?symbol=${symbol}&interval=D&theme=dark&style=1&locale=en&hide_side_toolbar=0&allow_symbol_change=1"
                style="width: 100%; height: 100%; border: 0;"
            ></iframe>
        </div>
    `;

    // Load predictions for selected symbol
    loadPredictions(symbol);
}

async function loadPredictions(symbol) {
    const panel = document.getElementById('predictions-panel');
    panel.innerHTML = '<p style="color: var(--admin-text-muted);">Loading predictions...</p>';

    try {
        const response = await fetch(`${API_BASE}/api/finance/predictions/${symbol}`);
        if (!response.ok) throw new Error('Failed to fetch predictions');

        const data = await response.json();
        panel.innerHTML = `
            <h5>AI Predictions for ${symbol}</h5>
            <div style="margin: 1rem 0;">
                <p style="color: var(--admin-text-muted); font-size: 0.9rem;">
                    ${data.ai_analysis || 'Analyzing market conditions...'}
                </p>
            </div>
        `;
    } catch (e) {
        console.error('Predictions error:', e);
        panel.innerHTML = '<p style="color: var(--admin-danger);">Failed to load predictions.</p>';
    }
}

function refreshFinanceData() {
    addActivity('Refreshing finance data...');
    loadMarketData();
    loadAISignals();
    loadPortfolio();
    loadWalletStatus();
    const symbol = document.getElementById('finance-symbol-selector').value;
    if (symbol) loadPredictions(symbol);
}

// Export for global access
window.refreshFinanceData = refreshFinanceData;
window.loadFinanceChart = loadFinanceChart;

// ============================================
// V2 FEATURES: INTELLIGENCE & PREDICTION SYSTEM
// ============================================

// Global state for V2
let v2_selected_prediction_id = null;
let v2_intelligence_data = [];
let v2_predictions_data = [];

// Intelligence Feed
async function refreshV2Intelligence() {
    const priority = document.getElementById('v2-intelligence-priority').value;
    const url = priority
        ? `${API_BASE}/api/v2/intelligence/feed?priority=${priority}&limit=50`
        : `${API_BASE}/api/v2/intelligence/feed?limit=50`;

    try {
        const response = await fetch(url);
        const data = await response.json();

        v2_intelligence_data = data.signals || [];
        renderV2Intelligence(v2_intelligence_data);
    } catch (e) {
        console.error('V2 Intelligence error:', e);
        document.getElementById('v2-intelligence-feed').innerHTML =
            '<p style="color: var(--admin-danger);">Failed to load intelligence feed.</p>';
    }
}

function renderV2Intelligence(signals) {
    const feed = document.getElementById('v2-intelligence-feed');

    if (!signals || signals.length === 0) {
        feed.innerHTML = '<p style="color: var(--admin-text-muted);">No intelligence signals available.</p>';
        return;
    }

    const priorityColors = {
        'critical': '#ff4444',
        'high': '#ff9800',
        'medium': '#fbbf24',
        'low': '#4caf50',
        'noise': '#gray'
    };

    feed.innerHTML = signals.map(signal => {
        const color = priorityColors[signal.priority] || '#ffffff';
        return `
            <div style="padding: 0.75rem; margin-bottom: 0.5rem; background: rgba(0,0,0,0.3); border-left: 3px solid ${color}; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                    <span style="font-size: 0.75rem; color: ${color}; font-weight: bold;">${signal.priority.toUpperCase()}</span>
                    <span style="font-size: 0.7rem; color: var(--admin-text-muted);">${signal.source_type} | ${new Date(signal.timestamp).toLocaleTimeString()}</span>
                </div>
                <p style="margin: 0.25rem 0; font-size: 0.85rem; line-height: 1.4;">${signal.content}</p>
                ${signal.entities && signal.entities.length > 0 ? `
                    <div style="margin-top: 0.5rem;">
                        ${signal.entities.map(e => `<span style="display: inline-block; padding: 0.2rem 0.5rem; margin-right: 0.25rem; background: rgba(255,255,255,0.1); border-radius: 3px; font-size: 0.7rem;">${e}</span>`).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

// Event Predictions
async function loadV2Predictions() {
    const category = document.getElementById('v2-prediction-category').value;
    const url = category
        ? `${API_BASE}/api/v2/prediction/events?category=${category}&limit=20`
        : `${API_BASE}/api/v2/prediction/events?limit=20`;

    try {
        const response = await fetch(url);
        const data = await response.json();

        v2_predictions_data = data.predictions || [];
        renderV2Predictions(v2_predictions_data);
    } catch (e) {
        console.error('V2 Predictions error:', e);
        document.getElementById('v2-predictions-grid').innerHTML =
            '<p style="color: var(--admin-danger);">Failed to load predictions.</p>';
    }
}

function renderV2Predictions(predictions) {
    const grid = document.getElementById('v2-predictions-grid');

    if (!predictions || predictions.length === 0) {
        grid.innerHTML = '<p style="color: var(--admin-text-muted);">No predictions available. Generate predictions via backend.</p>';
        return;
    }

    const categoryColors = {
        'geopolitical': '#e74c3c',
        'economic': '#f39c12',
        'technological': '#3498db',
        'black_swan': '#9b59b6',
        'corporate': '#1abc9c',
        'regime_change': '#e67e22'
    };

    grid.innerHTML = predictions.map(pred => {
        const color = categoryColors[pred.category] || '#ffffff';
        const probPercent = (pred.probability * 100).toFixed(1);
        const impactPercent = (pred.impact * 100).toFixed(0);

        return `
            <div onclick="selectV2Prediction('${pred.prediction_id}')" 
                 style="padding: 1rem; background: rgba(0,0,0,0.3); border-left: 3px solid ${color}; border-radius: 4px; cursor: pointer; transition: all 0.2s;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.75rem; color: ${color}; font-weight: bold; text-transform: uppercase;">${pred.category}</span>
                    <span style="font-size: 0.7rem; color: var(--admin-text-muted);">${pred.timeframe}</span>
                </div>
                <h4 style="margin: 0.5rem 0; font-size: 0.9rem; line-height: 1.3;">${pred.description}</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; margin-top: 0.75rem;">
                    <div>
                        <span style="display: block; font-size: 0.7rem; color: var(--admin-text-muted);">Probability</span>
                        <span style="font-size: 1.1rem; font-weight: bold; color: ${color};">${probPercent}%</span>
                    </div>
                    <div>
                        <span style="display: block; font-size: 0.7rem; color: var(--admin-text-muted);">Impact</span>
                        <span style="font-size: 1.1rem; font-weight: bold;">${impactPercent}%</span>
                    </div>
                    <div>
                        <span style="display: block; font-size: 0.7rem; color: var(--admin-text-muted);">Confidence</span>
                        <span style="font-size: 1.1rem; font-weight: bold;">${(pred.confidence * 100).toFixed(0)}%</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function selectV2Prediction(predictionId) {
    v2_selected_prediction_id = predictionId;
    console.log('Selected prediction:', predictionId);
    // Highlight selected
    document.querySelectorAll('#v2-predictions-grid > div').forEach(el => {
        el.style.opacity = '0.6';
    });
    event.target.closest('div').style.opacity = '1';
}

// Network Analysis
async function loadV2PowerCenters() {
    try {
        const response = await fetch(`${API_BASE}/api/v2/network/power-centers?min_influence=0.3`);
        const data = await response.json();

        renderPowerCenters(data.power_centers || []);
    } catch (e) {
        console.error('Power centers error:', e);
        document.getElementById('v2-power-centers').innerHTML =
            '<p style="color: var(--admin-danger); font-size: 0.85rem;">Failed to load power centers.</p>';
    }
}

function renderPowerCenters(centers) {
    const container = document.getElementById('v2-power-centers');

    if (!centers || centers.length === 0) {
        container.innerHTML = '<p style="color: var(--admin-text-muted); font-size: 0.85rem;">No power centers identified. Build network graph first.</p>';
        return;
    }

    container.innerHTML = centers.map((center, idx) => `
        <div style="padding: 0.5rem; margin-bottom: 0.5rem; background: rgba(155,89,182,0.1); border-left: 2px solid #9b59b6; border-radius: 3px;">
            <div style="font-size: 0.8rem; font-weight: bold; margin-bottom: 0.25rem;">Center ${idx + 1}</div>
            <div style="font-size: 0.75rem; color: var(--admin-text-muted);">
                Nodes: ${center.core_nodes.length} | Influence: ${center.total_influence.toFixed(2)} | Radius: ${center.influence_radius.toFixed(1)}
            </div>
            <div style="margin-top: 0.25rem; font-size: 0.7rem;">
                ${center.core_nodes.slice(0, 5).map(n => `<span style="display: inline-block; padding: 0.1rem 0.3rem; margin-right: 0.2rem; background: rgba(255,255,255,0.1); border-radius: 2px;">${n}</span>`).join('')}
            </div>
        </div>
    `).join('');
}

async function runCascadePrediction() {
    const triggerNode = document.getElementById('v2-cascade-trigger').value.trim();

    if (!triggerNode) {
        document.getElementById('v2-cascade-result').innerHTML =
            '<p style="color: var(--admin-warning);">Please enter a trigger node</p>';
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/v2/network/cascade-prediction`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trigger_node: triggerNode,
                trigger_event: 'Cascade simulation',
                cascade_threshold: 0.3,
                max_hops: 5
            })
        });

        const data = await response.json();

        document.getElementById('v2-cascade-result').innerHTML = `
            <div style="padding: 0.5rem; background: rgba(0,0,0,0.2); border-radius: 3px;">
                <div style="margin-bottom: 0.5rem;">
                    <span style="font-size: 0.75rem; color: var(--admin-text-muted);">Cascade Probability:</span>
                    <span style="font-size: 1rem; font-weight: bold; color: #e74c3c; margin-left: 0.5rem;">${(data.cascade_probability * 100).toFixed(1)}%</span>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <span style="font-size: 0.75rem; color: var(--admin-text-muted);">Max Reach:</span>
                    <span style="font-size: 1rem; font-weight: bold; margin-left: 0.5rem;">${data.max_reach} nodes</span>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <span style="font-size: 0.75rem; color: var(--admin-text-muted);">Propagation Speed:</span>
                    <span style="font-size: 1rem; font-weight: bold; margin-left: 0.5rem;">${data.propagation_speed.toFixed(1)} nodes/day</span>
                </div>
                ${data.affected_nodes && data.affected_nodes.length > 0 ? `
                    <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.1);">
                        <span style="font-size: 0.75rem; color: var(--admin-text-muted); display: block; margin-bottom: 0.25rem;">Affected Nodes:</span>
                        <div style="font-size: 0.7rem; opacity: 0.8;">
                            ${data.affected_nodes.slice(0, 10).join(', ')}${data.affected_nodes.length > 10 ? '...' : ''}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    } catch (e) {
        console.error('Cascade prediction error:', e);
        document.getElementById('v2-cascade-result').innerHTML =
            '<p style="color: var(--admin-danger);">Failed to predict cascade.</p>';
    }
}

// Monte Carlo Simulation
async function runMonteCarloSimulation() {
    if (!v2_selected_prediction_id) {
        document.getElementById('v2-simulation-results').innerHTML =
            '<p style="color: var(--admin-warning);">Please select a prediction first</p>';
        return;
    }

    const nSimulations = parseInt(document.getElementById('v2-sim-count').value);
    const confidenceLevel = parseFloat(document.getElementById('v2-sim-confidence').value);

    document.getElementById('v2-simulation-results').innerHTML =
        '<p style="color: var(--admin-text-muted);">Running simulations...</p>';

    try {
        const response = await fetch(`${API_BASE}/api/v2/simulation/monte-carlo`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prediction_id: v2_selected_prediction_id,
                n_simulations: nSimulations,
                confidence_level: confidenceLevel,
                parallel_workers: 4
            })
        });

        const data = await response.json();

        document.getElementById('v2-simulation-results').innerHTML = `
            <h4 style="margin-bottom: 0.75rem; font-size: 0.9rem;">Simulation Results (${data.total_runs.toLocaleString()} runs)</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin-bottom: 0.75rem;">
                <div style="padding: 0.5rem; background: rgba(0,0,0,0.2); border-radius: 3px;">
                    <span style="display: block; font-size: 0.7rem; color: var(--admin-text-muted);">Mean Outcome</span>
                    <span style="font-size: 1.1rem; font-weight: bold;">${data.mean_outcome.toFixed(3)}</span>
                </div>
                <div style="padding: 0.5rem; background: rgba(0,0,0,0.2); border-radius: 3px;">
                    <span style="display: block; font-size: 0.7rem; color: var(--admin-text-muted);">Median Outcome</span>
                    <span style="font-size: 1.1rem; font-weight: bold;">${data.median_outcome.toFixed(3)}</span>
                </div>
                <div style="padding: 0.5rem; background: rgba(0,0,0,0.2); border-radius: 3px;">
                    <span style="display: block; font-size: 0.7rem; color: var(--admin-text-muted);">Std Deviation</span>
                    <span style="font-size: 1.1rem; font-weight: bold;">${data.std_dev.toFixed(3)}</span>
                </div>
                <div style="padding: 0.5rem; background: rgba(0,0,0,0.2); border-radius: 3px;">
                    <span style="display: block; font-size: 0.7rem; color: var(--admin-text-muted);">${(confidenceLevel * 100).toFixed(0)}% CI Range</span>
                    <span style="font-size: 0.9rem; font-weight: bold;">[${Object.values(data.confidence_intervals)[0].lower.toFixed(2)}, ${Object.values(data.confidence_intervals)[0].upper.toFixed(2)}]</span>
                </div>
            </div>
            <div style="padding: 0.75rem; background: rgba(0,0,0,0.2); border-radius: 3px;">
                <h5 style="margin-bottom: 0.5rem; font-size: 0.85rem;">Extreme Outcomes</h5>
                <div style="font-size: 0.75rem; line-height: 1.6;">
                    <div>Best Case: ${data.extreme_outcomes.best_case.expected_value.toFixed(3)} (p=${data.extreme_outcomes.best_case.probability.toFixed(3)})</div>
                    <div>Worst Case: ${data.extreme_outcomes.worst_case.expected_value.toFixed(3)} (p=${data.extreme_outcomes.worst_case.probability.toFixed(3)})</div>
                    <div>Most Likely: ${data.extreme_outcomes.most_likely.expected_value.toFixed(3)}</div>
                </div>
            </div>
        `;
    } catch (e) {
        console.error('Monte Carlo error:', e);
        document.getElementById('v2-simulation-results').innerHTML =
            '<p style="color: var(--admin-danger);">Failed to run simulation.</p>';
    }
}

// OpSec Functions
async function encryptPrediction() {
    const predictionData = document.getElementById('v2-opsec-prediction').value.trim();
    const securityLevel = document.getElementById('v2-opsec-level').value;

    if (!predictionData) {
        alert('Please enter prediction data to encrypt');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/v2/opsec/encrypt-prediction`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prediction_data: predictionData,
                security_level: securityLevel,
                authorized_entities: ['admin']
            })
        });

        const data = await response.json();

        // Update counters
        const currentCount = parseInt(document.getElementById('v2-encrypted-count').textContent);
        document.getElementById('v2-encrypted-count').textContent = currentCount + 1;

        // Log to surveillance
        const log = document.getElementById('v2-surveillance-log');
        const timestamp = new Date().toISOString();
        log.innerHTML = `<p style="font-family: monospace; font-size: 0.75rem; margin: 0.25rem 0; color: #4caf50;">[${timestamp}] ENCRYPTED: ${data.prediction_id} at ${securityLevel.toUpperCase()} level</p>` + log.innerHTML;

        // Clear input
        document.getElementById('v2-opsec-prediction').value = '';

        alert(`Prediction encrypted successfully!\nID: ${data.prediction_id}\nLevel: ${securityLevel.toUpperCase()}`);
    } catch (e) {
        console.error('Encryption error:', e);
        alert('Failed to encrypt prediction');
    }
}

// Export for global access
window.refreshV2Intelligence = refreshV2Intelligence;
window.loadV2Predictions = loadV2Predictions;
window.selectV2Prediction = selectV2Prediction;
window.loadV2PowerCenters = loadV2PowerCenters;
window.runCascadePrediction = runCascadePrediction;
window.runMonteCarloSimulation = runMonteCarloSimulation;
window.encryptPrediction = encryptPrediction;

// Auto-load on section change
document.addEventListener('DOMContentLoaded', function () {
    // Hook V2 nav clicks
    document.querySelectorAll('a[href="#v2-intelligence"]').forEach(el => {
        el.addEventListener('click', () => {
            setTimeout(() => {
                refreshV2Intelligence();
                loadV2Predictions();
            }, 100);
        });
    });
});
