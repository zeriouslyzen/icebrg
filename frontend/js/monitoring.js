import { API_BASE, formatNumber, showNotification } from './utils.js';

// ============================================
// Charts Logic
// ============================================
let charts = {};

export function initCharts() {
    // Check if Chart is loaded
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded');
        return;
    }

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

export function getChartOptions(yLabel) {
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
// Metrics Logic
// ============================================

export async function loadMetrics(addActivityCallback) {
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
            if (addActivityCallback) addActivityCallback('Metrics loaded from API');
        } else {
            throw new Error('API returned error');
        }
    } catch (e) {
        console.warn('Could not fetch from API, using defaults:', e);
        metrics = getDefaultMetrics();
    }

    updateMetricsUI(metrics);
}

function updateMetricsUI(metrics) {
    const el = (id) => document.getElementById(id);
    const set = (id, val) => { if (el(id)) el(id).textContent = val; };

    // Overview metrics
    set('avg-latency', metrics.avgLatency + 'ms');
    set('total-requests', metrics.totalRequests.toLocaleString());
    set('accuracy-score', metrics.accuracy + '%');
    set('total-tokens', formatNumber(metrics.totalTokens));

    // Performance metrics
    set('ttft', metrics.ttft);
    set('tpot', metrics.tpot);
    set('tps', metrics.tps);
    set('rps', metrics.rps);

    // Cost metrics
    set('input-tokens', formatNumber(metrics.inputTokens));
    set('output-tokens', formatNumber(metrics.outputTokens));
    set('total-cost', '$' + metrics.totalCost.toFixed(2));
    set('cost-per-query', '$' + metrics.costPerQuery.toFixed(4));

    // Quality metrics
    set('quality-accuracy', metrics.accuracy);
    set('avg-confidence', metrics.avgConfidence);
    set('hallucination-rate', metrics.hallucinationRate);
    set('symbolic-rate', metrics.symbolicRate);

    // Resource metrics
    set('memory-usage', metrics.memoryGB);
    if (el('memory-bar')) el('memory-bar').style.width = metrics.memoryPercent + '%';
    set('gpu-usage', metrics.gpuPercent);
    if (el('gpu-bar')) el('gpu-bar').style.width = metrics.gpuPercent + '%';
}

export function getDefaultMetrics() {
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
