import { addActivity } from './dashboard.js';
import { API_BASE } from './utils.js';

// ============================================
// Training & Evolution (V4)
// ============================================

export async function loadTrainingData() {
    try {
        const response = await fetch(`${API_BASE}/api/admin/training/data-stats`);
        if (response.ok) {
            const data = await response.json();

            // Update metrics
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
            set('train-conv-count', data.total_conversations || 0);
            set('train-conv-tokens', `${(data.total_tokens || 0).toLocaleString()} tokens`);

            // Update chart if we have chart.js loaded
            updateTrainingChart(data);
        }
    } catch (e) {
        console.warn('Could not fetch training stats:', e);
    }
}

export async function loadTuningJobs() {
    try {
        const response = await fetch(`${API_BASE}/api/admin/training/jobs`);
        if (response.ok) {
            const data = await response.json();
            const jobs = data.jobs || [];

            // Update active count
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
            set('train-jobs-active', jobs.filter(j => !j.success && !j.error).length);
            set('train-jobs-completed', `${jobs.filter(j => j.success).length} completed`);

            // Update table
            const tbody = document.getElementById('tuning-jobs-tbody');
            if (tbody) {
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
        }
    } catch (e) {
        console.warn('Could not fetch tuning jobs:', e);
    }
}

export async function loadEvolutionHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/admin/evolution/history`);
        if (response.ok) {
            const data = await response.json();
            const metrics = data.metrics || {};
            const history = data.recommendations || [];

            // Update metrics
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
            set('train-evolved-count', metrics.evolved_count || 0);
            set('train-evolved-rate', `+${((metrics.avg_improvement || 0) * 100).toFixed(1)}% improvement`);

            // Update timeline
            const timeline = document.getElementById('evolution-timeline');
            if (timeline) {
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
        }
    } catch (e) {
        console.warn('Could not fetch evolution history:', e);
    }
}

export function updateTrainingChart(data) {
    const ctx = document.getElementById('trainingDataChart');
    if (!ctx || typeof Chart === 'undefined') return;

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

export function exportTrainingData() {
    if (window.addActivity) window.addActivity('Exported training data'); // Fallback if not imported correctly
    window.open(`${API_BASE}/api/admin/training/data-stats?export=jsonl`, '_blank');
}
