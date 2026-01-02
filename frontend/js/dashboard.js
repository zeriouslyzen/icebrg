import { API_BASE, showNotification } from './utils.js';

// ============================================
// Navigation & Core Dashboard Logic
// ============================================

export function initNavigation() {
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
            const targetEl = document.getElementById(targetSection);
            if (targetEl) targetEl.classList.add('active');

            // Update title
            sectionTitle.textContent = this.querySelector('span:last-child').textContent;
        });
    });
}

// ============================================
// Activity Feed
// ============================================

export function addActivity(text) {
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

// ============================================
// Settings
// ============================================

export async function loadSettings() {
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

    // Helper to safely set value
    const setVal = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
    const setChk = (id, val) => { const el = document.getElementById(id); if (el) el.checked = val; };

    // Model settings
    if (settings.defaultModel) setVal('default-model', settings.defaultModel);
    if (settings.synthesistModel) setVal('synthesist-model', settings.synthesistModel);
    if (settings.vjepaModel) setVal('vjepa-model', settings.vjepaModel);

    // Feature flags
    setChk('formal-reasoning', settings.formalReasoning !== false);
    setChk('vjepa-enabled', settings.vjepaEnabled !== false);
    setChk('smart-routing', settings.smartRouting !== false);
    setChk('hybrid-mode', settings.hybridMode === true);

    // Provider settings
    if (settings.ollamaUrl) setVal('ollama-url', settings.ollamaUrl);
    if (settings.timeout) setVal('timeout', settings.timeout);
}

export async function saveSettings() {
    const getVal = (id) => document.getElementById(id) ? document.getElementById(id).value : '';
    const getChk = (id) => document.getElementById(id) ? document.getElementById(id).checked : false;

    const settings = {
        defaultModel: getVal('default-model'),
        synthesistModel: getVal('synthesist-model'),
        vjepaModel: getVal('vjepa-model'),
        formalReasoning: getChk('formal-reasoning'),
        vjepaEnabled: getChk('vjepa-enabled'),
        smartRouting: getChk('smart-routing'),
        hybridMode: getChk('hybrid-mode'),
        ollamaUrl: getVal('ollama-url'),
        timeout: parseInt(getVal('timeout') || '30')
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

export function resetSettings() {
    localStorage.removeItem('iceburg-settings');
    loadSettings();
    addActivity('Settings reset to defaults');
    showNotification('Settings reset to defaults');
}

// ============================================
// Alerts & Hallucinations (V3)
// ============================================

export async function refreshAlerts() {
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

export async function loadHallucinationStats() {
    try {
        const response = await fetch(`${API_BASE}/api/admin/hallucination-stats`);
        if (response.ok) {
            const data = await response.json();

            const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
            set('hall-total', data.total_checked || 0);
            set('hall-detected', data.hallucinations_detected || 0);
            set('hall-rate', ((data.detection_rate || 0) * 100).toFixed(1) + '%');
        }
    } catch (e) {
        console.warn('Could not fetch hallucination stats:', e);
    }
}
