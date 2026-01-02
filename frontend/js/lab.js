import { API_BASE, formatBytes, showNotification } from './utils.js';

// ============================================
// The Lab (V6)
// ============================================

export async function loadLabModels() {
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

export async function runLabExperiment() {
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

export async function pollExperimentStatus(expId, logDiv) {
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

export function updateInjectionPayload() {
    const type = document.getElementById('lab-injection-type').value;
    const payloadInput = document.getElementById('lab-injection-payload');
    let template = {};
    if (type === 'memory_edit') template = { "key": "memory", "value": "New content", "op": "overwrite" };
    else if (type === 'force_output') template = { "text": "Forced response", "action": "stop" };
    payloadInput.value = JSON.stringify(template, null, 2);
}

export async function executeInjection() {
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
