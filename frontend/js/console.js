import { API_BASE } from './utils.js';

// ============================================
// Console & Control Logic
// ============================================

export async function executeOverride(action) {
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

export function handleConsoleCommand(e) {
    if (e.key === 'Enter') {
        const cmd = e.target.value;
        writeConsole(`$ ${cmd}`);
        e.target.value = '';

        // Parse basic commands
        if (cmd === '/help') {
            writeConsole("AVAILABLE COMMANDS: /target [id], /freeze, /kill, /clear");
        } else if (cmd === '/clear') {
            document.getElementById('console-output').innerHTML = '';
        } else if (cmd.startsWith('/target ')) {
            const target = cmd.split(' ')[1];
            if (target) {
                window.selectedAgentId = target;
                writeConsole(`TARGET ACQUIRED: ${target}`);
            }
        } else {
            writeConsole("UNKNOWN COMMAND. TRY /help.");
        }
    }
}

export function writeConsole(text) {
    const out = document.getElementById('console-output');
    if (!out) return;
    out.innerHTML += `<div>> ${text}</div>`;
    out.scrollTop = out.scrollHeight;
}

export function copyToConsole(id) {
    const input = document.getElementById('console-input');
    if (!input) return;
    input.value = `/target ${id}`;
    input.focus();
    // Also simulate entering it effectively
    window.selectedAgentId = id;
    writeConsole(`TARGET SELECTOR ACTIVATED: ${id}`);
}

// Make accessible for onclick handlers
window.copyToConsole = copyToConsole;
