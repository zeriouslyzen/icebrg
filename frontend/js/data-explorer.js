import { API_BASE, formatBytes, showNotification, downloadFile } from './utils.js';

// ============================================
// Data Explorer (V5)
// ============================================

let currentPath = "";
let currentFile = null;

export async function refreshFileList(path = "") {
    const treeContainer = document.getElementById('file-tree');
    if (!treeContainer) return;

    if (!path) treeContainer.innerHTML = '<div class="file-item loading">Loading...</div>';

    try {
        const response = await fetch(`${API_BASE}/api/admin/data/files?path=${encodeURIComponent(path)}`);
        if (response.ok) {
            const data = await response.json();
            currentPath = data.current_path;
            const pathEl = document.getElementById('current-path');
            if (pathEl) pathEl.textContent = currentPath || '/';

            renderFileTree(data.files, treeContainer);
        }
    } catch (e) {
        console.warn('Could not fetch file list:', e);
        treeContainer.innerHTML = '<div class="file-item error">Error loading files</div>';
    }
}

export function renderFileTree(files, container) {
    if (files.length === 0) {
        container.innerHTML = '<div class="file-item empty">No files found</div>';
        return;
    }

    // Add ".." if not at root
    let html = '';
    if (currentPath) {
        // We use data attributes instead of onclick for better security/compatibility
        html += `<div class="file-item folder back" data-action="up">
            <span class="icon">📁</span> ..
        </div>`;
    }

    html += files.map(file => `
        <div class="file-item ${file.type}" data-action="${file.type === 'directory' ? 'navigate' : 'load'}" data-path="${file.path}">
            <span class="icon">${file.type === 'directory' ? '📁' : '📄'}</span>
            <span class="name">${file.name}</span>
            ${file.type === 'file' ? `<span class="size">${formatBytes(file.size)}</span>` : ''}
        </div>
    `).join('');

    container.innerHTML = html;

    // Attach event listeners for the newly created elements
    // This replaces the inline onclick handlers
    container.querySelectorAll('.file-item').forEach(el => {
        el.addEventListener('click', () => {
            const action = el.dataset.action;
            const path = el.dataset.path;

            if (action === 'up') navigateUp();
            else if (action === 'navigate') refreshFileList(path);
            else if (action === 'load') loadFile(path);
        });
    });
}

export async function loadFile(path) {
    const viewer = document.getElementById('json-viewer');
    const nameEl = document.getElementById('file-name');
    const actionsEl = document.getElementById('file-actions');
    const sizeEl = document.getElementById('file-size');
    const approveBtn = document.getElementById('quarantine-approve-btn');
    const rejectBtn = document.getElementById('quarantine-reject-btn');

    if (viewer) viewer.value = 'Loading...';
    currentFile = path;
    if (nameEl) nameEl.textContent = path.split('/').pop();

    // Check if it's a quarantine file
    const isQuarantine = path.includes('quarantined');
    if (actionsEl) actionsEl.style.display = 'flex';
    if (approveBtn) approveBtn.style.display = isQuarantine ? 'inline-block' : 'none';
    if (rejectBtn) rejectBtn.style.display = isQuarantine ? 'inline-block' : 'none';

    try {
        const response = await fetch(`${API_BASE}/api/admin/data/content?path=${encodeURIComponent(path)}`);
        if (response.ok) {
            const data = await response.json();

            if (data.type === 'json' || data.type === 'jsonl') {
                if (viewer) viewer.value = JSON.stringify(data.content, null, 2);
            } else {
                if (viewer) viewer.value = data.content;
            }

            if (sizeEl && viewer) sizeEl.textContent = formatBytes(viewer.value.length);
        } else {
            const err = await response.json();
            if (viewer) viewer.value = `Error: ${err.error}`;
        }
    } catch (e) {
        console.warn('Could not load file:', e);
        if (viewer) viewer.value = 'Error loading file content';
    }
}

export function navigateUp() {
    if (!currentPath) return;
    const parts = currentPath.split('/');
    parts.pop();
    refreshFileList(parts.join('/'));
}

export async function reviewQuarantine(action) {
    if (!currentFile) return;

    // Logic to extract ID from filename would go here
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
