/**
 * Utility functions and configuration for ICEBURG Admin
 */

export const API_BASE = 'http://localhost:8000';

export function formatNumber(num) {
    if (num === undefined || num === null) return '0';
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

export function formatBytes(bytes, decimals = 2) {
    if (!+bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

export function showNotification(message, type = 'info') {
    // Simple notification - could be enhanced with toast library
    console.log(`[ICEBURG Admin] [${type.toUpperCase()}] ${message}`);
    // You could implement a UI toast here if desired
    // For now we just keep the console log as per original code
}

export function downloadFile(filename, content, type) {
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

// Make globally available for legacy calls if needed (mostly unnecessary with modules)
window.formatNumber = formatNumber;
window.formatBytes = formatBytes;
