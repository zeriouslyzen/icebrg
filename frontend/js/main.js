import { initNavigation, loadSettings, addActivity } from './dashboard.js';
import { initCharts, loadMetrics } from './monitoring.js';
import {
    runBenchmark, runABTest, runPlayground, loadTemplate, saveTemplate,
    searchSessions, exportSessions, exportAuditLog
} from './testing.js';
import {
    refreshAlerts, loadHallucinationStats, saveSettings, resetSettings
} from './dashboard.js'; // Re-import from dashboard
import { loadTrainingData, loadTuningJobs, loadEvolutionHistory, exportTrainingData } from './training.js';
import { refreshFileList, navigateUp, reviewQuarantine, loadFile } from './data-explorer.js';
import { loadLabModels, runLabExperiment, updateInjectionPayload, executeInjection } from './lab.js';
import { initNeuralGraph, loadNodeInspector } from './command-center.js';
import { handleConsoleCommand, executeOverride, copyToConsole } from './console.js';

// ============================================
// Initialization & Global Exports
// ============================================

// Expose functions globally for HTML event handlers (legacy support)
window.runBenchmark = runBenchmark;
window.runABTest = runABTest;
window.runPlayground = runPlayground;
window.loadTemplate = loadTemplate;
window.saveTemplate = saveTemplate;
window.searchSessions = searchSessions;
window.exportSessions = exportSessions;
window.exportAuditLog = exportAuditLog;
window.saveSettings = saveSettings;
window.resetSettings = resetSettings;
window.refreshData = refreshAllData; // Override with orchestrator
window.refreshFileList = refreshFileList;
window.navigateUp = navigateUp;
window.loadFile = loadFile;
window.reviewQuarantine = reviewQuarantine;
window.runLabExperiment = runLabExperiment;
window.updateInjectionPayload = updateInjectionPayload;
window.executeInjection = executeInjection;
window.handleConsoleCommand = handleConsoleCommand;
window.executeOverride = executeOverride;
window.copyToConsole = copyToConsole;
window.loadNodeInspector = loadNodeInspector;
window.refreshTrace = () => { addActivity('Trace refreshed'); console.log('Trace refreshed'); }; // Simple stub
window.applyDateRange = () => { addActivity('Date range applied'); }; // Simple stub

function refreshAllData() {
    addActivity('Refreshing all data...');
    loadMetrics();
    refreshAlerts();
    loadHallucinationStats();
    loadTrainingData();
    loadTuningJobs();
    loadEvolutionHistory();
    // refreshFileList(); // Don't reset file view on general refresh

    // Update timestamp
    const now = new Date().toLocaleTimeString();
    const el = document.getElementById('training-updated');
    if (el) el.textContent = now;
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('[ICEBURG] Initializing Admin Modules...');

    initNavigation();
    initCharts(); // This checks for Chart.js

    // Load initial data
    refreshAllData();
    loadSettings();
    refreshFileList();
    loadLabModels();

    // Initialize Neural Graph (if visible or delayed)
    // Check if we start on the lab tab
    if (location.hash === '#lab' || document.getElementById('lab').classList.contains('active')) {
        setTimeout(initNeuralGraph, 500);
    }

    // Hook nav clicks to init graph when switching to Lab
    document.querySelectorAll('[data-section="lab"]').forEach(el => {
        el.addEventListener('click', () => setTimeout(initNeuralGraph, 100));
    });

    console.log('[ICEBURG] Admin Modules Ready.');
});
