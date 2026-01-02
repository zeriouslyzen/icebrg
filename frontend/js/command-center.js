import { API_BASE } from './utils.js';
import { copyToConsole } from './console.js'; // Import to ensure it's loaded/available

// ============================================
// Neural Command Center (V7)
// ============================================

let cy = null;

export function initNeuralGraph() {
    if (cy) return; // Already initialized

    const container = document.getElementById('cy');
    if (!container) return;

    if (typeof cytoscape === 'undefined') {
        console.warn('Cytoscape not loaded');
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
                    'text-margin-y': 5,
                    'color': '#8899bb'
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

export async function pollSwarmTopology() {
    if (document.hidden || !document.getElementById('cy').offsetParent) {
        setTimeout(pollSwarmTopology, 2000); // Keep loop alive but don't fetch
        return;
    }

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

export function updateNeuralGraph(data) {
    if (!cy) return;

    // Update stats
    const nodesEl = document.getElementById('stat-nodes');
    const edgesEl = document.getElementById('stat-edges');
    if (nodesEl) nodesEl.innerText = data.elements.nodes.length;
    if (edgesEl) edgesEl.innerText = data.elements.edges.length;

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

export async function loadNodeInspector(nodeData) {
    const inspector = document.getElementById('inspector-content');
    if (!inspector) return;

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
