/**
 * Astro-Physiology Advanced Visualization
 * V2: Real-time dashboards, celestial charts, meridian graphs, progress charts, trajectory plots
 */

// Chart.js is assumed to be loaded globally

/**
 * Create real-time health dashboard
 */
function createHealthDashboard(algorithmicData, containerId = 'health-dashboard') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Container ${containerId} not found`);
        return null;
    }
    
    const behavioralPredictions = algorithmicData.behavioral_predictions || {};
    const tcmPredictions = algorithmicData.tcm_predictions || {};
    
    // Create canvas for chart
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    
    // Prepare data
    const labels = Object.keys(behavioralPredictions).map(key => 
        key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    );
    const values = Object.values(behavioralPredictions).map(v => v * 100);
    
    return new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Biophysical Parameters',
                data: values,
                backgroundColor: 'rgba(99, 102, 241, 0.2)',
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    min: -100,
                    max: 100
                }
            }
        }
    });
}

/**
 * Create celestial position chart
 */
function createCelestialChart(celestialPositions, containerId = 'celestial-chart') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Container ${containerId} not found`);
        return null;
    }
    
    // Create canvas for chart
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    
    // Prepare data (simplified - would use actual celestial positions)
    const bodies = Object.keys(celestialPositions).filter(k => !k.startsWith('_'));
    const positions = bodies.map(body => {
        const pos = celestialPositions[body];
        return Array.isArray(pos) ? pos[0] : 0;  // Right ascension
    });
    
    return new Chart(ctx, {
        type: 'polarArea',
        data: {
            labels: bodies.map(b => b.charAt(0).toUpperCase() + b.slice(1)),
            datasets: [{
                label: 'Celestial Positions',
                data: positions,
                backgroundColor: [
                    'rgba(255, 99, 132, 0.6)',
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(255, 206, 86, 0.6)',
                    'rgba(75, 192, 192, 0.6)',
                    'rgba(153, 102, 255, 0.6)',
                    'rgba(255, 159, 64, 0.6)',
                    'rgba(199, 199, 199, 0.6)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true
        }
    });
}

/**
 * Create meridian connection graph (network visualization)
 */
function createMeridianGraph(synthesizedAnalysis, containerId = 'meridian-graph') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Container ${containerId} not found`);
        return null;
    }
    
    // Create SVG for network graph
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '400');
    container.appendChild(svg);
    
    const meridianConnections = synthesizedAnalysis.meridian_connections || {};
    
    // Simplified network visualization
    // In production, would use D3.js or similar for proper network graphs
    const nodes = Object.keys(meridianConnections);
    const centerX = 200;
    const centerY = 200;
    const radius = 150;
    
    nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / nodes.length;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        
        // Draw node
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', 20);
        circle.setAttribute('fill', '#6366f1');
        svg.appendChild(circle);
        
        // Draw label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x);
        text.setAttribute('y', y + 5);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', '#fff');
        text.setAttribute('font-size', '10px');
        text.textContent = node.substring(0, 3).toUpperCase();
        svg.appendChild(text);
        
        // Draw connection to center
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', centerX);
        line.setAttribute('y1', centerY);
        line.setAttribute('x2', x);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', '#6366f1');
        line.setAttribute('stroke-width', '1');
        line.setAttribute('opacity', '0.3');
        svg.insertBefore(line, circle);
    });
    
    return svg;
}

/**
 * Create intervention progress chart (timeline visualization)
 */
function createProgressChart(interventions, containerId = 'progress-chart') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Container ${containerId} not found`);
        return null;
    }
    
    // Create canvas for chart
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    
    // Prepare data (simplified - would use actual intervention progress)
    const trackingMetadata = interventions.tracking_metadata || {};
    const createdAt = trackingMetadata.created_at ? new Date(trackingMetadata.created_at) : new Date();
    const durationDays = trackingMetadata.estimated_duration_days || 30;
    
    // Generate timeline data
    const labels = [];
    const progressData = [];
    for (let i = 0; i <= durationDays; i += 7) {  // Weekly data points
        const date = new Date(createdAt);
        date.setDate(date.getDate() + i);
        labels.push(date.toLocaleDateString());
        // Simulated progress (would use actual tracking data)
        progressData.push(Math.min(100, (i / durationDays) * 100));
    }
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Intervention Progress',
                data: progressData,
                borderColor: 'rgba(99, 102, 241, 1)',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

/**
 * Create predictive trajectory plot (future health predictions)
 */
function createTrajectoryPlot(healthTrajectory, containerId = 'trajectory-plot') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Container ${containerId} not found`);
        return null;
    }
    
    // Create canvas for chart
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    
    const shortTerm = healthTrajectory.short_term || {};
    const mediumTerm = healthTrajectory.medium_term || {};
    const longTerm = healthTrajectory.long_term || {};
    
    // Prepare data from daily indicators
    const dailyIndicators = shortTerm.daily_indicators || [];
    const labels = dailyIndicators.map(d => {
        const date = new Date(d.date);
        return date.toLocaleDateString();
    });
    const energyLevels = dailyIndicators.map(d => d.energy_level * 100);
    const stressLevels = dailyIndicators.map(d => d.stress_susceptibility * 100);
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Energy Level',
                    data: energyLevels,
                    borderColor: 'rgba(34, 197, 94, 1)',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Stress Susceptibility',
                    data: stressLevels,
                    borderColor: 'rgba(239, 68, 68, 1)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

/**
 * Create 3D molecular visualization (if library available)
 */
function createMolecularVisualization(molecularImprint, containerId = 'molecular-viz') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Container ${containerId} not found`);
        return null;
    }
    
    // Placeholder for 3D visualization
    // Would use Three.js or similar library
    const placeholder = document.createElement('div');
    placeholder.style.padding = '20px';
    placeholder.style.textAlign = 'center';
    placeholder.style.color = '#666';
    placeholder.innerHTML = `
        <p>3D Molecular Visualization</p>
        <p style="font-size: 12px;">Requires Three.js or similar 3D library</p>
        <p style="font-size: 10px;">Voltage Gates: ${JSON.stringify(molecularImprint.voltage_gates || {})}</p>
    `;
    container.appendChild(placeholder);
    
    return placeholder;
}

/**
 * Initialize all visualizations for astro-physiology results
 */
function initializeAstroPhysiologyVisualizations(resultData) {
    const algorithmicData = resultData.algorithmic_data || resultData;
    
    // Create visualization container if it doesn't exist
    let vizContainer = document.getElementById('astro-physiology-visualizations');
    if (!vizContainer) {
        vizContainer = document.createElement('div');
        vizContainer.id = 'astro-physiology-visualizations';
        vizContainer.style.display = 'grid';
        vizContainer.style.gridTemplateColumns = 'repeat(auto-fit, minmax(400px, 1fr))';
        vizContainer.style.gap = '20px';
        vizContainer.style.marginTop = '20px';
        
        // Find the astro-physiology card and append visualizations
        const astroCard = document.querySelector('.astro-physiology-card');
        if (astroCard) {
            astroCard.appendChild(vizContainer);
        }
    }
    
    const charts = {};
    
    // Health Dashboard
    const dashboardContainer = document.createElement('div');
    dashboardContainer.id = 'health-dashboard';
    dashboardContainer.style.height = '300px';
    vizContainer.appendChild(dashboardContainer);
    charts.healthDashboard = createHealthDashboard(algorithmicData);
    
    // Celestial Chart
    if (algorithmicData.molecular_imprint && algorithmicData.molecular_imprint.celestial_positions) {
        const celestialContainer = document.createElement('div');
        celestialContainer.id = 'celestial-chart';
        celestialContainer.style.height = '300px';
        vizContainer.appendChild(celestialContainer);
        charts.celestialChart = createCelestialChart(algorithmicData.molecular_imprint.celestial_positions);
    }
    
    // Meridian Graph
    if (algorithmicData.synthesized_analysis) {
        const meridianContainer = document.createElement('div');
        meridianContainer.id = 'meridian-graph';
        meridianContainer.style.height = '400px';
        vizContainer.appendChild(meridianContainer);
        charts.meridianGraph = createMeridianGraph(algorithmicData.synthesized_analysis);
    }
    
    // Progress Chart
    if (resultData.interventions) {
        const progressContainer = document.createElement('div');
        progressContainer.id = 'progress-chart';
        progressContainer.style.height = '300px';
        vizContainer.appendChild(progressContainer);
        charts.progressChart = createProgressChart(resultData.interventions);
    }
    
    // Trajectory Plot
    if (algorithmicData.health_trajectory) {
        const trajectoryContainer = document.createElement('div');
        trajectoryContainer.id = 'trajectory-plot';
        trajectoryContainer.style.height = '300px';
        vizContainer.appendChild(trajectoryContainer);
        charts.trajectoryPlot = createTrajectoryPlot(algorithmicData.health_trajectory);
    }
    
    return charts;
}

// Export functions for use in main.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createHealthDashboard,
        createCelestialChart,
        createMeridianGraph,
        createProgressChart,
        createTrajectoryPlot,
        createMolecularVisualization,
        initializeAstroPhysiologyVisualizations
    };
}

