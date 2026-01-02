/**
 * AGI Civilization - Interactive Game Engine
 * 
 * Canvas-based world visualization with real-time updates,
 * agent tracking, and interactive controls.
 */

// ==================== Configuration ====================
const CONFIG = {
    API_BASE: '/api/civilization',
    UPDATE_INTERVAL: 3000,  // ms between state refreshes (3 seconds to avoid rate limiting)
    RENDER_FPS: 60,
    WORLD_SIZE: { width: 100, height: 100 },
    AGENT_RADIUS: 8,
    RESOURCE_RADIUS: 6,
    COLORS: {
        background: '#06060a',
        grid: 'rgba(30, 144, 255, 0.05)',
        agents: {
            researcher: '#1e90ff',
            coordinator: '#ffcc66',
            specialist: '#ff66ff',
            generalist: '#66ffcc',
            leader: '#ffd700',
            follower: '#aaaaaa'
        },
        resources: {
            energy: '#ffcc00',
            knowledge: '#1e90ff',
            materials: '#ff6b6b',
            compute: '#ff66ff',
            default: '#4cd964'
        }
    }
};

// ==================== State ====================
let state = {
    isRunning: false,
    isPaused: false,
    simulationStep: 0,
    speed: 1,
    zoom: 1,
    pan: { x: 0, y: 0 },
    dragging: false,
    lastMouse: { x: 0, y: 0 },
    agents: [],
    resources: [],
    events: [],
    norms: {},
    economy: {},
    selectedAgent: null,
    hoveredAgent: null
};

// ==================== DOM Elements ====================
let canvas, ctx;
let tooltip;
let updateLoop = null;
let animationFrame = null;
let lastFrameTime = 0;
let frameCount = 0;
let lastFpsUpdate = 0;

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', () => {
    initCanvas();
    initControls();
    initTabs();
    initModal();
    startRenderLoop();
    fetchStatus();
});

function initCanvas() {
    canvas = document.getElementById('worldCanvas');
    ctx = canvas.getContext('2d');
    tooltip = document.getElementById('agentTooltip');

    // Set canvas size
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Mouse events for interaction
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('wheel', handleWheel);
    canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
        state.hoveredAgent = null;
    });
    canvas.addEventListener('click', handleClick);
}

function resizeCanvas() {
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
}

function initControls() {
    // Start button
    document.getElementById('startBtn').addEventListener('click', startSimulation);

    // Pause button
    document.getElementById('pauseBtn').addEventListener('click', pauseSimulation);

    // Step button
    document.getElementById('stepBtn').addEventListener('click', stepSimulation);

    // Speed selector
    document.getElementById('speedSelect').addEventListener('change', (e) => {
        state.speed = parseInt(e.target.value);
    });

    // Zoom controls
    document.getElementById('zoomInBtn').addEventListener('click', () => {
        state.zoom = Math.min(5, state.zoom * 1.2);
    });

    document.getElementById('zoomOutBtn').addEventListener('click', () => {
        state.zoom = Math.max(0.2, state.zoom / 1.2);
    });

    document.getElementById('resetViewBtn').addEventListener('click', () => {
        state.zoom = 1;
        state.pan = { x: 0, y: 0 };
    });
}

function initTabs() {
    const tabs = document.querySelectorAll('.civ-tab');
    const contents = document.querySelectorAll('.civ-tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.dataset.tab;

            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Show corresponding content
            contents.forEach(c => c.classList.remove('active'));
            document.getElementById(`${tabName}Tab`).classList.add('active');
        });
    });
}

function initModal() {
    const modal = document.getElementById('addAgentModal');

    document.getElementById('addAgentBtn').addEventListener('click', () => {
        modal.style.display = 'flex';
    });

    document.getElementById('closeModal').addEventListener('click', () => {
        modal.style.display = 'none';
    });

    document.getElementById('cancelAddAgent').addEventListener('click', () => {
        modal.style.display = 'none';
    });

    document.getElementById('confirmAddAgent').addEventListener('click', addNewAgent);

    // Close on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
}

// ==================== API Calls ====================
async function apiCall(endpoint, method = 'GET', body = null) {
    try {
        const options = {
            method,
            headers: { 'Content-Type': 'application/json' }
        };
        if (body) {
            options.body = JSON.stringify(body);
        }

        const response = await fetch(`${CONFIG.API_BASE}${endpoint}`, options);
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API call failed: ${endpoint}`, error);
        return null;
    }
}

async function fetchStatus() {
    const data = await apiCall('/status');
    if (data) {
        state.isRunning = data.is_running;
        state.simulationStep = data.simulation_step || 0;
        updateStatusUI(data);
    }
}

async function fetchWorldState() {
    const data = await apiCall('/world');
    if (data) {
        state.resources = data.resources || [];
        updateResourceCounter(data.resources?.length || 0);
    }
}

async function fetchAgents() {
    const data = await apiCall('/agents');
    if (data) {
        state.agents = data.agents || [];
        updateAgentCounter(data.total || 0);
        renderAgentList();
    }
}

async function fetchEvents() {
    const data = await apiCall('/events');
    if (data) {
        state.events = data.events || [];
        renderEventList();
    }
}

async function fetchNorms() {
    const data = await apiCall('/norms');
    if (data) {
        state.norms = data.norms || {};
        renderNormList(data);
    }
}

async function fetchEconomy() {
    const data = await apiCall('/economy');
    if (data) {
        state.economy = data;
        renderEconomy(data);
    }
}

async function refreshAllData() {
    await Promise.all([
        fetchStatus(),
        fetchWorldState(),
        fetchAgents(),
        fetchEvents(),
        fetchNorms(),
        fetchEconomy()
    ]);
}

// ==================== Simulation Control ====================
async function startSimulation() {
    setStatusMessage('Starting simulation...');

    const data = await apiCall('/start', 'POST', {
        world_size: [100, 100],
        max_agents: 20,
        initial_resources: []
    });

    if (data && data.status === 'started') {
        state.isRunning = true;
        state.isPaused = false;
        updateControlButtons();
        setStatusMessage('Simulation running');
        startUpdateLoop();
        await refreshAllData();
    } else {
        setStatusMessage('Failed to start simulation');
    }
}

async function pauseSimulation() {
    state.isPaused = !state.isPaused;
    updateControlButtons();

    if (state.isPaused) {
        stopUpdateLoop();
        setStatusMessage('Simulation paused');
    } else {
        startUpdateLoop();
        setStatusMessage('Simulation running');
    }
}

async function stepSimulation() {
    const steps = state.speed * 10;
    setStatusMessage(`Simulating ${steps} steps...`);

    const data = await apiCall('/step', 'POST', { steps, speed: state.speed });

    if (data) {
        state.simulationStep = data.current_step;
        document.getElementById('stepCounter').textContent = `Step: ${data.current_step}`;
        await refreshAllData();
        setStatusMessage(`Completed ${steps} steps`);
    }
}

async function addNewAgent() {
    const agentId = document.getElementById('agentIdInput').value.trim();
    const role = document.getElementById('agentRoleSelect').value;

    if (!agentId) {
        alert('Please enter an agent ID');
        return;
    }

    const data = await apiCall('/agents', 'POST', {
        agent_id: agentId,
        role: role
    });

    if (data && data.status === 'created') {
        document.getElementById('addAgentModal').style.display = 'none';
        document.getElementById('agentIdInput').value = '';
        await fetchAgents();
        setStatusMessage(`Added agent: ${agentId}`);
    } else {
        setStatusMessage('Failed to add agent');
    }
}

// ==================== Update Loop ====================
function startUpdateLoop() {
    if (updateLoop) return;

    updateLoop = setInterval(async () => {
        if (!state.isPaused && state.isRunning) {
            await refreshAllData();
        }
    }, CONFIG.UPDATE_INTERVAL);
}

function stopUpdateLoop() {
    if (updateLoop) {
        clearInterval(updateLoop);
        updateLoop = null;
    }
}

// ==================== Render Loop ====================
function startRenderLoop() {
    animationFrame = requestAnimationFrame(render);
}

function render(timestamp) {
    // FPS calculation
    frameCount++;
    if (timestamp - lastFpsUpdate >= 1000) {
        document.getElementById('fpsCounter').textContent = frameCount;
        frameCount = 0;
        lastFpsUpdate = timestamp;
    }

    // Clear canvas
    ctx.fillStyle = CONFIG.COLORS.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Apply camera transform
    ctx.save();
    ctx.translate(canvas.width / 2 + state.pan.x, canvas.height / 2 + state.pan.y);
    ctx.scale(state.zoom, state.zoom);
    ctx.translate(-canvas.width / 2, -canvas.height / 2);

    // Draw grid
    drawGrid();

    // Draw resources
    drawResources();

    // Draw agents
    drawAgents();

    // Draw connections between nearby agents
    drawConnections();

    ctx.restore();

    // Continue loop
    animationFrame = requestAnimationFrame(render);
}

function drawGrid() {
    const gridSize = 50;
    ctx.strokeStyle = CONFIG.COLORS.grid;
    ctx.lineWidth = 0.5;

    for (let x = 0; x < canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }

    for (let y = 0; y < canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
}

function drawResources() {
    state.resources.forEach(resource => {
        const pos = worldToScreen(resource.location[0], resource.location[1]);
        const color = CONFIG.COLORS.resources[resource.name] || CONFIG.COLORS.resources.default;
        const radius = CONFIG.RESOURCE_RADIUS * (resource.amount / resource.max_amount + 0.5);

        // Glow effect
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius * 2, 0, Math.PI * 2);
        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, radius * 2);
        gradient.addColorStop(0, color.replace(')', ', 0.3)').replace('rgb', 'rgba'));
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.fill();

        // Core
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
    });
}

function drawAgents() {
    state.agents.forEach(agent => {
        const pos = worldToScreen(agent.location[0], agent.location[1]);
        const color = CONFIG.COLORS.agents[agent.role] || CONFIG.COLORS.agents.generalist;
        const radius = CONFIG.AGENT_RADIUS;
        const isSelected = state.selectedAgent === agent.agent_id;
        const isHovered = state.hoveredAgent === agent.agent_id;

        // Selection ring
        if (isSelected) {
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius + 6, 0, Math.PI * 2);
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Hover ring
        if (isHovered) {
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Glow
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 3, 0, Math.PI * 2);
        const gradient = ctx.createRadialGradient(pos.x, pos.y, radius, pos.x, pos.y, radius + 10);
        gradient.addColorStop(0, color);
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.fill();

        // Body
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Energy indicator (arc around agent)
        const energy = agent.energy || 0.5;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 2, -Math.PI / 2, -Math.PI / 2 + (Math.PI * 2 * energy));
        ctx.strokeStyle = '#4cd964';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

function drawConnections() {
    // Draw lines between agents that are close to each other
    const threshold = 30;

    ctx.strokeStyle = 'rgba(30, 144, 255, 0.15)';
    ctx.lineWidth = 1;

    for (let i = 0; i < state.agents.length; i++) {
        for (let j = i + 1; j < state.agents.length; j++) {
            const a = state.agents[i];
            const b = state.agents[j];

            const posA = worldToScreen(a.location[0], a.location[1]);
            const posB = worldToScreen(b.location[0], b.location[1]);

            const dx = posA.x - posB.x;
            const dy = posA.y - posB.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < threshold * state.zoom) {
                ctx.beginPath();
                ctx.moveTo(posA.x, posA.y);
                ctx.lineTo(posB.x, posB.y);
                ctx.stroke();
            }
        }
    }
}

// ==================== Coordinate Conversion ====================
function worldToScreen(wx, wy) {
    // Map world coordinates (0-100) to screen space
    const scale = Math.min(canvas.width, canvas.height) / CONFIG.WORLD_SIZE.width;
    return {
        x: (wx / CONFIG.WORLD_SIZE.width) * canvas.width,
        y: (wy / CONFIG.WORLD_SIZE.height) * canvas.height
    };
}

function screenToWorld(sx, sy) {
    // Inverse of worldToScreen with camera transform
    const tx = (sx - canvas.width / 2 - state.pan.x) / state.zoom + canvas.width / 2;
    const ty = (sy - canvas.height / 2 - state.pan.y) / state.zoom + canvas.height / 2;

    return {
        x: (tx / canvas.width) * CONFIG.WORLD_SIZE.width,
        y: (ty / canvas.height) * CONFIG.WORLD_SIZE.height
    };
}

// ==================== Mouse Handlers ====================
function handleMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Handle panning
    if (state.dragging) {
        state.pan.x += mx - state.lastMouse.x;
        state.pan.y += my - state.lastMouse.y;
        state.lastMouse = { x: mx, y: my };
        return;
    }

    // Check for agent hover
    const worldPos = screenToWorld(mx, my);
    let hoveredAgent = null;

    for (const agent of state.agents) {
        const aPos = worldToScreen(agent.location[0], agent.location[1]);
        const dx = mx - (aPos.x * state.zoom + state.pan.x + canvas.width / 2 - canvas.width / 2 * state.zoom);
        const dy = my - (aPos.y * state.zoom + state.pan.y + canvas.height / 2 - canvas.height / 2 * state.zoom);
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < CONFIG.AGENT_RADIUS * state.zoom + 5) {
            hoveredAgent = agent;
            break;
        }
    }

    state.hoveredAgent = hoveredAgent ? hoveredAgent.agent_id : null;

    if (hoveredAgent) {
        showTooltip(hoveredAgent, e.clientX, e.clientY);
    } else {
        tooltip.style.display = 'none';
    }
}

function handleMouseDown(e) {
    const rect = canvas.getBoundingClientRect();
    state.dragging = true;
    state.lastMouse = { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function handleMouseUp(e) {
    state.dragging = false;
}

function handleWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    state.zoom = Math.max(0.2, Math.min(5, state.zoom * delta));
}

function handleClick(e) {
    if (state.hoveredAgent) {
        state.selectedAgent = state.hoveredAgent;
        renderAgentList();
    }
}

// ==================== Tooltip ====================
function showTooltip(agent, x, y) {
    document.getElementById('tooltipName').textContent = agent.agent_id;
    document.getElementById('tooltipRole').textContent = agent.role.toUpperCase();
    document.getElementById('tooltipEnergy').style.width = `${(agent.energy || 0.5) * 100}%`;
    document.getElementById('tooltipMood').style.width = `${((agent.mood || 0) + 1) / 2 * 100}%`;
    document.getElementById('tooltipReputation').style.width = `${(agent.reputation || 0.5) * 100}%`;

    const goal = agent.goals?.active > 0 ? `Active goals: ${agent.goals.active}` : 'No active goals';
    document.getElementById('tooltipGoal').textContent = goal;

    tooltip.style.display = 'block';
    tooltip.style.left = `${x + 15}px`;
    tooltip.style.top = `${y + 15}px`;
}

// ==================== UI Updates ====================
function updateStatusUI(data) {
    const statusEl = document.getElementById('civStatus');

    if (data.is_running) {
        statusEl.textContent = state.isPaused ? 'PAUSED' : 'RUNNING';
        statusEl.className = 'civ-status ' + (state.isPaused ? 'paused' : 'running');
    } else {
        statusEl.textContent = 'OFFLINE';
        statusEl.className = 'civ-status';
    }

    document.getElementById('stepCounter').textContent = `Step: ${data.simulation_step || 0}`;
}

function updateControlButtons() {
    document.getElementById('startBtn').disabled = state.isRunning && !state.isPaused;
    document.getElementById('pauseBtn').disabled = !state.isRunning;

    // Update pause button icon
    const pauseBtn = document.getElementById('pauseBtn');
    if (state.isPaused) {
        pauseBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
        `;
    } else {
        pauseBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="4" width="4" height="16"/>
                <rect x="14" y="4" width="4" height="16"/>
            </svg>
        `;
    }
}

function updateAgentCounter(count) {
    document.getElementById('agentCounter').textContent = `Agents: ${count}`;
}

function updateResourceCounter(count) {
    document.getElementById('resourceCounter').textContent = `Resources: ${count}`;
}

function setStatusMessage(message) {
    document.getElementById('statusMessage').textContent = message;
}

// ==================== Sidebar Renderers ====================
function renderAgentList() {
    const container = document.getElementById('agentList');

    if (state.agents.length === 0) {
        container.innerHTML = `
            <div class="civ-empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                    <circle cx="9" cy="7" r="4"/>
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
                    <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                </svg>
                <p>No agents yet</p>
                <span>Start simulation to spawn agents</span>
            </div>
        `;
        return;
    }

    container.innerHTML = state.agents.map(agent => `
        <div class="civ-agent-card ${state.selectedAgent === agent.agent_id ? 'selected' : ''}"
             onclick="selectAgent('${agent.agent_id}')">
            <div class="civ-agent-card-header">
                <span class="civ-agent-name">${agent.agent_id}</span>
                <span class="civ-agent-role ${agent.role}">${agent.role.toUpperCase()}</span>
            </div>
            <div class="civ-agent-stats">
                <div class="civ-agent-mini-bar">
                    <div class="civ-agent-mini-bar-fill energy" style="width: ${(agent.energy || 0.5) * 100}%"></div>
                </div>
                <div class="civ-agent-mini-bar">
                    <div class="civ-agent-mini-bar-fill mood" style="width: ${((agent.mood || 0) + 1) / 2 * 100}%"></div>
                </div>
            </div>
        </div>
    `).join('');
}

function selectAgent(agentId) {
    state.selectedAgent = agentId;
    renderAgentList();

    // Center view on agent
    const agent = state.agents.find(a => a.agent_id === agentId);
    if (agent) {
        const pos = worldToScreen(agent.location[0], agent.location[1]);
        state.pan.x = -(pos.x - canvas.width / 2);
        state.pan.y = -(pos.y - canvas.height / 2);
    }
}

function renderEventList() {
    const container = document.getElementById('eventList');

    if (state.events.length === 0) {
        container.innerHTML = `
            <div class="civ-empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                </svg>
                <p>No emergence events</p>
                <span>Events will appear as patterns emerge</span>
            </div>
        `;
        return;
    }

    container.innerHTML = state.events.slice().reverse().map(event => `
        <div class="civ-event-card ${event.type}">
            <div class="civ-event-header">
                <span class="civ-event-type">${event.type}</span>
                <span class="civ-event-severity ${event.severity.toLowerCase()}">${event.severity}</span>
            </div>
            <div class="civ-event-desc">${event.description}</div>
            <div class="civ-event-time">${formatTimestamp(event.timestamp)}</div>
        </div>
    `).join('');
}

function renderNormList(data) {
    const normList = document.getElementById('normList');
    const norms = Object.entries(data.norms || {});

    // Update stats
    document.getElementById('activeNorms').textContent = norms.length;
    document.getElementById('totalViolations').textContent = data.stats?.total_violations || 0;
    document.getElementById('avgAdherence').textContent =
        `${((data.stats?.avg_adherence_rate || 0) * 100).toFixed(0)}%`;

    if (norms.length === 0) {
        normList.innerHTML = `
            <div class="civ-empty-state">
                <p>No active norms</p>
                <span>Norms will emerge from agent behavior</span>
            </div>
        `;
        return;
    }

    normList.innerHTML = norms.map(([id, norm]) => `
        <div class="civ-norm-card">
            <div class="civ-norm-header">
                <span class="civ-norm-type">${norm.type.toUpperCase()}</span>
                <span class="civ-norm-strength">Strength: ${(norm.strength * 100).toFixed(0)}%</span>
            </div>
            <div class="civ-norm-desc">${norm.description}</div>
        </div>
    `).join('');
}

function renderEconomy(data) {
    document.getElementById('totalTrades').textContent = data.stats?.total_trades || 0;
    document.getElementById('giniCoeff').textContent = (data.stats?.gini_coefficient || 0).toFixed(3);
    document.getElementById('totalVolume').textContent = (data.stats?.total_volume || 0).toFixed(1);

    // Market prices
    const priceList = document.getElementById('priceList');
    const prices = Object.entries(data.market_prices || {});

    priceList.innerHTML = prices.map(([name, price]) => `
        <div class="civ-price-item">
            <span class="civ-price-name">${name}</span>
            <span class="civ-price-value">${price.toFixed(2)}</span>
        </div>
    `).join('') || '<div class="civ-empty-state"><span>No market data</span></div>';

    // Recent trades
    const tradeList = document.getElementById('tradeList');
    const trades = data.recent_trades || [];

    tradeList.innerHTML = trades.slice(0, 5).map(trade => `
        <div class="civ-trade-item">
            <span>${trade.seller_id} → ${trade.buyer_id}</span>
            <span>${trade.amount_offered} ${trade.resource_offered}</span>
        </div>
    `).join('') || '<div class="civ-empty-state"><span>No trades yet</span></div>';
}

// ==================== Utilities ====================
function formatTimestamp(ts) {
    const date = new Date(ts * 1000);
    return date.toLocaleTimeString();
}

// Make selectAgent available globally
window.selectAgent = selectAgent;
