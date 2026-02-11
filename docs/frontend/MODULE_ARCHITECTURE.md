# Frontend Module Architecture
## ICEBURG Frontend Modularization Plan

**Created:** February 2026  
**Purpose:** Document module boundaries and API contracts for frontend refactoring

---

## Module Structure

```
frontend/
├── main.js                    # Core orchestration (< 2,000 lines)
├── connection.js              # SSE connection module (existing)
├── config.js                  # Unified configuration (existing)
└── src/
    ├── streaming.js           # Streaming abstraction layer
    ├── rendering.js           # Markdown/code/math rendering
    └── components/
        ├── MessageRenderer.js # Chat message rendering
        ├── ActionList.js      # Action/status carousel
        ├── StepCards.js       # Research step cards
        └── SettingsPanel.js   # Settings UI
```

---

## Module API Contracts

### `frontend/src/streaming.js`

**Purpose:** Abstraction layer for streaming queries

**Exports:**
```javascript
/**
 * Send a query with streaming support
 * @param {Object} options - Query options
 * @param {string} options.query - User query
 * @param {string} options.mode - Query mode ('fast', 'research', etc.)
 * @param {Object} options.settings - Model settings
 * @param {Object} options.data - Additional data
 * @param {function} options.onChunk - Called for each text chunk
 * @param {function} options.onMessage - Called for all message types
 * @param {function} options.onError - Called on error
 * @param {function} options.onDone - Called when complete
 * @returns {function} Abort function to cancel request
 */
export function sendQuery(options)

/**
 * Register callback for connection status changes
 * @param {function(boolean)} callback - Called with (isConnected)
 * @returns {function} Unsubscribe function
 */
export function onConnectionChange(callback)

/**
 * Get current connection status
 * @returns {boolean} True if connected
 */
export function getConnectionStatus()
```

**Imports:**
- `connection.js` - For actual SSE implementation

**Dependencies:** None (pure abstraction)

---

### `frontend/src/rendering.js`

**Purpose:** All markdown, code, and math rendering logic

**Exports:**
```javascript
/**
 * Render markdown text to HTML
 * @param {string} text - Markdown text
 * @returns {string} HTML string
 */
export function renderMarkdown(text)

/**
 * Render code block with syntax highlighting
 * @param {string} code - Code text
 * @param {string} language - Language identifier
 * @returns {string} HTML string with highlighted code
 */
export function renderCode(code, language)

/**
 * Render LaTeX math formula
 * @param {string} formula - LaTeX formula
 * @param {boolean} displayMode - True for block math, false for inline
 * @returns {string} HTML string with rendered math
 */
export function renderMath(formula, displayMode = false)

/**
 * Render complete message content (markdown + code + math)
 * @param {string} content - Raw message content
 * @returns {string} Fully rendered HTML
 */
export function renderMessage(content)
```

**Imports:**
- `marked` (global) - Markdown parser
- `hljs` (global) - Code highlighting
- `katex` (global) - Math rendering

**Dependencies:** CDN-loaded libraries (marked, hljs, katex)

---

### `frontend/src/components/MessageRenderer.js`

**Purpose:** Render chat messages with all formatting

**Exports:**
```javascript
/**
 * Render a chat message element
 * @param {Object} data - Message data
 * @param {string} data.type - 'user' or 'assistant'
 * @param {string} data.content - Message content
 * @param {Object} data.metadata - Additional metadata
 * @returns {HTMLElement} Rendered message element
 */
export function render(data)

/**
 * Update an existing message element
 * @param {HTMLElement} element - Existing message element
 * @param {Object} data - Updated data
 */
export function update(element, data)

/**
 * Clean up message element
 * @param {HTMLElement} element - Message element to destroy
 */
export function destroy(element)
```

**Imports:**
- `../rendering.js` - For content rendering

---

### `frontend/src/components/ActionList.js`

**Purpose:** Render action/status carousel for research progress

**Exports:**
```javascript
/**
 * Render action list carousel
 * @param {HTMLElement} container - Container element
 * @param {Array} actions - Array of action items
 * @returns {HTMLElement} Action list element
 */
export function render(container, actions)

/**
 * Add action item to carousel
 * @param {HTMLElement} listElement - Action list element
 * @param {Object} action - Action data
 */
export function addAction(listElement, action)

/**
 * Update action item status
 * @param {HTMLElement} listElement - Action list element
 * @param {string} actionId - Action identifier
 * @param {string} status - New status ('processing', 'complete', 'error')
 */
export function updateAction(listElement, actionId, status)

/**
 * Remove action list
 * @param {HTMLElement} listElement - Action list element to destroy
 */
export function destroy(listElement)
```

**Imports:** None (self-contained)

---

### `frontend/src/components/StepCards.js`

**Purpose:** Render research step completion cards

**Exports:**
```javascript
/**
 * Render step cards container
 * @param {HTMLElement} container - Container element
 * @returns {HTMLElement} Step cards container
 */
export function render(container)

/**
 * Add or update step card
 * @param {HTMLElement} container - Step cards container
 * @param {Object} stepData - Step data with type 'step_complete'
 */
export function addStep(container, stepData)

/**
 * Remove step cards
 * @param {HTMLElement} container - Step cards container to destroy
 */
export function destroy(container)
```

**Imports:** None (self-contained)

---

### `frontend/src/components/SettingsPanel.js`

**Purpose:** Settings UI panel

**Exports:**
```javascript
/**
 * Render settings panel
 * @param {HTMLElement} container - Container element
 * @param {Object} currentSettings - Current settings values
 * @returns {HTMLElement} Settings panel element
 */
export function render(container, currentSettings)

/**
 * Update settings values
 * @param {HTMLElement} panelElement - Settings panel element
 * @param {Object} newSettings - Updated settings
 */
export function update(panelElement, newSettings)

/**
 * Get current settings from panel
 * @param {HTMLElement} panelElement - Settings panel element
 * @returns {Object} Current settings values
 */
export function getSettings(panelElement)

/**
 * Remove settings panel
 * @param {HTMLElement} panelElement - Settings panel to destroy
 */
export function destroy(panelElement)
```

**Imports:** None (self-contained)

---

## Migration Strategy

1. **Extract streaming.js** - Move streaming query logic from main.js
2. **Extract rendering.js** - Move markdown/code/math rendering
3. **Extract components one at a time** - Start with MessageRenderer, then ActionList, StepCards, SettingsPanel
4. **Update main.js** - Import modules and use them instead of inline code
5. **Verify** - Test after each extraction to ensure no regressions

---

## File Size Targets

- `main.js`: < 2,000 lines (currently 8,716 lines)
- `streaming.js`: ~200-300 lines
- `rendering.js`: ~300-400 lines
- Each component: ~200-400 lines

---

## Dependencies

- **CDN Libraries:** marked, hljs, katex (loaded globally)
- **ES Modules:** connection.js, config.js
- **No Build Step:** Modules use ES6 imports, loaded directly by browser

---

**This architecture enables incremental refactoring while maintaining functionality.**
