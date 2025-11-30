# ICEBURG 2.0 Frontend

Mobile-first web application for ICEBURG 2.0 with futuristic design, morphing animations, and real-time streaming capabilities.

## 📁 Index of Files

```
frontend/
├── index.html          # Main HTML structure
├── main.js            # Core application logic (WebSocket, UI, rendering)
├── styles.css         # All styles (mobile-first, animations, themes)
├── vite.config.js     # Vite build configuration
├── package.json       # Dependencies and scripts
├── package-lock.json  # Locked dependency versions
├── .env.example       # Environment variable template
├── .gitignore        # Git ignore rules
├── security.md        # Security documentation
└── README.md         # This file
```

### File Descriptions

- **`index.html`**: Root HTML document with semantic structure, meta tags, and external CDN links (KaTeX, Highlight.js, Chart.js)
- **`main.js`**: Core JavaScript application (~1744 lines)
  - WebSocket connection management
  - Message handling and streaming
  - Markdown/LaTeX/code rendering
  - UI interactions (sidebar, settings, file uploads)
  - Action tracking and document viewer
  - Voice input/output
  - Chart rendering
  - Keyboard shortcuts
- **`styles.css`**: Complete styling system (~2000+ lines)
  - Mobile-first responsive design
  - Dark theme (black background, white text)
  - Morphing animations
  - Action tracking UI
  - Document viewer styles
  - Sidebar, settings panel, toast notifications
- **`vite.config.js`**: Vite configuration for dev server (port 3000) and production builds
- **`package.json`**: Dependency management with exact versions

## 🏗️ Architecture

### Technology Stack

- **Build Tool**: Vite 5.0 (fast HMR, optimized builds)
- **Language**: Vanilla JavaScript (ES6+ modules, no framework)
- **Styling**: Modern CSS (custom properties, animations, Grid/Flexbox)
- **Markdown**: `marked` 17.0
- **Code Highlighting**: `highlight.js` 11.11.1
- **Math Rendering**: `katex` 0.16.25
- **Charts**: Chart.js 4.4.0 (CDN)

### Architecture Layers

```
┌─────────────────────────────────────┐
│         Presentation Layer          │
│  (HTML/CSS - UI Components)         │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│      Application Logic Layer        │
│  (main.js - State, Events, UI)     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│      Communication Layer            │
│  (WebSocket/HTTP - API Integration) │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│         Backend API Server           │
│  (FastAPI - Port 8000)              │
└─────────────────────────────────────┘
```

### Component Structure

1. **WebSocket Manager**: Handles real-time bidirectional communication
2. **Message Handler**: Processes streaming messages (thinking, chunks, actions, done)
3. **Renderer**: Markdown → HTML, LaTeX → MathML, Code → Highlighted HTML
4. **UI Controller**: Sidebar, settings, file uploads, voice, keyboard shortcuts
5. **Action Tracker**: Displays organized list of thoughts, actions, websites browsed
6. **Document Viewer**: Pop-out viewer for PDFs and markdown documents

### Data Flow

```
User Input → WebSocket → Backend Processing → Streaming Response
    ↓                                                              ↓
UI Update ← Message Handler ← WebSocket ← Backend Stream
```

## 🔌 Backend Connection

### Connection Details

- **HTTP API Endpoint**: `http://localhost:8000/api/query` (default)
- **WebSocket Endpoint**: `ws://localhost:8000/ws` (default)
- **Production**: Automatically switches to `https://` and `wss://` when `PROD=true`

### Environment Variables

Create a `.env` file in the `frontend/` directory:

```bash
# API Configuration
VITE_API_URL=http://localhost:8000/api/query
VITE_WS_URL=ws://localhost:8000/ws

# Feature Flags
VITE_ENABLE_WEB_SEARCH=true
VITE_ENABLE_IMAGE_GENERATION=true

# Production (auto-detected)
# PROD=true  # Automatically set during build
```

### WebSocket Message Format

#### Outgoing (Client → Server)

```json
{
  "query": "User's question",
  "mode": "standard|fast|deep",
  "agent": "surveyor|dissident|synthesist|oracle|archaeologist|supervisor|scribe|weaver|scrutineer|ide|auto",
  "degradation_mode": false,
  "settings": {
    "primaryModel": "llama3.1:8b",
    "temperature": 0.7,
    "maxTokens": 2000
  },
  "files": [
    {
      "name": "file.pdf",
      "type": "application/pdf",
      "content": "base64_encoded_content"
    }
  ]
}
```

#### Incoming (Server → Client)

**Thinking Message**:
```json
{
  "type": "thinking",
  "content": "Analyzing query..."
}
```

**Action Message** (Prompt Interpreter, Web Search, etc.):
```json
{
  "type": "action",
  "action": "prompt_interpreter|web_search|reading_document",
  "status": "starting|complete|error",
  "description": "Analyzing query intent...",
  "intent": "general|technical|creative",
  "domain": "general|science|technology",
  "complexity": 0.5,
  "routing": "standard|fast|deep",
  "websites": ["https://example.com"],
  "thoughts": ["Thought 1", "Thought 2"],
  "document": {
    "title": "Document Title",
    "filename": "doc.pdf",
    "type": "application/pdf",
    "url": "data:application/pdf;base64,..."
  }
}
```

**Chunk Message** (Streaming Response):
```json
{
  "type": "chunk",
  "content": "Partial response text..."
}
```

**Informatics Message** (Sources):
```json
{
  "type": "informatics",
  "sources": ["Source 1", "Source 2"],
  "confidence": 0.95
}
```

**Conclusion Message**:
```json
{
  "type": "conclusion",
  "conclusions": ["Conclusion 1", "Conclusion 2"]
}
```

**Done Message**:
```json
{
  "type": "done"
}
```

**Error Message**:
```json
{
  "type": "error",
  "message": "Error description"
}
```

### CORS Configuration

The backend must allow the frontend origin. Default allowed origins:
- `http://localhost:3000` (Vite dev server)
- `http://localhost:5173` (Vite default port)
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`

Set `ALLOWED_ORIGINS` environment variable on the backend to customize.

## ⚙️ How It Works

### 1. Initialization

- Loads environment variables
- Initializes WebSocket connection
- Sets up event listeners (keyboard shortcuts, file uploads, voice)
- Loads conversation history from `localStorage`
- Renders initial UI state

### 2. Query Processing Flow

1. **User Input**: User types query and clicks send (or presses Enter)
2. **Validation**: Checks query length, file size limits
3. **WebSocket Connection**: Establishes or reuses WebSocket connection
4. **Message Send**: Sends query with mode, agent, settings, files
5. **Streaming Response**: Receives and processes streaming messages:
   - **Action Messages**: Displayed in action tracking list (clickable, expandable)
   - **Thinking Messages**: Shown as thinking indicators
   - **Chunk Messages**: Appended to message content (markdown rendered)
   - **Informatics Messages**: Displayed as sources
   - **Conclusion Messages**: Shown as conclusions
   - **Done Message**: Re-enables input, scrolls to bottom
   - **Error Messages**: Displayed as toast notifications
6. **Rendering**: Markdown → HTML, LaTeX → MathML, Code → Highlighted
7. **UI Update**: Scrolls to bottom, updates conversation history

### 3. Action Tracking

- **Prompt Interpreter**: Analyzes query intent, domain, complexity, routing
- **Web Search**: Tracks websites browsed (if enabled)
- **Document Reading**: Shows documents being read
- **Agent Actions**: Displays agent-specific thinking and actions

Each action item is:
- **Clickable**: Expands to show details (intent, domain, complexity, websites, thoughts, documents)
- **Status Indicators**: Shows "starting", "complete", or "error" status
- **Document Viewer**: Click "View Document" to open pop-out viewer

### 4. Document Viewer

- **Black Background, White Text**: Futuristic aesthetic
- **PDF Support**: Renders via `<iframe>`
- **Markdown Support**: Renders markdown with code highlighting
- **Close Options**: Escape key, close button, backdrop click

### 5. Conversation Management

- **localStorage**: Stores conversations in browser
- **Search**: Filter conversations by title/content
- **Delete**: Remove conversations
- **New Chat**: Start fresh conversation

### 6. Settings Panel

- **Model Selection**: Choose primary model (llama3.1:8b, mistral:7b, etc.)
- **Temperature**: Adjust creativity (0.0-1.0)
- **Max Tokens**: Set response length limit
- **Fast Mode**: Enable fast mode (lightweight models, reduced pipeline)
- **Degradation Mode**: Enable slow degradation for agent communication

### 7. File Uploads

- **Supported Types**: Images, PDFs, text files, code files, documents (.doc, .docx)
- **Size Limit**: 10MB per file
- **Processing**: Files are base64-encoded and sent to backend
- **Display**: Attached files shown in input area

### 8. Voice Input/Output

- **Input**: Web Speech API (SpeechRecognition)
- **Output**: Web Speech API (SpeechSynthesis)
- **Browser Support**: Chrome, Edge, Safari (partial)

### 9. Keyboard Shortcuts

- **Enter**: Send message (Shift+Enter for new line)
- **Escape**: Close modals, clear input
- **Ctrl/Cmd + K**: Toggle sidebar
- **Ctrl/Cmd + /**: Show keyboard shortcuts
- **Ctrl/Cmd + ,**: Open settings

### 10. Rendering Pipeline

```
Raw Text → Marked (Markdown) → HTML
                ↓
         KaTeX (LaTeX) → MathML
                ↓
         Highlight.js (Code) → Highlighted HTML
                ↓
         Final Rendered HTML
```

## 📦 Dependency Freezing

### Current Dependencies

**Production Dependencies** (in `package.json`):
```json
{
  "highlight.js": "^11.11.1",
  "katex": "^0.16.25",
  "marked": "^17.0.0",
  "vite": "^5.0.0"
}
```

**Development Dependencies**:
```json
{
  "eslint": "^8.55.0",
  "terser": "^5.44.1"
}
```

### Freezing Dependencies

To freeze dependencies to exact versions:

```bash
cd frontend
npm install --save-exact
```

This updates `package.json` to use exact versions (removes `^`).

### Lock File

`package-lock.json` already locks all transitive dependencies. Commit this file to ensure reproducible builds.

### Production Build

```bash
npm run build
```

This creates a `dist/` directory with:
- Minified JavaScript (Terser)
- Optimized CSS
- Static assets
- `index.html` with asset references

## 🚀 Easy Startup Script

### Quick Start Script

Create `start.sh` in the `frontend/` directory:

```bash
#!/bin/bash

# ICEBURG Frontend Startup Script

echo "🚀 Starting ICEBURG Frontend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Navigate to frontend directory
cd "$(dirname "$0")"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please configure it if needed."
    else
        echo "⚠️  .env.example not found. Using defaults."
    fi
fi

# Check if backend is running (optional check)
echo "🔍 Checking backend connection..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is running on port 8000"
else
    echo "⚠️  Backend not detected on port 8000. Make sure the API server is running."
    echo "   Start it with: cd ../.. && python -m uvicorn src.iceburg.api.server:app --host 0.0.0.0 --port 8000 --reload"
fi

# Start Vite dev server
echo "🌐 Starting Vite dev server on http://localhost:3000..."
npm run dev
```

Make it executable:

```bash
chmod +x frontend/start.sh
```

### Usage

```bash
cd frontend
./start.sh
```

Or from project root:

```bash
./frontend/start.sh
```

### Alternative: npm Scripts

Already available in `package.json`:

```bash
# Development (with HMR)
npm run dev

# Production build
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## 🔧 Development

### Prerequisites

- **Node.js**: 18.0.0 or higher
- **npm**: 9.0.0 or higher (comes with Node.js)
- **Backend API**: Running on port 8000 (see backend README)

### Setup

```bash
cd frontend
npm install
cp .env.example .env  # Optional: configure environment variables
npm run dev
```

Opens at `http://localhost:3000`

### Hot Module Replacement (HMR)

Vite provides instant HMR. Changes to `main.js` or `styles.css` are reflected immediately without full page reload.

### Debugging

- **Browser DevTools**: Open DevTools (F12) to see WebSocket messages, console logs, network requests
- **Vite DevTools**: Vite provides detailed build information in terminal
- **WebSocket Inspector**: Use browser DevTools → Network → WS to inspect WebSocket messages

## 🏗️ Build & Deploy

### Development Build

```bash
npm run build
```

Outputs to `dist/` directory.

### Production Deployment

1. **Build**:
   ```bash
   npm run build
   ```

2. **Serve** (using a static file server):
   ```bash
   npm run preview  # Vite preview server
   ```

   Or use any static file server:
   - **Nginx**: Serve `dist/` directory
   - **Apache**: Serve `dist/` directory
   - **Vercel/Netlify**: Deploy `dist/` directory
   - **Docker**: Copy `dist/` to nginx container

3. **Environment Variables**: Set production environment variables:
   ```bash
   VITE_API_URL=https://api.iceburg.ai
   VITE_WS_URL=wss://api.iceburg.ai
   PROD=true
   ```

### Docker Deployment (Optional)

Create `Dockerfile`:

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 🔒 Security

See `security.md` for detailed security documentation.

Key security features:
- **Input Validation**: Client-side and server-side validation
- **XSS Prevention**: Sanitized markdown rendering
- **CORS**: Configured for allowed origins
- **HTTPS/WSS**: Automatic in production
- **File Upload Limits**: 10MB per file, type validation

## 📱 Browser Support

- **Chrome/Edge**: Full support (latest 2 versions)
- **Firefox**: Full support (latest 2 versions)
- **Safari**: Full support (latest 2 versions)
- **Mobile Browsers**: Optimized for iOS Safari, Chrome Mobile

## 🐛 Troubleshooting

### WebSocket Connection Failed

- **Check Backend**: Ensure API server is running on port 8000
- **Check CORS**: Verify `ALLOWED_ORIGINS` includes frontend URL
- **Check Firewall**: Ensure ports 8000 (backend) and 3000 (frontend) are open

### Dependencies Not Installing

```bash
rm -rf node_modules package-lock.json
npm install
```

### Build Fails

```bash
npm run build -- --debug  # See detailed error messages
```

### Port Already in Use

Change port in `vite.config.js`:

```javascript
server: {
  port: 3001,  // Change to available port
}
```

## 📚 Additional Resources

- **Backend README**: See main project README for backend documentation
- **API Documentation**: `http://localhost:8000/docs` (Swagger UI when backend is running)
- **Vite Documentation**: https://vitejs.dev/
- **Marked Documentation**: https://marked.js.org/
- **KaTeX Documentation**: https://katex.org/
- **Highlight.js Documentation**: https://highlightjs.org/

## 📝 License

See main project LICENSE file.

---

**Version**: 2.0.0  
**Last Updated**: 2024  
**Maintainer**: ICEBURG Development Team
