#!/usr/bin/env python3
"""
Run ICEBURG API Server
"""

import uvicorn
from .server import app

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           ICEBURG 2.0 - Truth-Finding Civilization          ║
╚══════════════════════════════════════════════════════════════╝

🌐 API Server starting...
📡 WebSocket support enabled
🔗 API available at: http://localhost:8000
📚 API docs at: http://localhost:8000/docs

Press Ctrl+C to stop the server.
""")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )

