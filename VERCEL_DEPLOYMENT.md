# Vercel Deployment Guide for ICEBURG

## Overview

This guide explains how to deploy ICEBURG to Vercel. Note that Vercel has limitations for full ICEBURG functionality:

### What Works on Vercel:
- ✅ Frontend UI (static files)
- ✅ Encyclopedia API (serverless function)
- ✅ Basic API endpoints (serverless functions)
- ✅ Static file serving

### What Requires Additional Setup:
- ⚠️ WebSocket connections (requires Edge Functions or external service)
- ⚠️ Full FastAPI backend (needs conversion to serverless functions)
- ⚠️ Real-time streaming (requires Edge Functions or external service)

## Deployment Steps

### Option 1: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd /Users/jackdanger/Desktop/Projects/iceburg
vercel

# Follow prompts to link your project
```

### Option 2: Deploy via GitHub Integration

1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repository: `zeriouslyzen/icebrg`
3. Vercel will auto-detect the configuration
4. Deploy

## Configuration

### Files Included:
- `vercel.json` - Vercel configuration
- `api/encyclopedia.js` - Encyclopedia API endpoint
- `api/query.js` - Query endpoint (placeholder)
- `api/health.js` - Health check endpoint
- `frontend/` - All frontend files
- `data/celestial_encyclopedia.json` - Encyclopedia data

### Environment Variables

If you need to connect to external services, add these in Vercel dashboard:

```bash
# For LLM APIs (if using)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# For backend connection (if using external ICEBURG backend)
ICEBURG_BACKEND_URL=https://your-backend.vercel.app
```

## WebSocket Support

Vercel doesn't support traditional WebSocket connections in serverless functions. Options:

1. **Use Vercel Edge Functions** (recommended for streaming)
2. **Use external WebSocket service** (Ably, Pusher, etc.)
3. **Use Server-Sent Events (SSE)** instead of WebSockets
4. **Deploy backend separately** (Railway, Render, etc.) and connect frontend

## Frontend Configuration

The frontend in `frontend/main.js` currently expects:
- WebSocket at `ws://localhost:8000/ws`
- API at `http://localhost:8000/api`

For Vercel deployment, you'll need to:
1. Update API URLs to use Vercel domain
2. Replace WebSocket with SSE or Edge Functions
3. Or connect to external backend

## Custom Domain

After deployment, you can add a custom domain in Vercel dashboard:
- Settings → Domains
- Add your domain

## Monitoring

Vercel provides:
- Function logs
- Analytics
- Performance monitoring

Check the Vercel dashboard for deployment status and logs.

