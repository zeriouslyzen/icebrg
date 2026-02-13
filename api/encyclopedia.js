import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Vercel serverless function proxy for ICEBURG encyclopedia endpoint
export default async function handler(req, res) {
  const BACKEND_URL = process.env.ICEBURG_API_URL || 'http://localhost:8000';
  
  try {
    const response = await fetch(`${BACKEND_URL}/api/encyclopedia`);
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    const data = await response.json();
    return res.status(200).json(data);
  } catch (error) {
    console.error('Bridge proxy error:', error);
    return res.status(500).json({ 
      error: 'Backend unreachable',
      details: error.message,
      target: BACKEND_URL 
    });
  }
}
