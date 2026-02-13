// Vercel serverless function proxy for ICEBURG health endpoint
export default async function handler(req, res) {
  const BACKEND_URL = process.env.ICEBURG_API_URL || 'http://localhost:8000';
  
  try {
    const response = await fetch(`${BACKEND_URL}/v2/health`);
    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (error) {
    return res.status(500).json({ 
      error: 'Backend unreachable',
      details: error.message,
      target: BACKEND_URL 
    });
  }
}
