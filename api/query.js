// Vercel serverless function proxy for ICEBURG query endpoint
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const BACKEND_URL = process.env.ICEBURG_API_URL || 'http://localhost:8000';
  
  try {
    const response = await fetch(`${BACKEND_URL}/v2/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (error) {
    console.error('Bridge proxy error:', error);
    return res.status(500).json({ 
      error: 'Backend unreachable',
      details: error.message,
      target: BACKEND_URL 
    });
  }
}
