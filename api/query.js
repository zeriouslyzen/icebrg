// Vercel serverless function for ICEBURG query endpoint
// Note: Full ICEBURG functionality requires the full FastAPI backend
// This is a simplified version for Vercel deployment

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { query, mode } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    // For Vercel deployment, you'll need to:
    // 1. Use an external LLM API (OpenAI, Anthropic, etc.)
    // 2. Or connect to your ICEBURG backend via API
    // 3. Or use Vercel Edge Functions for streaming

    return res.status(200).json({
      response: 'ICEBURG query endpoint - connect to your backend API for full functionality',
      query,
      mode: mode || 'chat',
      note: 'Full ICEBURG requires the FastAPI backend. This is a placeholder endpoint.'
    });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}

