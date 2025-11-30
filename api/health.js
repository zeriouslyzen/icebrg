// Health check endpoint for Vercel
export default async function handler(req, res) {
  return res.status(200).json({
    status: 'ok',
    service: 'ICEBURG',
    version: '2.0.0',
    timestamp: new Date().toISOString()
  });
}

