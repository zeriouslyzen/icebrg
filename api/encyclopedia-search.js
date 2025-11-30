import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default async function handler(req, res) {
  try {
    const { query } = req.query;
    
    if (!query || query.length < 2) {
      return res.status(400).json({ error: 'Query must be at least 2 characters' });
    }

    // Load encyclopedia data
    const dataPath = path.join(process.cwd(), 'data', 'celestial_encyclopedia.json');
    
    if (!fs.existsSync(dataPath)) {
      return res.status(404).json({ error: 'Encyclopedia data not found' });
    }
    
    const encyclopediaData = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
    
    // Simple search - filter entries by query string
    const searchLower = query.toLowerCase();
    const entries = encyclopediaData.entries || [];
    const results = entries.filter(entry => {
      const name = (entry.name || '').toLowerCase();
      const description = (entry.description || '').toLowerCase();
      const category = (entry.category || '').toLowerCase();
      return name.includes(searchLower) || 
             description.includes(searchLower) || 
             category.includes(searchLower);
    });
    
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');
    return res.status(200).json({
      query,
      count: results.length,
      results: results.slice(0, 50) // Limit to 50 results
    });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
