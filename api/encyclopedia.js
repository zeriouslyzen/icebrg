import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default async function handler(req, res) {
  try {
    // Try multiple possible paths for the data file
    const possiblePaths = [
      path.join(process.cwd(), 'data', 'celestial_encyclopedia.json'),
      path.join(process.cwd(), '..', 'data', 'celestial_encyclopedia.json'),
      path.join(__dirname, '..', 'data', 'celestial_encyclopedia.json'),
      path.join(process.cwd(), 'celestial_encyclopedia.json')
    ];
    
    let dataPath = null;
    for (const testPath of possiblePaths) {
      if (fs.existsSync(testPath)) {
        dataPath = testPath;
        break;
      }
    }
    
    if (!dataPath) {
      console.error('Encyclopedia data not found. Tried paths:', possiblePaths);
      return res.status(404).json({ 
        error: 'Encyclopedia data not found',
        tried: possiblePaths.map(p => path.relative(process.cwd(), p))
      });
    }
    
    const data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
    
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');
    return res.status(200).json(data);
  } catch (error) {
    console.error('Encyclopedia API error:', error);
    return res.status(500).json({ 
      error: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
}
