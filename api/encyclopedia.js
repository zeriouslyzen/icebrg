import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Vercel serverless function proxy for ICEBURG encyclopedia endpoint
export default async function handler(req, res) {
  const BACKEND_URL = process.env.ICEBURG_API_URL || 'http://localhost:8000';
  
  try {
      path.join('/var/task', 'celestial_encyclopedia.json')
    ];
    
    let dataPath = null;
    let triedPaths = [];
    
    for (const testPath of possiblePaths) {
      triedPaths.push(testPath);
      if (fs.existsSync(testPath)) {
        dataPath = testPath;
        console.log('Found encyclopedia data at:', dataPath);
        break;
      }
    }
    
    if (!dataPath) {
      // List what's actually in the directory for debugging
      const cwdContents = fs.existsSync(process.cwd()) 
        ? fs.readdirSync(process.cwd()).join(', ') 
        : 'cwd does not exist';
      const dirContents = fs.existsSync(__dirname) 
        ? fs.readdirSync(__dirname).join(', ') 
        : 'dirname does not exist';
      
      console.error('Encyclopedia data not found. Tried paths:', triedPaths);
      console.error('process.cwd():', process.cwd(), 'Contents:', cwdContents);
      console.error('__dirname:', __dirname, 'Contents:', dirContents);
      
      return res.status(404).json({ 
        error: 'Encyclopedia data not found',
        cwd: process.cwd(),
        tried: triedPaths.map(p => path.relative(process.cwd(), p) || p),
        debug: {
          cwdContents,
          dirContents
        }
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
