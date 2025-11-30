import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default async function handler(req, res) {
  try {
    const { category } = req.query;
    
    if (!category) {
      return res.status(400).json({ error: 'Category is required' });
    }

    // Load encyclopedia data
    const dataPath = path.join(process.cwd(), 'data', 'celestial_encyclopedia.json');
    
    if (!fs.existsSync(dataPath)) {
      return res.status(404).json({ error: 'Encyclopedia data not found' });
    }
    
    const encyclopediaData = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
    
    // Filter by category
    const categoryData = encyclopediaData.categories?.[category] || 
                        encyclopediaData.entries?.filter(entry => 
                          entry.category === category || 
                          entry.categories?.includes(category)
                        ) || [];
    
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');
    return res.status(200).json({
      category,
      count: Array.isArray(categoryData) ? categoryData.length : Object.keys(categoryData).length,
      data: categoryData
    });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
