const fs = require('fs');
const path = require('path');

module.exports = async (req, res) => {
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
    
    const data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
    const results = [];
    const searchLower = query.toLowerCase();
    
    // Search across all categories
    for (const category in data) {
      if (category === 'metadata') continue;
      const categoryData = data[category];
      if (typeof categoryData !== 'object') continue;
      
      for (const entryId in categoryData) {
        const entry = categoryData[entryId];
        if (typeof entry !== 'object') continue;
        
        const searchableText = JSON.stringify(entry).toLowerCase();
        if (searchableText.includes(searchLower)) {
          results.push({...entry, category, id: entryId});
        }
      }
    }
    
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');
    return res.status(200).json(results);
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
};

