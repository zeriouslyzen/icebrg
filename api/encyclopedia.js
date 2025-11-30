const fs = require('fs');
const path = require('path');

module.exports = async (req, res) => {
  try {
    // Load encyclopedia data
    const dataPath = path.join(process.cwd(), 'data', 'celestial_encyclopedia.json');
    
    if (!fs.existsSync(dataPath)) {
      return res.status(404).json({ error: 'Encyclopedia data not found' });
    }
    
    const data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
    
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');
    return res.status(200).json(data);
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
};

