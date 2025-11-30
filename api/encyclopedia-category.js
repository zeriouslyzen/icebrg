const fs = require('fs');
const path = require('path');

module.exports = async (req, res) => {
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
    
    const data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
    
    if (category in data) {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Content-Type', 'application/json');
      return res.status(200).json(data[category]);
    } else {
      return res.status(404).json({ error: `Category '${category}' not found` });
    }
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
};

