import json
from pathlib import Path

def handler(request):
    """Vercel serverless function for encyclopedia API"""
    try:
        # Load encyclopedia data
        data_dir = Path(__file__).parent.parent / "data"
        encyclopedia_path = data_dir / "celestial_encyclopedia.json"
        
        if not encyclopedia_path.exists():
            return {
                "statusCode": 404,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Encyclopedia data not found"})
            }
        
        with open(encyclopedia_path, 'r', encoding='utf-8') as f:
            encyclopedia_data = json.load(f)
        
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(encyclopedia_data)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }

