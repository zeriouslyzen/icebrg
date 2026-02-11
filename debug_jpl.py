
import requests

def debug_jpl():
    url = "https://ssd.jpl.nasa.gov/api/horizons.api"
    params = {
        "format": "json",
        "COMMAND": "599", # Jupiter
        "CENTER": "'500@10'",
        "EPHEM_TYPE": "VECTORS",
        "START_TIME": "'2026-02-11 12:00'",
        "STOP_TIME": "'2026-02-11 12:01'",
        "STEP_SIZE": "'1d'"
    }
    
    print(f"URL: {url}")
    print(f"Params: {params}")
    
    response = requests.get(url, params=params)
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Text (First 500 chars): {response.text[:500]}")
    
    try:
        data = response.json()
        print("JSON parsed successfully!")
    except Exception as e:
        print(f"JSON parsing failed: {e}")

if __name__ == "__main__":
    debug_jpl()
