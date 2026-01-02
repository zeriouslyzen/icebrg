import sys
import traceback
import os

# Add project root to python path
sys.path.append(os.getcwd())

try:
    print("Attempting to import src.iceburg.monitoring.finance_controller...")
    from src.iceburg.monitoring import finance_controller
    print("Import successful!")
except Exception:
    traceback.print_exc()
