
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

print("Verifying imports...")

try:
    print("Importing factory...")
    from iceburg.providers import factory
    print("Factory imported successfully.")
except Exception as e:
    print(f"FAILED to import factory: {e}")
    sys.exit(1)

try:
    print("Importing finance_controller...")
    from iceburg.monitoring import finance_controller
    print("Finance controller imported successfully.")
except Exception as e:
    print(f"FAILED to import finance_controller: {e}")
    sys.exit(1)

try:
    print("Importing curiosity_engine...")
    from iceburg.curiosity import curiosity_engine
    # Test instantiation (mocking imports if needed or just checking the class exists)
    # The fix was in __init__ call inside the module, wait, the fix was in _get_memory_persistence
    # which is called during init if enable_persistence is True.
    # We won't simulate full init as it needs DB, but import should pass.
    print("Curiosity engine imported successfully.")
except Exception as e:
    print(f"FAILED to import curiosity_engine: {e}")
    sys.exit(1)

try:
    print("Importing trading cli...")
    from iceburg.trading import cli
    print("Trading CLI imported successfully.")
except Exception as e:
    print(f"FAILED to import trading cli: {e}")
    sys.exit(1)

print("ALL VERIFICATIONS PASSED")
