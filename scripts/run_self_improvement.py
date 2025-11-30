#!/usr/bin/env python3
"""
ICEBURG Self-Improvement Runner
Quick script to run self-improvement agents
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.self_improvement_activation import main

if __name__ == "__main__":
    main()

