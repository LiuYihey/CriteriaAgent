#!/usr/bin/env python3
"""Backward-compatible shim — use scripts/build_graph.py or trial_graph.build."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from trial_graph.build import main  # noqa: E402

if __name__ == "__main__":
    main()
