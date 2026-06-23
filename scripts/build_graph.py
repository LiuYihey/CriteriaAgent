#!/usr/bin/env python3
"""Build trial knowledge graph from a four-domain trial_profile JSON."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trial_graph.build import main  # noqa: E402

if __name__ == "__main__":
    main()
