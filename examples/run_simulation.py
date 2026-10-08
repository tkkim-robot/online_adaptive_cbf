#!/usr/bin/env python3
"""Run a navigation scenario and method selected by command-line flags."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oa_cbf_jax.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
