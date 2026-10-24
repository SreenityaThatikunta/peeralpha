"""Fetch market data (synthetic regimes or real CSI 300). Thin wrapper over graphalpha.data.

    python scripts/fetch_data.py --synthetic --regime relational --out data/synthetic_relational.npz
    python scripts/fetch_data.py --universe builtin --out data/csi300.npz     # real prices, fixed large-cap list
    python scripts/fetch_data.py --out data/csi300.npz                        # real; live constituents, auto fallback
    python scripts/fetch_data.py --limit 10 --out data/csi_test.npz           # quick connectivity test

Real-data downloads are cached under data/cache/raw and resume if interrupted.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root

from graphalpha.data import main

if __name__ == "__main__":
    main()
