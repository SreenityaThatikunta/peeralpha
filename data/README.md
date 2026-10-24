# Data

Market files are not tracked in git. Regenerate them from the repo root:

```bash
# real market: ~100 large-cap NSE (India) stocks, 2011-2024, Yahoo Finance (needs internet + yfinance)
python scripts/fetch_nse.py --out data/nse100.npz
python scripts/fetch_nse.py --limit 8 --out data/nse_test.npz        # quick connectivity test

# synthetic markets (offline, deterministic seed)
python scripts/fetch_data.py --synthetic --out data/synthetic.npz                                   # "mixed" regime
python scripts/fetch_data.py --synthetic --regime relational --out data/synthetic_relational.npz   # graph-favouring regime

# optional: CSI 300 via Baostock
python scripts/fetch_data.py --out data/csi300.npz
```

Downloads are cached per stock under `data/cache/` so an interrupted fetch resumes.
`data/cache/graphs_*.pt` are neighbour-graph caches built automatically on first use.

Each `.npz` holds the price/volume tensors (open, high, low, close, volume, vwap), dates,
tickers and sector ids, in the format produced by `graphalpha.data.assemble`.
