"""Export the data the website's formula lab needs.

    python web/build.py --data data/nifty200.npz

Writes web/assets/market.json (dates, tickers, sectors, sector graph) and web/assets/market.bin
(float32 arrays, days x stocks: close, close/vwap - 1, high/low - 1). The browser derives every
other feature from these, exactly as peeralpha.market.Market does.

The sector graph is built with peeralpha itself on 2012-2015 returns (the first walk-forward
window's training period), so it is causal for every test period on the site (2018 onwards).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from peeralpha.market import Market  # noqa: E402

START, END = "2015-07-01", "2024-07-31"     # warm-up for 250+120 day windows, plus 5 days of target
TRAIN = ["2012-01-01", "2015-12-31"]
TESTS = [["2018-01-01", "2019-12-31"], ["2020-01-01", "2021-12-31"], ["2022-01-01", "2024-06-30"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "nifty200.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "assets"))
    args = ap.parse_args()

    m = Market(args.data, horizon=5, splits={"train": TRAIN}, graph_names=["sector"],
               graph_k=8, device="cpu", gop_zscore=True, graph_residual=True)
    g = m.graph("sector")
    idx, mask = g.idx[0].numpy(), g.mask[0].numpy()
    nbrs = [[int(j) for j, ok in zip(idx[i], mask[i]) if ok] for i in range(m.N)]

    z = np.load(args.data, allow_pickle=False)
    dates = z["dates"].astype("datetime64[D]")
    lo = int(np.searchsorted(dates, np.datetime64(START)))
    hi = int(np.searchsorted(dates, np.datetime64(END), side="right"))
    close = z["close"][lo:hi].astype("<f4")
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap, high, low = (z[k][lo:hi].astype(np.float64) for k in ("vwap", "high", "low"))
        cvwap = np.where(vwap > 0, close / vwap - 1.0, np.nan)
        rng = np.where(low > 0, high / low - 1.0, np.nan)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "market.bin"), "wb") as f:
        f.write(close.tobytes())
        f.write(cvwap.astype("<f4").tobytes())
        f.write(rng.astype("<f4").tobytes())
    meta = {
        "universe": "Nifty 200 (145 stocks with full 2011-2024 history)",
        "T": int(hi - lo), "N": int(m.N), "horizon": 5,
        "dates": [str(d) for d in dates[lo:hi]],
        "codes": m.codes,
        "sectors": m.sector_names,
        "sector_ids": [int(s) for s in m.sector_ids],
        "graph_train": TRAIN,
        "sector_nbrs": nbrs,
        "tests": TESTS,
    }
    with open(os.path.join(args.out, "market.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    print(f"wrote {meta['T']} days x {meta['N']} stocks to {args.out}")


if __name__ == "__main__":
    main()
