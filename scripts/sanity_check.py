"""Sanity check: do hand-written graph formulas carry signal on the market?

    python scripts/sanity_check.py --config configs/graph_all.yaml

If a sector-relative mean-reversion formula has a clearly non-zero IC on the
synthetic market, the operators and graphs work and any weakness is search.
"""
import argparse

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root

from graphalpha.config import Config
from graphalpha.expr import Node, evaluate
from graphalpha.metrics import daily_ic, zscore_rows
from graphalpha.runner import build_market


def feat(n): return Node("feat", name=n)
def win(v): return Node("win", value=v)
def op(name, *a): return Node("op", name=name, args=tuple(a))
def graph(n): return Node("graph", name=n)


def candidates():
    # 5-day return = Delta(close,5)/Ref(close,5): scale-free, unlike a raw price difference
    r5 = op("Div", op("Delta", feat("close"), win(5)), op("Ref", feat("close"), win(5)))
    r10 = op("Div", op("Delta", feat("close"), win(10)), op("Ref", feat("close"), win(10)))
    return {
        "5d return  [no graph]": r5,
        "$ret5  [no graph]": feat("ret5"),
        "NbrRel($ret5, sector)": op("NbrRel", feat("ret5"), graph("sector")),
        "NbrMean($ret, lead)": op("NbrMean", feat("ret"), graph("lead")),
        "NbrRel(5d ret, sector)": op("NbrRel", r5, graph("sector")),
        "NbrRel(10d ret, sector)": op("NbrRel", r10, graph("sector")),
        "NbrRel(5d ret, corr)": op("NbrRel", r5, graph("corr")),
        "NbrRel(5d ret, lead)": op("NbrRel", r5, graph("lead")),
        "NbrMean(5d ret, lead)": op("NbrMean", r5, graph("lead")),
        "NbrMean(5d ret, sector)": op("NbrMean", r5, graph("sector")),
        "[raw price diff] Delta(close,5)": op("Delta", feat("close"), win(5)),
    }


def split_ic(node, mkt, name):
    lo, hi = mkt.ic_slice(name)
    a = evaluate(node, mkt, hi)
    z = zscore_rows(a[lo:hi])
    ic = daily_ic(z, mkt.target[lo:hi])
    ic = ic[torch.isfinite(ic)]
    return float(ic.mean()), float(ic.mean() / (ic.std() + 1e-9))


def lead_precision(mkt):
    """Share of followers whose planted leader is among their lead-graph neighbours (synthetic only)."""
    import numpy as np
    z = np.load(mkt.path)
    if "leader_of" not in z or "lead" not in mkt.graph_names:
        return None
    leader = torch.as_tensor(z["leader_of"], device=mkt.device)
    follower = leader != torch.arange(mkt.N, device=mkt.device)
    g = mkt.graph("lead")
    lo, hi = mkt.ic_slice("test")
    chunks = g.day2chunk[lo:hi].unique()
    hit = (g.idx[chunks] == leader.view(1, -1, 1)) & g.mask[chunks]
    return float(hit.any(-1)[:, follower].float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/graph_all.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. data_path=data/x.npz")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config, args.set)
    print(f"data: {cfg.data_path}")
    mkt = build_market(cfg)
    print(f"{'formula':36s} {'train':>8s} {'valid':>8s} {'test':>8s}   (mean daily IC)")
    for label, node in candidates().items():
        try:
            row = [split_ic(node, mkt, s)[0] for s in ("train", "valid", "test")]
        except Exception as e:  # graph not built in this config, etc.
            print(f"{label:36s} skipped: {e}")
            continue
        print(f"{label:36s} {row[0]:+8.4f} {row[1]:+8.4f} {row[2]:+8.4f}")
    p = lead_precision(mkt)
    if p is not None:
        print(f"lead graph: true leader in top-{cfg.graph_k} for {p:.1%} of followers (test period)")


if __name__ == "__main__":
    main()
