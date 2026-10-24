"""Collect every finished experiment into one clean, browsable summary (no retraining, no re-eval).

    python scripts/summarize.py                          # -> reports/SUMMARY.md + reports/all_runs.csv
    python scripts/summarize.py --tags real1m            # only some experiment tags
    python scripts/summarize.py --runs runs --out reports

Reads ``runs/*/results.json`` (written by evaluate.py) and ``config.json``.
Runs are named ``<tag>_<method>_s<seed>``; the tag identifies an experiment (data + budget), the
method is a ``configs/<method>.yaml``. Older untagged runs are grouped under ``legacy``.
Each experiment gets: an aggregate table (mean ± std over seeds), a per-seed table and the best formulas.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import OrderedDict, defaultdict
from typing import Dict, List

import numpy as np

COLS = [("valid_ic", "Valid IC"), ("test_ic", "Test IC"), ("test_rank_ic", "Test RankIC"),
        ("test_icir", "Test ICIR"), ("bt_sharpe", "Sharpe"), ("bt_annual_return", "AnnRet"),
        ("bt_max_drawdown", "MaxDD"), ("pool_mean_abs_corr", "|corr|"), ("uses_graph", "#graph")]


def methods_available(config_dir: str = "configs") -> List[str]:
    return sorted((os.path.basename(p)[:-5] for p in glob.glob(os.path.join(config_dir, "*.yaml"))),
                  key=len, reverse=True)


def split_name(name: str, methods: List[str]):
    """'real1m_graph_all' -> ('real1m', 'graph_all'); 'graph_all' -> ('legacy', 'graph_all')."""
    for m in methods:
        if name == m:
            return "legacy", m
        if name.endswith("_" + m):
            return name[: -len(m) - 1], m
    return "legacy", name


def load_runs(runs_dir: str) -> List[Dict]:
    methods, out = methods_available(), []
    for rj in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        d = os.path.dirname(rj)
        try:
            r = json.load(open(rj))
            cfg = json.load(open(os.path.join(d, "config.json")))
        except Exception:
            continue
        tag, method = split_name(r["name"], methods)
        r.update(tag=tag, method=method, run=os.path.basename(d), data=os.path.basename(cfg.get("data_path", "?")),
                 steps=cfg.get("total_timesteps"), pool_capacity=cfg.get("pool_capacity"),
                 weighting=cfg.get("weighting"))
        out.append(r)
    return out


def ms(vals: List[float]) -> str:
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0 or np.all(np.isnan(v)):
        return "n/a"
    return f"{np.nanmean(v):+.4f} ± {np.nanstd(v):.4f}" if len(v) > 1 else f"{v[0]:+.4f}"


def table(header: List[str], rows: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + \
           ["| " + " | ".join(r) + " |" for r in rows]


def build_summary(runs: List[Dict]) -> str:
    by_tag: "OrderedDict[str, List[Dict]]" = OrderedDict()
    for r in sorted(runs, key=lambda r: (r["tag"] == "legacy", r["tag"], r["method"], r["seed"])):
        by_tag.setdefault(r["tag"], []).append(r)

    L = ["# PeerAlpha — experiment summary", "",
         "Test metrics use the pool snapshot with the best *validation* IC. `±` = std over seeds.", "",
         "## Experiments", ""]
    idx = []
    for tag, rs in by_tag.items():
        idx.append([f"[{tag}](#{tag.lower()})", ", ".join(sorted({r['data'] for r in rs})),
                    ", ".join(sorted({str(r['steps']) for r in rs})), str(len({r['method'] for r in rs})),
                    str(len(rs))])
    L += table(["Experiment", "Data", "Steps", "Methods", "Runs"], idx) + [""]

    for tag, rs in by_tag.items():
        L += [f"## {tag}", ""]
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for r in rs:
            groups[r["method"]].append(r)
        # rank methods by mean test IC
        order = sorted(groups, key=lambda m: -np.nanmean([r["test_ic"] for r in groups[m]]))
        L += ["### Aggregate (mean ± std over seeds, sorted by Test IC)", ""]
        L += table(["Method", "seeds"] + [c[1] for c in COLS],
                   [[m, str(len(groups[m]))] + [ms([r.get(c[0]) for r in groups[m]]) for c in COLS] for m in order])
        L += ["", "<details><summary>Per-seed results</summary>", ""]
        L += table(["Method", "seed"] + [c[1] for c in COLS],
                   [[m, str(r["seed"])] + [f"{r.get(c[0], float('nan')):+.4f}" for c in COLS]
                    for m in order for r in sorted(groups[m], key=lambda r: r["seed"])])
        L += ["", "</details>", "", "### Best formulas (best validation seed per method)", ""]
        for m in order:
            best = max(groups[m], key=lambda r: r["valid_ic"] if r["valid_ic"] == r["valid_ic"] else -9)
            L.append(f"<details><summary><b>{m}</b> — seed {best['seed']}, valid {best['valid_ic']:+.4f}, "
                     f"test {best['test_ic']:+.4f}</summary>\n")
            for f in sorted(best["formulas"], key=lambda f: -abs(f["weight"])):
                L.append(f"- `{f['formula']}` (w={f['weight']:+.3f}, train IC={f['train_ic']:+.4f})")
            L += ["", "</details>", ""]
    return "\n".join(L) + "\n"


def write_csv(runs: List[Dict], path: str) -> None:
    keys = ["tag", "method", "seed", "data", "steps", "pool_capacity", "weighting"] + [c[0] for c in COLS] + ["run"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="only these methods, e.g. sector_only random_nograph")
    args = ap.parse_args()
    runs = load_runs(args.runs)
    if args.tags:
        runs = [r for r in runs if r["tag"] in args.tags]
    if args.methods:
        runs = [r for r in runs if r["method"] in args.methods]
    if not runs:
        raise SystemExit("no evaluated runs found (need runs/*/results.json from evaluate.py)")
    os.makedirs(args.out, exist_ok=True)
    md = build_summary(runs)
    with open(os.path.join(args.out, "SUMMARY.md"), "w") as f:
        f.write(md)
    write_csv(runs, os.path.join(args.out, "all_runs.csv"))
    print(md)
    print(f"-> {args.out}/SUMMARY.md, {args.out}/all_runs.csv")


if __name__ == "__main__":
    main()
