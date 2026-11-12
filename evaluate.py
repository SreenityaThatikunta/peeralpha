"""Evaluate trained runs on validation/test, aggregate seeds, and make the report figures.

    python evaluate.py runs/graph_all_s0 runs/graph_all_s1 runs/nograph_s0 ... --out reports

Runs are grouped by their config ``name``; metrics are averaged over seeds.
Each run is scored twice: the final pool (main result) and the snapshot with the best validation
IC (one year of validation is too noisy to pick on, so that one is secondary). The test set is
touched only here. ``--hand`` adds a fixed, unfitted sector baseline row and the report also
has a seed-ensemble table (the seeds' test signals averaged per method and window).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np
import torch

from peeralpha.backtest import longshort_backtest, topk_backtest
from peeralpha.config import Config
from peeralpha.expr import evaluate
from peeralpha.grammar import build_from_keys
from peeralpha.metrics import daily_ic, nanmean, newey_west_t, summarize, zscore_rows
from peeralpha.ops import FEATURES
from peeralpha.plotting import (plot_ablation, plot_cumulative, plot_pool_corr,
                                 plot_training_curves)
from peeralpha.pool import AlphaPool
from peeralpha.runner import build_market
from peeralpha.vocab import Vocab


SNAPSHOTS = {"final": "pool_latest.json", "best_valid": "pool_best_valid.json"}

# hand-made reference: sector momentum minus within-sector 5-day reversal (no fitting, signs by prior)
HAND_SECTOR = [(+1.0, ["feat:ret20", "graph:sector", "NbrMean"]), (-1.0, ["feat:ret5", "graph:sector", "NbrRel"])]
# the same plus 12-1 month own momentum: the fair bar once the long-horizon features are in the vocabulary
HAND_BASELINES = {"hand_sector": HAND_SECTOR, "hand_sector_mom": HAND_SECTOR + [(+1.0, ["feat:mom12_1"])]}


def _market_key(cfg: Config) -> Tuple:
    return (cfg.data_path, tuple(cfg.graphs), cfg.graph_k, cfg.graph_window, cfg.graph_step,
            tuple(cfg.train), tuple(cfg.valid), tuple(cfg.test), cfg.horizon,
            cfg.gop_zscore, cfg.graph_residual, cfg.min_sector_size)


def _market(cfg: Config, cache: Dict):
    key = _market_key(cfg)
    if key not in cache:
        cache[key] = build_market(cfg)
    return cache[key]


def _score_signal(sig: torch.Tensor, market, horizon: int) -> Dict:
    """Test metrics of a ``[hi, N]`` signal: IC stats, Newey-West t and both backtests."""
    lo, hi = market.ic_slice("test")
    ic = daily_ic(sig[lo:hi], market.target[lo:hi])
    s = summarize(ic)
    res = {"test_ic": s["ic"], "test_icir": s["icir"], "test_nw_t": newey_west_t(ic, horizon)}
    bt = topk_backtest(sig, market.ret, lo, hi, k=30, rebalance=horizon)
    for k, v in bt.items():
        if k not in ("port", "bench"):
            res[f"bt_{k}"] = v
    ls = longshort_backtest(sig, market.ret, lo, hi, k=max(5, market.N // 5), rebalance=horizon)
    for k in ("sharpe", "annual_return", "max_drawdown"):
        res[f"ls_{k}"] = ls[k]
    res["_port"], res["_bench"] = bt["port"], bt["bench"]
    return res


def evaluate_run(run_dir: str, market_cache: Dict, snapshot: str = "final") -> Dict:
    cfg = Config.from_json(os.path.join(run_dir, "config.json"))
    market = _market(cfg, market_cache)
    vocab = Vocab.from_config(cfg)
    pool = AlphaPool.load(market, cfg, vocab, os.path.join(run_dir, SNAPSHOTS[snapshot]))

    res = {"name": cfg.name, "seed": cfg.seed, "snapshot": snapshot, "pool_size": pool.size, "train_obj": pool.obj}
    for split in ("train", "valid", "test"):
        m = pool.evaluate_split(split)
        res[f"{split}_ic"], res[f"{split}_rank_ic"], res[f"{split}_icir"] = m["ic"], m["rank_ic"], m["icir"]
    if not pool.members:
        res.update({"test_nw_t": float("nan"), "pool_mean_abs_corr": float("nan"), "uses_graph": 0, "formulas": [],
                    "_sig": None, "_port": None, "_bench": None, "_corr": np.zeros((0, 0)), "_key": _market_key(cfg)})
        return res

    lo, hi = market.ic_slice("test")
    sig = pool.signal("test")
    res["test_nw_t"] = newey_west_t(daily_ic(sig[lo:hi], market.target[lo:hi]), cfg.horizon)
    bt = topk_backtest(sig, market.ret, lo, hi, k=30, rebalance=cfg.horizon)
    for k, v in bt.items():
        if k not in ("port", "bench"):
            res[f"bt_{k}"] = v
    ls = longshort_backtest(sig, market.ret, lo, hi, k=max(5, market.N // 5), rebalance=cfg.horizon)
    for k in ("sharpe", "annual_return", "max_drawdown"):
        res[f"ls_{k}"] = ls[k]

    # members for the gated seed union: sign-aligned full-length signal, valid IC, fixed flag
    full_sigs = pool.member_signals("test")
    vlo, vhi = market.ic_slice("valid")
    members = []
    for i, (m, s) in enumerate(zip(pool.members, full_sigs)):
        a = s * (1.0 if m.ic >= 0 else -1.0)
        members.append((str(m.node), i < pool.n_fixed, a, nanmean(daily_ic(a[vlo:vhi], market.target[vlo:vhi]))))

    # diversity: mean |pairwise corr| of the pool alphas on test
    sigs = [s[lo:hi] for s in full_sigs]
    n = len(sigs)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = nanmean(daily_ic(sigs[i], sigs[j]))
            corr[i, j] = corr[j, i] = 0.0 if c != c else c
    res["pool_mean_abs_corr"] = float(np.abs(corr[np.triu_indices(n, 1)]).mean()) if n > 1 else float("nan")
    res["uses_graph"] = int(sum(m.node.uses_graph() for m in pool.members))
    res["formulas"] = [{"formula": str(m.node), "weight": float(w), "train_ic": m.ic}
                       for m, w in zip(pool.members, pool.weights())]

    out = "results.json" if snapshot == "final" else f"results_{snapshot}.json"
    with open(os.path.join(run_dir, out), "w") as f:
        json.dump({k: v for k, v in res.items()}, f, indent=2)
    res["_port"], res["_bench"], res["_corr"] = bt["port"], bt["bench"], corr
    res["_members"] = members
    res["_sig"], res["_key"] = sig, _market_key(cfg)
    return res


def hand_baseline(results: List[Dict], cache: Dict, name: str = "hand_sector") -> List[Dict]:
    """Score ``HAND_BASELINES[name]`` once per window, replicated for every seed so paired tables line up."""
    terms = HAND_BASELINES[name]
    out, done = [], {}
    for r in results:
        cfg = Config.from_json(os.path.join(r["_dir"], "config.json"))
        hcfg = replace(cfg, graphs=["sector"], gop_zscore=True, graph_residual=True)   # fixed peers for every method
        key = _market_key(hcfg)
        if key not in done:
            market = _market(hcfg, cache)
            vocab = Vocab(["sector"], FEATURES)
            full = sum(sgn * torch.nan_to_num(zscore_rows(evaluate(build_from_keys(vocab, keys), market, market.T)),
                                              nan=0.0) for sgn, keys in terms)
            _, hi = market.ic_slice("test")
            sig = full[:hi]
            m = _score_signal(sig, market, hcfg.horizon)
            for split in ("train", "valid"):
                a, b = market.ic_slice(split)
                m[f"{split}_ic"] = nanmean(daily_ic(full[a:b], market.target[a:b]))
            m.update({"test_rank_ic": float("nan"), "pool_mean_abs_corr": float("nan"), "uses_graph": 2,
                      "pool_size": len(terms), "formulas": [], "_corr": np.zeros((0, 0)), "_sig": sig})
            done[key] = m
        seeds = {(x["seed"], x["window"]) for x in out if x["window"] == r["window"]}
        if (r["seed"], r["window"]) in seeds:
            continue
        row = dict(done[key])
        row.update({"name": name, "method": name, "seed": r["seed"], "window": r["window"],
                    "snapshot": r["snapshot"], "_key": key})
        out.append(row)
    return out


def seed_ensemble_table(groups: "OrderedDict[str, List[Dict]]", cache: Dict) -> str:
    """Average the z-scored test signals of all seeds of a method within each window."""
    windows = sorted({r["window"] for rs in groups.values() for r in rs})
    lines = ["Seed ensemble: test IC of the seeds' averaged signals (single-seed mean in brackets)", "",
             "| Method | " + " | ".join(w or "all" for w in windows) + " | mean |",
             "|---|" + "|".join("---" for _ in windows) + "|---|"]
    for name, rs in groups.items():
        cells, vals = [], []
        for w in windows:
            sub = [r for r in rs if r["window"] == w and r.get("_sig") is not None]
            if not sub:
                cells.append("")
                continue
            market = cache[sub[0]["_key"]]
            lo, hi = market.ic_slice("test")
            sig = torch.stack([torch.nan_to_num(zscore_rows(r["_sig"][:hi]), nan=0.0) for r in sub]).mean(0)
            v = nanmean(daily_ic(sig[lo:hi], market.target[lo:hi]))
            vals.append(v)
            cells.append(f"{v:+.4f} ({np.mean([r['test_ic'] for r in sub]):+.4f})")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {np.mean(vals):+.4f} |" if vals else f"| {name} |")
    return "\n".join(lines)


def union_gate_table(groups: "OrderedDict[str, List[Dict]]", cache: Dict) -> str:
    """Seed union: every member of every seed's pool (fixed prior members once) whose sign-aligned valid
    IC is > 0, equally weighted. Uses only train (signs) and valid (gate), never test."""
    windows = sorted({r["window"] for rs in groups.values() for r in rs})
    lines = ["Gated seed union: equal-weight union of all seeds' members with valid IC > 0 (members kept)", "",
             "| Method | " + " | ".join(w or "all" for w in windows) + " | mean |",
             "|---|" + "|".join("---" for _ in windows) + "|---|"]
    for name, rs in groups.items():
        cells, vals = [], []
        for w in windows:
            sub = [r for r in rs if r["window"] == w and r.get("_members")]
            if not sub:
                cells.append("")
                continue
            market = cache[sub[0]["_key"]]
            lo, hi = market.ic_slice("test")
            seen, keep = set(), []
            for r in sub:
                for f, fixed, a, v in r["_members"]:
                    if f in seen and fixed:
                        continue
                    seen.add(f)
                    if fixed or v > 0:
                        keep.append(a[:hi])
            if not keep:
                cells.append("")
                continue
            sig = torch.nan_to_num(zscore_rows(torch.stack(keep).sum(0)), nan=0.0)
            v = nanmean(daily_ic(sig[lo:hi], market.target[lo:hi]))
            vals.append(v)
            cells.append(f"{v:+.4f} ({len(keep)})")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {np.mean(vals):+.4f} |" if vals else f"| {name} |")
    return "\n".join(lines)


def _window_and_method(name: str):
    """``v2nse_w1_v2_graph_all`` -> (``w1``, ``v2_graph_all``); names without ``_w<k>_`` keep the tag."""
    parts = name.split("_")
    for i, p in enumerate(parts):
        if len(p) > 1 and p[0] == "w" and p[1:].isdigit():
            return p, "_".join(parts[i + 1:])
    return "", name


def _ms(vals: List[float]) -> str:
    v = np.array(vals, dtype=float)
    return f"{np.nanmean(v):+.4f} ± {np.nanstd(v):.4f}" if len(v) > 1 else f"{v[0]:+.4f}"


def paired_table(groups: "OrderedDict[str, List[Dict]]", baseline: str, metric: str = "test_ic") -> str:
    """Per-seed difference of each method to ``baseline`` (same seed / window), mean ± s.e. and t."""
    base = {(r["seed"], r.get("window", "")): r[metric] for r in groups[baseline]}
    lines = [f"Paired {metric} difference vs `{baseline}` (matched seed and window)", "",
             "| Method | pairs | mean diff | s.e. | t | wins |", "|---|---|---|---|---|---|"]
    for name, rs in groups.items():
        if name == baseline:
            continue
        d = np.array([r[metric] - base[(r["seed"], r.get("window", ""))] for r in rs
                      if (r["seed"], r.get("window", "")) in base], dtype=float)
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        t = d.mean() / se if se == se and se > 0 else float("nan")
        lines.append(f"| {name} | {len(d)} | {d.mean():+.4f} | {se:.4f} | {t:+.2f} | {int((d > 0).sum())}/{len(d)} |")
    return "\n".join(lines)


def _report(results: List[Dict], key: str, baseline: str, cache: Dict) -> Tuple[str, "OrderedDict[str, List[Dict]]"]:
    groups: "OrderedDict[str, List[Dict]]" = OrderedDict()
    for r in results:
        groups.setdefault(r[key], []).append(r)
    cols = [("valid_ic", "Valid IC"), ("test_ic", "Test IC"), ("test_nw_t", "NW t"), ("test_rank_ic", "Test RankIC"),
            ("test_icir", "Test ICIR"), ("bt_sharpe", "Sharpe"), ("bt_annual_return", "AnnRet"),
            ("bt_max_drawdown", "MaxDD"), ("ls_sharpe", "LS Sharpe"), ("ls_annual_return", "LS AnnRet"),
            ("pool_mean_abs_corr", "|corr|"), ("uses_graph", "#graph")]
    lines = ["| Method | seeds | " + " | ".join(c[1] for c in cols) + " |",
             "|---|---|" + "|".join("---" for _ in cols) + "|"]
    for name, rs in groups.items():
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(_ms([r.get(c[0], float("nan")) for r in rs]) for c in cols) + " |")
    table = "\n".join(lines)
    if baseline:
        base = baseline if baseline in groups else next((g for g in groups if g.endswith(baseline)), None)
        if base is None:
            print(f"[eval] baseline '{baseline}' not found; skipping paired table")
        else:
            table += "\n\n" + paired_table(groups, base) + "\n\n" + paired_table(groups, base, "ls_sharpe")
        for hname in HAND_BASELINES:
            if hname in groups and base != hname:
                table += "\n\n" + paired_table(groups, hname)
    table += "\n\n" + seed_ensemble_table(groups, cache) + "\n\n" + union_gate_table(groups, cache)
    return table, groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--baseline", default=None, help="method name for the paired comparison table")
    ap.add_argument("--group", default="name", choices=["name", "method"],
                    help="'method' strips the window/tag prefix so walk-forward windows pool together")
    ap.add_argument("--snapshot", default="both", choices=["final", "best_valid", "both"],
                    help="pool to score; with 'both' the final pool is the main table")
    ap.add_argument("--hand", action="store_true", help="add the hand-made sector baseline (needs sector labels)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    cache: Dict = {}
    key = "method" if args.group == "method" else "name"
    snaps = ["final", "best_valid"] if args.snapshot == "both" else [args.snapshot]
    titles = {"final": "Final pool (main result)", "best_valid": "Best-validation snapshot"}
    sections, main_groups, main_dirs = [], None, None
    for snap in snaps:
        runs = [d for d in args.runs if os.path.exists(os.path.join(d, SNAPSHOTS[snap]))]
        results = []
        for d in runs:
            r = evaluate_run(d, cache, snap)
            r["_dir"] = d
            r["window"], r["method"] = _window_and_method(r["name"])
            results.append(r)
        if args.hand and results:
            hand = hand_baseline(results, cache)
            if any("mom12_1" in Config.from_json(os.path.join(r["_dir"], "config.json")).features for r in results):
                hand += hand_baseline(results, cache, "hand_sector_mom")
            results += hand
        table, groups = _report(results, key, args.baseline, cache)
        sections.append(f"## {titles[snap]}\n\n{table}")
        if main_groups is None:
            main_groups = groups
            main_dirs = OrderedDict()
            for r in results:
                if "_dir" in r:
                    main_dirs.setdefault(r[key], []).append(r["_dir"])
    report = "\n\n".join(sections)
    with open(os.path.join(args.out, "results_table.md"), "w") as f:
        f.write(report + "\n")
    print(report)

    groups, dirs = main_groups, main_dirs
    pick = lambda rs: max(rs, key=lambda r: r["valid_ic"] if r["valid_ic"] == r["valid_ic"] else -9)
    flines = []
    for name, rs in groups.items():
        best = pick(rs)
        if not best["formulas"]:
            continue
        flines.append(f"### {name} (seed {best['seed']}, valid IC {best['valid_ic']:+.4f}, test IC {best['test_ic']:+.4f})")
        for m in sorted(best["formulas"], key=lambda m: -abs(m["weight"])):
            flines.append(f"- `{m['formula']}`  (w={m['weight']:+.3f}, train IC={m['train_ic']:+.4f})")
        flines.append("")
    with open(os.path.join(args.out, "formulas.md"), "w") as f:
        f.write("\n".join(flines))

    fig = os.path.join(args.out, "figures")
    plot_training_curves(dirs, os.path.join(fig, "training_curves.png"))
    plot_ablation({n: {k: [r.get(k, float("nan")) for r in rs] for k in ("valid_ic", "test_ic", "test_rank_ic")}
                   for n, rs in groups.items()}, os.path.join(fig, "ablation.png"))
    curves, bench = {}, None
    for name, rs in groups.items():
        best = pick(rs)
        if best["_port"] is None:
            continue
        curves[name], bench = best["_port"], best["_bench"]
        n = best["_corr"].shape[0]
        if n > 1:
            plot_pool_corr(best["_corr"], [f"a{i+1}" for i in range(n)],
                           os.path.join(fig, f"pool_corr_{name}.png"), title=f"{name}: pool alpha correlation (test)")
    if bench is not None:
        plot_cumulative(curves, bench, os.path.join(fig, "cumulative_return.png"))
    print(f"\nSaved table, formulas and figures under {args.out}/")


if __name__ == "__main__":
    main()
