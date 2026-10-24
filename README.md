# GraphAlpha-RL

Graph-aware symbolic alpha mining with reinforcement learning.

A **MaskablePPO** agent writes stock-selection formulas one token at a time. Besides ordinary
time-series operators (`Delta`, `Mean`, `Corr`, ...), the formula language contains **graph operators**
(`NbrMean`, `NbrRel`, `NbrStd`, `NbrMax`, `NbrRank`, `Diffuse`) that aggregate a quantity over each
stock's neighbours on a **day-varying stock-relationship graph** (sector peers, rolling-correlation
peers, lead-lag leaders). The agent is rewarded for how much a finished formula improves a small,
diverse **pool** of alphas, measured by cross-sectional Information Coefficient (IC) against the
20-day forward return.

> Everything that *learns* is the RL policy. Pool weights are parameter-free (`w_i = IC_i`), graphs
> are deterministic functions of past returns, and the backtest is a fixed evaluation rule.

Prior work acknowledged: the token-by-token MDP / synergistic alpha pool idea comes from
*Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning* (Yu et al., KDD 2023).
This repository is an independent implementation; the graph operators, dynamic graphs, diversity /
complexity-shaped reward, curriculum and the AST graph-attention policy are new here.

---

## RL formulation

| | |
|---|---|
| **State** | Dict: `tokens` (`L=15` token ids, 0-padded), `adj` (`15x15` operator→operand edges of the partial formula tree), `top` (positions of the top-3 stack subtrees), `ctx` (4 numbers: remaining budget, graph-ops-enabled flag, pool fill, current pool IC). |
| **Action** | Discrete next token (≈ 50): 6 features, 10 constants, 5 windows, 3 graph ids, 3 unary + 6 binary + 9 rolling + 2 pair-rolling + 6 graph operators, `<SEP>`. A grammar stack provides an **action mask** so only completable, type-correct tokens are legal. |
| **Reward** | 0 until the formula ends, then `R = s·(ΔIC_pool + β·(1−max\|ρ\|)·\|IC\|) − γ·len/budget`; invalid formula `−0.5`, near-duplicate (`\|ρ\|>0.99`) `−0.2`. |
| **Environment** | Custom Gymnasium env (`graphalpha/env.py`): a formula builder + GPU/CPU evaluator on daily stock data (NSE-100 or CSI 300, or a synthetic market) + the alpha pool. |
| **Algorithm** | MaskablePPO (`sb3-contrib`) with a custom **graph-attention** feature extractor over the partial formula tree (or an LSTM baseline). |

## Repository layout

```
graphalpha/
  config.py     experiment config (dataclass <-> YAML, CLI overrides)
  data.py       market assembly + Baostock CSI 300 downloader
  universe.py   stock universes (live / built-in constituent lists)
  synthetic.py  synthetic market with planted cross-stock structure (regimes: mixed, relational)
  market.py     tensors, forward-return target, date splits (with purging), graphs
  graphs.py     sector / rolling-correlation / lead-lag neighbour graphs (causal)
  ops.py        operator library incl. graph operators (all causal)
  expr.py       formula tree + evaluator
  vocab.py      token vocabulary (= action space)
  grammar.py    postfix stack grammar + action masking
  metrics.py    z-score, daily IC / RankIC
  pool.py       alpha pool + reward
  env.py        Gymnasium environment
  policy.py     graph-attention and LSTM feature extractors
  runner.py     training loops (PPO + random baseline), logging
  backtest.py   long-only top-K evaluation
  plotting.py   report figures
configs/        one YAML per experiment / ablation
scripts/        fetch_nse.py, fetch_data.py (data), summarize.py, progress.py, sanity_check.py
tests/          unit tests (grammar completeness, operator causality, graph no-lookahead, metrics, env)
reports/        evaluated results: per-experiment tables, formulas, figures + SUMMARY.md
data/           market .npz files (not in git, see data/README.md)
runs/           training outputs (not in git)
train.py  evaluate.py  run_suite.py
```

## Installation

```bash
conda create -n graphalpha python=3.10 -y && conda activate graphalpha
pip install -r requirements.txt
pytest -q                         # ~1 min, runs offline on synthetic data
```

A CUDA GPU is optional (formula evaluation uses it automatically when present; the policy runs on CPU).

## Data

Data files are not in git; see [`data/README.md`](data/README.md) to regenerate them.

```bash
python scripts/fetch_nse.py --out data/nse100.npz                     # main real dataset (needs internet + yfinance)
python scripts/fetch_data.py --synthetic --out data/synthetic.npz     # offline synthetic market
python scripts/fetch_data.py --synthetic --regime relational --out data/synthetic_relational.npz
python scripts/fetch_data.py --out data/csi300.npz                    # optional: CSI 300 via Baostock
```

* **Main real dataset: NSE-100.** ~102 large-cap NSE (India) stocks, 2011-2024, 15 sectors, from Yahoo
  Finance (split/dividend adjusted). Yahoo has no VWAP, so `vwap` = typical price (H+L+C)/3. Sectors come
  from a hand-written map. Universe = today's large caps listed before 2012 (survivorship bias).
* **CSI 300 (optional).** Baostock, CSRC industry code as sector. Same survivorship caveat.
* **Synthetic.** `mixed` (original planted market; a plain 5-day reversal carries most signal) and
  `relational` (designed so cross-stock / graph formulas carry the signal). Report these as designed data.
* Features: open, high, low, close, volume, vwap. Target = 20-day forward return.
* Splits (`configs/*.yaml`): train 2012-2019, valid 2020-2021, test 2022-mid-2024. The last 20 days of
  train/valid are purged so label windows never cross a split boundary.
* No lookahead: all operators are causal; the graph serving day *t* is built from returns up to *t-1*.

## Running

```bash
# smoke test (≈ 2 min, synthetic data)
python train.py --config configs/smoke.yaml

# main experiment + ablations (3 seeds each)
for cfg in graph_all nograph sector_only corr_only lead_only all_lstm all_nocurriculum all_nodiv random_graph random_nograph; do
  for s in 0 1 2; do python train.py --config configs/$cfg.yaml --seed $s; done
done

# evaluate best-validation pools on the untouched test set + figures
python evaluate.py runs/*_s* --out reports
```

Full suites (methods x seeds on one dataset, resumable, then evaluated into `reports/<tag>/`):

```bash
python run_suite.py --data data/nse100.npz --tag nse300k --seeds 0 1 2 3 4 --steps 300000
python scripts/summarize.py                 # -> reports/SUMMARY.md + reports/all_runs.csv
python scripts/progress.py nse300k          # live progress of running suites
python scripts/sanity_check.py --config configs/graph_all.yaml   # do hand-written graph formulas carry signal?
```

Outputs: `runs/<name>_s<seed>/{config.json, log.csv, pool_best_valid.json, pool_latest.json, model.zip}`;
`reports/{results_table.md, formulas.md, figures/*.png}`.
Override any config value: `python train.py --config configs/graph_all.yaml --set total_timesteps=100000 graph_k=6`.

## Experiments

| Config | What it tests |
|---|---|
| `graph_all` | full model |
| `nograph` | same agent without graph operators (AlphaGen-style language) |
| `sector_only`, `corr_only`, `lead_only` | which graph helps |
| `all_lstm` | graph-attention policy vs. flat LSTM policy |
| `all_nocurriculum`, `all_nodiv` | curriculum and diversity-bonus ablations |
| `random_graph`, `random_nograph` | random valid-action baseline (no learning) |
| `alphagen_like` | AlphaGen-style baseline (no graph ops) |
| `sector_dense`, `corr_dense`, `alphagen_dense` | same with `dense_reward: true` |
| `graph_all_robust`, `nograph_robust`, `sector_only_robust`, `corr_robust` | stability-aware reward (`robust_folds`) |

New config flags:

* `dense_reward` - when the pool is full, rejected candidates still get a graded (<= 0) reward instead of a
  flat zero, which otherwise freezes learning once the pool saturates.
* `robust_folds`, `robust_lambda` - with `robust_folds >= 2`, the pool objective is
  mean(fold IC) - lambda * std(fold IC) over contiguous train sub-periods.

## Results

Evaluated results live in `reports/` (`SUMMARY.md`, `all_runs.csv`, and one folder per experiment tag with
`results_table.md`, `formulas.md`, `figures/`). Caveats:

* `reports/syn1m_snapshot` comes from 1M-step runs that stopped at ~233k steps; its numbers match the 300k
  suite and are **not** a valid 1M-step result.
* On NSE-100, graph variants have so far not beaten the no-graph or random-search baselines on test IC, and
  long-only top-K Sharpe is similar across all methods (largely market beta).

## Reproducibility

Seeds are set for Python, NumPy, PyTorch and SB3. Each run saves its full config; `evaluate.py`
reloads pools from the saved token sequences, so reported formulas can be re-evaluated exactly.
