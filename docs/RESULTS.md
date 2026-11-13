# NSE walk-forward results

The results can also be browsed at https://peeralpha.sreenitya.in (Results tab).

Universe: ~100 NSE large caps (`data/nse100.npz`) unless noted; v5 also runs on Nifty 200
(`data/nifty200.npz`, 145 stocks with full history, 18 NSE industries). 3 expanding walk-forward windows, 3 seeds,
300k steps per run. Metric: mean daily test IC of the final pool. Full tables (Newey-West t,
backtests, paired tests, formulas) are in `reports/<tag>_wf/results_table.md`.

| Window | Train | Valid | Test |
|---|---|---|---|
| w0 | 2012-2015 | 2016-2017 | 2018-2019 |
| w1 | 2012-2017 | 2018-2019 | 2020-2021 |
| w2 | 2012-2019 | 2020-2021 | 2022-2024H1 |

(v2 used 1 validation year, so its train windows are one year longer.)

Baselines (no learning, no fitting):

- `hand_sector`: `NbrMean($ret20, sector) - NbrRel($ret5, sector)` (sector momentum minus
  within-sector 5-day reversal).
- `hand_sector_mom`: `hand_sector` plus 12-1 month own momentum `$mom12_1` (added in v4).

## Summary

| Version | Horizon | Best learned method | Test IC | hand_sector | hand_sector_mom | Gate |
|---|---|---|---|---|---|---|
| v2 (`reports/v2nse_wf_rescore`) | 20d | v2_graph_all | 0.038 | 0.033 | - | not set |
| v3 (`reports/v3nse_wf`) | 5d | v3_random_graph | 0.029 | 0.039 | - | failed |
| v4 (`reports/v4nse_wf`) | 5d | v4_sector_prior | 0.037 (0.040 seed avg) | 0.039 | **0.044** | passed by 0.0006 (noise) |
| v5, 100 stocks (`reports/v5nse_wf`) | 5d | hand_sector_mom + pool blend | 0.044 | 0.039 | **0.044** | failed |
| v5, Nifty 200 (`reports/v5n200_wf`) | 5d | hand_sector_mom + pool blend | **0.050** | 0.038 | 0.047 | passed narrowly (2/3 windows) |

The unlearned `hand_sector_mom` baseline is still about as strong as anything learned. Learned
pools clearly beat the no-graph agent. On Nifty 200, blending the learned pool into
`hand_sector_mom` (weight picked on validation) edges past it, but the margin is within noise.

## v3 (2026-11-12)

Changes: 5-day horizon, scale-free features only, out-of-fold ridge reward, sign-stability gate,
pool of 10 with `ridge_lambda` 1, `swap_needs_gain`, 2 validation years.

| Method | w0 | w1 | w2 | Mean |
|---|---|---|---|---|
| v3_random_graph | 0.037 | 0.024 | 0.025 | 0.029 |
| v3_sector | 0.038 | 0.011 | 0.024 | 0.024 |
| v3_graph_all | 0.029 | 0.012 | 0.023 | 0.022 |
| v3_nograph | 0.026 | -0.009 | 0.023 | 0.013 |
| hand_sector | 0.047 | 0.044 | 0.027 | 0.039 |

Gate failed: v3_graph_all lost to `hand_sector` (paired t -3.8, 1/9 wins) and to random search.

What the follow-up analysis found:

- The drop from v2 is mostly the horizon. Re-scored at 5 days, v2 graph_all gives 0.025, about
  the same as v3.
- The overfitting is in which formulas get picked, not in the weights. Equal sign weights match
  ridge, and the out-of-fold objective is close to the in-sample one (0.105 vs 0.108).
- Pool members shrink about 80% out of sample (train IC 0.036, valid 0.013, test 0.008), and
  pools correlate 0.55-0.66 with `hand_sector` on test.
- Sector-graph members do best (test IC 0.013 vs 0.005 without). The corr and lead graphs are rare
  in pools (3-4% of members) and add nothing. `$vrel` is overused.
- The vocabulary could not express 6-12 month momentum (features stop at 20 days, windows at 40).

## v4 (2026-11-12)

Changes on top of v3:

- Features `$ret60`, `$ret120`, `$mom12_1` (return from t-250 to t-20) and `$vol20`.
- Windows 60 and 120.
- No `$vrel`. Only the sector graph. No NbrMax, NbrStd or Diffuse. `max_len` 10.
- `prior: hand_sector`: the pool starts with the two hand factors as fixed members that are never
  dropped, so the agent is paid only for what it adds on top.
- The report has a `hand_sector_mom` row and a gated seed-union table.

Test IC. Each window is the 3 seeds' averaged signal; single-seed mean in brackets.

| Method | w0 | w1 | w2 | Mean |
|---|---|---|---|---|
| hand_sector_mom | 0.055 | 0.034 | 0.043 | **0.044** |
| v4_sector_prior | 0.053 | 0.035 | 0.032 | 0.040 (0.037) |
| hand_sector | 0.047 | 0.044 | 0.027 | 0.039 |
| v4_sector | 0.050 | 0.029 | 0.034 | 0.038 (0.035) |
| v4_random_prior | 0.045 | 0.036 | 0.029 | 0.037 (0.034) |
| v4_nograph | 0.041 | 0.005 | 0.028 | 0.025 (0.022) |

Paired per seed and window, final pool:

| Comparison | Mean diff | t | Wins |
|---|---|---|---|
| v4_sector_prior vs v4_nograph | +0.015 | +3.9 | 9/9 |
| v4_sector_prior vs hand_sector | -0.003 | -1.0 | 3/9 |
| v4_sector_prior vs hand_sector_mom | -0.008 | -3.3 | 2/9 |

Takeaways:

- Graph operators clearly beat no-graph, and PPO beats random search when both have the same
  prior and vocabulary.
- The prior adds about 0.002 over v4_sector.
- v4 is much better than v3: single-seed mean 0.037 vs 0.022 for the main method.
- The gated seed union (members with validation IC > 0, equal weight) is worse than plain seed
  averaging (0.028 for v4_sector_prior). Validation is too noisy to filter single members.
- Outside the protocol: adding the v4_sector_prior seed-averaged signal at half weight to
  `hand_sector_mom` gives 0.059 / 0.038 / 0.041 (mean 0.046 vs 0.044). The weight was picked
  after seeing test, so this is only a lead.

## v5 (2026-11-12)

Changes on top of v4:

- `prior: hand_sector_mom`: the pool starts with `hand_sector` and `$mom12_1` as three fixed
  members.
- Prior blend: final signal = z(`hand_sector_mom`) + lam * z(seed-averaged pool), with lam picked
  on validation from {0, 0.25, 0.5, 1, 2}. This is the protocol version of the v4 lead.
- New universe: Nifty 200 constituents (`scripts/fetch_nse.py --universe_csv`, built on Kaggle by
  `scripts/kaggle_fetch_nifty200.py`).

### 100 stocks

Test IC, seed-averaged signal (single-seed mean in brackets):

| Method | w0 | w1 | w2 | Mean |
|---|---|---|---|---|
| hand_sector_mom | 0.055 | 0.034 | 0.043 | **0.044** |
| blend: hand_sector_mom + v5_sector_prior | 0.053 (lam 2) | 0.039 (lam 2) | 0.039 (lam 2) | 0.044 |
| v5_sector_prior | 0.047 | 0.038 | 0.034 | 0.040 (0.037) |
| hand_sector | 0.047 | 0.044 | 0.027 | 0.039 |
| v5_random_prior | 0.045 | 0.036 | 0.034 | 0.038 (0.035) |

Gate (beat `hand_sector_mom` on the mean and in 2 of 3 windows) failed: 0.0437 vs 0.0441.
Validation picked the largest pool weight in every window, but test did not agree.
v5_sector_prior vs `hand_sector_mom`, paired per seed and window: -0.007, t -2.8, 2/9 wins.

### Nifty 200 (145 stocks)

| Method | w0 | w1 | w2 | Mean | LS Sharpe |
|---|---|---|---|---|---|
| blend: hand_sector_mom + v5_sector_prior | 0.070 (lam 2) | 0.027 (lam 1) | 0.053 (lam 0) | **0.050** | - |
| blend: hand_sector_mom + v5_random_prior | 0.067 (lam 0.5) | 0.029 (lam 1) | 0.046 (lam 2) | 0.047 | - |
| hand_sector_mom | 0.063 | 0.025 | 0.053 | 0.047 | 1.19 |
| v5_sector_prior | 0.068 | 0.025 | 0.042 | 0.045 (0.042) | 1.50 |
| v5_random_prior | 0.063 | 0.030 | 0.040 | 0.044 (0.041) | 1.40 |
| v4_sector | 0.063 | 0.027 | 0.032 | 0.041 (0.038) | 1.46 |
| hand_sector | 0.045 | 0.028 | 0.043 | 0.038 | 0.93 |
| v4_nograph | 0.058 | 0.007 | 0.029 | 0.031 (0.029) | - |

LS Sharpe is the single-seed mean of the final pool's long-short backtest.

Takeaways:

- The gate passed narrowly: the sector-prior blend beats `hand_sector_mom` in w0 and w1 and ties
  in w2, where validation set lam to 0. The gain (+0.003) is within noise.
- The random-search prior blend also ties the baseline, so PPO is not clearly what adds the value
  over random search with the same vocabulary.
- Learned pools have a clearly higher long-short Sharpe than the hand factors (1.50 vs 1.19).
- Graph operators still beat no-graph by a wide margin (v5_sector_prior vs v4_nograph +0.014,
  t 5.6, 9/9).

## Next

1. More seeds or windows on Nifty 200 to tell whether the blend's edge is real.
2. Score portfolios on long-short returns as well as IC, since that is where pools beat the hand
   factors.

## Reproduce

Runs on Kaggle CPU, one kernel per window (`scripts/kaggle_walk_forward.py`, about 1.3-4.3 h per
window with 4 jobs). Locally:

```bash
python scripts/walk_forward.py --data data/nse100.npz --tag v4nse --seeds 0 1 2 --valid_years 2 \
  --methods v4_sector_prior v4_sector v4_nograph v4_random_prior --steps 300000 \
  --baseline v4_nograph --hand --jobs 4 --threads 1
```

v5 on Nifty 200: `--data data/nifty200.npz --tag v5n200 --methods v5_sector_prior v4_sector
v4_nograph v5_random_prior` with the same other flags.
