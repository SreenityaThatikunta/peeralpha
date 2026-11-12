# NSE walk-forward results

Universe: ~100 NSE large caps (`data/nse100.npz`), 3 expanding walk-forward windows, 3 seeds,
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

The strongest signal so far is the unlearned `hand_sector_mom` baseline. Learned pools now clearly
beat the no-graph agent, but not that baseline.

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

## Next

1. Use `hand_sector + mom12_1` as both the pool prior and the bar to beat. Pick the mix weight
   between prior and learned pool on validation.
2. Move to a Nifty 200 universe (more peers per sector, less noisy daily IC).

## Reproduce

Runs on Kaggle CPU, one kernel per window (`scripts/kaggle_walk_forward.py`, about 1.3-4.3 h per
window with 4 jobs). Locally:

```bash
python scripts/walk_forward.py --data data/nse100.npz --tag v4nse --seeds 0 1 2 --valid_years 2 \
  --methods v4_sector_prior v4_sector v4_nograph v4_random_prior --steps 300000 \
  --baseline v4_nograph --hand --jobs 4 --threads 1
```
