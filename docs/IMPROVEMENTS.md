# Design log

What was changed in each version and why. Results are in `docs/RESULTS.md`.

## v2: fixing the collapse (Oct 2026)

### Diagnosis

Almost every PPO run in `runs/` collapsed within 20k-50k steps to one-token formulas
(`$close <SEP>`): 2.00 steps/episode, mean return exactly `-gamma_len/15 = -0.0033`, pool
frozen for the rest of the run, graph usage 0%. Only `gamma_len: 0` configs (alphagen_like,
alphagen_dense) and random search kept exploring, which is why they looked better. The
existing graph vs no-graph tables therefore say nothing about graph operators yet.

### Status

| # | Change | Status |
|---|---|---|
| 1 | Stop the collapse: `gamma_len` 0, `<SEP>` masked before `min_tokens=3`, flat `repeat_penalty` for already-submitted formulas, smaller invalid/dup penalties, `ent_coef` 0.05 | done |
| 1 | Collapse alarms in `log.csv` (`mean_len`, `frac_unique`, `n_accepted`) and `freeze_patience` early stop | done |
| 2 | Ridge combiner (`weighting: ridge`, drop smallest \|w\| when full), `pool_capacity` 20, `dup_threshold` 0.9 | done |
| 3 | Scale-free features `$ret`, `$ret5`, `$vrel`; `gop_zscore` z-scores graph-operator inputs | done |
| 4 | Synthetic: `leader_of` saved, `graph_residual` (corr/lead on sector-demeaned returns), lead-graph precision + new formulas in `scripts/sanity_check.py`, `configs/v2_*.yaml` | done |
| 5 | NSE: sectors under `min_sector_size=3` merged into the closest large sector; residual corr graph via `graph_residual` | done |
| 6 | Evaluation: `scripts/walk_forward.py` (3 expanding windows), paired seed/window table vs `--baseline` in `evaluate.py` | done |
| 6 | Evaluation: 10 seeds, H=5 on NSE | run-time choice (`--seeds`, `--set horizon=5`) |
| 7 | Backtest: long-short top/bottom 20% (`LS Sharpe`, `LS AnnRet` columns) | done |
| 7 | Point-in-time universe (survivorship) | todo |
| 8 | Cache formula evaluations; regenerate `reports/SUMMARY.md` | todo |

## v3: generalization (Nov 2026)

What the v2 NSE walk-forward showed:
Members barely keep their sign out of sample, ridge over 20 of them inflates train IC to 0.12-0.16,
80-94% of members use raw price levels, and validation IC does not predict test IC.

| # | Change | Status |
|---|---|---|
| 1 | `evaluate.py` scores the final pool (main table) and the best-valid snapshot | done |
| 1 | Newey-West t of daily test IC (`NW t`), seed-ensemble table, `--hand` sector baseline row | done |
| 1 | `walk_forward.py --valid_years 2` | done |
| 2 | Out-of-fold reward (`oof_folds`): each train fold scored with ridge fitted on the others | done |
| 2 | Sign-stability gate (`stable_folds`, `stable_min_frac`, `unstable_penalty`) | done |
| 2 | `swap_needs_gain`: a full pool only swaps a member out if the (out-of-fold) objective improves | done |
| 2 | `features` option; new scale-free features `$ret20`, `$cvwap` (close/vwap-1), `$range` (high/low-1) | done |
| 2 | `configs/v3_*.yaml`: H=5, scale-free features, pool 10, `ridge_lambda` 1, OOF 4 folds, gate 3/4, `swap_needs_gain` | done |

Re-scoring the v2 runs (`reports/v2nse_wf_rescore/`), final pool, mean test IC over 3 windows x 3 seeds:
graph_all 0.038 (NW t 2.0), random_graph 0.032, random_nograph 0.029, nograph 0.021, sector 0.017,
hand sector baseline 0.033. Averaging the 3 seeds' signals lifts graph_all to 0.047 (positive in every window).

```bash
python scripts/walk_forward.py --data data/nse100.npz --tag v3nse --seeds 0 1 2 --valid_years 2 \
  --methods v3_nograph v3_sector v3_graph_all v3_random_graph --steps 300000 \
  --baseline v3_nograph --hand --jobs 5 --threads 1
```

Pass gate: v3_graph_all beats `hand_sector` on average, beats v3_nograph with t > 2 and
v3_random_graph with t > 1.5, under both snapshot rules.

Old run configs keep their behaviour: `Config.from_json` fills `gop_zscore`/`graph_residual`
with the values those runs used, and their saved `weighting`/penalties are explicit.

## v4: long-horizon features and a fixed prior (Nov 2026)

Results for v3 and v4 are in `docs/RESULTS.md`.

| # | Change | Status |
|---|---|---|
| 1 | Features `$ret60`, `$ret120`, `$mom12_1` (t-250..t-20), `$vol20`; config `windows` (adds 60, 120) and `ops` (operator subset) | done |
| 2 | `prior: hand_sector`: the pool starts with fixed members that are never dropped (`pool.PRIORS`) | done |
| 3 | `configs/v4_*.yaml`: sector graph only, no `$vrel`, no NbrMax/NbrStd/Diffuse, `max_len` 10 | done |
| 4 | `evaluate.py`: `hand_sector_mom` baseline row, gated seed-union table, hand baseline always on residual sector peers | done |
| 5 | `scripts/kaggle_walk_forward.py`: one walk-forward window per Kaggle kernel | done |
| 6 | `hand_sector + mom12_1` as the prior; mix weight between prior and pool picked on validation | done in v5 |
| 7 | Nifty 200 universe | done in v5 |

## v5: stronger prior and a bigger universe (Nov 2026)

| # | Change | Status |
|---|---|---|
| 1 | `prior: hand_sector_mom`: `hand_sector` plus `$mom12_1` as fixed pool members | done |
| 2 | `evaluate.py` prior-blend table: z(prior) + lam * z(seed-averaged pool), lam picked on validation | done |
| 3 | `configs/v5_sector_prior.yaml`, `configs/v5_random_prior.yaml` | done |
| 4 | `scripts/fetch_nse.py --universe_csv` (universe and industries from an index CSV); `scripts/kaggle_fetch_nifty200.py` | done |
| 5 | Nifty 200 walk-forward (`reports/v5n200_wf`) | done |
| 6 | Point-in-time universe (survivorship) | todo |
| 7 | More seeds / windows on Nifty 200 to test whether the blend's edge is real | todo |

Outcome: on 100 stocks the blend ties `hand_sector_mom` (0.044); on Nifty 200 it reaches 0.050 vs
0.047, a narrow pass that is within noise.

## Sanity check on the relational synthetic market

`python scripts/sanity_check.py --set data_path=data/synthetic_relational_v2.npz gop_zscore=true ...`
(mean daily IC on test; the sign is irrelevant, the pool flips it)

| Setting | own 5d reversal | NbrRel($ret5, sector) | lead graph precision |
|---|---|---|---|
| H=20, window 60, raw returns (old) | -0.014 | -0.040 | 17% |
| H=5, window 60, raw returns | -0.013 | -0.089 | 17% |
| H=5, window 250, residual returns | -0.013 | -0.087 | 100% |

With H=5 the sector graph alpha is about 7x stronger than any own-history alpha, so this is the
setting where graph operators must win before trying NSE.

## Commands

```bash
# regenerate the relational market with ground-truth leaders
python -m peeralpha.data --synthetic --regime relational --out data/synthetic_relational_v2.npz

# synthetic test of graph operators (H=5)
python run_suite.py --data data/synthetic_relational_v2.npz --tag v2rel --seeds 0 1 2 \
  --methods v2_nograph v2_sector v2_graph_all v2_random_nograph v2_random_graph \
  --steps 200000 --set horizon=5 --jobs 2 --threads 2

# then NSE
python run_suite.py --data data/nse100.npz --tag v2nse --seeds 0 1 2 3 4 \
  --methods v2_nograph v2_sector v2_graph_all v2_random_nograph \
  --steps 300000 --jobs 2 --threads 2

# walk-forward on NSE (3 windows x 3 seeds; report in reports/v2nse_wf/)
python scripts/walk_forward.py --data data/nse100.npz --tag v2nse --seeds 0 1 2 \
  --methods v2_nograph v2_sector v2_graph_all v2_random_nograph --steps 300000 --jobs 3
```

Add `--baseline v2_nograph` to `run_suite.py` (or `evaluate.py`) for the paired table;
a method is only better if the mean paired difference is clearly positive (|t| > 2).

Check `log.csv` while it runs: `mean_len` should stay above 5, `frac_unique` above 0.5, and
`n_accepted` should keep growing past 50k steps.
