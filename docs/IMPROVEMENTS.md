# Improvement plan (Oct 2026)

Full write-up with evidence: https://claude.ai/artifact/SfsR2CkixGpFecwADZvQTi

## Diagnosis

Almost every PPO run in `runs/` collapsed within 20k-50k steps to one-token formulas
(`$close <SEP>`): 2.00 steps/episode, mean return exactly `-gamma_len/15 = -0.0033`, pool
frozen for the rest of the run, graph usage 0%. Only `gamma_len: 0` configs (alphagen_like,
alphagen_dense) and random search kept exploring, which is why they looked better. The
existing graph vs no-graph tables therefore say nothing about graph operators yet.

## Status

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

Old run configs keep their behaviour: `Config.from_json` fills `gop_zscore`/`graph_residual`
with the values those runs used, and their saved `weighting`/penalties are explicit.

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
python -m graphalpha.data --synthetic --regime relational --out data/synthetic_relational_v2.npz

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
