# PeerAlpha — experiment summary

Test metrics use the pool snapshot with the best *validation* IC. `±` = std over seeds.

## Experiments

| Experiment | Data | Steps | Methods | Runs |
|---|---|---|---|---|
| [pilot_e0.005_n1024](#pilot_e0.005_n1024) | synthetic.npz | 100000 | 1 | 2 |
| [pilot_e0.005_n2048](#pilot_e0.005_n2048) | synthetic.npz | 100000 | 1 | 2 |
| [pilot_e0.02_n1024](#pilot_e0.02_n1024) | synthetic.npz | 100000 | 1 | 2 |
| [pilot_e0.02_n2048](#pilot_e0.02_n2048) | synthetic.npz | 100000 | 1 | 2 |

## pilot_e0.005_n1024

### Aggregate (mean ± std over seeds, sorted by Test IC)

| Method | seeds | Valid IC | Test IC | Test RankIC | Test ICIR | Sharpe | AnnRet | MaxDD | |corr| | #graph |
|---|---|---|---|---|---|---|---|---|---|---|
| sector_dense | 2 | +0.0402 ± 0.0040 | +0.0421 ± 0.0040 | +0.0386 ± 0.0031 | +0.3104 ± 0.0312 | +0.7054 ± 0.0608 | +0.0930 ± 0.0087 | -0.1369 ± 0.0128 | +0.0971 ± 0.0405 | +3.5000 ± 0.5000 |

<details><summary>Per-seed results</summary>

| Method | seed | Valid IC | Test IC | Test RankIC | Test ICIR | Sharpe | AnnRet | MaxDD | |corr| | #graph |
|---|---|---|---|---|---|---|---|---|---|---|
| sector_dense | 0 | +0.0362 | +0.0461 | +0.0417 | +0.3417 | +0.7662 | +0.1017 | -0.1241 | +0.1376 | +4.0000 |
| sector_dense | 1 | +0.0442 | +0.0381 | +0.0355 | +0.2792 | +0.6447 | +0.0843 | -0.1496 | +0.0566 | +3.0000 |

</details>

### Best formulas (best validation seed per method)

<details><summary><b>sector_dense</b> — seed 1, valid +0.0442, test +0.0381</summary>

- `Sub(Greater(-0.5, $low), Add(0.5, WMA($open, 20)))` (w=-0.050, train IC=-0.0498)
- `Div(-2, Div($close, Sub(Mul(-5, $vwap), $close)))` (w=+0.035, train IC=+0.0354)
- `WMA(NbrMean(Sub(2, EMA($vwap, 10)), G_sector), 1)` (w=-0.019, train IC=-0.0185)
- `Mul(-5, CsRank(CsRank(Div(Std($high, 20), $high))))` (w=-0.016, train IC=-0.0163)
- `CsRank(Greater(-0.01, CsRank(Delta(NbrStd($close, G_sector), 5))))` (w=-0.016, train IC=-0.0162)
- `Greater(-1, Add(Greater(Max($volume, 5), 2), 0.5))` (w=-0.011, train IC=-0.0107)
- `CsRank(Delta(Ref(Mul(-0.01, $vwap), 20), 40))` (w=-0.010, train IC=-0.0095)
- `Mul(0.5, Delta(Delta(Mean($volume, 10), 40), 5))` (w=+0.009, train IC=+0.0093)
- `Div(Abs(Min(Cov($volume, $open, 20), 20)), $volume)` (w=-0.007, train IC=-0.0072)
- `Abs(Add(0.5, Log(Std(NbrRel($close, G_sector), 40))))` (w=+0.007, train IC=+0.0067)

</details>

## pilot_e0.005_n2048

### Aggregate (mean ± std over seeds, sorted by Test IC)

| Method | seeds | Valid IC | Test IC | Test RankIC | Test ICIR | Sharpe | AnnRet | MaxDD | |corr| | #graph |
|---|---|---|---|---|---|---|---|---|---|---|
| sector_dense | 2 | +0.0339 ± 0.0072 | +0.0439 ± 0.0005 | +0.0411 ± 0.0004 | +0.3217 ± 0.0243 | +0.6621 ± 0.0259 | +0.0870 ± 0.0036 | -0.1183 ± 0.0055 | +0.1324 ± 0.0598 | +4.5000 ± 0.5000 |

<details><summary>Per-seed results</summary>

| Method | seed | Valid IC | Test IC | Test RankIC | Test ICIR | Sharpe | AnnRet | MaxDD | |corr| | #graph |
|---|---|---|---|---|---|---|---|---|---|---|
| sector_dense | 0 | +0.0267 | +0.0444 | +0.0415 | +0.2974 | +0.6362 | +0.0833 | -0.1238 | +0.0726 | +5.0000 |
| sector_dense | 1 | +0.0411 | +0.0435 | +0.0408 | +0.3460 | +0.6880 | +0.0906 | -0.1128 | +0.1922 | +4.0000 |

</details>

### Best formulas (best validation seed per method)

<details><summary><b>sector_dense</b> — seed 1, valid +0.0411, test +0.0435</summary>

- `Sub(Greater(-0.5, $low), Add(0.5, WMA($open, 20)))` (w=-0.050, train IC=-0.0498)
- `Mul(5, Div($close, Mul(-2, Less($open, $high))))` (w=+0.037, train IC=+0.0367)
- `WMA(NbrMean(Sub(2, EMA($vwap, 10)), G_sector), 1)` (w=-0.019, train IC=-0.0185)
- `Mul(-5, CsRank(CsRank(Div(Std($high, 20), $high))))` (w=-0.016, train IC=-0.0163)
- `Less(Abs(Min(Cov($volume, $open, 20), 20)), $volume)` (w=-0.012, train IC=-0.0117)
- `Add($vwap, Greater(-1, Greater(Max($volume, 5), 2)))` (w=-0.011, train IC=-0.0107)
- `Abs(Add(0.5, Log(Std(NbrRel($close, G_sector), 40))))` (w=+0.007, train IC=+0.0067)
- `Add(-0.5, Mul(Abs($volume), CsRank(Sub($close, 5))))` (w=+0.003, train IC=+0.0026)
- `Mean(Div(2, Ref(NbrStd($open, G_sector), 20)), 20)` (w=+0.002, train IC=+0.0019)
- `Mul(-0.01, NbrStd($close, G_sector))` (w=+0.002, train IC=+0.0016)

</details>

## pilot_e0.02_n1024

### Aggregate (mean ± std over seeds, sorted by Test IC)

| Method | seeds | Valid IC | Test IC | Test RankIC | Test ICIR | Sharpe | AnnRet | MaxDD | |corr| | #graph |
|---|---|---|---|---|---|---|---|---|---|---|
| sector_dense | 2 | +0.0402 ± 0.0040 | +0.0421 ± 0.0040 | +0.0386 ± 0.0031 | +0.3104 ± 0.0312 | +0.7054 ± 0.0608 | +0.0930 ± 0.0087 | -0.1369 ± 0.0128 | +0.0971 ± 0.0405 | +3.5000 ± 0.5000 |

<details><summary>Per-seed results</summary>

| Method | seed | Valid IC | Test IC | Test RankIC | Test ICIR | Sharpe | AnnRet | MaxDD | |corr| | #graph |
|---|---|---|---|---|---|---|---|---|---|---|
| sector_dense | 0 | +0.0362 | +0.0461 | +0.0417 | +0.3417 | +0.7662 | +0.1017 | -0.1241 | +0.1376 | +4.0000 |
| sector_dense | 1 | +0.0442 | +0.0381 | +0.0355 | +0.2792 | +0.6447 | +0.0843 | -0.1496 | +0.0566 | +3.0000 |

</details>

### Best formulas (best validation seed per method)

<details><summary><b>sector_dense</b> — seed 1, valid +0.0442, test +0.0381</summary>

- `Sub(Greater(-0.5, $low), Add(0.5, WMA($open, 20)))` (w=-0.050, train IC=-0.0498)
- `Div(-2, Div($close, Sub(Mul(-5, $vwap), $close)))` (w=+0.035, train IC=+0.0354)
- `WMA(NbrMean(Sub(2, EMA($vwap, 10)), G_sector), 1)` (w=-0.019, train IC=-0.0185)
- `Mul(-5, CsRank(CsRank(Div(Std($high, 20), $high))))` (w=-0.016, train IC=-0.0163)
- `CsRank(Greater(-0.01, CsRank(Delta(NbrStd($close, G_sector), 5))))` (w=-0.016, train IC=-0.0162)
- `Greater(-1, Add(Greater(Max($volume, 5), 2), 0.5))` (w=-0.011, train IC=-0.0107)
- `CsRank(Delta(Ref(Mul(-0.01, $vwap), 20), 40))` (w=-0.010, train IC=-0.0095)
- `Mul(0.5, Delta(Delta(Mean($volume, 10), 40), 5))` (w=+0.009, train IC=+0.0093)
- `Div(Abs(Min(Cov($volume, $open, 20), 20)), $volume)` (w=-0.007, train IC=-0.0072)
- `Abs(Add(0.5, Log(Std(NbrRel($close, G_sector), 40))))` (w=+0.007, train IC=+0.0067)

</details>

## pilot_e0.02_n2048

### Aggregate (mean ± std over seeds, sorted by Test IC)

| Method | seeds | Valid IC | Test IC | Test RankIC | Test ICIR | Sharpe | AnnRet | MaxDD | |corr| | #graph |
|---|---|---|---|---|---|---|---|---|---|---|
| sector_dense | 2 | +0.0423 ± 0.0012 | +0.0400 ± 0.0035 | +0.0380 ± 0.0028 | +0.3111 ± 0.0349 | +0.7015 ± 0.0135 | +0.0926 ± 0.0019 | -0.1062 ± 0.0066 | +0.1330 ± 0.0592 | +3.0000 ± 1.0000 |

<details><summary>Per-seed results</summary>

| Method | seed | Valid IC | Test IC | Test RankIC | Test ICIR | Sharpe | AnnRet | MaxDD | |corr| | #graph |
|---|---|---|---|---|---|---|---|---|---|---|
| sector_dense | 0 | +0.0434 | +0.0365 | +0.0352 | +0.2762 | +0.7149 | +0.0945 | -0.0995 | +0.0738 | +2.0000 |
| sector_dense | 1 | +0.0411 | +0.0435 | +0.0408 | +0.3460 | +0.6880 | +0.0906 | -0.1128 | +0.1922 | +4.0000 |

</details>

### Best formulas (best validation seed per method)

<details><summary><b>sector_dense</b> — seed 0, valid +0.0434, test +0.0365</summary>

- `Sub(Add(2, Delta(Abs($vwap), 5)), CsRank($low))` (w=-0.041, train IC=-0.0408)
- `Greater(-1, Div($open, Abs(Div(Abs($vwap), 2))))` (w=+0.036, train IC=+0.0365)
- `Greater(5, Less($close, Mul(0.5, NbrMean($low, G_sector))))` (w=+0.032, train IC=+0.0324)
- `Div(Max(Ref(Add($low, 0.5), 10), 10), $open)` (w=+0.025, train IC=+0.0252)
- `Div($low, Sub(Less($open, -0.5), WMA($close, 10)))` (w=+0.024, train IC=+0.0241)
- `Delta(Greater(Sum(Ref($volume, 20), 10), 5), 40)` (w=-0.019, train IC=-0.0192)
- `Less(5, CsRank(Mul($open, Cov($vwap, $volume, 40))))` (w=+0.015, train IC=+0.0150)
- `Add(0.01, Log(CsRank(Mean(Greater($volume, $volume), 20))))` (w=-0.013, train IC=-0.0134)
- `Std(Add(-2, NbrRank(Max($low, 40), G_sector)), 10)` (w=+0.012, train IC=+0.0123)
- `Div(Ref(Mul(-0.01, $open), 40), Sum($vwap, 20))` (w=+0.010, train IC=+0.0101)

</details>

