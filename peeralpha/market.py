"""Market container: price tensors, forward-return target, date splits and graphs."""
from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .graphs import NeighborGraph, build_graphs
from .ops import BASE_FEATURES, FEATURES, op_mean, op_std


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Market:
    """All data the environment needs, held as tensors on ``device``.

    feat   : [F, T, N]  (open, high, low, close, volume, vwap, ret, ret5, vrel, ret20, cvwap, range,
                         ret60, ret120, mom12_1, vol20)
    ret    : [T, N]     close-to-close daily return (day t vs t-1)
    target : [T, N]     forward return close[t+H]/close[t]-1  (NaN near the end)
    """

    def __init__(
        self,
        path: str,
        horizon: int,
        splits: Dict[str, List[str]],
        graph_names: List[str],
        graph_k: int = 8,
        graph_window: int = 60,
        graph_step: int = 5,
        device: str = "auto",
        cache_dir: Optional[str] = None,
        gop_zscore: bool = False,
        graph_residual: bool = False,
        min_sector_size: int = 1,
    ):
        z = np.load(path, allow_pickle=False)
        self.path = path
        self.device = resolve_device(device)
        self.horizon = horizon
        self.dates = z["dates"].astype("datetime64[D]")
        self.codes = [str(c) for c in z["codes"]]
        self.sector_ids = z["sector_ids"].astype(np.int64)
        self.sector_names = [str(s) for s in z["sector_names"]]
        self.T, self.N = len(self.dates), len(self.codes)

        self.gop_zscore = gop_zscore      # read by expr.evaluate for graph operators
        base = torch.as_tensor(np.stack([z[f] for f in BASE_FEATURES]).astype(np.float32), device=self.device)
        close = base[BASE_FEATURES.index("close")]
        volume = base[BASE_FEATURES.index("volume")]

        ret = torch.full_like(close, float("nan"))
        ret[1:] = close[1:] / close[:-1] - 1.0
        self.ret = ret
        ret5 = torch.full_like(close, float("nan"))
        ret5[5:] = close[5:] / close[:-5] - 1.0
        def back(a: int, b: int = 0) -> torch.Tensor:      # return from day t-a to day t-b
            out = torch.full_like(close, float("nan"))
            out[a:] = close[a - b:close.shape[0] - b] / close[:-a] - 1.0
            return out

        ret20 = back(20)
        vmean = op_mean(volume, 20)
        vrel = volume / torch.where(vmean > 0, vmean, torch.full_like(vmean, float("nan")))
        pos = lambda x: torch.where(x > 0, x, torch.full_like(x, float("nan")))
        cvwap = close / pos(base[BASE_FEATURES.index("vwap")]) - 1.0
        rng = base[BASE_FEATURES.index("high")] / pos(base[BASE_FEATURES.index("low")]) - 1.0
        derived = {"ret": ret, "ret5": ret5, "vrel": vrel, "ret20": ret20, "cvwap": cvwap, "range": rng,
                   "ret60": back(60), "ret120": back(120), "mom12_1": back(250, 20), "vol20": op_std(ret, 20)}
        self.feat = torch.cat([base, torch.stack([derived[f] for f in FEATURES[len(BASE_FEATURES):]])])
        target = torch.full_like(close, float("nan"))
        target[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
        self.target = target

        self._ranges = {k: self._date_range(*v) for k, v in splits.items()}
        if min_sector_size > 1:
            self.sector_ids = self._merge_small_sectors(ret, min_sector_size)
        self.graph_names = list(graph_names)
        self._graphs: Dict[str, NeighborGraph] = {}
        if graph_names:
            train_hi = self._ranges["train"][1]
            gret = self._sector_demeaned(ret) if graph_residual else ret
            built = build_graphs(
                gret.cpu(), self.sector_ids, graph_names, train_hi,
                graph_window, graph_step, graph_k, cache_dir=cache_dir,
                cache_key=f"{os.path.abspath(path)}|{os.path.getmtime(path)}|res={int(graph_residual)}"
                          f"|minsec={min_sector_size}",
            )
            self._graphs = {n: g.to(self.device) for n, g in built.items()}

    # ------------------------------------------------------------------
    def _merge_small_sectors(self, ret: torch.Tensor, min_size: int) -> np.ndarray:
        """Move stocks of sectors smaller than ``min_size`` into the large sector whose mean train
        return they track best (a sector of 1-2 stocks gives NaN or single-peer graph features)."""
        lo, hi = self._ranges["train"]
        R = ret[lo:hi].cpu().numpy()
        sec = self.sector_ids.copy()
        counts = np.bincount(sec, minlength=sec.max() + 1)
        big = [s for s in range(len(counts)) if counts[s] >= min_size]
        if not big:
            return sec
        with warnings.catch_warnings():                 # days before any listing are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.stack([np.nanmean(R[:, sec == s], axis=1) for s in big], axis=1)   # [T, S]
        for i in np.flatnonzero(counts[sec] < min_size):
            x = R[:, i]
            ok = np.isfinite(x) & np.isfinite(means).all(1)
            if ok.sum() < 20:
                continue
            c = [np.corrcoef(x[ok], means[ok, j])[0, 1] for j in range(len(big))]
            sec[i] = big[int(np.nanargmax(c))]
        return sec

    def _sector_demeaned(self, ret: torch.Tensor) -> torch.Tensor:
        """Daily return minus its sector's mean that day (idiosyncratic part for corr/lead graphs)."""
        out = ret.clone()
        sec = torch.as_tensor(self.sector_ids, device=ret.device)
        for s in sec.unique():
            cols = sec == s
            if int(cols.sum()) < 2:
                continue
            x = ret[:, cols]
            ok = torch.isfinite(x)
            mean = torch.where(ok, x, torch.zeros_like(x)).sum(1, keepdim=True) / ok.sum(1, keepdim=True).clamp(min=1)
            out[:, cols] = x - mean
        return out

    def _date_range(self, start: str, end: str) -> Tuple[int, int]:
        lo = int(np.searchsorted(self.dates, np.datetime64(start), side="left"))
        hi = int(np.searchsorted(self.dates, np.datetime64(end), side="right"))
        return lo, hi

    def split_range(self, split: str) -> Tuple[int, int]:
        """Raw ``[lo, hi)`` day indices of a split."""
        return self._ranges[split]

    def ic_slice(self, split: str) -> Tuple[int, int]:
        """Day range used to score alphas.

        Train/valid targets reach ``horizon`` days beyond the split end, so those
        final days are purged to stop target information crossing split boundaries.
        """
        lo, hi = self._ranges[split]
        if split != "test":
            hi = hi - self.horizon
        return lo, hi

    def graph(self, name: str) -> NeighborGraph:
        return self._graphs[name]

    @property
    def feature_names(self) -> List[str]:
        return list(FEATURES)

    def summary(self) -> str:
        parts = [f"{k}=[{self.dates[a]}..{self.dates[min(b, self.T) - 1]}]({b - a}d)" for k, (a, b) in self._ranges.items()]
        return f"Market(T={self.T}, N={self.N}, device={self.device}, " + ", ".join(parts) + ")"
