"""Market container: price tensors, forward-return target, date splits and graphs."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .graphs import NeighborGraph, build_graphs
from .ops import BASE_FEATURES, FEATURES, op_mean


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Market:
    """All data the environment needs, held as tensors on ``device``.

    feat   : [F, T, N]  (open, high, low, close, volume, vwap, ret, ret5, vrel)
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
        vmean = op_mean(volume, 20)
        vrel = volume / torch.where(vmean > 0, vmean, torch.full_like(vmean, float("nan")))
        derived = {"ret": ret, "ret5": ret5, "vrel": vrel}
        self.feat = torch.cat([base, torch.stack([derived[f] for f in FEATURES[len(BASE_FEATURES):]])])
        target = torch.full_like(close, float("nan"))
        target[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
        self.target = target

        self._ranges = {k: self._date_range(*v) for k, v in splits.items()}
        self.graph_names = list(graph_names)
        self._graphs: Dict[str, NeighborGraph] = {}
        if graph_names:
            train_hi = self._ranges["train"][1]
            gret = self._sector_demeaned(ret) if graph_residual else ret
            built = build_graphs(
                gret.cpu(), self.sector_ids, graph_names, train_hi,
                graph_window, graph_step, graph_k, cache_dir=cache_dir,
                cache_key=f"{os.path.abspath(path)}|{os.path.getmtime(path)}|res={int(graph_residual)}",
            )
            self._graphs = {n: g.to(self.device) for n, g in built.items()}

    # ------------------------------------------------------------------
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
