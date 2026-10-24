"""Alpha pool: the part of the environment that turns a finished formula into a reward.

The pool keeps at most ``capacity`` alphas. Each member is z-scored cross-sectionally
every day and the members are combined linearly. ``weighting`` picks the weights:

  * ``ridge`` : ridge regression of the z-scored train target on the members (AlphaGen-style
                linear combiner, refit on every change; frozen at evaluation time)
  * ``ic``    : ``w_i = IC_i`` (signed, so a negatively-predictive alpha is flipped)
  * ``equal`` : ``w_i = sign(IC_i)``

Adding a candidate:
  1. invalid / constant / mostly-NaN                  -> penalty
  2. near-duplicate of a member (|mutual IC| > thr)   -> penalty
  3. otherwise tentatively add; if over capacity drop one member (possibly the newcomer):
     the smallest |ridge weight| for ``ridge``, else the member whose removal hurts the
     ensemble IC least.
Reward = scale * (Delta ensemble IC + beta * novelty bonus) - gamma * length penalty.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .expr import Node, evaluate
from .grammar import build_from_keys
from .metrics import daily_ic, daily_rank_ic, nanmean, summarize, zscore_rows
from .vocab import Vocab


@dataclass
class Member:
    node: Node
    keys: List[str]
    ic: float
    z: torch.Tensor                      # filled z-score on the train slice [T_tr, N]


@dataclass
class AddResult:
    reward: float
    status: str                          # invalid | duplicate | rejected | accepted
    ic: float = float("nan")
    max_mutual: float = float("nan")
    delta_obj: float = 0.0
    parts: Dict[str, float] = field(default_factory=dict)


class AlphaPool:
    def __init__(self, market, cfg, vocab: Vocab):
        self.mkt = market
        self.cfg = cfg
        self.vocab = vocab
        self.capacity = cfg.pool_capacity
        self.lo, self.hi = market.ic_slice("train")
        self.y = market.target[self.lo:self.hi]
        yz = zscore_rows(self.y)
        self._ymask = torch.isfinite(yz).to(self.y.dtype)
        self._yz = torch.nan_to_num(yz, nan=0.0)
        self.members: List[Member] = []
        self.w = torch.zeros(0, device=market.device)     # current combination weights
        self._G = torch.zeros(0, 0, device=market.device)  # member Gram matrix (ridge)
        self._b = torch.zeros(0, device=market.device)     # member-target products (ridge)
        self.obj = 0.0                   # ensemble IC on train slice
        self.eval_count = 0
        self.n_accepted = 0

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.members)

    def _weights(self, ics: List[float]) -> torch.Tensor:
        w = torch.tensor(ics, dtype=torch.float32, device=self.mkt.device)
        return torch.sign(w) if self.cfg.weighting == "equal" else w

    def weights(self) -> np.ndarray:
        return self.w.cpu().numpy()

    def _ridge(self, G: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        k = G.shape[0]
        lam = self.cfg.ridge_lambda * float(G.diagonal().mean().clamp_min(1e-12))
        return torch.linalg.solve(G + lam * torch.eye(k, dtype=G.dtype, device=G.device), b)

    def _fit(self, ics: List[float], G: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._ridge(G, b) if self.cfg.weighting == "ridge" else self._weights(ics)

    def _stats(self, zf: torch.Tensor):
        """Gram row of ``zf`` against the members, its own norm and its target product."""
        zm = zf * self._ymask
        row = torch.stack([(zm * m.z).sum() for m in self.members]) if self.members \
            else torch.zeros(0, device=zf.device)
        return row, (zm * zf).sum(), (zm * self._yz).sum()

    def _refresh(self) -> None:
        """Recompute Gram/target products and weights from scratch (used after load)."""
        k = len(self.members)
        self._G = torch.zeros(k, k, device=self.mkt.device)
        self._b = torch.zeros(k, device=self.mkt.device)
        for i, m in enumerate(self.members):
            zm = m.z * self._ymask
            self._b[i] = (zm * self._yz).sum()
            for j in range(i + 1):
                self._G[i, j] = self._G[j, i] = (zm * self.members[j].z).sum()
        self.w = self._fit([m.ic for m in self.members], self._G, self._b) if k \
            else torch.zeros(0, device=self.mkt.device)

    def ensemble(self) -> torch.Tensor:
        """Current ensemble on the train slice."""
        return torch.einsum("k,ktn->tn", self.w, torch.stack([m.z for m in self.members]))

    def _objective(self, ensemble: torch.Tensor) -> float:
        ic = daily_ic(ensemble, self.y)
        k = self.cfg.robust_folds
        if k >= 2:
            # stability-aware objective: mean - lambda * std of the IC over contiguous train folds
            folds = [nanmean(c) for c in torch.tensor_split(ic, k)]
            folds = [f for f in folds if f == f]
            if len(folds) >= 2:
                f = np.asarray(folds)
                return float(f.mean() - self.cfg.robust_lambda * f.std())
        v = nanmean(ic)
        return 0.0 if v != v else v

    # ------------------------------------------------------------------
    def _candidate(self, node: Node):
        try:
            a = evaluate(node, self.mkt, self.hi)
        except Exception:
            return None
        if not torch.is_tensor(a) or tuple(a.shape) != (self.hi, self.mkt.N):
            return None
        z = zscore_rows(a[self.lo:self.hi])
        if float(torch.isfinite(z).float().mean()) < 0.3:
            return None
        ic = nanmean(daily_ic(z, self.y))
        if ic != ic:
            return None
        return z, ic

    def try_add(self, node: Node, n_tokens: int, budget: int) -> AddResult:
        cfg = self.cfg
        self.eval_count += 1
        cand = self._candidate(node)
        if cand is None:
            return AddResult(-cfg.invalid_penalty, "invalid")
        z, ic = cand
        zf = torch.nan_to_num(z, nan=0.0)

        mutual = [abs(nanmean(daily_ic(zf, m.z))) for m in self.members]
        mutual = [0.0 if v != v else v for v in mutual]
        max_mut = max(mutual) if mutual else 0.0
        if max_mut > cfg.dup_threshold:
            return AddResult(-cfg.dup_penalty, "duplicate", ic, max_mut)

        ics = [m.ic for m in self.members] + [ic]
        Z = torch.stack([m.z for m in self.members] + [zf])
        K = len(ics)
        row, own, bn = self._stats(zf)
        G = torch.zeros(K, K, device=zf.device)
        G[:-1, :-1], G[-1, :-1], G[:-1, -1], G[-1, -1] = self._G, row, row, own
        b = torch.cat([self._b, bn.view(1)])
        w = self._fit(ics, G, b)
        ens = torch.einsum("k,ktn->tn", w, Z)

        drop: Optional[int] = None
        w_new = w
        if K <= self.capacity:
            obj_new = self._objective(ens)
            retained = True
        elif cfg.weighting == "ridge":
            j = int(w.abs().argmin())
            if j == K - 1:                   # newcomer gets the smallest weight -> pool unchanged
                obj_new, retained = self.obj, False
            else:
                keep = [i for i in range(K) if i != j]
                w_new = self._fit([ics[i] for i in keep], G[keep][:, keep], b[keep])
                obj_new = self._objective(torch.einsum("k,ktn->tn", w_new, Z[keep]))
                retained, drop = True, j
        else:
            best_obj, best_j = -1e9, -1
            for j in range(K):
                o = self._objective(ens - w[j] * Z[j])
                if o > best_obj:
                    best_obj, best_j = o, j
            if best_j == K - 1:              # newcomer is the weakest link -> pool unchanged
                obj_new, retained = self.obj, False
            else:
                obj_new, retained, drop = best_obj, True, best_j
                keep = [i for i in range(K) if i != drop]
                w_new = self._weights([ics[i] for i in keep])

        delta = obj_new - self.obj
        r_delta = delta
        if cfg.dense_reward and not retained:
            # saturated pool: still tell the agent how much the rejected candidate would have changed the
            # ensemble (<= 0, graded) instead of a flat zero
            r_delta = self._objective(ens) - self.obj
        novelty = cfg.beta_div * (1.0 - max_mut) * abs(ic) if (retained or cfg.dense_reward) else 0.0
        length_pen = cfg.gamma_len * n_tokens / max(budget, 1)
        reward = cfg.reward_scale * (r_delta + novelty) - length_pen

        if retained:
            self.members.append(Member(node, [], ic, zf))   # keys filled by caller
            keep = list(range(K))
            if drop is not None:
                keep.remove(drop)
                # keep the newcomer last so ``set_last_keys`` still addresses it
                self.members = [self.members[i] for i in keep]
            self._G, self._b, self.w = G[keep][:, keep], b[keep], w_new
            self.obj = obj_new
            self.n_accepted += 1
        parts = {"delta": cfg.reward_scale * r_delta, "novelty": cfg.reward_scale * novelty, "length": -length_pen}
        return AddResult(reward, "accepted" if retained else "rejected", ic, max_mut, delta, parts)

    def set_last_keys(self, keys: List[str]) -> None:
        if self.members:
            self.members[-1].keys = list(keys)

    # ------------------------------------------------------------------
    # evaluation on any split (weights stay frozen at their training values)
    # ------------------------------------------------------------------
    def member_signals(self, split: str) -> List[torch.Tensor]:
        lo, hi = self.mkt.ic_slice(split)
        return [
            torch.nan_to_num(zscore_rows(evaluate(m.node, self.mkt, hi)[:hi]), nan=0.0)
            for m in self.members
        ]

    def signal(self, split: str) -> torch.Tensor:
        """Ensemble signal ``[hi, N]`` for days ``[0, hi)`` of the split's scoring window."""
        sigs = self.member_signals(split)
        return torch.einsum("k,ktn->tn", self.w, torch.stack(sigs))

    def evaluate_split(self, split: str) -> Dict[str, float]:
        if not self.members:
            return {"ic": float("nan"), "rank_ic": float("nan"), "icir": float("nan"), "n_days": 0}
        lo, hi = self.mkt.ic_slice(split)
        sig = self.signal(split)[lo:hi]
        y = self.mkt.target[lo:hi]
        out = summarize(daily_ic(sig, y))
        out["rank_ic"] = nanmean(daily_rank_ic(sig, y))
        return out

    # ------------------------------------------------------------------
    def to_json(self) -> Dict:
        return {
            "objective_train": self.obj,
            "capacity": self.capacity,
            "weighting": self.cfg.weighting,
            "members": [
                {"formula": str(m.node), "keys": m.keys, "train_ic": m.ic, "weight": float(w)}
                for m, w in zip(self.members, self.weights())
            ],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(cls, market, cfg, vocab: Vocab, path: str) -> "AlphaPool":
        """Rebuild a pool from JSON; member ICs are recomputed on the train slice."""
        with open(path) as f:
            blob = json.load(f)
        pool = cls(market, cfg, vocab)
        for item in blob["members"]:
            node = build_from_keys(vocab, item["keys"])
            cand = pool._candidate(node)
            if cand is None:
                continue
            z, ic = cand
            pool.members.append(Member(node, item["keys"], ic, torch.nan_to_num(z, nan=0.0)))
        pool._refresh()
        if pool.members:
            pool.obj = pool._objective(pool.ensemble())
        return pool
