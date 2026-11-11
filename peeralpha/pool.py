"""Alpha pool: the part of the environment that turns a finished formula into a reward.

The pool keeps at most ``capacity`` alphas. Each member is z-scored cross-sectionally
every day and the members are combined linearly. ``weighting`` picks the weights:

  * ``ridge`` : ridge regression of the z-scored train target on the members (AlphaGen-style
                linear combiner, refit on every change; frozen at evaluation time)
  * ``ic``    : ``w_i = IC_i`` (signed, so a negatively-predictive alpha is flipped)
  * ``equal`` : ``w_i = sign(IC_i)``

Adding a candidate:
  1. invalid / constant / mostly-NaN                  -> penalty
  2. IC sign unstable across train folds (optional)   -> penalty
  3. near-duplicate of a member (|mutual IC| > thr)   -> penalty
  4. otherwise tentatively add; if over capacity drop one member (possibly the newcomer):
     the smallest |ridge weight| for ``ridge``, else the member whose removal hurts the
     ensemble IC least.
Reward = scale * (Delta ensemble IC + beta * novelty bonus) - gamma * length penalty.

With ``oof_folds >= 2`` the ensemble IC in the reward is out-of-fold: the train slice is cut
into contiguous folds (the last ``horizon`` days of each are purged), and each fold is scored
with ridge weights fitted on the other folds. A member only earns reward if it helps on data
the combiner did not see. The weights used at evaluation time are still fitted on all of train.
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
    status: str                          # invalid | unstable | duplicate | rejected | accepted
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
        if cfg.oof_folds >= 2 and cfg.weighting != "ridge":
            raise ValueError("oof_folds needs weighting: ridge")
        self._M, self._fold_of = self._part_masks(self.hi - self.lo, cfg.oof_folds, market.horizon, market.device)
        P = self._M.shape[1]
        self.members: List[Member] = []
        self.w = torch.zeros(0, device=market.device)     # current combination weights
        self._G = torch.zeros(P, 0, 0, device=market.device)  # member Gram matrix per part (ridge)
        self._b = torch.zeros(P, 0, device=market.device)     # member-target products per part (ridge)
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

    @staticmethod
    def _part_masks(T: int, folds: int, horizon: int, device):
        """Day masks ``[T, P]``: column 0 is the whole train slice, then one column per OOF fold.
        Fold ``f`` holds a contiguous block minus its last ``horizon`` days, so no fold's target
        window overlaps another fold. ``fold_of[t]`` is the fold of day ``t`` (``folds`` if purged)."""
        if folds < 2:
            return torch.ones(T, 1, device=device), None
        fold_of = torch.full((T,), folds, dtype=torch.long, device=device)
        for f, idx in enumerate(torch.tensor_split(torch.arange(T, device=device), folds)):
            fold_of[idx[: max(len(idx) - horizon, 0)]] = f
        M = torch.zeros(T, 1 + folds, device=device)
        M[:, 0] = 1.0
        M[fold_of < folds, 1 + fold_of[fold_of < folds]] = 1.0
        return M, fold_of

    def _ridge(self, G: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        k = G.shape[0]
        lam = self.cfg.ridge_lambda * float(G.diagonal().mean().clamp_min(1e-12))
        return torch.linalg.solve(G + lam * torch.eye(k, dtype=G.dtype, device=G.device), b)

    def _fit(self, ics: List[float], G: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Weights fitted on the whole train slice (part 0 of ``G``/``b``)."""
        return self._ridge(G[0], b[0]) if self.cfg.weighting == "ridge" else self._weights(ics)

    def _stats(self, zf: torch.Tensor, members: Optional[List[Member]] = None):
        """Per-part Gram row of ``zf`` against the members ``[P, k]``, its own norm ``[P]`` and
        its target product ``[P]``."""
        members = self.members if members is None else members
        zm = zf * self._ymask
        if members:
            row = torch.stack([(zm * m.z).sum(1) for m in members]) @ self._M        # [k, P]
        else:
            row = torch.zeros(0, self._M.shape[1], device=zf.device)
        return row.T, (zm * zf).sum(1) @ self._M, (zm * self._yz).sum(1) @ self._M

    @staticmethod
    def _grow(G: torch.Tensor, b: torch.Tensor, row: torch.Tensor, own: torch.Tensor, bn: torch.Tensor):
        """Append one member to per-part Gram ``[P, k, k]`` and target products ``[P, k]``."""
        P, k = G.shape[0], G.shape[1]
        G2 = torch.zeros(P, k + 1, k + 1, device=G.device)
        G2[:, :k, :k], G2[:, k, :k], G2[:, :k, k], G2[:, k, k] = G, row, row, own
        return G2, torch.cat([b, bn.view(P, 1)], dim=1)

    def _refresh(self) -> None:
        """Recompute Gram/target products and weights from scratch (used after load)."""
        P = self._M.shape[1]
        G = torch.zeros(P, 0, 0, device=self.mkt.device)
        b = torch.zeros(P, 0, device=self.mkt.device)
        for i, m in enumerate(self.members):
            G, b = self._grow(G, b, *self._stats(m.z, self.members[:i]))
        self._G, self._b = G, b
        self.w = self._fit([m.ic for m in self.members], G, b) if self.members \
            else torch.zeros(0, device=self.mkt.device)

    def _score(self, w: torch.Tensor, G: torch.Tensor, b: torch.Tensor, Z: torch.Tensor) -> float:
        """Pool objective for members ``Z`` with full-train weights ``w`` (out-of-fold if enabled)."""
        if self._fold_of is None:
            return self._objective(torch.einsum("k,ktn->tn", w, Z))
        F = self._M.shape[1] - 1
        rest_G, rest_b = G[1:].sum(0), b[1:].sum(0)
        Wf = torch.stack([self._ridge(rest_G - G[1 + f], rest_b - b[1 + f]) for f in range(F)])
        Wf = torch.cat([Wf, torch.zeros(1, Wf.shape[1], device=Wf.device)])      # purged days -> 0
        ens = torch.einsum("tk,ktn->tn", Wf[self._fold_of], Z)
        v = nanmean(daily_ic(ens, self.y))
        return 0.0 if v != v else v

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
    def _stable(self, ic_days: torch.Tensor, ic: float) -> bool:
        k = self.cfg.stable_folds
        if k < 2:
            return True
        agree = sum(1 for c in torch.tensor_split(ic_days, k) if (m := nanmean(c)) == m and m * ic > 0)
        return agree >= int(np.ceil(self.cfg.stable_min_frac * k - 1e-9))

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
        ic_days = daily_ic(z, self.y)
        ic = nanmean(ic_days)
        if ic != ic:
            return None
        return z, ic, ic_days

    def try_add(self, node: Node, n_tokens: int, budget: int) -> AddResult:
        cfg = self.cfg
        self.eval_count += 1
        cand = self._candidate(node)
        if cand is None:
            return AddResult(-cfg.invalid_penalty, "invalid")
        z, ic, ic_days = cand
        if not self._stable(ic_days, ic):
            return AddResult(-cfg.unstable_penalty, "unstable", ic)
        zf = torch.nan_to_num(z, nan=0.0)

        mutual = [abs(nanmean(daily_ic(zf, m.z))) for m in self.members]
        mutual = [0.0 if v != v else v for v in mutual]
        max_mut = max(mutual) if mutual else 0.0
        if max_mut > cfg.dup_threshold:
            return AddResult(-cfg.dup_penalty, "duplicate", ic, max_mut)

        ics = [m.ic for m in self.members] + [ic]
        Z = torch.stack([m.z for m in self.members] + [zf])
        K = len(ics)
        G, b = self._grow(self._G, self._b, *self._stats(zf))
        w = self._fit(ics, G, b)
        ens = torch.einsum("k,ktn->tn", w, Z)

        drop: Optional[int] = None
        w_new = w
        if K <= self.capacity:
            obj_new = self._score(w, G, b, Z)
            retained = True
        elif cfg.weighting == "ridge":
            j = int(w.abs().argmin())
            if j == K - 1:                   # newcomer gets the smallest weight -> pool unchanged
                obj_new, retained = self.obj, False
            else:
                keep = [i for i in range(K) if i != j]
                Gk, bk = G[:, keep][:, :, keep], b[:, keep]
                w_new = self._fit([ics[i] for i in keep], Gk, bk)
                obj_new = self._score(w_new, Gk, bk, Z[keep])
                retained, drop = True, j
                if cfg.swap_needs_gain and obj_new <= self.obj:   # a swap that lowers the objective is refused
                    obj_new, retained, drop, w_new = self.obj, False, None, w
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
            r_delta = self._score(w, G, b, Z) - self.obj
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
            self._G, self._b, self.w = G[:, keep][:, :, keep], b[:, keep], w_new
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
            z, ic, _ = cand
            pool.members.append(Member(node, item["keys"], ic, torch.nan_to_num(z, nan=0.0)))
        pool._refresh()
        if pool.members:
            pool.obj = pool._score(pool.w, pool._G, pool._b, torch.stack([m.z for m in pool.members]))
        return pool
