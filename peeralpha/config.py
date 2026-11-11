"""Experiment configuration (single dataclass, YAML-loadable, CLI-overridable)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml


# behaviour of runs saved before a key existed (only keys that change formula values)
LEGACY_DEFAULTS: Dict[str, Any] = {"gop_zscore": False, "graph_residual": False, "min_sector_size": 1}


@dataclass
class Config:
    # ---- bookkeeping -------------------------------------------------
    name: str = "graph_all"
    seed: int = 0
    out_dir: str = "runs"
    agent: str = "ppo"                  # "ppo" | "random"

    # ---- data --------------------------------------------------------
    data_path: str = "data/csi300.npz"
    horizon: int = 20                   # forward-return horizon (days)
    train: List[str] = field(default_factory=lambda: ["2012-01-01", "2019-12-31"])
    valid: List[str] = field(default_factory=lambda: ["2020-01-01", "2021-12-31"])
    test: List[str] = field(default_factory=lambda: ["2022-01-01", "2024-06-30"])

    # ---- graphs ------------------------------------------------------
    graphs: List[str] = field(default_factory=lambda: ["sector", "corr", "lead"])
    graph_k: int = 8                    # neighbours kept per stock
    graph_window: int = 60              # rolling window for dynamic graphs
    graph_step: int = 5                 # graph refresh period (days)
    graph_residual: bool = False        # build corr/lead graphs on sector-demeaned returns
    gop_zscore: bool = True             # z-score graph-operator inputs per day (scale-free peers)
    min_sector_size: int = 3            # smaller sectors are merged into the closest large one

    # ---- formula language / environment ------------------------------
    max_len: int = 15
    min_tokens: int = 3                 # <SEP> masked before this many tokens (no bare-feature alphas)
    features: List[str] = field(default_factory=list)   # leaf features; empty = v2 set (prices + ret, ret5, vrel)
    curriculum: List[List[Any]] = field(
        default_factory=lambda: [[0, 9, True], [2000, 12, True], [6000, 15, True]]
    )  # rows: [episode_threshold, max_len, graph_ops_allowed]

    # ---- alpha pool / reward -----------------------------------------
    pool_capacity: int = 20
    weighting: str = "ridge"            # "ridge" | "ic" | "equal"
    ridge_lambda: float = 0.1           # ridge strength, relative to the mean Gram diagonal
    dup_threshold: float = 0.9
    reward_scale: float = 10.0
    beta_div: float = 0.3               # diversity bonus weight
    gamma_len: float = 0.0              # complexity penalty weight (>0 drove the 1-token collapse)
    invalid_penalty: float = 0.1
    dup_penalty: float = 0.1
    repeat_penalty: float = 0.1         # formula already submitted earlier in this run
    dense_reward: bool = False          # rejected candidates still get graded reward (fixes frozen-pool zero signal)
    robust_folds: int = 0               # >=2: reward uses mean(fold IC) - lambda*std(fold IC) over train folds
    robust_lambda: float = 1.0          # penalty on IC instability across train sub-periods
    oof_folds: int = 0                  # >=2: reward IC is out-of-fold (ridge fit without the fold it is scored on)
    stable_folds: int = 0               # >=2: candidate IC must keep its sign in most of these train folds
    stable_min_frac: float = 0.75       # share of stable_folds that must agree with the overall IC sign
    unstable_penalty: float = 0.05
    swap_needs_gain: bool = False       # full ridge pool: only swap a member out if the objective improves

    # ---- policy / PPO ------------------------------------------------
    policy: str = "gat"                 # "gat" | "lstm"
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    total_timesteps: int = 300_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 128
    n_epochs: int = 4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    ent_coef: float = 0.05
    clip_range: float = 0.2

    # ---- devices -----------------------------------------------------
    compute_device: str = "auto"        # formula evaluation (GPU if available)
    policy_device: str = "cpu"
    freeze_patience: int = 0            # >0: stop once the pool has not changed for this many log rows

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.data_path == "data/csi300.npz" and not os.path.exists(cfg.data_path) and os.path.exists("data/synthetic.npz"):
            cfg.data_path = "data/synthetic.npz"
        return cfg

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[List[str]] = None) -> "Config":
        with open(path, "r") as f:
            cfg = cls.from_dict(yaml.safe_load(f) or {})
        if overrides:
            cfg.apply_overrides(overrides)
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load a saved run config; keys it predates get the value that run actually used."""
        with open(path, "r") as f:
            d = json.load(f)
        for k, v in LEGACY_DEFAULTS.items():
            d.setdefault(k, v)
        return cls.from_dict(d)

    def apply_overrides(self, items: List[str]) -> None:
        """Apply ``key=value`` strings (value parsed as YAML)."""
        valid = {f.name for f in fields(self)}
        for item in items:
            key, _, raw = item.partition("=")
            key = key.strip()
            if key not in valid:
                raise KeyError(f"Unknown config key: {key}")
            setattr(self, key, yaml.safe_load(raw))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @property
    def use_graph_ops(self) -> bool:
        return len(self.graphs) > 0

    @property
    def run_name(self) -> str:
        return f"{self.name}_s{self.seed}"
