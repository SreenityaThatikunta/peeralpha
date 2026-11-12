"""Token vocabulary = the RL agent's discrete action space."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .ops import CONSTANTS, DEFAULT_FEATURES, FEATURES, REGISTRY, WINDOWS

SEP = "<SEP>"


@dataclass(frozen=True)
class Tok:
    id: int
    kind: str       # unary|binary|rolling|pair|gop|feat|const|win|graph|sep
    name: str       # printable token name
    value: float = 0.0


class Vocab:
    """Ordered token table. Graph operators/tokens exist only if graphs are enabled.

    ``features`` picks the leaf features (default: the v2 set, so old runs keep their token ids),
    ``windows`` the rolling windows (default ``WINDOWS``) and ``ops`` the operators (default: all).
    """

    def __init__(self, graphs: List[str], features: Optional[List[str]] = None,
                 windows: Optional[List[int]] = None, ops: Optional[List[str]] = None):
        self.graphs = list(graphs)
        chosen = set(features or DEFAULT_FEATURES)
        unknown = chosen - set(FEATURES)
        if unknown:
            raise KeyError(f"Unknown features: {sorted(unknown)}")
        allowed = set(ops or REGISTRY)
        unknown = allowed - set(REGISTRY)
        if unknown:
            raise KeyError(f"Unknown ops: {sorted(unknown)}")
        toks: List[Tok] = []

        def add(kind: str, name: str, value: float = 0.0) -> None:
            toks.append(Tok(len(toks), kind, name, value))

        for spec in REGISTRY.values():
            if (spec.kind == "gop" and not graphs) or spec.name not in allowed:
                continue
            add(spec.kind, spec.name)
        for f in FEATURES:
            if f in chosen:
                add("feat", f)
        for c in CONSTANTS:
            add("const", f"{c:g}", c)
        for w in (windows or WINDOWS):
            add("win", f"{w}", float(w))
        for g in graphs:
            add("graph", g)
        add("sep", SEP)

        self.tokens: List[Tok] = toks
        self.by_name: Dict[str, Tok] = {}
        for t in toks:
            key = t.name if t.kind in ("unary", "binary", "rolling", "pair", "gop", "sep") else f"{t.kind}:{t.name}"
            self.by_name[key] = t
        self.groups: Dict[str, List[int]] = {}
        for t in toks:
            self.groups.setdefault(t.kind, []).append(t.id)
        self.sep_id = self.groups["sep"][0]

    @classmethod
    def from_config(cls, cfg) -> "Vocab":
        return cls(cfg.graphs, cfg.features, cfg.windows or None, cfg.ops or None)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int) -> Tok:
        return self.tokens[i]

    def key(self, tok: Tok) -> str:
        """Stable string used when saving a formula (inverse of ``lookup``)."""
        return tok.name if tok.kind in ("unary", "binary", "rolling", "pair", "gop", "sep") else f"{tok.kind}:{tok.name}"

    def lookup(self, key: str) -> Tok:
        return self.by_name[key]
