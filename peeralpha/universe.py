"""Stock-universe providers for the real-data download.

Each provider returns ``List[str]`` of Baostock codes (``sh.600519`` / ``sz.000858``).
``resolve_universe`` tries them in order, so one flaky endpoint cannot block a run:

    1. ``file``     -- user-supplied text/CSV (one code per line, or a ``code`` column)
    2. ``baostock`` -- live CSI 300 constituents (``query_hs300_stocks``), with a timeout
    3. ``builtin``  -- a fixed list of large caps below

Caveats (state them in the report): both ``baostock`` and ``builtin`` use *current*
membership, which is survivorship-biased; ``builtin`` is a ~100-stock subset of CSI 300-class
large caps, not the exact index. Codes that have no data are skipped by the downloader.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional

BUILTIN_LARGE_CAPS: List[str] = [
    # --- Shanghai ---
    "sh.600000", "sh.600009", "sh.600011", "sh.600015", "sh.600016", "sh.600019", "sh.600028",
    "sh.600029", "sh.600030", "sh.600031", "sh.600036", "sh.600048", "sh.600050", "sh.600104",
    "sh.600115", "sh.600196", "sh.600276", "sh.600309", "sh.600436", "sh.600519", "sh.600570",
    "sh.600585", "sh.600690", "sh.600703", "sh.600837", "sh.600887", "sh.600900", "sh.601006",
    "sh.601012", "sh.601088", "sh.601111", "sh.601166", "sh.601169", "sh.601186", "sh.601211",
    "sh.601288", "sh.601318", "sh.601328", "sh.601390", "sh.601398", "sh.601601", "sh.601628",
    "sh.601633", "sh.601668", "sh.601688", "sh.601818", "sh.601857", "sh.601888", "sh.601899",
    "sh.601939", "sh.601988",
    # --- Shenzhen ---
    "sz.000001", "sz.000002", "sz.000063", "sz.000069", "sz.000100", "sz.000157", "sz.000166",
    "sz.000333", "sz.000338", "sz.000425", "sz.000538", "sz.000568", "sz.000625", "sz.000651",
    "sz.000725", "sz.000768", "sz.000776", "sz.000786", "sz.000858", "sz.000895", "sz.000963",
    "sz.002024", "sz.002027", "sz.002142", "sz.002230", "sz.002304", "sz.002352", "sz.002415",
    "sz.002460", "sz.002475", "sz.002594", "sz.002714", "sz.300059", "sz.300750",
]


def from_file(path: str) -> List[str]:
    codes: List[str] = []
    with open(path, "r") as f:
        for line in f:
            tok = line.strip().split(",")[0].strip().strip('"')
            if tok.lower() in ("", "code"):
                continue
            codes.append(tok)
    if not codes:
        raise ValueError(f"no codes found in {path}")
    return sorted(set(codes))


def from_baostock(bs, query: Callable) -> List[str]:
    cons = query(lambda: bs.query_hs300_stocks())
    return sorted(cons["code"].tolist())


def resolve_universe(
    bs,
    query: Callable,
    codes_file: Optional[str] = None,
    source: str = "auto",
    cache_path: Optional[str] = None,
) -> List[str]:
    """source: auto | file | baostock | builtin. ``auto`` = file (if given) -> cache -> baostock -> builtin."""
    def log(msg: str) -> None:
        print(f"[universe] {msg}", flush=True)

    if source == "builtin":
        log(f"using built-in list ({len(BUILTIN_LARGE_CAPS)} large caps)")
        return list(BUILTIN_LARGE_CAPS)

    if source in ("auto", "file") and codes_file:
        codes = from_file(codes_file)
        log(f"{len(codes)} codes from {codes_file}")
        return codes
    if source == "file":
        raise ValueError("--universe file requires --codes_file")

    if source == "auto" and cache_path and os.path.exists(cache_path):
        codes = from_file(cache_path)
        log(f"{len(codes)} codes from cached constituent list {cache_path}")
        return codes

    try:
        codes = from_baostock(bs, query)
        log(f"{len(codes)} CSI 300 constituents from Baostock")
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "w") as f:
                f.write("\n".join(codes) + "\n")
        return codes
    except Exception as e:
        if source == "baostock":
            raise
        log(f"Baostock constituent query failed ({e}); falling back to built-in list "
            f"({len(BUILTIN_LARGE_CAPS)} large caps)")
        return list(BUILTIN_LARGE_CAPS)
