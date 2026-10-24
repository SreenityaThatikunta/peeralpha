"""Walk-forward evaluation: run the same suite on several expanding train/valid/test windows.

    python scripts/walk_forward.py --data data/nse100.npz --tag v2nse --seeds 0 1 2 \
        --methods v2_nograph v2_sector v2_graph_all v2_random_nograph --steps 300000 --jobs 2

Each window ``i`` runs ``run_suite.py`` with tag ``<tag>_w<i>`` (finished runs are skipped, so
this is safe to re-run), then all windows are evaluated together into ``reports/<tag>_wf/``
with one row per method and a paired comparison against ``--baseline``.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# expanding windows over 2012-2024: (train, valid, test)
WINDOWS = [
    (["2012-01-01", "2016-12-31"], ["2017-01-01", "2017-12-31"], ["2018-01-01", "2019-12-31"]),
    (["2012-01-01", "2018-12-31"], ["2019-01-01", "2019-12-31"], ["2020-01-01", "2021-12-31"]),
    (["2012-01-01", "2020-12-31"], ["2021-01-01", "2021-12-31"], ["2022-01-01", "2024-06-30"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--windows", nargs="+", type=int, default=list(range(len(WINDOWS))))
    ap.add_argument("--baseline", default="v2_nograph")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out_dir", default="runs")
    args = ap.parse_args()

    run_dirs = []
    for i in args.windows:
        train, valid, test = WINDOWS[i]
        tag = f"{args.tag}_w{i}"
        overrides = [f"train={json.dumps(train)}", f"valid={json.dumps(valid)}", f"test={json.dumps(test)}"]
        cmd = [sys.executable, "run_suite.py", "--data", args.data, "--tag", tag,
               "--methods", *args.methods, "--seeds", *map(str, args.seeds), "--steps", str(args.steps),
               "--jobs", str(args.jobs), "--threads", str(args.threads), "--out_dir", args.out_dir,
               "--no_eval", "--set", *overrides, *args.set]
        print(f"[wf] window {i}: test {test[0]}..{test[1]}", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        for m in args.methods:
            run_dirs += sorted(glob.glob(os.path.join(ROOT, args.out_dir, f"{tag}_{m}_s*[0-9]")))

    done = [d for d in run_dirs if os.path.exists(os.path.join(d, "pool_best_valid.json"))]
    cmd = [sys.executable, "evaluate.py", *done, "--out", os.path.join("reports", f"{args.tag}_wf"),
           "--group", "method", "--baseline", args.baseline]
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
