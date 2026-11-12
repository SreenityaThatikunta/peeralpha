"""Kaggle kernel script: one walk-forward window on Kaggle CPU.

Upload the repo (code + ``data/nse100.npz``) as a private dataset, attach it to a script kernel
with internet on, set ``WINDOW``/``TAG``/``METHODS`` and push one kernel per window. The kernel
saves ``runs/`` and ``reports/`` as ``/kaggle/working/<TAG>_w<WINDOW>.tar.gz``; fetch it with
``kaggle kernels output <user>/<kernel> -p <dir>`` and evaluate all windows together locally:

    python evaluate.py runs/<TAG>_w*_s? --group method --baseline <baseline> --hand --out reports/<TAG>_wf
"""
import glob
import os
import shutil
import subprocess
import sys
import time

WINDOW = 0
TAG = "v4nse"
METHODS = ["v4_sector_prior", "v4_sector", "v4_nograph", "v4_random_prior"]
BASELINE = "v4_nograph"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gymnasium==0.29.1",
                "stable-baselines3==2.3.2", "sb3-contrib==2.3.0"], check=True)
src = os.path.dirname(sorted(glob.glob("/kaggle/input/**/run_suite.py", recursive=True))[0])
print("source:", src, flush=True)
work = "/kaggle/working/peeralpha"
shutil.copytree(src, work, dirs_exist_ok=True)
subprocess.run(["chmod", "-R", "u+w", work])
print("cpus:", os.cpu_count(), flush=True)
cmd = [sys.executable, "scripts/walk_forward.py", "--data", "data/nse100.npz", "--tag", TAG,
       "--seeds", "0", "1", "2", "--valid_years", "2", "--methods", *METHODS,
       "--steps", "300000", "--baseline", BASELINE, "--hand",
       "--jobs", "4", "--threads", "1", "--windows", str(WINDOW)]
t0 = time.time()
try:
    rc = subprocess.run(cmd, cwd=work, timeout=11 * 3600).returncode
except subprocess.TimeoutExpired:
    rc = "timeout"
print(f"walk_forward rc={rc} after {(time.time() - t0) / 3600:.2f}h", flush=True)
out = "/kaggle/working/out"
os.makedirs(out, exist_ok=True)
for d in ("runs", "reports"):
    if os.path.exists(os.path.join(work, d)):
        shutil.copytree(os.path.join(work, d), os.path.join(out, d), dirs_exist_ok=True)
shutil.rmtree(work)
subprocess.run(["tar", "czf", f"/kaggle/working/{TAG}_w{WINDOW}.tar.gz", "-C", out, "."])
shutil.rmtree(out)
print("saved", os.listdir("/kaggle/working"), flush=True)
