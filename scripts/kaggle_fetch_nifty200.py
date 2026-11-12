"""Kaggle kernel script: build data/nifty200.npz (Nifty 200 constituents, NSE industry labels) from Yahoo.

Attach the repo dataset, turn internet on; outputs nifty200.npz and ind_nifty200list.csv in /kaggle/working.
"""
import glob, os, shutil, subprocess, sys, urllib.request
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance"], check=True)
src = os.path.dirname(sorted(glob.glob("/kaggle/input/**/run_suite.py", recursive=True))[0])
work = "/kaggle/working/peeralpha"
shutil.copytree(src, work, dirs_exist_ok=True)
subprocess.run(["chmod", "-R", "u+w", work])
csv = os.path.join(work, "data", "ind_nifty200list.csv")
urls = ["https://niftyindices.com/IndexConstituent/ind_nifty200list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"]
for u in urls:
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"})
        data = urllib.request.urlopen(req, timeout=30).read()
        if b"Symbol" in data[:300]:
            open(csv, "wb").write(data)
            print("universe from", u, flush=True)
            break
    except Exception as e:
        print("failed", u, e, flush=True)
else:
    raise SystemExit("could not download the Nifty 200 list")
rc = subprocess.run([sys.executable, "scripts/fetch_nse.py", "--universe_csv", csv, "--out", "data/nifty200.npz"],
                    cwd=work).returncode
print("fetch rc", rc, flush=True)
for f in ("data/nifty200.npz", "data/ind_nifty200list.csv"):
    if os.path.exists(os.path.join(work, f)):
        shutil.copy(os.path.join(work, f), "/kaggle/working/")
shutil.rmtree(work)
print("saved", os.listdir("/kaggle/working"), flush=True)
