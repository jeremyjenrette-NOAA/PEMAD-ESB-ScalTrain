#!/usr/bin/env python3
"""
Upload the staged dataset to a Zenodo DRAFT (never publishes).

  export ZENODO_TOKEN=...            # personal token with deposit:write (never commit it)
  python config/zenodo/zenodo_upload.py --stage zenodo_stage --check          # sizes/limits only
  python config/zenodo/zenodo_upload.py --stage zenodo_stage                  # create draft + upload
  python config/zenodo/zenodo_upload.py --stage zenodo_stage --deposition ID  # resume / add files

Files are uploaded flat (Zenodo has no folders): weights/x.pt -> weights__x.pt, etc.
Re-running skips files already in the draft with a matching MD5. Use --sandbox to test first.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True)
ap.add_argument("--metadata", default=str(Path(__file__).with_name("metadata.json")))
ap.add_argument("--deposition", default=None, help="Existing draft ID to resume")
ap.add_argument("--sandbox", action="store_true")
ap.add_argument("--check", action="store_true", help="Only list files and check limits")
a = ap.parse_args()

API = "https://sandbox.zenodo.org/api" if a.sandbox else "https://zenodo.org/api"
stage = Path(a.stage)
files = sorted(p for p in stage.rglob("*") if p.is_file() and not p.name.startswith("."))
names = {p: "__".join(p.relative_to(stage).parts) for p in files}
total = sum(p.stat().st_size for p in files)
print(f"{len(files)} files, {total/1e9:.2f} GB in {stage}")
for p in files:
    print(f"  {names[p]:60s} {p.stat().st_size/1e9:8.3f} GB")
problems = []
if len(files) > 100: problems.append(f"{len(files)} files > 100-file limit")
if total > 50e9: problems.append(f"{total/1e9:.1f} GB > 50 GB record limit (request a quota increase)")
if not any(n == "README.md" for n in names.values()): problems.append("README.md missing at stage root")
if problems:
    print("PROBLEMS:\n  " + "\n  ".join(problems)); sys.exit(1)
if a.check:
    print("Limits OK."); sys.exit(0)

tok = os.environ.get("ZENODO_TOKEN") or sys.exit("Set ZENODO_TOKEN")
H = {"Authorization": f"Bearer {tok}"}

if a.deposition:
    r = requests.get(f"{API}/deposit/depositions/{a.deposition}", headers=H); r.raise_for_status()
else:
    meta = json.load(open(a.metadata))
    for c in meta["metadata"]["creators"]:
        if not c.get("orcid"): c.pop("orcid", None)
    r = requests.post(f"{API}/deposit/depositions", headers=H, json=meta); r.raise_for_status()
dep = r.json()
bucket = dep["links"]["bucket"]
doi = dep["metadata"].get("prereserve_doi", {}).get("doi")
print(f"Draft deposition {dep['id']}  reserved DOI: {doi}\n  edit: {dep['links'].get('html')}")

existing = {f["filename"]: f["checksum"].split(":")[-1] for f in requests.get(
    f"{API}/deposit/depositions/{dep['id']}/files", headers=H).json()}

def md5(p, bs=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()

for p in files:
    n = names[p]; local = md5(p)
    if existing.get(n) == local:
        print(f"skip (already uploaded): {n}"); continue
    print(f"uploading {n} ({p.stat().st_size/1e9:.2f} GB) ...", flush=True)
    with open(p, "rb") as fh:
        r = requests.put(f"{bucket}/{n}", data=fh, headers=H, timeout=None)
    r.raise_for_status()
    remote = r.json().get("checksum", "").split(":")[-1]
    print(f"  {'OK' if remote == local else 'CHECKSUM MISMATCH'}  md5 {local}")
print(f"\nDone. Draft {dep['id']} is NOT published. Review it in the web UI, then publish there.")
