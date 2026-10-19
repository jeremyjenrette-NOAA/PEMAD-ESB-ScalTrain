#!/usr/bin/env python3
"""
Upload the staged dataset to a Zenodo DRAFT (never publishes).

  export ZENODO_TOKEN=...            # personal token with deposit:write (never commit it)
  python config/zenodo/zenodo_upload.py --stage zenodo_stage --check          # sizes/limits only
  python config/zenodo/zenodo_upload.py --stage zenodo_stage                  # create draft + upload
  python config/zenodo/zenodo_upload.py --stage zenodo_stage --deposition ID  # resume / add files
  python config/zenodo/zenodo_upload.py --stage zenodo_stage --deposition ID --replace --prune
      # replace draft files whose content changed; delete draft files no longer staged

Files are uploaded flat (Zenodo has no folders): weights/x.pt -> weights__x.pt, etc.
Re-running skips files already in the draft with a matching MD5. Use --sandbox to test first.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True)
ap.add_argument("--metadata", default=str(Path(__file__).with_name("metadata.json")))
ap.add_argument("--deposition", default=None, help="Existing draft ID to resume")
ap.add_argument("--sandbox", action="store_true")
ap.add_argument("--check", action="store_true", help="Only list files and check limits")
ap.add_argument("--replace", action="store_true",
                help="If a draft file has the same name but a different MD5, delete it and upload the staged one")
ap.add_argument("--prune", action="store_true", help="Delete draft files that are not in the stage")
ap.add_argument("--retries", type=int, default=4, help="Attempts per file (backoff 1, 2, 4 min)")
ap.add_argument("--dry_run", action="store_true", help="With --deposition: report what would change, change nothing")
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

# Remember the draft so a rerun never creates a second record (and a second DOI)
dep_file = stage / ".zenodo_deposition"
if not a.deposition and dep_file.exists():
    a.deposition = dep_file.read_text().strip()
    print(f"Reusing draft {a.deposition} from {dep_file} (delete that file to start a new record)")
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
if not a.sandbox:
    dep_file.write_text(str(dep["id"]) + "\n")

def refresh_remote():
    r = requests.get(f"{API}/deposit/depositions/{dep['id']}/files", headers=H); r.raise_for_status()
    fl = r.json()
    return ({f["filename"]: f["checksum"].split(":")[-1] for f in fl}, {f["filename"]: f["id"] for f in fl})
existing, remote_id = refresh_remote()

def delete_remote(n):
    if a.dry_run:
        print(f"  [dry run] would delete {n}"); return
    rr = requests.delete(f"{API}/deposit/depositions/{dep['id']}/files/{remote_id[n]}", headers=H)
    rr.raise_for_status(); print(f"  deleted {n}")

staged_names = set(names.values())
stale = sorted(set(existing) - staged_names)
if stale:
    print(f"{len(stale)} draft files are not in the stage:", *stale, sep="\n  ")
    if a.prune:
        for n in stale: delete_remote(n)

def md5(p, bs=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()

failed = []
for p in files:
    n = names[p]; local = md5(p)
    if existing.get(n) == local:
        print(f"skip (already uploaded): {n}"); continue
    if n in existing:
        if not a.replace:
            print(f"DIFFERENT CONTENT in draft, not replaced (use --replace): {n}"); continue
        print(f"changed: {n}"); delete_remote(n)
    if a.dry_run:
        print(f"  [dry run] would upload {n}"); continue
    ok = False
    for attempt in range(1, a.retries + 1):
        print(f"uploading {n} ({p.stat().st_size/1e9:.2f} GB), attempt {attempt}/{a.retries} ...", flush=True)
        t0 = time.time()
        try:
            with open(p, "rb") as fh:
                r = requests.put(f"{bucket}/{n}", data=fh, headers=H, timeout=(60, 3600))
            r.raise_for_status()
            remote = r.json().get("checksum", "").split(":")[-1]
            mb_s = p.stat().st_size / 1e6 / max(time.time() - t0, 1)
            print(f"  {'OK' if remote == local else 'CHECKSUM MISMATCH'}  md5 {local}  ({mb_s:.1f} MB/s)")
            ok = remote == local
            if ok:
                break
        except requests.exceptions.RequestException as e:
            print(f"  failed after {(time.time() - t0)/60:.1f} min: {type(e).__name__}: {str(e)[:200]}")
        # remove any partial object before retrying
        try:
            existing, remote_id = refresh_remote()
            if n in existing:
                delete_remote(n)
        except requests.exceptions.RequestException as e:
            print(f"  (could not check for a partial upload: {e})")
        if attempt < a.retries:
            wait = 60 * 2 ** (attempt - 1)
            print(f"  retrying in {wait} s"); time.sleep(wait)
    if not ok:
        failed.append(n)
if failed:
    print("\nFAILED uploads (rerun the same command to retry; finished files are skipped):", *failed, sep="\n  ")
print(f"\nDone. Draft {dep['id']} is NOT published. Review it in the web UI, then publish there.")
if failed:
    sys.exit(1)
