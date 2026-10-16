#!/usr/bin/env python3
"""
Package the Calibration-set images for the Zenodo record: ZIP (stored; PNGs are already
compressed) by year x region, split so no archive exceeds --max_gb. Writes images_manifest.csv
(imagename -> archive) and checksums_images.md5. Zenodo allows 100 files per record.
"""
import argparse, hashlib, zipfile
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--image_metadata", required=True, help="image_metadata.csv from ScalPred s/build_zenodo_tables.py")
ap.add_argument("--src_dir", required=True, help="Dir holding the images (symlinks followed), e.g. data2226/yolo/images/val")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--max_gb", type=float, default=9.5)
a = ap.parse_args()

meta = pd.read_csv(a.image_metadata, low_memory=False)
meta = meta[meta["image_in_record"].astype(str).str.lower().eq("true")].sort_values(["year", "region", "imagename"])
src, out = Path(a.src_dir), Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
limit = a.max_gb * 1e9
rows, missing, archives = [], [], []

def md5(p, bs=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()

for (yr, reg), g in meta.groupby(["year", "region"]):
    part, size, zf, zname = 1, 0.0, None, None
    for n in g["imagename"]:
        p = src / n
        if not p.exists():
            missing.append(n); continue
        s = p.stat().st_size
        if zf is None or size + s > limit:
            if zf:
                zf.close(); part += 1
            zname = out / f"images_{yr}_{reg}_part{part:02d}.zip"
            archives.append(zname)
            zf = zipfile.ZipFile(zname, "w", compression=zipfile.ZIP_STORED, allowZip64=True); size = 0.0
        zf.write(p.resolve(), arcname=n); size += s
        rows.append({"imagename": n, "archive": zname.name, "bytes": s})
    if zf:
        zf.close()

pd.DataFrame(rows).to_csv(out / "images_manifest.csv", index=False)
with open(out / "checksums_images.md5", "w") as fh:
    for z in archives:
        fh.write(f"{md5(z)}  {z.name}\n")
print(f"Packaged {len(rows):,} images ({sum(r['bytes'] for r in rows)/1e9:.1f} GB) into {len(archives)} archives; {len(missing)} missing")
for z in archives:
    print(f"  {z.name}  {z.stat().st_size/1e9:.2f} GB")
if missing:
    pd.Series(missing, name="imagename").to_csv(out / "images_missing.csv", index=False)
