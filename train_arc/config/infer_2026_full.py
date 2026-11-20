#!/usr/bin/env python3
"""
Full-survey 2026 inference with the Tier B (2022-2024) YOLOv12n weights.

Same detector settings as the Tier B evaluation (job/eval_tierb_2026.sh ->
config/eval_yolo_detections_hung.py): imgsz 640, conf >= 0.1, NMS IoU 0.65,
max_det 600, FP16 on GPU, and each stereo PNG cropped to its LEFT camera frame
(columns [0, 1936)) before inference. Images are read with OpenCV (BGR), exactly as in
the evaluator. There is no ground truth here, so no matching: every detection is written.

Modes
-----
probe    Reads a small sample per group x drive (and duplicate pairs) and reports image
         size, read/decode time, colour/glare statistics, the pixel difference between
         the two copies of duplicated images, and the GPU/library setup. Run this first.
infer    Processes the manifest chunk by chunk (build_2026_manifest.py). A chunk is done
         when its images_cNNNNN.csv exists, so re-running the same command resumes.
compare  Checks a --subset_csv run of annotated images against the Tier B evaluation's
         autotest CSV (same boxes and confidences expected, up to GPU/FP16 noise).
combine  Concatenates the per-chunk files into detections_2026_all.csv.gz and
         images_2026_all.csv.gz and prints a summary.

Outputs (in --out_dir)
----------------------
chunks/dets_cNNNNN.csv    one row per detection: image_uid, Imagename, drive, chunk_id,
                          Detectid (within chunk), TLx, TLy, BRx, BRy (pixels in the left
                          frame), Conf, bw, bh, boxsize, bin, Spname
chunks/images_cNNNNN.csv  one row per file: status/error, img_w, img_h, eval_w, cropped,
                          n_det, max_conf, colour stats of the left frame (mean_b/g/r,
                          gray_mean, gray_std, sat_frac = share of pixels >= 250: glare)
run_config.json           settings, weights SHA-256, library and GPU versions
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

# torch must be imported BEFORE numpy/pandas/cv2 on Windows: if another package loads its own
# (older) MSVC runtime DLL first, torch's c10.dll fails with "WinError 1114".
try:
    import torch  # noqa: F401
    _TORCH_ERR = ""
except Exception as _e:  # reported by env_info / load_model
    _TORCH_ERR = f"{type(_e).__name__}: {_e}"

import numpy as np
import pandas as pd

try:
    import cv2
    cv2.setNumThreads(1)  # we parallelise over images with threads
except ImportError:  # pragma: no cover
    sys.exit("OpenCV (cv2) is required: it ships with ultralytics (opencv-python).")

DET_COLS = ["image_uid", "Imagename", "drive", "chunk_id", "Detectid", "TLx", "TLy", "BRx", "BRy",
            "Conf", "bw", "bh", "boxsize", "bin", "Spname"]
IMG_COLS = ["image_uid", "Imagename", "drive", "chunk_id", "group", "status", "error", "img_w", "img_h",
            "eval_w", "cropped", "n_det", "max_conf", "mean_b", "mean_g", "mean_r", "gray_mean",
            "gray_std", "sat_frac", "read_s"]
MAN_DTYPES = {"imagename": str, "drive": str, "src_path": str, "cruise": str, "date": str, "month": str,
              "time": str, "group": str, "other_drives": str, "exclude_reason": str}


# ----------------------------------------------------------------------------- utilities
def to_native(p: str) -> str:
    """'/v/habcam/x.png' -> 'V:/habcam/x.png' on Windows (Git Bash/MSYS style paths)."""
    if os.name == "nt":
        m = re.match(r"^/([A-Za-z])(/.*)?$", str(p))
        if m:
            return f"{m.group(1).upper()}:{m.group(2) or '/'}"
    return str(p)


def parse_ranges(spec: str) -> set[int]:
    out: set[int] = set()
    for part in filter(None, (s.strip() for s in spec.split(","))):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "opencv": cv2.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}
    try:
        if _TORCH_ERR:
            raise ImportError(_TORCH_ERR)
        import torch
        info.update(torch=torch.__version__, cuda_available=bool(torch.cuda.is_available()),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except Exception as e:
        info["torch"] = f"not importable: {e}"
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception as e:
        info["ultralytics"] = f"not importable: {e}"
    return info


def read_image(path: str, crop_w: int, retries: int, save_dir: str | None, rel: str | None):
    """Read one PNG, keep the left frame, compute colour stats. Never raises."""
    t0 = time.perf_counter()
    rec = {"status": "ok", "error": "", "img_w": np.nan, "img_h": np.nan, "eval_w": np.nan, "cropped": False}
    im = None
    for attempt in range(retries + 1):
        try:
            buf = np.fromfile(path, dtype=np.uint8)  # handles any path; one sequential read
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is None:
                rec.update(status="error", error="decode failed (truncated/corrupt?)")
            break
        except (OSError, ValueError) as e:  # network hiccup or missing file
            rec.update(status="error", error=f"{type(e).__name__}: {e}")
            if isinstance(e, FileNotFoundError) or attempt == retries:
                break
            time.sleep(2 ** attempt)
    if im is not None:
        rec["status"], rec["error"] = "ok", ""
        h, w = im.shape[:2]
        rec.update(img_w=w, img_h=h)
        if crop_w and w > crop_w:
            im = np.ascontiguousarray(im[:, :crop_w])
            rec["cropped"] = True
        rec["eval_w"] = im.shape[1]
        small = cv2.resize(im, (max(1, im.shape[1] // 8), max(1, im.shape[0] // 8)), interpolation=cv2.INTER_AREA)
        b, g, r = small.reshape(-1, 3).mean(0)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        rec.update(mean_b=round(float(b), 2), mean_g=round(float(g), 2), mean_r=round(float(r), 2),
                   gray_mean=round(float(gray.mean()), 2), gray_std=round(float(gray.std()), 2),
                   sat_frac=round(float((gray >= 250).mean()), 5))
        if save_dir and rel:
            dst = Path(save_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    rec["read_s"] = round(time.perf_counter() - t0, 4)
    return im, rec


def prefetch(rows, fn, workers: int, depth: int):
    """Yield (row, result) in order with at most `depth` images decoded ahead (bounded memory)."""
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        q: collections.deque = collections.deque()
        it = iter(rows)
        for row in it:
            q.append((row, ex.submit(fn, row)))
            if len(q) >= depth:
                break
        while q:
            row, fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt, ex.submit(fn, nxt)))
            yield row, fut.result()


def load_manifest(path: str) -> pd.DataFrame:
    m = pd.read_csv(to_native(path), dtype=MAN_DTYPES, low_memory=False, keep_default_na=False,
                    na_values={"chunk_id": [""], "frame": [""]})
    for c in ("is_duplicate", "color_corrected_assumed", "is_preferred_copy", "annotated_2026", "selected"):
        if c in m and m[c].dtype == object:
            m[c] = m[c].astype(str).str.lower().eq("true")
    m["chunk_id"] = pd.to_numeric(m["chunk_id"], errors="coerce").astype("Int64")
    return m


# ----------------------------------------------------------------------------- probe
def run_probe(a, man: pd.DataFrame):
    out = Path(to_native(a.out_dir)); out.mkdir(parents=True, exist_ok=True)
    info = env_info()
    print("Environment:", json.dumps(info, indent=2))
    w = to_native(a.weights)
    print(f"Weights: {w}  exists={os.path.exists(w)}")
    rng = np.random.default_rng(a.seed)
    sel = man[man["exclude_reason"].eq("")]
    samples = []
    for (g, d), sub in sel.groupby(["group", "drive"]):
        samples.append(sub.iloc[rng.choice(len(sub), min(a.probe_n, len(sub)), replace=False)].assign(probe="sample"))
    dup_names = sel.loc[sel["is_duplicate"], "imagename"].unique()
    if len(dup_names):
        pick = rng.choice(dup_names, min(a.probe_n, len(dup_names)), replace=False)
        samples.append(sel[sel["imagename"].isin(pick)].assign(probe="dup_pair"))
    s = pd.concat(samples, ignore_index=True)
    print(f"\nProbing {len(s)} files ({a.workers} reader threads) ...")
    arrays = {}
    recs = []
    t0 = time.perf_counter()
    for row, (im, rec) in prefetch(s.to_dict("records"),
                                   lambda r: read_image(to_native(r["src_path"]), a.crop_w, a.retries, None, None),
                                   a.workers, a.workers * 2):
        rec.update(image_uid=row["image_uid"], imagename=row["imagename"], drive=row["drive"], group=row["group"],
                   probe=row["probe"], color_corrected_assumed=row["color_corrected_assumed"],
                   file_mb=round(os.path.getsize(to_native(row["src_path"])) / 1e6, 2)
                   if rec["status"] == "ok" else np.nan)
        if row["probe"] == "dup_pair" and im is not None:
            arrays[row["image_uid"]] = cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4), interpolation=cv2.INTER_AREA)
        recs.append(rec)
    wall = time.perf_counter() - t0
    r = pd.DataFrame(recs)
    print(f"Read {len(r)} files in {wall:.1f}s = {len(r)/wall:.1f} img/s with {a.workers} threads "
          f"(median {r['read_s'].median():.2f}s per file per thread)")

    print("\n=== Image size, read time and colour by group x drive (sample) ===")
    g = r[r["probe"].eq("sample")].groupby(["group", "drive"])
    print(g.agg(n=("status", "size"), errors=("status", lambda x: int((x != "ok").sum())),
                widths=("img_w", lambda x: ",".join(map(str, sorted(set(x.dropna().astype(int)))))),
                heights=("img_h", lambda x: ",".join(map(str, sorted(set(x.dropna().astype(int)))))),
                file_mb=("file_mb", "median"), read_s=("read_s", "median"),
                gray_mean=("gray_mean", "median"), gray_std=("gray_std", "median"),
                sat_frac=("sat_frac", "median"), mean_b=("mean_b", "median"), mean_g=("mean_g", "median"),
                mean_r=("mean_r", "median")).to_string())
    odd = r[(r["status"] == "ok") & (r["img_w"] != 2 * a.crop_w)]
    if len(odd):
        print(f"\nNOTE: {len(odd)} probed files are not {2*a.crop_w} px wide; for these the 'left half' is not "
              f"[0,{a.crop_w}). Widths seen: {sorted(set(odd['img_w'].astype(int)))}. Check --crop_w for them.")

    d = r[r["probe"].eq("dup_pair") & r["status"].eq("ok")]
    if len(d):
        piv = d.pivot_table(index="imagename", columns="drive",
                            values=["gray_mean", "gray_std", "sat_frac", "mean_b", "mean_g", "mean_r"])
        diffs = []
        for name, sub in d.groupby("imagename"):
            if len(sub) == 2:
                u1, u2 = sub["image_uid"].tolist()
                x1, x2 = arrays.get(u1), arrays.get(u2)
                if x1 is not None and x2 is not None and x1.shape == x2.shape:
                    diffs.append({"imagename": name, "mean_abs_pixel_diff": float(np.abs(x1.astype(np.int16) - x2).mean()),
                                  "identical": bool(np.array_equal(x1, x2))})
        dd = pd.DataFrame(diffs)
        print("\n=== Duplicate pairs: median colour stats per drive ===")
        print(piv.median().unstack().round(3).to_string())
        if len(dd):
            print(f"\nPairs compared: {len(dd)};  pixel-identical: {int(dd['identical'].sum())};  "
                  f"median mean |diff| (0-255): {dd['mean_abs_pixel_diff'].median():.2f}")
            dd.to_csv(out / "probe_dup_pairs.csv", index=False)
    r.to_csv(out / "probe_images.csv", index=False)
    (out / "probe_env.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"\nWrote {out/'probe_images.csv'}")

    if a.probe_model:
        model, kw = load_model(a)
        ok = [rr for rr in s.to_dict("records")][: a.batch]
        ims = [read_image(to_native(rr["src_path"]), a.crop_w, a.retries, None, None)[0] for rr in ok]
        ims = [x for x in ims if x is not None]
        model.predict(source=ims[:1], **kw)  # warm-up
        t1 = time.perf_counter()
        res = model.predict(source=ims, **kw)
        dt = time.perf_counter() - t1
        print(f"\nModel: {len(ims)} images in {dt:.2f}s on GPU/CPU = {len(ims)/dt:.1f} img/s (excludes reading); "
              f"detections per image: {[len(x.boxes) for x in res]}")


# ----------------------------------------------------------------------------- model
def load_model(a):
    if _TORCH_ERR:
        sys.exit("PyTorch cannot be imported in this Python:\n  " + _TORCH_ERR + "\n"
                 "Check with:  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"\n"
                 "WinError 1114 on c10.dll usually means the Microsoft Visual C++ Redistributable (x64) is missing/"
                 "outdated, or a CPU/CUDA wheel mismatch; see the notes in job/infer_2026_full.sh.")
    import torch
    from ultralytics import YOLO
    dev = str(a.device)
    use_cpu = dev.lower() == "cpu"
    if not use_cpu and not torch.cuda.is_available():
        sys.exit(f"--device {dev} requested but torch sees no CUDA GPU. Check the environment or use --device cpu.")
    model = YOLO(to_native(a.weights))
    kw = dict(imgsz=a.imgsz, conf=a.conf, iou=a.nms_iou, max_det=a.max_det, device=dev,
              verbose=False, half=(not use_cpu) and not a.no_half)
    return model, kw


# ----------------------------------------------------------------------------- infer
def write_csv(path: Path, cols: list[str], rows: list[dict]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    return tmp


def run_infer(a, man: pd.DataFrame):
    out = Path(to_native(a.out_dir))
    if a.subset_csv:  # a subset only partly fills chunks: keep it apart from the main run
        out = out / f"subset_{Path(to_native(a.subset_csv)).stem}"
        print(f"--subset_csv: outputs go to {out}")
    cdir = out / "chunks"; cdir.mkdir(parents=True, exist_ok=True)
    m = man[man["selected"] & man["chunk_id"].notna()]
    if a.subset_csv:  # listed images regardless of frame thinning; preferred copy only
        keep = set(pd.read_csv(to_native(a.subset_csv), usecols=["imagename"])["imagename"].astype(str))
        m = man[man["exclude_reason"].eq("") & man["is_preferred_copy"] & man["imagename"].isin(keep)].copy()
        m["chunk_id"] = np.arange(len(m)) // 10000 + 1
        print(f"--subset_csv: {len(m):,} manifest rows match {len(keep):,} listed names")
    if a.groups:
        m = m[m["group"].isin([g.strip() for g in a.groups.split(",")])]
    chunks = sorted(int(c) for c in m["chunk_id"].unique())
    if a.chunks:
        want = parse_ranges(a.chunks)
        chunks = [c for c in chunks if c in want]
    todo = [c for c in chunks if not (cdir / f"images_c{c:05d}.csv").exists()]
    n_done = len(chunks) - len(todo)
    if a.max_chunks:
        todo = todo[: a.max_chunks]
    print(f"Chunks selected: {len(chunks)};  already done: {n_done};  "
          f"to run now: {len(todo)}  ({int(m['chunk_id'].isin(todo).sum()):,} files)")
    if not todo:
        return

    info = env_info()
    cfg = {k: v for k, v in vars(a).items()}
    cfg.update(env=info, weights_native=to_native(a.weights), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not a.dry_run:
        cfg["weights_sha256"] = sha256(to_native(a.weights))
        model, kw = load_model(a)
        print(f"Model loaded: {to_native(a.weights)}  ({info.get('gpu') or 'CPU'})  settings: {kw}")
    else:
        model, kw = None, {}
        print("DRY RUN: images are read, cropped and summarised, but no model is run.")
    (out / f"run_config_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(cfg, indent=2, default=str) + "\n")

    save_dir = to_native(a.save_crops) if a.save_crops else None
    t_all, n_all = time.perf_counter(), 0
    stop_file = out / "STOP"
    if stop_file.exists():
        stop_file.unlink()
    print(f"To stop cleanly after the current chunk: touch {stop_file}")
    for k, c in enumerate(todo, 1):
        if stop_file.exists():
            print(f"STOP file found: stopping before chunk {c}. Re-run the same command to resume.")
            stop_file.unlink()
            break
        rows = m[m["chunk_id"] == c].to_dict("records")
        t0 = time.perf_counter()
        img_rows, det_rows = [], []
        detectid = 0
        batch_ims, batch_meta = [], []

        def flush():
            nonlocal detectid
            if not batch_ims:
                return
            if model is None:
                results = [None] * len(batch_ims)
            else:
                try:
                    results = model.predict(source=batch_ims, batch=len(batch_ims), **kw)
                except Exception as e:  # fall back to one image at a time
                    print(f"  WARNING: batch predict failed ({e}); retrying images one by one")
                    results = []
                    for im, (_, rec) in zip(batch_ims, batch_meta):
                        try:
                            results.append(model.predict(source=[im], batch=1, **kw)[0])
                        except Exception as e1:
                            rec.update(status="error", error=f"predict failed: {e1}")
                            results.append(None)
            for res, (row, rec) in zip(results, batch_meta):
                n, mx = 0, np.nan
                if res is not None and res.boxes is not None and len(res.boxes):
                    xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float64)
                    conf = res.boxes.conf.cpu().numpy().astype(np.float64)
                    n, mx = len(conf), float(conf.max())
                    for (x1, y1, x2, y2), cf_ in zip(xyxy, conf):
                        detectid += 1
                        bw, bh = x2 - x1, y2 - y1
                        det_rows.append({"image_uid": row["image_uid"], "Imagename": row["imagename"],
                                         "drive": row["drive"], "chunk_id": c, "Detectid": detectid,
                                         "TLx": round(x1, 2), "TLy": round(y1, 2), "BRx": round(x2, 2),
                                         "BRy": round(y2, 2), "Conf": round(float(cf_), 5),
                                         "bw": round(bw, 2), "bh": round(bh, 2), "boxsize": round(bw * bh, 1),
                                         "bin": int(round(cf_ * 100)), "Spname": "scallop"})
                rec.update(n_det=n if (model is not None and rec["status"] == "ok") else np.nan, max_conf=mx)
            batch_ims.clear(); batch_meta.clear()

        def reader(row):
            rel = f"{row['drive']}/{row['date']}/{row['imagename']}"
            return read_image(to_native(row["src_path"]), a.crop_w, a.retries, save_dir, rel)

        for row, (im, rec) in prefetch(rows, reader, a.workers, max(a.batch * 3, a.workers * 2)):
            rec.update(image_uid=row["image_uid"], Imagename=row["imagename"], drive=row["drive"],
                       chunk_id=c, group=row["group"])
            img_rows.append(rec)
            if im is None:
                continue
            batch_ims.append(im); batch_meta.append((row, rec))
            if len(batch_ims) >= a.batch:
                flush()
        flush()

        n_err = sum(r["status"] != "ok" for r in img_rows)
        if n_err == len(rows) and len(rows) >= 50:  # a whole chunk unreadable = mount problem, not bad files
            sys.exit(f"Every file in chunk {c} failed to read (first error: {img_rows[0]['error']}). "
                     "Is the drive mounted? Stopping without marking the chunk done.")
        # atomic finish: images_cNNNNN.csv appears last and marks the chunk done
        td = write_csv(cdir / f"dets_c{c:05d}.csv", DET_COLS, det_rows)
        ti = write_csv(cdir / f"images_c{c:05d}.csv", IMG_COLS, img_rows)
        os.replace(td, cdir / f"dets_c{c:05d}.csv")
        os.replace(ti, cdir / f"images_c{c:05d}.csv")

        dt = time.perf_counter() - t0
        n_all += len(rows)
        rate = n_all / (time.perf_counter() - t_all)
        left = int(m["chunk_id"].isin(todo[k:]).sum())
        print(f"[{time.strftime('%H:%M:%S')}] chunk {c:5d} ({k}/{len(todo)}): {len(rows):,} files, "
              f"{len(det_rows):,} detections, {n_err} errors, {len(rows)/dt:.1f} img/s; "
              f"ETA for this run {left / max(rate, 1e-9) / 3600:.1f} h", flush=True)


# ----------------------------------------------------------------------------- combine
def run_combine(a, man: pd.DataFrame):
    out = Path(to_native(a.out_dir)); cdir = out / "chunks"
    imgs = sorted(cdir.glob("images_c*.csv"))
    if not imgs:
        sys.exit(f"No chunk outputs in {cdir}")
    im = pd.concat((pd.read_csv(p, low_memory=False) for p in imgs), ignore_index=True)
    dets = [cdir / p.name.replace("images_", "dets_") for p in imgs]
    de = pd.concat((pd.read_csv(p, low_memory=False) for p in dets if p.exists()), ignore_index=True)
    keep = ["image_uid", "date", "time", "datetime", "frame", "is_duplicate", "other_drives",
            "color_corrected_assumed", "is_preferred_copy", "annotated_2026"]
    im = im.merge(man[keep], on="image_uid", how="left")
    im.to_csv(out / "images_2026_all.csv.gz", index=False, compression="gzip")
    de.to_csv(out / "detections_2026_all.csv.gz", index=False, compression="gzip")
    sel = man[man["selected"]]
    print(f"Chunks combined: {len(imgs)};  files processed: {len(im):,} of {len(sel):,} selected")
    print(im.groupby(["group", "drive"]).agg(files=("status", "size"),
                                             errors=("status", lambda s: int((s != "ok").sum())),
                                             with_det=("n_det", lambda s: int((s > 0).sum())),
                                             detections=("n_det", "sum"),
                                             conf50=("max_conf", lambda s: int((s >= 0.5).sum()))).to_string())
    if (im["status"] != "ok").any():
        im[im["status"] != "ok"].to_csv(out / "errors_2026.csv", index=False)
        print(f"Errors listed in {out/'errors_2026.csv'}")
    print(f"Wrote {out/'images_2026_all.csv.gz'} and {out/'detections_2026_all.csv.gz'} ({len(de):,} detections)")


# ----------------------------------------------------------------------------- compare
def run_compare(a, man: pd.DataFrame):
    """Check this pipeline against the Tier B evaluation output on the same images."""
    out = Path(to_native(a.out_dir))
    if not list((out / "chunks").glob("images_c*.csv")):
        sys.exit(f"No chunk outputs under {out/'chunks'}: run the --subset_csv inference first.")
    im = pd.concat((pd.read_csv(p) for p in sorted((out / "chunks").glob("images_c*.csv"))), ignore_index=True)
    de = pd.concat((pd.read_csv(p) for p in sorted((out / "chunks").glob("dets_c*.csv"))), ignore_index=True)
    ref = pd.read_csv(to_native(a.ref_csv), low_memory=False)
    im = im[im["status"].eq("ok")].drop_duplicates("Imagename")
    names = set(im["Imagename"])
    de = de[de["Imagename"].isin(names)].drop_duplicates(["Imagename", "TLx", "TLy", "BRx", "BRy"])
    ref = ref[ref["Imagename"].isin(names)]
    n_new = de.groupby("Imagename").size().reindex(sorted(names), fill_value=0)
    n_ref = ref.groupby("Imagename").size().reindex(sorted(names), fill_value=0)
    print(f"Images compared: {len(names):,};  detections new {len(de):,} vs Tier B eval {len(ref):,}")
    print(f"Images with the same number of detections: {(n_new == n_ref).mean():.1%}")
    rows = []
    for name in names:
        p = de.loc[de["Imagename"].eq(name), ["TLx", "TLy", "BRx", "BRy", "Conf"]].to_numpy(float)
        r = ref.loc[ref["Imagename"].eq(name), ["TLx", "TLy", "BRx", "BRy", "Conf"]].to_numpy(float)
        if len(r) == 0:
            continue
        for b in r:
            if len(p) == 0:
                rows.append({"iou": 0.0, "dconf": np.nan, "ref_conf": b[4]}); continue
            ix = np.clip(np.minimum(p[:, 2], b[2]) - np.maximum(p[:, 0], b[0]), 0, None)
            iy = np.clip(np.minimum(p[:, 3], b[3]) - np.maximum(p[:, 1], b[1]), 0, None)
            inter = ix * iy
            iou = inter / ((p[:, 2]-p[:, 0])*(p[:, 3]-p[:, 1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
            k = int(np.argmax(iou))
            rows.append({"iou": float(iou[k]), "dconf": float(p[k, 4] - b[4]), "ref_conf": b[4]})
    r = pd.DataFrame(rows)
    if len(r):
        hit = r["iou"] >= 0.9
        print(f"Tier B eval boxes with a new box at IoU >= 0.9: {hit.mean():.1%} "
              f"(conf >= 0.5 boxes: {hit[r['ref_conf'] >= 0.5].mean():.1%})")
        print(f"|conf difference| for those: median {r.loc[hit, 'dconf'].abs().median():.4f}, "
              f"95th pct {r.loc[hit, 'dconf'].abs().quantile(0.95):.4f}")
        print("Small differences are expected (GPU model, FP16, batch size); misses concentrated near "
              "conf 0.1 are threshold noise. Large gaps mean the crop or settings differ.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["probe", "infer", "combine", "compare"], required=True)
    ap.add_argument("--manifest", required=True, help="manifest_2026.csv.gz from build_2026_manifest.py")
    ap.add_argument("--out_dir", required=True, help="Local output folder, e.g. D:/scallop2026_tierb")
    ap.add_argument("--weights", default="/c/Users/jeremy.jenrette/PEMAD-ESB-ScalPred/models/yolov12n_2022-2024_tierB.pt")
    # Tier B evaluation settings (job/eval_tierb_2026.sh); change only deliberately
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--nms_iou", type=float, default=0.65)
    ap.add_argument("--max_det", type=int, default=600)
    ap.add_argument("--crop_w", type=int, default=1936, help="Keep columns [0, crop_w) of wider images (left camera)")
    ap.add_argument("--no_half", action="store_true", help="Disable FP16 on GPU (Tier B eval used FP16)")
    # throughput
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8, help="Reader/decoder threads")
    ap.add_argument("--retries", type=int, default=3, help="Re-reads after a read error (network drives)")
    # selection
    ap.add_argument("--chunks", default="", help="e.g. '1-20,25' (default: all)")
    ap.add_argument("--groups", default="", help="e.g. habcam_primary or auv (default: all)")
    ap.add_argument("--max_chunks", type=int, default=0, help="Stop after this many chunks (0 = no limit)")
    ap.add_argument("--subset_csv", default="", help="CSV with an imagename column: process only these; outputs go to <out_dir>/subset_<csv name>")
    ap.add_argument("--save_crops", default="", help="Also write each left-frame PNG under this folder")
    ap.add_argument("--dry_run", action="store_true", help="Read/crop/summarise only; no model")
    # probe
    ap.add_argument("--probe_n", type=int, default=20, help="Files per group x drive (and duplicate pairs)")
    ap.add_argument("--probe_model", action="store_true", help="Also load the model and time one batch")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ref_csv", default="", help="compare: autotest_tierb2026_*.csv from the Tier B evaluation")
    a = ap.parse_args()

    t = time.perf_counter()
    man = load_manifest(a.manifest)
    print(f"Manifest: {len(man):,} rows ({time.perf_counter()-t:.0f}s)")
    {"probe": run_probe, "infer": run_infer, "combine": run_combine, "compare": run_compare}[a.mode](a, man)


if __name__ == "__main__":
    main()
