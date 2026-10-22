#!/usr/bin/env python3
"""
N8: inference throughput for the Ultralytics detectors (YOLOv12n, RT-DETR-L).

Times, per image, (1) reading/decoding the PNG and (2) model inference, after a warm-up,
so the paper can separate detector speed from disk/IO speed. Same settings as evaluation
(imgsz 640, conf 0.1, NMS IoU 0.65, max_det 600). Cascade R-CNN (VIAME) is timed from
its pipeline log instead (see the note printed at the end).

  # A40 (Slurm):  python config/bench_inference.py --device 0
  # laptop:       python config/bench_inference.py --device mps   (Apple) or --device cpu
  python config/bench_inference.py --weights a.pt b.pt --images <dir> --n 500 --device 0
"""
import argparse, json, platform, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--weights", nargs="+", default=[      # the weights published on Zenodo
    "zenodo_stage/weights/yolov12n_2022-2026.pt",
    "zenodo_stage/weights/rtdetr_l_2022-2026.pt"])
ap.add_argument("--images", default="data2226/yolo/images/val")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--device", default="0")
ap.add_argument("--batch", type=int, default=1, help="1 = per-image latency; >1 = batched throughput")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out", default="bench_inference.json")
ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = leave torch default)")
a = ap.parse_args()

import cv2, torch
from ultralytics import YOLO, __version__ as ul_version

imgs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
rng = np.random.default_rng(1)
pick = rng.choice(len(imgs), size=min(a.n + a.warmup, len(imgs)), replace=False)   # same draw as before
# spares, used only if some picked files cannot be read (e.g. truncated copies on a laptop)
rest = np.setdiff1d(np.arange(len(imgs)), pick)
spare = np.random.default_rng(2).permutation(rest)[:500]
imgs = [imgs[i] for i in np.concatenate([pick, spare])]
sync = (lambda: torch.cuda.synchronize()) if a.device not in ("cpu", "mps") and torch.cuda.is_available() else \
       (lambda: torch.mps.synchronize()) if a.device == "mps" else (lambda: None)

if a.threads > 0: torch.set_num_threads(a.threads)
hw = {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ul_version,
      "machine": platform.platform(), "processor": platform.processor(), "device": a.device,
      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and a.device not in ("cpu", "mps") else None,
      "cpu_threads": torch.get_num_threads()}
results = {"hardware": hw, "settings": vars(a), "models": {}}
print(json.dumps(hw, indent=1))

for w in a.weights:
    if not Path(w).exists():
        print(f"skip {w}: not found"); continue
    model = YOLO(w)
    kw = dict(imgsz=a.imgsz, conf=0.1, iou=0.65, max_det=600, device=a.device, verbose=False,
              half=a.device not in ("cpu", "mps"))
    t_read, t_inf = [], []
    it, done, skipped = iter(imgs), 0, []
    while done < a.n + a.warmup:
        t0 = time.perf_counter()
        arr = []
        while len(arr) < a.batch:
            p = next(it, None)
            if p is None: break
            x = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if x is None:                                   # unreadable file: skip, take a spare
                skipped.append(p.name); t0 = time.perf_counter(); continue
            arr.append(x[:, :1936] if x.shape[1] > 1936 else x)   # left frame, as in evaluation
        if not arr: break
        t1 = time.perf_counter()
        model.predict(source=arr, **kw); sync()
        t2 = time.perf_counter()
        if done >= a.warmup:
            t_read.append((t1 - t0) / len(arr)); t_inf.append((t2 - t1) / len(arr))
        done += len(arr)
    if skipped: print(f"  skipped {len(skipped)} unreadable images, e.g. {skipped[:3]}")
    r, f = np.array(t_read), np.array(t_inf)
    res = {"n_images": int(len(r) * a.batch), "n_unreadable_skipped": len(skipped), "read_ms_median": 1e3 * float(np.median(r)),
           "inference_ms_median": 1e3 * float(np.median(f)),
           "inference_images_per_s": float(1 / np.mean(f)),
           "end_to_end_images_per_s": float(1 / np.mean(r + f)),
           "gpu_peak_mem_GB": float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() and a.device not in ("cpu", "mps") else None}
    results["models"][Path(w).stem] = res
    print(Path(w).stem, json.dumps(res, indent=1))
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

Path(a.out).write_text(json.dumps(results, indent=1))
print(f"Wrote {a.out}")
print("Cascade R-CNN (VIAME): use images processed / wall time from the VIAME pipeline log of the "
      "holdout or inference run on the same node (and note whether it includes image reading).")
