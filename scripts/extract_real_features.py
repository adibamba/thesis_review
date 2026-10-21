"""Real-data pilot: pretrained YOLOv8n (COCO) features -> predictive-coding surprise -> controller replay.

What it does (all on real COCO images, no simulated signal):
  1. Downloads coco128 (128 real COCO images) from the Ultralytics GitHub release if missing.
  2. Runs the pretrained YOLOv8n and takes the 256-channel deep feature map (layer 9, SPPF),
     averaged over space -> a 256-number vector per image (the same vector sent to Azure).
  3. Fits a small predictive-coding (PC) layer (256 inputs, 64 latent units, 16,384 weights) on
     CLEAN images only. Its "free energy" F (the prediction error left after K relaxation steps)
     is the surprise signal.
  4. Pilot 1: creates damaged copies (noise, blur, darkness, fog) and measures how well F
     separates clean from damaged frames (AUROC) on images the PC layer has never seen.
  5. Calibrates the two controller lines from calm frames only (mean + 3 sd, mean + 1 sd).
  6. Builds a 120-frame stream (calm -> damaged -> calm) from real images and writes
     logs/real_stream.json, which the surveillance script can read instead of the simulation.

Run:  python scripts/extract_real_features.py
Test without PyTorch (NOT a result, pipeline test only):  python scripts/extract_real_features.py --extractor stub
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import requests
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
CORRUPTIONS = ["noise", "blur", "dark", "fog"]


# ----------------------------------------------------------------------------- data
def ensure_coco128(root: Path) -> Path:
    img_dir = root / "data" / "coco128" / "images" / "train2017"
    if img_dir.exists() and any(img_dir.glob("*.jpg")):
        return img_dir
    zpath = root / "data" / "coco128.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading coco128 (about 7 MB) ...")
    with requests.get(COCO128_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root / "data")
    return img_dir


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------- damage
def _rng(name: str, sev: int, tag: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{name}|{sev}|{tag}".encode()))


def corrupt(img: np.ndarray, kind: str, sev: int, tag: str, official: bool = False) -> np.ndarray:
    """Simplified damage inspired by COCO-C. Use --official for the imagecorruptions package."""
    if official:
        from imagecorruptions import corrupt as ic  # pip install imagecorruptions
        names = {"noise": "gaussian_noise", "blur": "gaussian_blur", "fog": "fog", "dark": None}
        if names.get(kind):
            return np.uint8(ic(img, corruption_name=names[kind], severity=sev))
    x = img.astype(np.float32) / 255.0
    r = _rng(kind, sev, tag)
    if kind == "noise":
        x = x + r.normal(0.0, [0.08, 0.12, 0.18, 0.26, 0.38][sev - 1], x.shape)
    elif kind == "blur":
        sigma = [2, 4, 6, 8, 12][sev - 1]
        x = cv2.GaussianBlur(x, (0, 0), sigma)
    elif kind == "dark":
        x = x * [0.7, 0.5, 0.35, 0.25, 0.15][sev - 1]
    elif kind == "fog":
        a = [0.2, 0.3, 0.4, 0.5, 0.6][sev - 1]
        low = cv2.resize(r.random((6, 6)).astype(np.float32), (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)
        x = (1 - a) * x + a * (0.6 + 0.4 * low)[..., None]
    else:
        raise ValueError(kind)
    return np.uint8(np.clip(x, 0.0, 1.0) * 255.0 + 0.5)


# ----------------------------------------------------------------------------- feature extractors
def letterbox(img: np.ndarray, size: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = im
    return canvas


class YoloExtractor:
    """Pretrained YOLOv8n (COCO). Feature = layer-9 (SPPF) map, 256 channels, averaged over space."""
    name = "yolov8n"

    def __init__(self, weights: str, imgsz: int = 640):
        import torch
        from ultralytics import YOLO
        self.torch, self.imgsz, self._f = torch, imgsz, {}
        self.net = YOLO(weights).model.eval().float()
        self.net.model[9].register_forward_hook(lambda m, i, o: self._f.__setitem__("f", o))

    def __call__(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, int, float]:
        torch = self.torch
        x = np.ascontiguousarray(letterbox(img_rgb, self.imgsz).transpose(2, 0, 1)).astype(np.float32) / 255.0
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.net(torch.from_numpy(x)[None])
        ms = (time.perf_counter() - t0) * 1000.0
        pred = out[0] if isinstance(out, (list, tuple)) else out          # (1, 84, anchors)
        feat = self._f["f"][0].mean(dim=(1, 2)).cpu().numpy().astype(np.float64)
        assert feat.shape == (256,), feat.shape
        cls_scores = pred[0, 4:, :]                                       # 80 class rows
        label = int(cls_scores.max(dim=1).values.argmax())
        return feat, label, ms


class StubExtractor:
    """PIPELINE TEST ONLY: random projection of a 16x16 thumbnail. NOT YOLO, NOT a result."""
    name = "stub"

    def __init__(self):
        self.P = np.random.default_rng(0).standard_normal((768, 256)) / np.sqrt(768)

    def __call__(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, int, float]:
        t0 = time.perf_counter()
        small = cv2.resize(img_rgb, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0
        v = small.reshape(-1)
        f = (v - 0.45) @ self.P
        return f, 0, (time.perf_counter() - t0) * 1000.0


# ----------------------------------------------------------------------------- predictive coding layer
class PCLayer:
    """x (256) is explained by W z (z has 64 units). Free energy F = 0.5 * ||x - W z||^2 after K steps."""

    def __init__(self, d: int = 256, k: int = 64, K: int = 5, eta_z: float = 0.3, lam: float = 0.02, seed: int = 0):
        r = np.random.default_rng(seed)
        self.W = r.standard_normal((d, k)) / np.sqrt(d)
        self.K, self.eta_z, self.lam = K, eta_z, lam
        self.mu = np.zeros(d)
        self.sd = np.ones(d)

    def _infer(self, x: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
        z = np.zeros(self.W.shape[1])
        for _ in range(K):
            eps = x - self.W @ z
            z = z + self.eta_z * (self.W.T @ eps - self.lam * z)
        return z, x - self.W @ z

    def fit(self, X: np.ndarray, epochs: int = 60, eta_w: float = 0.005, K_fit: int = 20, seed: int = 0) -> None:
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        r = np.random.default_rng(seed)
        for _ in range(epochs):
            for i in r.permutation(len(Z)):
                z, eps = self._infer(Z[i], K_fit)
                self.W += eta_w * np.outer(eps, z) / np.sqrt(len(eps))   # local update: error x activity
                self.W /= np.maximum(1.0, np.linalg.norm(self.W, axis=0, keepdims=True))  # keep columns bounded

    def free_energy(self, x: np.ndarray) -> float:
        _, eps = self._infer((x - self.mu) / self.sd, self.K)
        return float(0.5 * eps @ eps)


# ----------------------------------------------------------------------------- controller replay
def replay(F: np.ndarray, up: float, low: float) -> np.ndarray:
    mode, out = 0, []
    for f in F:
        if mode == 0 and f > up:
            mode = 1
        elif mode == 1 and f < low:
            mode = 0
        out.append(mode)
    return np.array(out)


# ----------------------------------------------------------------------------- main
def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", choices=["yolo", "stub"], default="yolo")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--images", default=None, help="folder of .jpg images (default: coco128, downloaded)")
    ap.add_argument("--out", default=str(ROOT / "logs"))
    ap.add_argument("--K", type=int, default=5, help="PC relaxation steps at test time")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--severities", default="1,3,5")
    ap.add_argument("--stream-corruption", default="fog", choices=CORRUPTIONS)
    ap.add_argument("--stream-severity", type=int, default=3)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--k-up", type=float, default=3.0, help="upper line = calm mean + k_up * sd")
    ap.add_argument("--k-low", type=float, default=1.0, help="lower line = calm mean + k_low * sd")
    ap.add_argument("--official", action="store_true", help="use the imagecorruptions package")
    ap.add_argument("--max-images", type=int, default=400, help="use at most this many images (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.images) if args.images else ensure_coco128(ROOT)
    files = sorted(img_dir.glob("*.jpg"))
    if len(files) < 60:
        raise SystemExit(f"need at least 60 images in {img_dir}, found {len(files)}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(files))
    files = [files[i] for i in order]
    if args.max_images and len(files) > args.max_images:
        files = files[:args.max_images]
    n = len(files)
    n_fit, n_cal = int(0.31 * n), int(0.19 * n)
    fit_f, cal_f, test_f = files[:n_fit], files[n_fit:n_fit + n_cal], files[n_fit + n_cal:]
    print(f"images: {n} from {img_dir}  | fit {len(fit_f)}  calibrate {len(cal_f)}  test {len(test_f)}")

    if args.extractor == "stub":
        print("*** EXTRACTOR = stub: PIPELINE TEST ONLY, these numbers are NOT results ***")
        ext: Callable = StubExtractor()
    else:
        ext = YoloExtractor(args.weights)
    ext_name = ext.name

    cache: Dict[Tuple[str, int, str], Tuple[np.ndarray, int, float]] = {}

    def feats(path: Path, kind: str = "clean", sev: int = 0):
        key = (path.name, sev, kind)
        if key not in cache:
            img = read_rgb(path)
            if kind != "clean":
                img = corrupt(img, kind, sev, path.name, args.official)
            cache[key] = ext(img)
        return cache[key]

    t0 = time.time()
    X_fit = np.stack([feats(p)[0] for p in fit_f])
    pc = PCLayer(256, args.latent, args.K)
    pc.fit(X_fit)
    print(f"PC layer fitted on {len(fit_f)} clean images ({pc.W.size} weights)")

    def F_of(path, kind="clean", sev=0):
        t = time.perf_counter()
        f = pc.free_energy(feats(path, kind, sev)[0])
        return f, (time.perf_counter() - t) * 1000.0

    F_cal = np.array([F_of(p)[0] for p in cal_f])
    mu, sd = float(F_cal.mean()), float(F_cal.std() + 1e-9)
    up, low = mu + args.k_up * sd, mu + args.k_low * sd
    print(f"calm F: mean {mu:.3f}  sd {sd:.3f}  -> upper line {up:.3f}, lower line {low:.3f}")

    # ---- Pilot 1: does F separate clean from damaged frames? (held-out test images)
    F_clean = np.array([F_of(p)[0] for p in test_f])
    sevs = [int(s) for s in args.severities.split(",")]
    table = []
    for kind in CORRUPTIONS:
        for sev in sevs:
            F_bad = np.array([F_of(p, kind, sev)[0] for p in test_f])
            y = np.r_[np.zeros(len(F_clean)), np.ones(len(F_bad))]
            auc = float(roc_auc_score(y, np.r_[F_clean, F_bad]))
            auc2 = float(roc_auc_score(y, np.abs(np.r_[F_clean, F_bad] - mu) / sd))   # two-sided: also catches F dropping
            hit = float((F_bad > up).mean())
            table.append({"damage": kind, "severity": sev, "auroc": round(auc, 3), "auroc_two_sided": round(auc2, 3),
                          "F_clean_mean": round(float(F_clean.mean()), 3), "F_damaged_mean": round(float(F_bad.mean()), 3),
                          "alarm_rate_damaged": round(hit, 3)})
    false_alarm = float((F_clean > up).mean())
    print("\nPilot 1: AUROC of the surprise signal (clean vs damaged, held-out images)")
    print(f"{'damage':8s} {'sev':>3s} {'AUROC':>6s} {'AUROC|z|':>9s} {'F clean':>8s} {'F dmg':>8s} {'alarm%':>7s}")
    for r in table:
        print(f"{r['damage']:8s} {r['severity']:3d} {r['auroc']:6.3f} {r['auroc_two_sided']:9.3f} {r['F_clean_mean']:8.2f} {r['F_damaged_mean']:8.2f} {100*r['alarm_rate_damaged']:6.0f}%")
    mean_auc = float(np.mean([r["auroc"] for r in table]))
    mean_auc2 = float(np.mean([r["auroc_two_sided"] for r in table]))
    print(f"mean AUROC {mean_auc:.3f} (one-sided: F up = damaged) | {mean_auc2:.3f} (two-sided) | false alarms on clean test frames {100*false_alarm:.0f}% | target >= 0.90")
    print("note: if AUROC is below 0.5 for a damage, F went DOWN under that damage (the signal inverts): use the two-sided score.")

    # ---- stream: calm -> damaged -> calm, built from real test images
    N = args.frames
    a, b = int(0.25 * N), int(0.625 * N)
    frames, Fs, yolo_ms = [], [], []
    for i in range(N):
        p = test_f[(i * 7) % len(test_f)]
        damaged = a <= i < b
        kind, sev = (args.stream_corruption, args.stream_severity) if damaged else ("clean", 0)
        feat, label, ms = feats(p, kind, sev)
        f, pc_ms = F_of(p, kind, sev)
        Fs.append(f)
        yolo_ms.append(ms)
        frames.append({"frame_idx": i, "scene": "damaged" if damaged else "calm", "image": p.name,
                       "free_energy": round(f, 5), "pseudo_label": label, "yolo_ms": round(ms, 2),
                       "pc_ms": round(pc_ms, 3), "features": [round(float(v), 5) for v in feat]})
    Fs = np.array(Fs)
    mode = replay(Fs, up, low)
    onset = a
    first_alarm = next((i for i in range(onset, N) if mode[i] == 1), None)
    calm_idx = [i for i in range(N) if not (a <= i < b)]
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": f"real_images_coco128_{ext_name}" if ext_name == "yolov8n" else "stub_pipeline_test",
        "extractor": ext_name, "weights": args.weights if ext_name == "yolov8n" else None,
        "dataset": "coco128 (128 images from COCO train2017; YOLOv8n was trained on COCO train2017)" if args.images is None else str(img_dir),
        "split": {"fit": len(fit_f), "calibrate": len(cal_f), "test": len(test_f)},
        "pc_layer": {"inputs": 256, "latent": args.latent, "weights": int(pc.W.size), "K": args.K},
        "calm_stats": {"mean": mu, "sd": sd},
        "thresholds": {"tau_upper": up, "tau_lower": low, "k_up": args.k_up, "k_low": args.k_low, "hysteresis": 0.0},
        "stream": {"frames": N, "damaged_from": a, "damaged_to": b, "damage": args.stream_corruption, "severity": args.stream_severity},
        "pilot1": {"table": table, "mean_auroc": round(mean_auc, 3), "mean_auroc_two_sided": round(mean_auc2, 3), "false_alarm_rate_clean": round(false_alarm, 3), "target_auroc": 0.90},
        "replay": {"cloud_frames": int(mode.sum()), "duty_cycle": round(float(mode.mean()), 3),
                   "alarm_delay_frames": None if first_alarm is None else int(first_alarm - onset),
                   "false_alarm_frames_in_calm": int(mode[calm_idx].sum())},
        "measured_here_ms": {"extractor_mean": round(float(np.mean(yolo_ms)), 2), "note": "measured on this computer; not used for the energy model"},
        "seed": args.seed, "official_corruptions": bool(args.official),
    }
    with open(out / "real_stream.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "frames": frames}, fh)
    with open(out / "real_pilot1.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": meta}, fh, indent=2)
    # every feature vector computed in this run, so the scoring can be tuned offline without re-running YOLO
    keys = list(cache.keys())
    split_of = {p.name: "fit" for p in fit_f}
    split_of.update({p.name: "cal" for p in cal_f})
    split_of.update({p.name: "test" for p in test_f})
    np.savez_compressed(
        out / "real_features.npz",
        image=np.array([k[0] for k in keys]), severity=np.array([k[1] for k in keys]),
        damage=np.array([k[2] for k in keys]), split=np.array([split_of[k[0]] for k in keys]),
        label=np.array([cache[k][1] for k in keys]), feature=np.stack([cache[k][0] for k in keys]),
        extractor=np.array(ext_name))
    print(f"saved {len(keys)} feature vectors: {out / 'real_features.npz'}")
    print(f"\nstream: {meta['replay']['cloud_frames']} of {N} frames sent to the cloud (duty cycle {meta['replay']['duty_cycle']}), "
          f"alarm delay {meta['replay']['alarm_delay_frames']} frames, false-alarm frames in calm: {meta['replay']['false_alarm_frames_in_calm']}")
    print(f"written: {out / 'real_stream.json'}   ({time.time() - t0:.0f} s)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.axvspan(a, b, color="#f3d9a8", alpha=0.5, label="damaged scene")
        ax.plot(Fs, color="#1f3a5f", lw=1.4, label="surprise F (real features)")
        ax.axhline(up, color="#1f3a5f", ls="--", lw=1, label="upper line")
        ax.axhline(low, color="#17776f", ls="--", lw=1, label="lower line")
        ax.step(range(N), mode * (Fs.max() * 0.95), color="#b3402e", lw=1, where="post", label="cloud mode")
        ax.set_xlabel("frame")
        ax.set_ylabel("surprise F")
        ax.legend(ncol=3, fontsize=8, loc="upper left")
        ax.set_title(f"{meta['data_source']}: calm -> {args.stream_corruption} (sev {args.stream_severity}) -> calm")
        fig.tight_layout()
        fig.savefig(out / "real_pilot_signal.png", dpi=160)
        print(f"plot: {out / 'real_pilot_signal.png'}")
    except Exception as ex:  # plotting is optional
        print("plot skipped:", ex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())