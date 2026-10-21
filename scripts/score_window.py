"""Windowed surprise score on saved YOLOv8n features (no YOLO needed).

Score: F_W = 0.5 * W * (xbar - mu)^T P (xbar - mu), xbar = mean of last W frame features,
(mu, P) = mean and Ledoit-Wolf precision of CLEAN reference features.
No test leakage: reference = fit+cal images; thresholds = cross-fitted calm windows
(each fold of the 63 non-test images is scored against a reference built from the other folds).
Test images are used only for AUROC and the stream.
"""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

def load(p):
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}

class Ref:
    def __init__(s, R):
        lw = LedoitWolf().fit(R); s.mu, s.P = lw.location_, lw.precision_
    def score(s, xbar, W):
        dx = xbar - s.mu
        return 0.5 * W * np.einsum('ij,jk,ik->i', dx, s.P, dx)

def windows(X, W, n, rng):
    return np.array([X[rng.choice(len(X), W, replace=True)].mean(0) for _ in range(n)])

def main(a):
    D = load(a.npz); img, sev, dmg, spl, X = D['image'], D['severity'], D['damage'], D['split'], D['feature']
    W = a.window
    def get(dm, sv, split):
        m = (dmg == dm) & (sev == sv) & (spl == split); o = np.argsort(img[m]); return img[m][o], X[m][o]
    rng = np.random.default_rng(a.seed)
    _, Xfit = get('clean', 0, 'fit'); _, Xcal = get('clean', 0, 'cal'); nt, Xte = get('clean', 0, 'test')
    pool = np.vstack([Xfit, Xcal]); n = len(pool)
    folds = np.array_split(rng.permutation(n), 3)
    cal_scores = []
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        r = Ref(pool[tr]); cal_scores.append(r.score(windows(pool[f], min(W, len(f)), 300, rng), W))
    cal_scores = np.concatenate(cal_scores)
    mu_c, sd_c = cal_scores.mean(), cal_scores.std()
    up, low = mu_c + a.k_up * sd_c, mu_c + a.k_low * sd_c
    ref = Ref(pool)
    print(f"images: fit {len(Xfit)} cal {len(Xcal)} test {len(Xte)} | window {W} | reference = {n} clean images")
    print(f"calm windows (cross-fitted, no test images): mean {mu_c:.1f} sd {sd_c:.1f} -> upper {up:.1f} lower {low:.1f}")
    sc_clean = ref.score(windows(Xte, W, 400, np.random.default_rng(11)), W)
    fa = float((sc_clean > up).mean())
    print(f"false-alarm rate on clean TEST windows: {100*fa:.1f}%")
    table = []
    print(f"{'damage':7s} {'sev':>3s} {'AUROC':>6s} {'alarm%':>7s}")
    for dm in ['noise', 'blur', 'dark', 'fog']:
        for sv in [1, 3, 5]:
            if not ((dmg == dm) & (sev == sv)).any(): continue
            _, Xd = get(dm, sv, 'test')
            sd_ = ref.score(windows(Xd, W, 400, np.random.default_rng(12)), W)
            y = np.r_[np.zeros(len(sc_clean)), np.ones(len(sd_))]
            auc = float(roc_auc_score(y, np.r_[sc_clean, sd_])); al = float((sd_ > up).mean())
            table.append({"damage": dm, "severity": sv, "auroc": round(auc, 3), "alarm_rate_damaged": round(al, 3)})
            print(f"{dm:7s} {sv:3d} {auc:6.3f} {100*al:6.0f}%")
    mean_all = float(np.mean([t['auroc'] for t in table]))
    hi = [t['auroc'] for t in table if t['severity'] >= 3]
    mean_hi = float(np.mean(hi))
    print(f"mean AUROC all {len(table)}: {mean_all:.3f} | severity>=3: {mean_hi:.3f} | pre-registered target >= 0.90 -> {'PASS' if mean_all >= 0.90 else 'NOT MET'}")
    # ---- stream replay
    N = a.frames; on, off = int(0.25 * N), int(0.625 * N)
    names = sorted(set(nt)); _, Xd = get(a.stream_corruption, a.stream_severity, 'test')
    nd, _x = get(a.stream_corruption, a.stream_severity, 'test'), None
    ndn, Xd = get(a.stream_corruption, a.stream_severity, 'test')
    idx = {nm: i for i, nm in enumerate(nt)}; idd = {nm: i for i, nm in enumerate(ndn)}
    feats, labs = [], []
    lab = {(i, s, d): l for i, s, d, l in zip(img, sev, dmg, D['label'])}
    for i in range(N):
        nm = nt[(i * 7) % len(nt)]; dam = on <= i < off
        feats.append(Xd[idd[nm]] if dam else Xte[idx[nm]])
        labs.append(int(lab[(nm, a.stream_severity, a.stream_corruption)] if dam else lab[(nm, 0, 'clean')]))
    feats = np.array(feats); Fs = np.full(N, np.nan)
    for t in range(W - 1, N): Fs[t] = ref.score(feats[t - W + 1:t + 1].mean(0, keepdims=True), W)[0]
    mode = np.zeros(N, int); m = 0
    for t in range(N):
        f = Fs[t]
        if not np.isnan(f):
            if m == 0 and f > up: m = 1
            elif m == 1 and f < low: m = 0
        mode[t] = m
    first = next((i for i in range(on, N) if mode[i]), None)
    calm = [i for i in range(N) if not on <= i < off]
    print(f"stream {a.stream_corruption} sev{a.stream_severity}: {mode.sum()}/{N} frames to cloud, alarm delay {None if first is None else first-on}, false-alarm frames in calm {int(mode[calm].sum())}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fe = np.nan_to_num(Fs, nan=0.0)
    frames = [{"frame_idx": i, "scene": "damaged" if on <= i < off else "calm", "free_energy": round(float(fe[i]), 4),
               "pseudo_label": labs[i], "features": [round(float(v), 5) for v in feats[i]]} for i in range(N)]
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(), "data_source": "real_images_coco128_yolov8n_windowed",
            "signal": f"windowed Gaussian free energy, W={W}, Ledoit-Wolf, cross-fitted calibration",
            "split": {"fit": len(Xfit), "calibrate": len(Xcal), "test": len(Xte)},
            "thresholds": {"tau_upper": float(up), "tau_lower": float(low), "k_up": a.k_up, "k_low": a.k_low, "hysteresis": 0.0},
            "stream": {"frames": N, "damaged_from": on, "damaged_to": off, "damage": a.stream_corruption, "severity": a.stream_severity, "warmup_frames": W - 1},
            "pilot1": {"table": table, "mean_auroc": round(mean_all, 3), "mean_auroc_sev_ge3": round(mean_hi, 3),
                       "false_alarm_rate_clean": round(fa, 3), "target_auroc": 0.90, "target_met": bool(mean_all >= 0.90)},
            "replay": {"cloud_frames": int(mode.sum()), "duty_cycle": round(float(mode.mean()), 3),
                       "alarm_delay_frames": None if first is None else int(first - on), "false_alarm_frames_in_calm": int(mode[calm].sum())},
            "seed": a.seed}
    json.dump({"meta": meta, "frames": frames}, open(out / "real_stream_windowed.json", "w"))
    json.dump({"meta": meta}, open(out / "real_pilot1_windowed.json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.2)); ax.axvspan(on, off, color="#f3d9a8", alpha=.5, label="damaged scene")
        ax.plot(Fs, color="#1f3a5f", label=f"windowed surprise (W={W})"); ax.axhline(up, ls="--", color="#1f3a5f", label="upper"); ax.axhline(low, ls="--", color="#17776f", label="lower")
        ax.step(range(N), mode * np.nanmax(Fs) * .95, color="#b3402e", where="post", label="cloud mode"); ax.set_yscale("log"); ax.legend(ncol=3, fontsize=8)
        ax.set_title(f"calm -> {a.stream_corruption} sev{a.stream_severity} -> calm"); fig.tight_layout(); fig.savefig(out / "real_pilot_signal_windowed.png", dpi=160)
    except Exception as e: print("plot skipped", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="logs/real_features.npz"); ap.add_argument("--out", default="logs")
    ap.add_argument("--window", type=int, default=15); ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--stream-corruption", default="blur"); ap.add_argument("--stream-severity", type=int, default=3)
    ap.add_argument("--k-up", type=float, default=3.0); ap.add_argument("--k-low", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    main(ap.parse_args())