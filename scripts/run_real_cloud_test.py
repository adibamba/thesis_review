r"""Send REAL YOLO features with REAL labels to a cloud BP server, one training step per photo, and log it.

Why: the old runner sent random numbers with label 0, so the server had nothing to learn.
Here every request carries the 256 real numbers of a real photo and its real label.

Usage (server must be running; see the step-by-step guide):
  python scripts\run_real_cloud_test.py --url http://127.0.0.1:8000     --out logs\metrics_local_server.json
  python scripts\run_real_cloud_test.py --url http://20.123.219.119:8000 --out logs\metrics_azure.json
Same --seed and --epochs on both => the same photos in the same order. Call /api/v1/reset (same seed) before each run.
"""
import argparse, json, time
from datetime import datetime
import numpy as np
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True, help="server base address, e.g. http://127.0.0.1:8000")
ap.add_argument("--npz", default="logs/real_features.npz")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=5, help="passes over the clean training photos")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--timeout", type=float, default=10.0)
a = ap.parse_args()
base = a.url.rstrip("/")

d = np.load(a.npz, allow_pickle=False)
img, dmg, spl, X, lab = d["image"], d["damage"], d["split"], d["feature"], d["label"]
m = (dmg == "clean") & np.isin(spl, ["fit", "cal"])          # the 63 clean training photos
order = np.argsort(img[m]); Xs, Ls = X[m][order].astype(np.float32), lab[m][order].astype(int)
assert Xs.shape[1] == 256, "the server expects exactly 256 features"
rng = np.random.default_rng(a.seed)
plan = np.concatenate([rng.permutation(len(Xs)) for _ in range(a.epochs)])
print(f"{len(Xs)} clean photos x {a.epochs} epochs = {len(plan)} cloud steps -> {base}")

# pre-flight: is a server answering at all?
try:
    h = requests.get(base + "/health", timeout=a.timeout); print("server answers /health:", h.status_code, h.text[:150])
except Exception as e:
    raise SystemExit(f"\nCANNOT REACH THE SERVER at {base}\n  {type(e).__name__}: {str(e)[:300]}\n"
                     "Is the server running in another terminal (python -m uvicorn azure_server:app --port 8000), and is the address right?")

def weights():
    try:
        return requests.get(base + "/api/v1/weights", timeout=a.timeout).json()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

w_before = weights()
telemetry = []
for step, i in enumerate(plan):
    payload = {"frame_id": int(step), "features": [float(v) for v in Xs[i]], "pseudo_label": int(Ls[i]), "free_energy": 0.0}
    row = {"frame_idx": int(step), "photo_index": int(i), "label": int(Ls[i]), "decision": 1, "offload_success": False,
           "loss": None, "rtt_ms": 0.0, "cloud_compute_ms": 0.0, "egress_kb": 0.0, "http_status": None, "error": None}
    t0 = time.perf_counter()
    try:
        r = requests.post(base + "/api/v1/cloud_bp_step", json=payload, timeout=a.timeout)
        row["rtt_ms"] = round((time.perf_counter() - t0) * 1000, 3); row["http_status"] = r.status_code
        row["egress_kb"] = round(len(r.request.body) / 1024, 3)
        if r.status_code == 200:
            j = r.json(); row.update(offload_success=True, loss=float(j["loss"]), cloud_compute_ms=float(j["cloud_compute_ms"]))
        else:
            row["error"] = r.text[:200]
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    telemetry.append(row)
    if row["error"] and step < 3:
        print(f"step {step + 1} FAILED: {row['error']}")
    if step == 4 and not any(t["offload_success"] for t in telemetry):
        raise SystemExit("First 5 requests all failed - stopping. Fix the server first (see the error lines above).")
    if (step + 1) % 50 == 0 or step == 0:
        print(f"step {step + 1:4d}  loss={row['loss']}  rtt={row['rtt_ms']} ms")
w_after = weights()

ok = [t for t in telemetry if t["offload_success"]]
loss = np.array([t["loss"] for t in ok])
summary = {"total_frames": len(telemetry), "successful_offloads": len(ok), "url": base, "seed": a.seed, "epochs": a.epochs,
           "hybrid_energy_j": 0.0, "total_egress_kb": round(sum(t["egress_kb"] for t in telemetry), 3),   # energy not modelled in this test
           "first10_loss": round(float(loss[:10].mean()), 4) if len(loss) else None,
           "last10_loss": round(float(loss[-10:].mean()), 4) if len(loss) else None,
           "weights_before": w_before, "weights_after": w_after}
json.dump({"generated_at": datetime.utcnow().isoformat() + "Z", "summary": summary, "telemetry": telemetry}, open(a.out, "w", encoding="utf-8"), indent=2)
print("first-10 loss", summary["first10_loss"], " last-10 loss", summary["last10_loss"], " ->", a.out)