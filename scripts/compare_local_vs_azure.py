r"""Compare a run where the cloud step ran on THIS PC with a run where it ran on AZURE.

How to produce the two logs (same stream, same settings, only the server location differs):
  1) start the server on your own PC:     python -m uvicorn azure_server:app --port 8000
     set endpoint_url in config\yolo_controller_config.yaml to http://127.0.0.1:8000/api/v1/cloud_bp_step
     python scripts\run_yolo_surveillance.py --source real --stream logs\real_stream.json
     copy logs\yolo_surveillance_metrics.json  logs\metrics_local_server.json
  2) set endpoint_url back to the Azure address and run the same command again
     copy logs\yolo_surveillance_metrics.json  logs\metrics_azure.json
Then:
  python scripts\compare_local_vs_azure.py --local logs\metrics_local_server.json --azure logs\metrics_azure.json --out logs
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--local", required=True); ap.add_argument("--azure", required=True); ap.add_argument("--out", default="logs")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

def load(p):
    d = json.load(open(p, encoding="utf-8")); t = d["telemetry"]
    off = [x for x in t if x.get("decision") == 1 and x.get("offload_success")]
    g = lambda k: np.array([x[k] for x in off if x.get(k) is not None], float)
    return d, t, off, g
rows = []; data = {}
for name, p in (("local PC", a.local), ("Azure", a.azure)):
    d, t, off, g = load(p); loss, rtt, cc = g("loss"), g("rtt_ms"), g("cloud_compute_ms")
    data[name] = (t, off, loss, rtt)
    rows.append((name, len(t), len(off), f"{np.median(rtt):.1f}" if len(rtt) else "-", f"{np.percentile(rtt, 95):.1f}" if len(rtt) else "-",
                 f"{np.median(cc):.2f}" if len(cc) else "-", f"{loss[0]:.3f}" if len(loss) else "-", f"{loss[-1]:.3f}" if len(loss) else "-",
                 f"{loss[-5:].mean():.3f}" if len(loss) >= 5 else "-", f"{d['summary']['hybrid_energy_j']:.2f}", f"{d['summary']['total_egress_kb']:.1f}"))
hdr = ["server", "frames", "cloud steps", "RTT median ms", "RTT p95 ms", "cloud compute ms", "first loss", "last loss", "last-5 mean", "hybrid energy J", "egress KB"]
w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
for r in rows: print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
(_, _, l1, r1), (_, _, l2, r2) = data["local PC"], data["Azure"]; n = min(len(l1), len(l2))
if n:
    print(f"\nloss difference over the first {n} cloud steps: mean |local - Azure| = {np.mean(np.abs(l1[:n] - l2[:n])):.4f}"
          "  (near 0 only if both servers started from the same weights and saw the same frames in the same order)")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(l1, color="#1f3a5f", label="server on local PC"); ax[0].plot(l2, color="#17776f", ls="--", label="server on Azure"); ax[0].set_xlabel("cloud step"); ax[0].set_ylabel("loss"); ax[0].set_title("loss per cloud step"); ax[0].legend(fontsize=8)
ax[1].boxplot([r1, r2], tick_labels=["local PC", "Azure"]); ax[1].set_ylabel("round trip (ms)"); ax[1].set_title("network delay")
ax[2].bar(["local PC", "Azure"], [float(rows[0][-2]), float(rows[1][-2])], color=["#1f3a5f", "#17776f"]); ax[2].set_title("hybrid energy (J)")
fig.tight_layout(); fig.savefig(out / "compare_local_vs_azure.png", dpi=150); print("saved", out / "compare_local_vs_azure.png")