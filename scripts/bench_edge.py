r"""Measure what the edge device (your PC, or the Raspberry Pi) really costs per learning step.

Run it on the machine you want to measure:
  python scripts/bench_edge.py                                   # compute timing only
  python scripts/bench_edge.py --url http://20.123.219.119:8000  # plus network delay to the server
Writes logs/edge_bench_<machine>.json. Replace the placeholder '12 ms' in the energy model with the
number measured here, and multiply by a measured power draw (USB power meter) to get joules.
"""
import argparse, importlib.util, json, platform, socket, statistics as st, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pc_vs_bp", ROOT / "src" / "learning" / "pc_vs_bp.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200, help="timed repetitions")
ap.add_argument("--url", default=None, help="server base address for a network-delay test")
a = ap.parse_args()

def timeit(fn, n):
    for _ in range(5): fn()                      # warm-up
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(st.median(t), 3), "p95_ms": round(float(np.percentile(t, 95)), 3)}

def cpu_temp():
    try: return round(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000, 1)
    except Exception: return None

W1, W2 = m.init_weights(seed=0)
r = np.random.default_rng(0); x = r.standard_normal(256); y = np.eye(80)[7]
out = {"machine": socket.gethostname(), "platform": platform.platform(), "processor": platform.processor() or platform.machine(),
       "python": platform.python_version(), "numpy": np.__version__, "cpu_temp_c_before": cpu_temp(),
       "bp_update": timeit(lambda: m.bp_update(W1, W2, x, y), a.n),
       "pc_update_60_settling_steps": timeit(lambda: m.pc_update(W1, W2, x, y, steps=60), a.n),
       "pc_update_5_settling_steps": timeit(lambda: m.pc_update(W1, W2, x, y, steps=5), a.n)}
out["cpu_temp_c_after"] = cpu_temp()
if a.url:
    import requests
    base = a.url.rstrip("/"); rtt = []
    for _ in range(30):
        t0 = time.perf_counter(); requests.get(base + "/health", timeout=10); rtt.append((time.perf_counter() - t0) * 1000)
    out["network_to_server_ms"] = {"url": base, "median_ms": round(st.median(rtt), 2), "p95_ms": round(float(np.percentile(rtt, 95)), 2)}
Path(ROOT / "logs").mkdir(exist_ok=True)
p = ROOT / "logs" / f"edge_bench_{out['machine']}.json"; json.dump(out, open(p, "w"), indent=2)
print(json.dumps(out, indent=2)); print("saved", p)