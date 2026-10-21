# Testing on a Raspberry Pi

Goal: replace the placeholder energy and timing constants with **measurements from the real edge device**, and check
that the edge-to-cloud path works from a real edge network.

The Pi plays the **edge** (camera side, PC learning, the client). The cloud BP server stays on Azure.

## 0. What you need
- Raspberry Pi 4 or 5, 64-bit Raspberry Pi OS (Bookworm), network access, SSH enabled.
- USB power meter (inline, USB-C) to read watts. Without it you can report time, not joules.
- The Azure server running, address `http://<VM_IP>:8000`, with your Pi's public IP allowed in the firewall (port 8000).

## 1. Get the code
```
sudo apt update && sudo apt install -y git python3-venv python3-pip
git clone https://github.com/<YOUR_USER>/<REPO>.git && cd <REPO>
python3 -m venv .venv && source .venv/bin/activate
python -m pip install numpy requests pyyaml matplotlib
```
(Private repo: use a personal access token, or add an SSH key to GitHub.)
For YOLO feature extraction also install `opencv-python-headless` and `ultralytics`. On a Pi this is large and slow;
start with the numpy-only tests below.

## 2. Tests, in this order
| # | Command | What it tells you |
|---|---|---|
| 1 | `python src/learning/pc_vs_bp.py` | code runs on ARM; expect "all checks passed" |
| 2 | `python scripts/bench_edge.py` | milliseconds per PC and BP update on the Pi |
| 3 | `python scripts/bench_edge.py --url http://<VM_IP>:8000` | adds the real network delay Pi -> Azure |
| 4 | `curl http://<VM_IP>:8000/health` | server reachable from the Pi |
| 5 | reset Azure (seed 0), then `python scripts/run_real_cloud_test.py --url http://<VM_IP>:8000 --out logs/metrics_pi_azure.json` | the full edge->cloud BP loop from the Pi (needs `logs/real_features.npz` copied over with scp) |
| 6 | YOLO extraction on the Pi (optional) | time per frame for the real backbone |

## 3. Measuring energy (what makes the numbers real)
1. Plug the Pi's power through the USB meter. Note idle watts.
2. Run a loop of the PC step (or `bench_edge.py` with a large `--n`) and note watts while it runs.
3. Do the same during the cloud step (the Pi sends and waits).
4. Energy per step = (watts under load) x (seconds per step). Subtract idle if you want the extra energy only.
5. Put the measured values into `config/yolo_controller_config.yaml` (`energy:` section) and mark them "measured on Raspberry Pi <model>, <date>".

Record: Pi model, RAM, OS version, cooling, CPU temperature (`bench_edge.py` logs it when available), room temperature.

## 4. What `bench_edge.py` does and does not measure
It times the small PC/BP network update in numpy (256 -> 64 -> 80). It does **not** time the YOLO backbone.
The backbone time is a separate number (test 6) and probably dominates on a Pi.

## 5. Expected pitfalls
- Wrong CPU architecture wheels: use 64-bit OS; if a package fails to install, try `pip install --prefer-binary`.
- `ultralytics`/`torch` are heavy on a Pi; consider exporting YOLOv8n to ONNX or NCNN for speed later.
- Thermal throttling during long runs: log temperature and use a heatsink or fan.
- Azure firewall: add the Pi's public IP; the home/lab IP can change.
- Use the same `--epochs` and seed as the laptop run when comparing.

## 6. Report template
Pi model / OS / python / numpy versions; ms per PC and BP update (median, p95); network delay Pi -> Azure; idle and load watts;
joules per step for local and cloud paths; the break-even point (cloud is cheaper only if the cloud path's energy is below the local one).