# Adaptive Energy-Aware Learning Selection between Predictive Coding and Backpropagation for Edge Vision Surveillance

MSc thesis project. An edge device watches a camera stream. A controller decides, frame by frame, whether to adapt
**locally with predictive coding (PC)** or to **offload to the cloud for backpropagation (BP)**, using a "surprise" signal
(free energy) and an energy model.

## Status (honest summary)
| Part | State |
|---|---|
| Cloud BP server (`server/azure_server.py`, Linear(256,80), Adam 1e-3, cross-entropy) | working locally and on Azure |
| PC and BP side by side with self-test (`src/learning/pc_vs_bp.py`) | working; PC update ~ BP gradient (cosine ~0.99) |
| Learning evidence on real YOLOv8n features (`scripts/learning_evidence.py`) | loss falls, weights move; **no generalisation shown with 63 training photos** |
| Windowed detection signal (`scripts/score_window.py`) | pilot result 0.80 overall, 0.88 for severity >= 3; target 0.90 not met |
| Energy constants (`config/yolo_controller_config.yaml`) | **placeholders**, to be replaced by measurements on the Raspberry Pi |
| Local-vs-Azure comparison (`scripts/compare_local_vs_azure.py`) | script ready; results to be added |

## Layout
```
config/   controller and energy settings (yaml)
server/   cloud BP server (FastAPI + PyTorch)
src/      decision_engine.py, cloud client, learning/pc_vs_bp.py
scripts/  experiments: extract_real_features, score_window, learning_evidence,
          run_real_cloud_test, compare_local_vs_azure, bench_edge, dashboard
docs/     RASPBERRY_PI.md  (how to test on the edge device)
logs/     outputs (mostly git-ignored)
```

## Quick start (Windows PowerShell shown; Linux/macOS use `source .venv/bin/activate`)
```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\learning\pc_vs_bp.py                      # self-test: PC vs BP, expect "all checks passed"
python scripts\learning_evidence.py --npz logs\real_features.npz --out logs
```
Cloud server (separate terminal, from `server/`):
```
python -m pip install -r requirements-server.txt
python -m uvicorn azure_server:app --port 8000
```
Test it (from the project root, second terminal):
```
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/reset -ContentType "application/json" -Body '{"seed":0}'
python scripts\run_real_cloud_test.py --url http://127.0.0.1:8000 --out logs\metrics_local_server.json
```
Always use `python -m pip`, not bare `pip`, if the project folder was ever moved (stale venv launchers).

## Reproducing the local-vs-Azure comparison
Run the same server code in both places, call `/api/v1/reset` with the same seed on each, run `run_real_cloud_test.py`
with the same `--epochs` against each, then `compare_local_vs_azure.py`. Expect near-identical loss curves and very
different round-trip times.

## Noise / damage injected (real-feature pilot)
Four types, **applied one at a time, never combined**, severity 1-5: Gaussian noise, blur, darkening, fog.
The original simulation (`run_yolo_surveillance.py`) injects no image damage; it uses a sine-wave free-energy signal.

## Known limits
- Held-out loss and accuracy are not better than a majority-class guess at 63 training photos; more data needed (COCO val2017 subset).
- Energy model constants are placeholders until measured on the target device.
- The 4 damage types are synthetic; real-camera damage is untested.

## Security
The server has no authentication. Restrict the VM firewall (ports 22 and 8000) to your own IP, and add an API key before any wider use.
Never commit `.pem`/`.key` files or passwords.

## Licences and data
- Ultralytics YOLOv8 is AGPL-3.0; check the implications before making this repository public or distributing it.
- COCO images carry their own licences (annotations CC BY 4.0, images per Flickr terms); the images are not included here.