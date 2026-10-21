"""Run YOLO surveillance pipeline with Azure cloud offloading integration.

This script is written to be robust when the rest of the full repository is not
present. It attempts to integrate with a YOLO/predictive-coding pipeline if
available, otherwise it simulates a bottleneck tensor and a free_energy
signal so telemetry can be exercised and exported.

Telemetry is written to: logs/yolo_surveillance_metrics.json

The script reads config/yolo_controller_config.yaml for thresholds and cloud
settings and uses src/cloud/azure_client.AzureCloudOffloader to perform POSTs
when the controller indicates Mode 1 (Cloud BP).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

# Make repository packages importable when launched by file path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.cloud.azure_client import AzureCloudOffloader
except Exception:
    try:
        from cloud.azure_client import AzureCloudOffloader  # type: ignore
    except Exception:
        AzureCloudOffloader = None  # type: ignore

try:
    from src.decision_engine import select_mode
except Exception:
    try:
        from decision_engine import select_mode  # type: ignore
    except Exception:
        select_mode = None  # type: ignore


LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
METRICS_PATH = LOG_DIR / "yolo_surveillance_metrics.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_yolo_surveillance")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def simulate_bottleneck(frame_idx: int) -> np.ndarray:
    # Simulate a compact bottleneck feature map (e.g., 64x64x8)
    rng = np.random.default_rng(seed=frame_idx)
    return rng.standard_normal((8, 64, 64)).astype(np.float32)


def resolve_cloud_url(config: Dict[str, Any]) -> str:
    configured = str(config.get("azure_cloud", {}).get("endpoint_url", "")).strip()
    base_url = os.environ.get("CLOUD_URL", "").strip().rstrip("/")
    if configured.startswith("${CLOUD_URL}"):
        suffix = configured[len("${CLOUD_URL}"):]
        if not base_url:
            raise RuntimeError("CLOUD_URL is required when endpoint_url uses ${CLOUD_URL}")
        return base_url + suffix
    if base_url:
        return base_url + "/api/v1/cloud_bp_step"
    if configured:
        return configured
    raise RuntimeError("Set CLOUD_URL or configure azure_cloud.endpoint_url")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_real_stream(path: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    stream_path = Path(path)
    if not stream_path.exists():
        raise FileNotFoundError(
            f"Real stream not found: {stream_path}. "
            "Provide a JSON file with {\"meta\": ..., \"frames\": [...]}."
        )
    with stream_path.open("r", encoding="utf-8") as stream_file:
        payload = json.load(stream_file)
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise ValueError("Real stream must be an object containing a list-valued 'frames' field")
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("Real stream 'meta' must be an object")
    for index, frame in enumerate(payload["frames"]):
        if not isinstance(frame, dict):
            raise ValueError(f"Real stream frame {index} must be an object")
        required = ("free_energy", "features", "pseudo_label")
        missing = [key for key in required if key not in frame]
        if missing:
            raise ValueError(f"Real stream frame {index} is missing: {', '.join(missing)}")
        if len(frame["features"]) != 256:
            raise ValueError(
                f"Real stream frame {index} must contain exactly 256 features; "
                f"got {len(frame['features'])}"
            )
    return meta, payload["frames"]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(PROJECT_ROOT) / "config" / "yolo_controller_config.yaml"))
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to run (simulation) or until stopped")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source", choices=("sim", "real"), default="sim")
    parser.add_argument(
        "--stream",
        default=str(Path(PROJECT_ROOT) / "logs" / "real_stream_windowed.json"),
        help="JSON stream used by --source real",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if select_mode is None:
        raise RuntimeError("select_mode could not be imported from src.decision_engine")

    config = load_config(args.config)
    real_meta: Dict[str, Any] = {}
    real_frames: List[Dict[str, Any]] = []
    if args.source == "real":
        real_meta, real_frames = load_real_stream(args.stream)
        if not real_frames:
            raise ValueError(f"Real stream contains no frames: {args.stream}")
    azure_conf = config.get("azure_cloud", {})
    azure_enabled = bool(azure_conf.get("enabled", False))
    endpoint_url = resolve_cloud_url(config)
    timeout_seconds = float(azure_conf.get("timeout_seconds", 2.5))

    # Initialize offloader if available and enabled
    offloader = None
    if azure_enabled:
        if AzureCloudOffloader is None:
            logger.error("AzureCloudOffloader class not found. Ensure src/cloud/azure_client.py is importable and dependencies (requests, numpy) are installed.")
            azure_enabled = False
        else:
            offloader = AzureCloudOffloader(endpoint_url=endpoint_url, timeout_seconds=timeout_seconds)
            logger.info("Azure offloader configured to %s (timeout %.2fs)", endpoint_url, timeout_seconds)

    # Controller thresholds
    if args.source == "real":
        stream_thresholds = real_meta.get("thresholds", {})
        tau_upper = float(stream_thresholds["tau_upper"])
        tau_lower = float(stream_thresholds["tau_lower"])
        safety_hysteresis = 0.0
    else:
        tau_upper = float(config.get("tau_upper", 0.78))
        tau_lower = float(config.get("tau_lower", 0.35))
        safety_hysteresis = float(config.get("safety_hysteresis", 0.05))
    energy_conf = config.get("energy", {})
    edge_pc_power = float(energy_conf["p_edge_pc_w"])
    edge_extraction_power = float(energy_conf["p_edge_ext_w"])
    transmit_power = float(energy_conf["p_tx_j_per_kb"])
    cloud_overhead_factor = float(energy_conf.get("cloud_overhead_factor_j", 0.293))
    wait_power = float(energy_conf.get("p_wait_j_per_ms", 0.0))
    breaker_conf = config.get("circuit_breaker", {})
    max_failures = int(breaker_conf.get("max_failures", 3))
    cooldown_frames = int(breaker_conf.get("cooldown_frames", 20))
    warmup_frames = (
        int(real_meta.get("stream", {}).get("warmup_frames", 0))
        if args.source == "real"
        else 0
    )

    # Telemetry list
    telemetry: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed=args.seed)

    # Simple hysteresis mode controller simulation
    current_state = 0  # 0 = Edge PC (local), 1 = Cloud BP
    consecutive_failures = 0
    cooldown_remaining = 0
    frame_count = len(real_frames) if args.source == "real" else args.frames
    for frame_idx in range(frame_count):
        ts = datetime.now(timezone.utc).isoformat()

        if args.source == "real":
            stream_frame = real_frames[frame_idx]
            free_energy = float(stream_frame["free_energy"])
            pseudo_label = int(stream_frame["pseudo_label"])
            bottleneck = np.asarray(stream_frame["features"], dtype=np.float32)
        else:
            # Simulate free_energy as noisy sinusoid + noise to create transitions
            t = frame_idx / max(1, args.frames) * 8.0
            free_energy = 0.5 + 0.45 * np.sin(t) + rng.normal(scale=0.03)
            free_energy = float(np.clip(free_energy, 0.0, 1.0))
            pseudo_label = 0
            bottleneck = simulate_bottleneck(frame_idx)

        # Hysteresis-based controller decision. The deadband preserves state.
        if frame_idx < warmup_frames:
            current_state = 0
        elif cooldown_remaining > 0:
            current_state = 0
            cooldown_remaining -= 1
        else:
            current_state = select_mode(
                current_state, free_energy, tau_upper, tau_lower, safety_hysteresis
            )

        baseline_egress_kb = (
            offloader.payload_egress_kb(frame_idx, bottleneck, free_energy, pseudo_label)
            if offloader is not None
            else 0.0
        )

        # Compute local processing and feature-extraction times (simulate).
        local_compute_ms = float(max(0.5, rng.normal(loc=6.0 if current_state == 1 else 12.0, scale=1.5)))
        local_extraction_ms = float(max(0.5, rng.normal(loc=6.0, scale=0.75)))

        rtt_ms = 0.0
        cloud_compute_ms = 0.0
        loss = None
        egress_kb = 0.0
        offload_success = False
        offload_error = None
        offload_error_detail = None
        http_status = None

        if current_state == 1 and azure_enabled and offloader is not None:
            # Offload to Azure
            try:
                offload_res = offloader.offload_frame(
                    frame_idx, bottleneck, free_energy, pseudo_label=pseudo_label
                )
                rtt_ms = float(offload_res.get("rtt_ms", 0.0))
                cloud_compute_ms = float(offload_res.get("cloud_compute_ms", 0.0))
                loss = offload_res.get("loss", None)
                egress_kb = float(offload_res.get("egress_kb", 0.0))
                offload_success = bool(offload_res.get("offload_success", False))
                offload_error = offload_res.get("offload_error") or offload_res.get("error")
                offload_error_detail = offload_error
                http_status = offload_res.get("http_status")

                if not offload_success:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        cooldown_remaining = cooldown_frames
                    # network fallback: do local compute instead
                    logger.debug("Frame %d: offload failed, falling back to local compute", frame_idx)
                    # simulate local compute penalty
                    local_compute_ms += 5.0
                else:
                    consecutive_failures = 0
            except Exception as ex:
                logger.exception("Unexpected error while offloading frame %d: %s", frame_idx, ex)
                offload_error = f"{type(ex).__name__}: {ex}"
        else:
            # Mode 0: local/predictive coding path — no egress
            rtt_ms = 0.0
            egress_kb = 0.0

        # Mode 0: full local edge-PC processing.
        edge_pc_energy_j = local_compute_ms * edge_pc_power
        # Mode 1: local feature extraction plus transmission to cloud BP.
        cloud_bp_energy_j = (
            local_extraction_ms * edge_extraction_power
            + egress_kb * transmit_power
            + cloud_overhead_factor
        )
        # Baseline: always perform cloud BP, including the fixed cloud overhead.
        baseline_bp_energy_j = (
            local_extraction_ms * edge_extraction_power
            + baseline_egress_kb * transmit_power
            + cloud_overhead_factor
        )
        if current_state == 1 and offload_success:
            hybrid_energy_j = cloud_bp_energy_j
        elif current_state == 1:
            hybrid_energy_j = (
                edge_pc_energy_j
                + egress_kb * transmit_power
                + rtt_ms * wait_power
            )
        else:
            hybrid_energy_j = edge_pc_energy_j
        hybrid_energy_excluded_j = (
            hybrid_energy_j - cloud_overhead_factor
            if current_state == 1 and offload_success
            else hybrid_energy_j
        )
        baseline_excluded_j = baseline_bp_energy_j - cloud_overhead_factor

        telemetry_item = {
                "frame_idx": int(frame_idx),
                "timestamp": ts,
                "current_state": int(current_state),
                "current_mode": int(current_state),
                "decision": int(current_state),
                "free_energy": float(round(free_energy, 6)),
                "local_compute_ms": float(round(local_compute_ms, 3)),
                "local_extraction_ms": float(round(local_extraction_ms, 3)),
                "rtt_ms": float(round(rtt_ms, 3)),
                "cloud_compute_ms": float(round(cloud_compute_ms, 3)),
                "offload_success": offload_success,
                "offload_error": offload_error,
                "offload_error_detail": offload_error_detail,
                "http_status": http_status,
                "loss": (None if loss is None else float(loss)),
                "egress_kb": float(round(egress_kb, 3)),
                "baseline_egress_kb": float(round(baseline_egress_kb, 3)),
                "edge_pc_energy_j": float(round(edge_pc_energy_j, 6)),
                "cloud_bp_energy_j": float(round(cloud_bp_energy_j, 6)),
                "hybrid_energy_j": float(round(hybrid_energy_j, 6)),
                "hybrid_energy_overhead_excluded_j": float(round(hybrid_energy_excluded_j, 6)),
                "baseline_bp_energy_j": float(round(baseline_bp_energy_j, 6)),
                "baseline_bp_energy_overhead_excluded_j": float(round(baseline_excluded_j, 6)),
            }
        if args.source == "real":
            telemetry_item.update(
                {
                    "scene": stream_frame.get("scene"),
                    "image": stream_frame.get("image"),
                    "yolo_ms_measured": stream_frame.get("yolo_ms"),
                }
            )
        telemetry.append(telemetry_item)

        # Sleep a small amount to simulate real-time (can be tuned or removed)
        time.sleep(0.02)

    # Persist metrics and a comparative summary to JSON.
    hybrid_energy = sum(item["hybrid_energy_j"] for item in telemetry)
    baseline_bp_energy = sum(item["baseline_bp_energy_j"] for item in telemetry)
    hybrid_excluded = sum(item["hybrid_energy_overhead_excluded_j"] for item in telemetry)
    baseline_excluded = sum(item["baseline_bp_energy_overhead_excluded_j"] for item in telemetry)
    offloaded = [item for item in telemetry if item["decision"] == 1]
    successful_offloads = [
        item for item in offloaded
        if item["offload_success"] and item.get("http_status") == 200
    ]
    summary = {
        "total_frames": len(telemetry),
        "mode_0_frames": sum(item["current_state"] == 0 for item in telemetry),
        "mode_1_frames": len(offloaded),
        "successful_offloads": len(successful_offloads),
        "hybrid_energy_j": round(hybrid_energy, 6),
        "baseline_always_bp_energy_j": round(baseline_bp_energy, 6),
        "net_energy_savings_pct": round(
            0.0 if baseline_bp_energy == 0 else
            (1.0 - hybrid_energy / baseline_bp_energy) * 100.0,
            3,
        ),
        "net_energy_savings_pct_overhead_excluded": round(
            0.0 if baseline_excluded == 0 else
            (1.0 - hybrid_excluded / baseline_excluded) * 100.0,
            3,
        ),
        "mean_local_compute_ms": round(
            sum(item["local_compute_ms"] for item in telemetry) / max(1, len(telemetry)), 3
        ),
        "mean_cloud_rtt_ms": round(
            sum(item["rtt_ms"] for item in successful_offloads) / max(1, len(successful_offloads)), 3
        ),
        "total_egress_kb": round(sum(item["egress_kb"] for item in telemetry), 3),
        "energy_model": {
            "p_edge_pc_w": edge_pc_power,
            "p_edge_ext_w": edge_extraction_power,
            "p_tx_j_per_kb": transmit_power,
            "p_wait_j_per_ms": wait_power,
            "cloud_overhead_factor_j": cloud_overhead_factor,
        },
        "data_source": (
            "real_images_coco128_yolov8n_windowed"
            if args.source == "real"
            else "simulated"
        ),
        "seed": args.seed,
        "config_snapshot": config,
        "git_commit": git_commit(),
        "dataset": real_meta.get("dataset") if args.source == "real" else None,
        "pilot1_mean_auroc": (
            real_meta.get("pilot1", {}).get("mean_auroc")
            if args.source == "real"
            else None
        ),
        "pilot1_mean_auroc_two_sided": (
            real_meta.get("pilot1", {}).get("mean_auroc_two_sided")
            if args.source == "real"
            else None
        ),
        "stream": real_meta.get("stream") if args.source == "real" else None,
        "damaged_from": (
            real_meta.get("stream", {}).get("damaged_from")
            if args.source == "real"
            else None
        ),
        "damaged_to": (
            real_meta.get("stream", {}).get("damaged_to")
            if args.source == "real"
            else None
        ),
        "replay": real_meta.get("replay") if args.source == "real" else None,
        "thresholds": {
            "tau_upper_plus_h": tau_upper + safety_hysteresis,
            "tau_lower_minus_h": tau_lower - safety_hysteresis,
        },
    }
    sensitivity = []
    for overhead in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
        for pc_power_w in (5.0, 10.0, 21.5):
            hybrid_sensitive = 0.0
            baseline_sensitive = 0.0
            for item in telemetry:
                baseline_sensitive += (
                    item["local_extraction_ms"] * edge_extraction_power
                    + item["baseline_egress_kb"] * transmit_power
                    + overhead
                )
                if item["decision"] == 1 and item["offload_success"]:
                    hybrid_sensitive += (
                        item["local_extraction_ms"] * edge_extraction_power
                        + item["egress_kb"] * transmit_power
                        + overhead
                    )
                elif item["decision"] == 1:
                    hybrid_sensitive += (
                        item["local_compute_ms"] * pc_power_w / 1000.0
                        + item["egress_kb"] * transmit_power
                        + item["rtt_ms"] * wait_power
                    )
                else:
                    hybrid_sensitive += item["local_compute_ms"] * pc_power_w / 1000.0
            sensitivity.append(
                {
                    "cloud_overhead_j": overhead,
                    "pc_power_w": pc_power_w,
                    "net_savings_pct": round(
                        (1.0 - hybrid_sensitive / baseline_sensitive) * 100.0
                        if baseline_sensitive else 0.0,
                        3,
                    ),
                }
            )
    summary["sensitivity"] = sensitivity
    with open(METRICS_PATH, "w", encoding="utf-8") as out:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "data_source": summary["data_source"],
                "meta": real_meta if args.source == "real" else None,
                "summary": summary,
                "telemetry": telemetry,
            },
            out,
            indent=2,
        )

    logger.info("Telemetry written to %s (frames=%d)", METRICS_PATH, len(telemetry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
