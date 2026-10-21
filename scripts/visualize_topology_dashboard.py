"""Streamlit dashboard for edge/cloud surveillance telemetry."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st


LOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "logs", "yolo_surveillance_metrics.json")
)
CLOUD_URL = os.environ.get("CLOUD_URL", "").rstrip("/")

st.set_page_config(page_title="Edge-Cloud Surveillance Telemetry", layout="wide")
st.title("Edge AI Surveillance — Edge/Cloud Telemetry")


@st.cache_data
def load_metrics(path: str, mtime: float) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"generated_at": None, "summary": {}, "telemetry": []}
    with open(path, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


data = load_metrics(LOG_PATH, os.path.getmtime(LOG_PATH) if os.path.exists(LOG_PATH) else 0.0)
telemetry: List[Dict[str, Any]] = data.get("telemetry", [])
if not telemetry:
    st.warning(f"No telemetry found at {LOG_PATH}. Run the surveillance script first.")
    st.stop()

data_source = data.get("data_source") or data.get("summary", {}).get("data_source", "")
if data_source == "simulated":
    st.warning(
        "SIMULATED: free energy, features and compute times are synthetic; "
        "only the Azure round trip is real."
    )
elif str(data_source).startswith("real_images"):
    st.info(
        "REAL YOLOv8n features (coco128, offline replay). Energy values are "
        "modelled constants."
    )
elif data_source == "stub_pipeline_test":
    st.error("Stub pipeline test data is not suitable for production telemetry.")

frames = np.array([item.get("frame_idx", index) for index, item in enumerate(telemetry)])
summary = data.get("summary", {})
meta = data.get("meta") or {}
pilot1 = meta.get("pilot1", {})
replay = meta.get("replay") or summary.get("replay") or {}
energy_model = summary["energy_model"]
edge_ext_power = float(energy_model["p_edge_ext_w"])
tx_energy_per_kb = float(energy_model["p_tx_j_per_kb"])
cloud_overhead = float(energy_model["cloud_overhead_factor_j"])
free_energy = np.array([item.get("free_energy", 0.0) for item in telemetry], dtype=float)
mode = np.array(
    [item.get("decision", item.get("current_mode", item.get("current_state", 0))) for item in telemetry],
    dtype=int,
)
local_latency = np.array([item.get("local_compute_ms", 0.0) for item in telemetry], dtype=float)
rtt = np.array(
    [
        item.get("rtt_ms", np.nan) if item.get("decision", 0) == 1 else np.nan
        for item in telemetry
    ],
    dtype=float,
)
egress = np.array([item.get("egress_kb", 0.0) for item in telemetry], dtype=float)
hybrid_energy = np.array(
    [
        item.get(
            "hybrid_energy_j",
            item.get("cloud_bp_energy_j", 0.0)
            if item.get("decision", 0) == 1 and item.get("offload_success", False)
            else item.get("edge_pc_energy_j", 0.0),
        )
        for item in telemetry
    ],
    dtype=float,
)
baseline_energy = np.array(
    [
        item.get(
            "baseline_bp_energy_j",
            item.get("local_extraction_ms", 0.0) * edge_ext_power
            + item.get("baseline_egress_kb", item.get("egress_kb", 0.0)) * tx_energy_per_kb
            + cloud_overhead,
        )
        for item in telemetry
    ],
    dtype=float,
)
thresholds = summary.get("thresholds", {})
upper = float(thresholds.get("tau_upper_plus_h", 0.83))
lower = float(thresholds.get("tau_lower_minus_h", 0.30))

total_hybrid = float(summary["hybrid_energy_j"])
total_baseline = float(summary["baseline_always_bp_energy_j"])
savings = float(summary["net_energy_savings_pct"])
successful_mask = np.array(
    [item.get("offload_success", False) and item.get("http_status") == 200 for item in telemetry],
    dtype=bool,
)
successful_rtt = rtt[successful_mask]
mean_rtt = float(summary["mean_cloud_rtt_ms"])
median_rtt = float(np.median(successful_rtt) if successful_rtt.size else 0.0)
p95_rtt = float(np.percentile(successful_rtt, 95) if successful_rtt.size else 0.0)
mean_local = float(summary["mean_local_compute_ms"])
total_egress = float(summary["total_egress_kb"])
overhead_excluded = float(summary["net_energy_savings_pct_overhead_excluded"])


def format_data_volume(kilobytes: float) -> str:
    if kilobytes > 1024.0:
        return f"{kilobytes / 1024.0:.2f} MB"
    return f"{kilobytes:.2f} KB"

cards = st.columns(7)
cards[0].metric("Hybrid Energy", f"{total_hybrid:.3f} J")
cards[1].metric("Baseline Always-BP", f"{total_baseline:.3f} J")
cards[2].metric("Net Energy Savings", f"{savings:.2f}%")
cards[2].caption("Overhead counted on both sides")
cards[3].metric("Savings (overhead excluded)", f"{overhead_excluded:.2f}%")
cards[3].caption("Overhead excluded from both sides")
cards[4].metric("Mean Local Compute", f"{mean_local:.3f} ms")
cards[5].metric("Azure RTT", f"{mean_rtt:.3f} ms")
cards[5].caption(f"Median {median_rtt:.3f} ms | p95 {p95_rtt:.3f} ms")
cards[6].metric("Total Egress", format_data_volume(total_egress))
if str(data_source).startswith("real_images"):
    pilot_auroc = summary.get("pilot1_mean_auroc")
    auroc_col, replay_col = st.columns(2)
    auroc_col.metric(
        "Pilot 1 mean AUROC",
        "N/A" if pilot_auroc is None else f"{float(pilot_auroc):.3f}",
    )
    replay_col.metric("Alarm delay", f"{replay.get('alarm_delay_frames', 'N/A')} frames")
    replay_col.caption(f"Cloud frames: {replay.get('cloud_frames', 'N/A')}")

st.subheader("Panel A — Cumulative Energy Consumption")
fig_energy = go.Figure()
fig_energy.add_trace(go.Scatter(x=frames, y=np.cumsum(baseline_energy), name="Baseline: Always Cloud BP"))
fig_energy.add_trace(go.Scatter(x=frames, y=np.cumsum(hybrid_energy), name="Hybrid Adaptive Controller"))
fig_energy.add_trace(
    go.Scatter(
        x=frames,
        y=np.cumsum(baseline_energy) - np.cumsum(hybrid_energy),
        name="Cumulative savings gap",
        fill="tozeroy",
        line=dict(dash="dot"),
    )
)
fig_energy.update_layout(xaxis_title="Frame", yaxis_title="Cumulative energy (J)")
st.plotly_chart(fig_energy, use_container_width=True)

st.subheader("Panel B — Free Energy and Controller State")
fig_state = go.Figure()
display_free_energy = np.where(free_energy > 0.0, free_energy, np.nan)
fig_state.add_trace(go.Scatter(x=frames, y=display_free_energy, name="Surprise F", line=dict(color="#1f77b4")))
fig_state.add_trace(go.Scatter(x=frames, y=np.full(frames.shape, upper), name="Upper switch threshold", line=dict(dash="dash")))
fig_state.add_trace(go.Scatter(x=frames, y=np.full(frames.shape, lower), name="Lower switch threshold", line=dict(dash="dash")))
fig_state.add_hrect(
    y0=lower,
    y1=upper,
    fillcolor="lightgray",
    opacity=0.2,
    line_width=0,
    annotation_text="Hysteresis deadband",
)
if str(data_source).startswith("real_images"):
    damaged_from = summary.get("damaged_from", meta.get("stream", {}).get("damaged_from"))
    damaged_to = summary.get("damaged_to", meta.get("stream", {}).get("damaged_to"))
    if damaged_from is not None and damaged_to is not None:
        fig_state.add_vrect(
            x0=float(damaged_from) - 0.5,
            x1=float(damaged_to) - 0.5,
            fillcolor="#d62728",
            opacity=0.12,
            line_width=0,
            layer="below",
            annotation_text="damaged interval",
        )
fig_state.add_trace(
    go.Scatter(
        x=frames,
        y=mode,
        name="Mode (0=Edge, 1=Cloud)",
        mode="lines",
        line=dict(shape="hv", color="#d62728"),
        yaxis="y2",
    )
)
fig_state.update_layout(
    xaxis_title="Frame",
    yaxis=dict(title="Surprise F", type="log"),
    yaxis2=dict(
        title="Controller mode",
        overlaying="y",
        side="right",
        range=[-0.05, 1.05],
        tickvals=[0, 1],
    ),
)
st.plotly_chart(fig_state, use_container_width=True)

if str(data_source).startswith("real_images"):
    st.subheader("Pilot 1: signal AUROC")
    pilot_table = pilot1.get("table", [])
    if pilot_table:
        st.dataframe(pilot_table, use_container_width=True, hide_index=True)
    severity_rows = [
        float(row["auroc"])
        for row in pilot_table
        if float(row.get("severity", 0)) >= 3 and row.get("auroc") is not None
    ]
    mean_auroc = pilot1.get("mean_auroc", summary.get("pilot1_mean_auroc"))
    false_alarm_rate = pilot1.get("false_alarm_rate_clean")
    target_auroc = pilot1.get("target_auroc", 0.90)
    criterion_met = bool(meta.get("pilot1", {}).get("target_met", False))
    status_text = "MET" if criterion_met else "NOT MET"
    status_color = "green" if criterion_met else "red"
    st.markdown(
        f"Mean AUROC: **{mean_auroc if mean_auroc is not None else 'N/A'}**  \n"
        f"Severity >=3 mean: **{np.mean(severity_rows):.3f}**  \n"
        f"False-alarm rate on clean test windows: **"
        f"{false_alarm_rate if false_alarm_rate is not None else 'N/A'}**  \n"
        f"Target: **{target_auroc:.2f}**  \n"
        f":{status_color}[**{status_text}**]"
    )

st.subheader("Panel C — Latency Breakdown and Egress")
fig_latency = go.Figure()
fig_latency.add_trace(go.Bar(x=frames, y=local_latency, name="Local edge compute (ms)"))
fig_latency.add_trace(go.Scatter(x=frames, y=rtt, name="Azure RTT (ms)", mode="lines+markers", yaxis="y2"))
fig_latency.add_trace(go.Scatter(x=frames, y=egress, name="Egress (KB)", mode="lines", yaxis="y3"))
fig_latency.update_layout(
    xaxis_title="Frame",
    yaxis=dict(title="Local compute (ms)"),
    yaxis2=dict(title="Azure RTT (ms)", overlaying="y", side="right"),
    yaxis3=dict(title="Egress (KB)", anchor="free", overlaying="y", side="right", position=0.95),
)
st.plotly_chart(fig_latency, use_container_width=True)

st.subheader("Panel D — Network Architecture and Node Status")
edge_col, cloud_col, distribution_col = st.columns(3)
edge_col.success("Edge Node: ONLINE\nLocal processing available")
cloud_reachable = False
health_detail = "CLOUD_URL is not set"
if CLOUD_URL:
    try:
        health = requests.get(f"{CLOUD_URL}/health", timeout=2)
        cloud_reachable = 200 <= health.status_code < 300
        health_detail = f"HTTP {health.status_code}"
    except requests.RequestException as exc:
        health_detail = str(exc)
if cloud_reachable:
    cloud_col.success(f"Azure Cloud Node: REACHABLE\n{health_detail}")
else:
    cloud_col.error(f"Azure Cloud Node: UNREACHABLE\n{health_detail}")
cloud_paths = int(np.sum(successful_mask))
edge_paths = len(telemetry) - cloud_paths
controller_edge_paths = int(np.sum(mode == 0))
controller_cloud_paths = int(np.sum(mode == 1))
distribution_col.metric(
    "Frame Distribution",
    f"Controller edge: {controller_edge_paths} | controller cloud: {controller_cloud_paths}",
)
distribution_col.caption(f"Actual path: cloud {cloud_paths} | edge/fallback {edge_paths}")
last_error = next((item.get("offload_error") for item in reversed(telemetry) if item.get("offload_error")), None)
if last_error:
    st.error(f"Last offload error: {last_error}")

topology_dot = """
    digraph G {
      rankdir=LR;
      node [shape=box, style=filled];
      Edge [label="Edge Node\\nLocal", fillcolor=lightgreen];
      Cloud [label="Azure Cloud Node\\nCloud BP", fillcolor=lightblue];
      Edge -> Cloud [label="Mode 1 HTTP POST"];
    }
    """
st.graphviz_chart(topology_dot)

st.subheader("Energy Sensitivity")
sensitivity_rows = summary.get("sensitivity", [])
if sensitivity_rows:
    sensitivity_table = {
        (row["cloud_overhead_j"], row["pc_power_w"]): row["net_savings_pct"]
        for row in sensitivity_rows
    }
    overheads = sorted({row["cloud_overhead_j"] for row in sensitivity_rows})
    powers = sorted({row["pc_power_w"] for row in sensitivity_rows})
    sensitivity_table_rows = []
    for overhead in overheads:
        row = {"cloud_overhead_j": overhead}
        for power in powers:
            row[f"PC {power:g} W savings (%)"] = sensitivity_table[(overhead, power)]
        sensitivity_table_rows.append(row)
    st.dataframe(sensitivity_table_rows, use_container_width=True)

st.subheader("Comparative Summary")
st.json(summary)
st.subheader("Recent Telemetry Samples")
st.dataframe(telemetry[-50:], use_container_width=True)
st.caption(f"Metrics file: {LOG_PATH}")
