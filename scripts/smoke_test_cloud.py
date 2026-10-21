"""Send one schema-valid request to the configured cloud endpoint."""
from __future__ import annotations

import os
import sys
import time

import requests


def main() -> int:
    base_url = os.environ.get("CLOUD_URL")
    if not base_url:
        print("ERROR: CLOUD_URL is not set", file=sys.stderr)
        return 2
    endpoint = base_url.rstrip("/") + "/api/v1/cloud_bp_step"
    payload = {
        "frame_id": 1,
        "features": [0.1] * 256,
        "free_energy": 0.5,
        "pseudo_label": 0,
    }
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["X-API-Key"] = os.environ["API_KEY"]
    started = time.perf_counter()
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
    except requests.RequestException as exc:
        print(f"request_error={exc}", file=sys.stderr)
        return 1
    latency_ms = (time.perf_counter() - started) * 1000.0
    print(f"status={response.status_code}")
    print(f"latency_ms={latency_ms:.3f}")
    print(f"response={response.text}")
    try:
        response_json = response.json()
    except ValueError:
        response_json = {}
    return 0 if response.status_code == 200 and response_json.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
