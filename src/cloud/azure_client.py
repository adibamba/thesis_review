"""AzureCloudOffloader

Provides AzureCloudOffloader.offload_frame(...) to POST bottleneck tensors to an
Azure VM endpoint, measure RTT, compute payload egress size, and return a
structured result. The implementation is resilient to network errors and timeouts
and returns fallback information so the video loop can continue.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

try:
    import numpy as np
except Exception:  # pragma: no cover - handled at runtime
    np = None

try:
    import requests
except Exception:  # pragma: no cover - handled at runtime
    requests = None

logger = logging.getLogger("azure_client")
logger.addHandler(logging.NullHandler())


class AzureCloudOffloader:
    """Offloads bottleneck feature maps to an Azure cloud endpoint via HTTP.

    Usage:
        offloader = AzureCloudOffloader(endpoint_url, timeout_seconds=2.5)
        result = offloader.offload_frame(frame_id, bottleneck_tensor, free_energy)

    Returns a dict with keys: offload_success (bool), rtt_ms (float),
    cloud_compute_ms (float), loss (float or None), egress_kb (float).
    """

    def __init__(self, endpoint_url: str, timeout_seconds: float = 2.5, session: Any = None):
        if requests is None:
            raise RuntimeError("requests is required by AzureCloudOffloader but is not installed")
        if np is None:
            raise RuntimeError("numpy is required by AzureCloudOffloader but is not installed")

        self.endpoint_url = endpoint_url
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()
        self.api_key = os.getenv("API_KEY")
        self._warm_up()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _build_request_body(
        self, frame_id: int, features: list[float], free_energy: float, pseudo_label: int
    ) -> bytes:
        return json.dumps(
            {
                "frame_id": int(frame_id),
                "pseudo_label": int(pseudo_label),
                "free_energy": float(free_energy),
                "features": features,
            },
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def _warm_up(self) -> None:
        """Prime the persistent HTTP session; warm-up is excluded from telemetry."""
        try:
            body = self._build_request_body(0, [0.0] * 256, 0.5, 0)
            response = self.session.post(
                self.endpoint_url,
                data=body,
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            if response.status_code != 200:
                logger.warning("Azure warm-up returned HTTP %s", response.status_code)
        except requests.RequestException as exc:
            logger.warning("Azure warm-up failed: %s", exc)

    def _tensor_to_payload(self, bottleneck_tensor) -> tuple[list[float], int]:
        """Global-average-pool a tensor into the server's 256-float feature vector.

        Channel-wise pooling is used for tensors with 256 channels. For the
        simulation's smaller tensor, the flattened values are partitioned into
        256 bins and averaged so the wire contract remains exactly 256 floats.
        """
        arr = np.asarray(bottleneck_tensor)
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim >= 2 and arr.shape[0] == 256:
            features = arr.mean(axis=tuple(range(1, arr.ndim)))
        else:
            flat = arr.reshape(-1)
            bins = np.array_split(flat, 256)
            features = np.array(
                [chunk.mean() if chunk.size else 0.0 for chunk in bins],
                dtype=np.float32,
            )
        return [float(value) for value in features], 256

    def payload_egress_kb(
        self, frame_id: int, bottleneck_tensor, free_energy: float, pseudo_label: int = 0
    ) -> float:
        features, _ = self._tensor_to_payload(bottleneck_tensor)
        return len(
            self._build_request_body(frame_id, features, free_energy, pseudo_label)
        ) / 1024.0

    def offload_frame(self, frame_id: int, bottleneck_tensor, free_energy: float, pseudo_label: int = 0) -> Dict[str, Any]:
        """Offload a single frame's bottleneck tensor to the cloud.

        Parameters
        - frame_id: integer frame index
        - bottleneck_tensor: numpy array or anything convertible to numpy array
        - free_energy: scalar value representing free energy at the edge
        - pseudo_label: optional int label for debugging/metadata

        Returns a dictionary with these keys:
        - offload_success: bool
        - rtt_ms: float (0.0 on failure)
        - cloud_compute_ms: float (0.0 on failure or if not returned)
        - loss: float or None
        - egress_kb: float
        """
        result = {
            "offload_success": False,
            "rtt_ms": 0.0,
            "cloud_compute_ms": 0.0,
            "loss": None,
            "egress_kb": 0.0,
            "error": None,
            "offload_error": None,
            "http_status": None,
        }
        request_size_bytes = 0
        started = time.perf_counter()

        try:
            payload_tensor, _ = self._tensor_to_payload(bottleneck_tensor)

            request_body = self._build_request_body(
                frame_id, payload_tensor, free_energy, pseudo_label
            )
            headers = self._headers()
            request_size_bytes = len(request_body)
            egress_kb = request_size_bytes / 1024.0
            # Send the exact bytes whose size is recorded as egress.
            resp = self.session.post(
                self.endpoint_url,
                data=request_body,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            rtt_ms = (time.perf_counter() - started) * 1000.0
            result["http_status"] = int(resp.status_code)
            resp.raise_for_status()
            if not 200 <= resp.status_code < 300:
                raise requests.exceptions.HTTPError(
                    f"HTTP {resp.status_code}", response=resp
                )
            if resp.status_code != 200:
                raise requests.exceptions.HTTPError(
                    f"Expected HTTP 200, received {resp.status_code}", response=resp
                )

            result.update({
                "rtt_ms": round(rtt_ms, 3),
                "egress_kb": round(egress_kb, 3),
                "http_status": int(resp.status_code),
            })

            # Parse response JSON for cloud timings/loss
            try:
                j = resp.json()
                # Response contract (best-effort): cloud_compute_ms, loss
                cloud_compute_ms = float(j.get("cloud_compute_ms", 0.0))
                loss = j.get("loss", None)
                try:
                    loss = None if loss is None else float(loss)
                except Exception:
                    # keep raw if not convertible
                    pass

                result.update({
                    "cloud_compute_ms": round(cloud_compute_ms, 3),
                    "loss": loss,
                    "offload_success": j.get("status") == "success",
                })
            except ValueError:
                # non-json response
                logger.debug("Azure response not JSON for frame %s: status %s", frame_id, resp.status_code)

            return result

        except requests.exceptions.Timeout as ex:
            rtt_ms = (time.perf_counter() - started) * 1000.0
            logger.warning("Azure offload timeout for frame %s after %.3fs: %s", frame_id, self.timeout_seconds, ex)
            result.update({
                "rtt_ms": round(rtt_ms, 3),
                "egress_kb": round(float(request_size_bytes) / 1024.0, 3),
                "error": f"{type(ex).__name__}: {ex}",
                "offload_error": f"{type(ex).__name__}: {ex}"[:300],
            })
            return result
        except requests.exceptions.RequestException as ex:
            rtt_ms = (time.perf_counter() - started) * 1000.0
            logger.warning("Azure offload request error for frame %s: %s", frame_id, ex)
            status_code = getattr(getattr(ex, "response", None), "status_code", None)
            if status_code is None:
                status_code = result.get("http_status")
            response_text = getattr(getattr(ex, "response", None), "text", "")
            logger.warning(
                "Azure offload response status=%s body=%s",
                status_code,
                response_text[:300],
            )
            result.update({
                "rtt_ms": round(rtt_ms, 3),
                "egress_kb": round(float(request_size_bytes) / 1024.0, 3),
                "error": f"{type(ex).__name__}: HTTP {status_code}",
                "offload_error": (
                    f"HTTP {status_code}: "
                    f"{response_text}"
                )[:300],
                "http_status": status_code,
            })
            return result
        except Exception as ex:  # pragma: no cover - defensive
            rtt_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("Unexpected error during Azure offload for frame %s: %s", frame_id, ex)
            result.update({
                "rtt_ms": round(rtt_ms, 3),
                "egress_kb": round(float(request_size_bytes) / 1024.0, 3),
                "error": f"{type(ex).__name__}: {ex}",
                "offload_error": f"{type(ex).__name__}: {ex}"[:300],
            })
            return result
