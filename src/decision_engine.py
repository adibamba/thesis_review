"""Controller decisions for the edge/cloud surveillance pipeline."""
from __future__ import annotations


def select_mode(
    current_mode: int,
    free_energy: float,
    tau_upper: float,
    tau_lower: float,
    hysteresis: float,
) -> int:
    """Return the next mode using the configured threshold deadband.

    Mode 0 is local edge processing and mode 1 is cloud Bayesian processing.
    """
    if current_mode == 0 and free_energy > (tau_upper + hysteresis):
        return 1
    if current_mode == 1 and free_energy < (tau_lower - hysteresis):
        return 0
    return current_mode
