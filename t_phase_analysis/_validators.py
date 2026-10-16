"""Input validators and time-window builders shared by pmcc, fdetector, and other modules.

All functions are private to the package (``_`` prefix).
"""

import numpy as np


def _require_three_traces(st, caller="function"):
    """Assert the stream has exactly 3 traces."""
    if len(st) != 3:
        raise ValueError(f"{caller} requires exactly 3 traces.")


def _require_positive_parameters(time_one_slice, time_slide, time_length,
                                 azimuth_step=None, speed=None,
                                 speed_min=None, speed_max=None, speed_step=None,
                                 max_cc_shift=None):
    """Validate positive-value parameters used by PMCC / F-detector."""
    if time_one_slice <= 0 or time_slide <= 0 or time_length <= 0:
        raise ValueError("time_one_slice/time_slide/time_length must be positive.")
    if azimuth_step is not None and azimuth_step <= 0:
        raise ValueError("azimuth_step must be positive.")
    if speed is not None and speed <= 0:
        raise ValueError("speed must be positive.")
    if speed_min is not None and speed_min <= 0:
        raise ValueError("speed_min must be positive.")
    if speed_max is not None and speed_max <= 0:
        raise ValueError("speed_max must be positive.")
    if speed_step is not None and speed_step <= 0:
        raise ValueError("speed_step must be positive.")
    if speed_min is not None and speed_max is not None and speed_max <= speed_min:
        raise ValueError("speed_max must be greater than speed_min.")
    if max_cc_shift is not None and max_cc_shift <= 0:
        raise ValueError("max_cc_shift must be positive.")


def _require_sampling_consistency(st):
    """Assert all traces in the stream have consistent sampling rates."""
    srs = np.array([tr.stats.sampling_rate for tr in st], dtype=float)
    if not np.allclose(srs, srs[0], rtol=0.0, atol=1e-6):
        raise ValueError(f"Sampling rates are inconsistent: {srs}")


def _build_time_windows(time_start, time_slide, time_length):
    """Build sliding time windows and return (nwindows, time_offsets)."""
    nwindows = int(np.floor(time_length / time_slide))
    if nwindows < 1:
        raise ValueError("No valid windows: increase time_length or reduce time_slide.")
    return nwindows, time_start + np.arange(nwindows) * time_slide
