"""
F-statistic 检测器：频率域相干谱扫描。

对 3 台站阵列计算每个时间窗 / 方位角的 F 统计量，
F 值越高表示该方向 / 时刻的波场越相干。

作者: Yong Zhou (zhouyong@scsio.ac.cn)

调用示例
--------
```python
from t_phase_analysis import f_statistic, plot_fdetector_diagnostics

res = f_statistic(st, time_start=0.0, time_one_slice=10.0, time_slide=2.0, time_length=200.0)
fig, axs = plot_fdetector_diagnostics(res)
print(res["meta"]["valid_window_ratio"])
```
"""

import numpy as np
from obspy.geodetics.base import gps2dist_azimuth

# F-detector 默认配置：集中定义，函数内部仅引用
DEFAULT_FD_SPEED = 1480.0
DEFAULT_FD_AZIMUTH_STEP = 1.0
DEFAULT_FD_MIN_SAMPLES_PER_WINDOW = 8
DEFAULT_FD_THRESHOLD_PERCENTILE = 75.0


def _require_three_traces(st):
    if len(st) != 3:
        raise ValueError("F-statistic requires exactly 3 traces.")


def _require_positive_parameters(time_one_slice, time_slide, time_length, speed, azimuth_step):
    if time_one_slice <= 0 or time_slide <= 0 or time_length <= 0:
        raise ValueError("time_one_slice/time_slide/time_length must be positive.")
    if speed <= 0:
        raise ValueError("speed must be positive.")
    if azimuth_step <= 0:
        raise ValueError("azimuth_step must be positive.")


def _require_sampling_consistency(st):
    srs = np.array([tr.stats.sampling_rate for tr in st], dtype=float)
    if not np.allclose(srs, srs[0], rtol=0.0, atol=1e-6):
        raise ValueError(f"Sampling rates are inconsistent: {srs}")


def _require_sac_geometry(st):
    for idx, tr in enumerate(st):
        if not hasattr(tr.stats, "sac"):
            raise ValueError(f"Trace {idx} has no SAC header.")
        for key in ("stla", "stlo"):
            if key not in tr.stats.sac:
                raise ValueError(f"Trace {idx} SAC header missing '{key}'.")


def _build_time_windows(time_start, time_slide, time_length):
    nwindows = int(np.floor(time_length / time_slide))
    if nwindows < 1:
        raise ValueError("No valid windows: increase time_length or reduce time_slide.")
    time_windows = time_start + np.arange(nwindows) * time_slide
    return nwindows, time_windows


def f_statistic(st, time_start, time_one_slice, time_slide, time_length,
                speed=DEFAULT_FD_SPEED,
                azimuth_step=DEFAULT_FD_AZIMUTH_STEP,
                min_samples_per_window=DEFAULT_FD_MIN_SAMPLES_PER_WINDOW):
    """
    F-statistic 检测器：频率域相干谱扫描。

    对 3 台站计算每个时间窗 / 方位角的 F 统计量，
    F 值越高表示该方向 / 时刻的波场越相干。

    Parameters
    ----------
    st : obspy.Stream
        恰好 3 道波形。
    time_start : float
        起始时间 (s，相对 st[0].stats.starttime)。
    time_one_slice : float
        每窗长度 (s)。
    time_slide : float
        窗滑动步长 (s)。
    time_length : float
        总时长 (s)。
    speed : float
        平面波视速度 (m/s)，默认 DEFAULT_FD_SPEED。
    azimuth_step : float
        方位角步长 (度)，默认 DEFAULT_FD_AZIMUTH_STEP。
    min_samples_per_window : int
        每窗最少有效采样点，默认 DEFAULT_FD_MIN_SAMPLES_PER_WINDOW。

    Returns
    -------
    dict
        time_windows, azimuths, F_statistic (n_azimuth × n_windows)
    """
    _require_three_traces(st)
    _require_positive_parameters(time_one_slice, time_slide, time_length, speed, azimuth_step)
    _require_sampling_consistency(st)
    _require_sac_geometry(st)

    nwindows, time_windows = _build_time_windows(time_start, time_slide, time_length)
    dist_meter_12, az_12, baz_12 = gps2dist_azimuth(
        st[0].stats.sac.stla, st[0].stats.sac.stlo,
        st[1].stats.sac.stla, st[1].stats.sac.stlo)
    dist_meter_13, az_13, baz_13 = gps2dist_azimuth(
        st[0].stats.sac.stla, st[0].stats.sac.stlo,
        st[2].stats.sac.stla, st[2].stats.sac.stlo)
    deg2rad = np.pi / 180.0

    azimuths = np.arange(0, 360.0, azimuth_step)
    nazimuths = len(azimuths)
    F = np.full((nazimuths, nwindows), np.nan, dtype=float)
    valid_counts = np.zeros(nwindows, dtype=int)

    for iaz, az in enumerate(azimuths):
        dt12_pred = -dist_meter_12 * np.cos((az_12 - az) * deg2rad) / speed
        dt13_pred = -dist_meter_13 * np.cos((az_13 - az) * deg2rad) / speed

        for iw in range(nwindows):
            t_pick = st[0].stats.starttime + time_start + iw * time_slide
            tr0 = st[0].slice(t_pick, t_pick + time_one_slice)
            tr1 = st[1].slice(t_pick + dt12_pred, t_pick + time_one_slice + dt12_pred)
            tr2 = st[2].slice(t_pick + dt13_pred, t_pick + time_one_slice + dt13_pred)

            try:
                tr_data = np.vstack([tr.data for tr in [tr0, tr1, tr2]])
            except ValueError:
                min_len = min(tr0.data.size, tr1.data.size, tr2.data.size)
                tr_data = np.vstack([tr.data[:min_len] for tr in [tr0, tr1, tr2]])

            if tr_data.shape[1] < min_samples_per_window:
                continue

            F_num = np.sum(np.sum(tr_data, axis=0)**2)
            F_den = np.sum(3.0 * np.var(tr_data, axis=0))
            F[iaz, iw] = (2.0 / 3.0) * (F_num / max(F_den, 1e-20))
            valid_counts[iw] += 1

    F_max = np.nanmax(F, axis=0)
    azimuth_peak = azimuths[np.nanargmax(np.where(np.isnan(F), -np.inf, F), axis=0)]
    valid_window_mask = np.isfinite(F_max)

    return {
        'time_windows': time_windows,
        'azimuths': azimuths,
        'F_statistic': F,
        'F_max': F_max,
        'azimuth_peak': azimuth_peak,
        'valid_window_mask': valid_window_mask,
        'meta': {
            'speed': speed,
            'azimuth_step': azimuth_step,
            'min_samples_per_window': min_samples_per_window,
            'valid_window_ratio': float(np.mean(valid_window_mask)),
            'valid_counts_per_window': valid_counts,
        },
    }


def plot_fdetector_diagnostics(result, waveform_trace=None, f_threshold=None, cmap_name="jet"):
    """绘制 F-detector 诊断图：热力图、峰值方位角、峰值强度。"""
    import matplotlib.pyplot as plt

    time_windows = result["time_windows"]
    azimuths = result["azimuths"]
    f_grid = result["F_statistic"]
    f_max = result["F_max"]
    az_peak = result["azimuth_peak"]
    valid_mask = result["valid_window_mask"]

    if f_threshold is None:
        f_threshold = float(np.nanpercentile(f_max, DEFAULT_FD_THRESHOLD_PERCENTILE))

    fig, axs = plt.subplots(3 if waveform_trace is None else 4, 1, figsize=(12, 10), sharex=True)

    mesh = axs[0].contourf(time_windows, azimuths, f_grid, levels=40, cmap=cmap_name)
    plt.colorbar(mesh, ax=axs[0], pad=0.01, label="F-statistic")
    axs[0].set_ylabel("BackAzimuth (deg)")
    axs[0].set_title("F(time, azimuth)")

    axs[1].plot(time_windows, az_peak, ".", ms=4, alpha=0.7, label="azimuth peak")
    axs[1].plot(time_windows[valid_mask], az_peak[valid_mask], ".", ms=5, label="valid windows")
    axs[1].set_ylabel("Peak Azimuth (deg)")
    axs[1].set_ylim(0.0, 360.0)
    axs[1].legend(loc="best")

    axs[2].plot(time_windows, f_max, color="tab:blue", lw=1.0, label="F_max")
    axs[2].axhline(f_threshold, color="tab:red", ls="--", lw=1.0, label="threshold")
    axs[2].set_ylabel("F max")
    axs[2].legend(loc="best")

    if waveform_trace is not None:
        t0 = waveform_trace.stats.starttime
        ts = np.arange(waveform_trace.stats.npts) * waveform_trace.stats.delta
        axs[3].plot(ts, waveform_trace.data, color="black", lw=0.8)
        axs[3].set_ylabel(f"Waveform ({waveform_trace.stats.station})")

    axs[-1].set_xlabel("Time (s, relative)")
    fig.tight_layout()
    return fig, axs
