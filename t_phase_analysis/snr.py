"""
信噪比 (SNR) 计算。

从波形中截取信号窗和噪声窗，计算信噪比。
支持标准差比 ('std') 和峰值比 ('max') 两种方法。

作者: Yong Zhou (zhouyong@scsio.ac.cn)

调用示例
--------
```python
from t_phase_analysis import snr

ratio = snr(st, t_signal=t_t, t_noise=t_n, twin=10.0, method="std", plot_fig=False)
print(ratio)
```
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_SNR_TWIN = 10.0
DEFAULT_SNR_METHOD = "std"
DEFAULT_SNR_DENOMINATOR_TOL = 1e-20


def _select_trace(st, trace_index):
    if len(st) == 0:
        raise ValueError("Empty stream: at least one trace is required.")
    if trace_index < 0 or trace_index >= len(st):
        raise ValueError(f"trace_index out of range: {trace_index}, len(st)={len(st)}")
    return st[trace_index]


def _validate_time_window(center_time, half_width, trace_start, trace_end, label):
    if not (center_time - half_width > trace_start and center_time + half_width < trace_end):
        warnings.warn(f"Time window for {label} out of range", UserWarning)
        return False
    return True


def snr(st, t_signal, t_noise, twin=DEFAULT_SNR_TWIN, method=DEFAULT_SNR_METHOD,
        plot_fig=True, trace_index=0):
    """
    计算给定信号和噪声时间窗口内的信噪比 (SNR)。

    Parameters
    ----------
    st : obspy.Stream
        地震波形。
    t_signal, t_noise : obspy.UTCDateTime
        信号窗、噪声窗的中心时刻。
    twin : float
        各窗半宽 (s)，实际截取 [中心 - twin, 中心 + twin]。
    method : str
        ``'std'``：方差比；``'max'``：峰值比。
    plot_fig : bool
        是否绘图。
    trace_index : int
        从 ``st`` 中选取用于计算 SNR 的道索引，默认 ``0``。

    Returns
    -------
    float
        信噪比；窗越界或噪声标准差为 0 时返回 ``nan``。
    """
    if twin <= 0:
        raise ValueError("twin must be positive.")

    trace = _select_trace(st, trace_index)
    trace_start_time = trace.stats.starttime
    trace_end_time = trace.stats.endtime

    if not _validate_time_window(t_signal, twin, trace_start_time, trace_end_time, "signal"):
        return np.nan

    if not _validate_time_window(t_noise, twin, trace_start_time, trace_end_time, "noise"):
        return np.nan

    signal_trace = trace.slice(t_signal - twin, t_signal + twin)
    noise_trace = trace.slice(t_noise - twin, t_noise + twin)

    if signal_trace.stats.npts == 0 or noise_trace.stats.npts == 0:
        warnings.warn("Signal/noise window is empty after slicing.", UserWarning)
        return np.nan

    method_normalized = method.lower().strip()
    signal_data = signal_trace.data
    noise_data = noise_trace.data

    if method_normalized == "std":
        noise_std = float(np.std(noise_data))
        if noise_std <= DEFAULT_SNR_DENOMINATOR_TOL:
            warnings.warn("Noise window std is near zero; SNR is undefined.", UserWarning)
            return np.nan
        signal_std = float(np.std(signal_data))
        signal_to_noise_ratio = (signal_std ** 2) / (noise_std ** 2)
    elif method_normalized == "max":
        # 峰值比应使用绝对振幅，避免符号导致负 SNR。
        noise_peak = float(np.max(np.abs(noise_data)))
        if noise_peak <= DEFAULT_SNR_DENOMINATOR_TOL:
            warnings.warn("Noise window peak is near zero; SNR is undefined.", UserWarning)
            return np.nan
        signal_peak = float(np.max(np.abs(signal_data)))
        signal_to_noise_ratio = signal_peak / noise_peak
    else:
        warnings.warn(
            f"Method '{method}' not defined; use 'std' or 'max'.",
            UserWarning,
        )
        return np.nan

    if plot_fig:
        fig, ax = plt.subplots(dpi=200)
        ax.plot(trace.times(), trace.data, label='data')
        ax.plot(
            signal_trace.times() + (signal_trace.stats.starttime - trace_start_time),
            signal_data,
            label='signal',
        )
        ax.plot(
            noise_trace.times() + (noise_trace.stats.starttime - trace_start_time),
            noise_data,
            label='noise',
        )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.set_title(f"SNR ({method_normalized}): {signal_to_noise_ratio:.4g}")
        plt.show()

    return float(signal_to_noise_ratio)
