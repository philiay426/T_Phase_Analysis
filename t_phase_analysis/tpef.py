"""
T 相能量因子 (TPEF) 计算。

从分析波形中提取 T 相信号窗，计算信号能量与噪声能量，
返回 T 相能量因子及相关度量。

作者: Yong Zhou (zhouyong@scsio.ac.cn)

调用示例
--------
```python
from t_phase_analysis import tpef

res = tpef(st, t_t=t_t, t_p=t_p, plot_fig=False)
print(res["tpef"], res["SNR"])
```
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.filter import envelope
from obspy.signal.util import smooth


def tpef(st, t_t, t_p, half_width=300, t_coef=1 / 3.0,
         mag_coef_t_smooth=3.0, t_shift_p=60.0, plot_fig=True):
    """
    计算 T 相能量因子 (T-phase energy factor) 及相关振幅、信噪比。

    在 T 相理论到时附近截取波形，用平滑包络确定 T 相窗，
    并在 P 相之前取等长窗估计噪声能量。

    Parameters
    ----------
    st : obspy.Stream
        三分量或单分量波形。
    t_t, t_p : obspy.UTCDateTime
        T 相、P 相理论到时。
    half_width : float
        以 T 相到时为中心的分析半窗宽 (s)。
    t_coef : float
        包络峰值的比例阈值，用于界定 T 相窗起止。
    mag_coef_t_smooth : float
        包络平滑窗口长度 = 该系数 × 采样率 (样点数)。
    t_shift_p : float
        P 相之前用于估计噪声的偏移 (s)。
    plot_fig : bool
        是否绘制诊断图。

    Returns
    -------
    dict or None
        ``time_duration``, ``tpef``, ``T_amplitude``, ``P_amplitude``, ``SNR``。
    """
    trace = st[0]
    time_axis_full = np.arange(trace.stats.npts) * trace.stats.delta

    stream_around_t = st.slice(t_t - half_width, t_t + half_width)
    if len(stream_around_t) == 0 or stream_around_t[0].stats.npts == 0:
        warnings.warn('T window is empty; check t_t and half_width.', UserWarning)
        return None
    trace_around_t = stream_around_t[0]
    envelope_amplitude = envelope(trace_around_t.data)
    time_axis_t_window = np.arange(trace_around_t.stats.npts) * trace_around_t.stats.delta
    sampling_rate_hz = trace_around_t.stats.sampling_rate

    smooth_window_samples = max(1, int(mag_coef_t_smooth * sampling_rate_hz))
    smoothed_envelope = smooth(envelope_amplitude, smooth_window_samples)

    # 在 T 到时后 0 ~ 0.4×half_width 内搜索包络峰值
    search_start = int(half_width * sampling_rate_hz)
    search_end = int(half_width * 1.4 * sampling_rate_hz)
    envelope_search_segment = smoothed_envelope[search_start:search_end]
    if len(envelope_search_segment) == 0:
        warnings.warn(
            'envelope search window is empty; check half_width and sampling rate.',
            UserWarning,
        )
        return None

    envelope_peak = float(np.max(envelope_search_segment))
    index_envelope_peak = int(np.argmax(envelope_search_segment)) + search_start
    amplitude_threshold = envelope_peak * t_coef

    indices_below_threshold = np.flatnonzero(
        smoothed_envelope[:index_envelope_peak] < amplitude_threshold,
    )
    index_window_start = int(indices_below_threshold[-1]) if len(indices_below_threshold) else 0

    indices_above_threshold = np.flatnonzero(
        smoothed_envelope[index_envelope_peak:] < amplitude_threshold,
    )
    index_window_end = (
        int(indices_above_threshold[0] + index_envelope_peak)
        if len(indices_above_threshold)
        else len(smoothed_envelope) - 1
    )
    if index_window_end <= index_window_start:
        warnings.warn('Invalid T window picked from envelope.', UserWarning)
        return None

    phase_duration_s = time_axis_t_window[index_window_end] - time_axis_t_window[index_window_start]
    t_phase_slice = trace_around_t.data[index_window_start:index_window_end]
    if len(t_phase_slice) == 0:
        warnings.warn('T window slice is empty after indexing.', UserWarning)
        return None
    t_phase_amplitude = max(abs(t_phase_slice))
    signal_energy = sum(
        t_phase_slice ** 2
        * trace_around_t.stats.delta,
    )

    # P 相之前、长度与 T 相窗相同的噪声窗
    stream_before_p = st.slice(trace.stats.starttime, t_p + t_shift_p)
    if len(stream_before_p) == 0 or stream_before_p[0].stats.npts == 0:
        warnings.warn('Noise stream before P is empty.', UserWarning)
        return None
    trace_before_p = stream_before_p[0]
    time_axis_before_p = np.arange(trace_before_p.stats.npts) * trace_before_p.stats.delta
    samples_per_shift = max(1, int(t_shift_p / trace_before_p.stats.delta))
    noise_index_end = trace_before_p.stats.npts - samples_per_shift
    if noise_index_end <= 0:
        warnings.warn('Noise window end index is invalid.', UserWarning)
        return None
    window_length_samples = index_window_end - index_window_start
    noise_index_start = noise_index_end - window_length_samples

    p_phase_tail = trace_before_p.data[noise_index_end:]
    if len(p_phase_tail) == 0:
        warnings.warn('P-phase tail window is empty.', UserWarning)
        p_phase_amplitude = np.nan
    else:
        p_phase_amplitude = max(abs(p_phase_tail))

    if 0 <= noise_index_start < noise_index_end:
        noise_energy = sum(
            trace_before_p.data[noise_index_start:noise_index_end] ** 2
            * trace_before_p.stats.delta,
        )
    else:
        warnings.warn('Noise window indices are invalid; using fallback noise energy.', UserWarning)
        noise_energy = 1e-9 * signal_energy

    tpef_value = signal_energy - noise_energy
    signal_to_noise_ratio = signal_energy / noise_energy

    trace_start = trace.stats.starttime
    if plot_fig:
        relative_t_arrival = t_t - trace_start
        relative_p_arrival = t_p - trace_start

        fig, ax = plt.subplots()
        ax.plot(time_axis_full, trace.data)
        ax.axvline(x=relative_t_arrival, c='r')
        ax.axvspan(
            xmin=relative_t_arrival - half_width,
            xmax=relative_t_arrival + half_width,
            ymin=0, ymax=1, color='m', alpha=0.5,
        )
        ax.axvspan(
            xmin=relative_p_arrival - phase_duration_s,
            xmax=relative_p_arrival - t_shift_p,
            ymin=0, ymax=1, color='orange', alpha=0.5,
        )
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(time_axis_t_window, trace_around_t.data)
        ax.plot(time_axis_t_window, smoothed_envelope)
        ax.axvspan(
            xmin=time_axis_t_window[index_window_start],
            xmax=time_axis_t_window[index_window_end],
            ymin=0, ymax=1, color='m', alpha=0.5,
        )
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(time_axis_before_p, trace_before_p.data)
        ax.plot(
            time_axis_before_p[noise_index_start:noise_index_end],
            trace_before_p.data[noise_index_start:noise_index_end],
        )
        plt.show()

    return {
        'time_duration': phase_duration_s,
        'tpef': tpef_value,
        'T_amplitude': t_phase_amplitude,
        'P_amplitude': p_phase_amplitude,
        'SNR': signal_to_noise_ratio,
    }
