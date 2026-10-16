#!/usr/bin/env python3
"""
PMCC（Progressive Multi-Channel Correlation）方位角/视速度反演。

基于 PMCC 核心逻辑，提取为独立函数，供其他脚本直接调用。

Yong Zhou / Hermes Agent, 2026-05-22

调用示例
--------
```python
from t_phase_analysis import pmcc, plot_pmcc_diagnostics

res = pmcc(st, time_start=-50, time_one_slice=10.0, time_slide=2.0, time_length=200.0)
fig, axs = plot_pmcc_diagnostics(res)
print(res["azimuth_mode1"], res["speed_mode1"])
```
"""
import warnings

import numpy as np
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.cross_correlation import xcorr_pick_correction as _obspy_xcorr


def _xcorr_pick_correction(pick1, trace1, pick2, trace2,
                           t_before, t_after, cc_maxlag,
                           plot=False, filename=None):
    """
    WENBO 风格包装：在 ObsPy xcorr_pick_correction 基础上加入 warning_tag。

    与 ObsPy 官方版差异：
    - 返回 (pick2_corr, coeff, warning_tag) 而非 (pick2_corr, coeff)
    - num_samples < 3 时不抛异常，设 warning_tag=6
    - CC_max < 0.8 不警告（T波/重复地震场景可接受低CC）
    - 静默处理 CC_max < 0（不打印警告，批量时避免刷屏）

    warning_tag 编码：
        0=正常, 1=trace id 不匹配, 2/3=滤波边界, 4=CC负值,
        5=拟合在最大lag边界, 6=拟合样本<5, 7=抛物线开口向上, 8=拟合残差大
    """
    warning_tag = 0

    # 检查 trace id
    if trace1.id != trace2.id:
        warnings.warn(f"Trace ids do not match: {trace1.id} != {trace2.id}")
        warning_tag = 1

    samp_rate = trace1.stats.sampling_rate

    # 用 ObsPy 版的逻辑复制一遍（避免重复导入 future 等依赖）
    # 在此简化：直接用 ObsPy 版计算，然后补 warning_tag 和静默处理
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pick2_corr, coeff = _obspy_xcorr(
                pick1, trace1, pick2, trace2,
                t_before, t_after, cc_maxlag,
                plot=plot, filename=filename)
            # 解析 ObsPy 版产生的 warning 来填充 warning_tag
            for w in caught:
                msg = str(w.message)
                if 'Fitting at maximum lag' in msg:
                    warning_tag = max(warning_tag, 5)
                elif 'Less than 5 samples' in msg:
                    warning_tag = max(warning_tag, 6)
                elif 'Fitted parabola opens upwards' in msg:
                    warning_tag = max(warning_tag, 7)
                elif 'Residual in quadratic fit' in msg:
                    warning_tag = max(warning_tag, 8)
                elif 'maximum is negative' in msg:
                    warning_tag = max(warning_tag, 4)
                elif 'filter' in msg.lower() or 'artifacts' in msg.lower():
                    warning_tag = max(warning_tag, 2)
    except Exception:
        # ObsPy 版在 num_samples < 3 时抛异常 → WENBO 版不抛
        pick2_corr, coeff = 0.0, 0.0
        warning_tag = max(warning_tag, 6)

    return pick2_corr, coeff, warning_tag


def _find_time_relative_to_b(st, ref_time):
    """计算 ref_time 相对于 trace 起始时间 'b' 的偏移 (s)。"""
    if ref_time.lower() not in ('b', 'o') and not ref_time.lower().startswith('t'):
        raise ValueError(f"Invalid ref_time: {ref_time}, must be b/o/t0–t9")
    return st.stats.sac[ref_time.lower()] - st.stats.sac.b


def _find_second_largest(arr):
    """返回数组中频率最高和次高的值。所有值相同时返回 (most, -12345)。"""
    unique_values, counts = np.unique(arr, return_counts=True)
    sorted_idx = np.argsort(-counts)
    most = unique_values[sorted_idx[0]]
    try:
        second = unique_values[sorted_idx[1]]
        return second, most
    except IndexError:
        return most, -12345


def _require_three_traces(st):
    if len(st) != 3:
        raise ValueError("PMCC requires exactly 3 traces.")


def _require_positive_parameters(time_one_slice, time_slide, time_length,
                                 speed_min, speed_max, speed_step,
                                 azimuth_step, max_cc_shift):
    if time_one_slice <= 0 or time_slide <= 0 or time_length <= 0:
        raise ValueError("time_one_slice/time_slide/time_length must be positive.")
    if speed_min <= 0 or speed_max <= 0 or speed_step <= 0:
        raise ValueError("speed_min/speed_max/speed_step must be positive.")
    if speed_max <= speed_min:
        raise ValueError("speed_max must be greater than speed_min.")
    if azimuth_step <= 0:
        raise ValueError("azimuth_step must be positive.")
    if max_cc_shift <= 0:
        raise ValueError("max_cc_shift must be positive.")


def _require_sampling_consistency(st):
    srs = np.array([tr.stats.sampling_rate for tr in st], dtype=float)
    if not np.allclose(srs, srs[0], rtol=0.0, atol=1e-6):
        raise ValueError(f"Sampling rates are inconsistent: {srs}")


def _require_sac_headers(st, ref_time_var):
    required = ("stla", "stlo", "b")
    ref_key = ref_time_var.lower()
    for idx, tr in enumerate(st):
        if not hasattr(tr.stats, "sac"):
            raise ValueError(f"Trace {idx} has no SAC header.")
        for key in required:
            if key not in tr.stats.sac:
                raise ValueError(f"Trace {idx} SAC header missing '{key}'.")
        if ref_key not in tr.stats.sac:
            raise ValueError(f"Trace {idx} SAC header missing '{ref_key}'.")


def _build_time_windows(time_start, time_slide, time_length):
    nwindows = int(np.floor(time_length / time_slide))
    if nwindows < 1:
        raise ValueError("No valid windows: increase time_length or reduce time_slide.")
    return nwindows, time_start + np.arange(nwindows) * time_slide

# ── 默认参数 ──
DEFAULT_SPEED_MIN = 1200.0      # m/s
DEFAULT_SPEED_MAX = 1800.0      # m/s
DEFAULT_SPEED_STEP = 1.0        # m/s
DEFAULT_AZIMUTH_STEP = 0.2      # degree
DEFAULT_CONSISTENCY_THRESHOLD = 0.05  # seconds
DEFAULT_MAX_CC_SHIFT = 2.0      # seconds


def pmcc(st, time_start, time_one_slice, time_slide, time_length,
         ref_time_var='t0',
         speed_min=DEFAULT_SPEED_MIN,
         speed_max=DEFAULT_SPEED_MAX,
         speed_step=DEFAULT_SPEED_STEP,
         azimuth_step=DEFAULT_AZIMUTH_STEP,
         consistency_threshold=DEFAULT_CONSISTENCY_THRESHOLD,
         correlation_threshold=0.3,
         max_cc_shift=DEFAULT_MAX_CC_SHIFT,
         ):
    """
    Progressive Multi-Channel Correlation：滑动时窗互相关，反演方位角与视速度。

    对 3 台站逐对计算互相关延迟，扫描方位角 (0–360°) 和速度，
    找到使预测延迟与观测延迟 misfit 最小的方位角-速度组合。

    Parameters
    ----------
    st : obspy.Stream
        恰好 3 道波形（SAC 格式，含台站坐标头段）。
    time_start : float
        相对参考时刻的起始时间 (s)。
    time_one_slice : float
        每个互相关窗的长度 (s)。
    time_slide : float
        窗滑动步长 (s)。
    time_length : float
        PMCC 总时长 (s)。
    ref_time_var : str
        SAC 头段中参考时刻的变量名（默认 't0' = 发震时刻）。
    speed_min, speed_max : float
        视速度搜索范围 (m/s)。
    speed_step : float
        速度搜索步长 (m/s)。
    azimuth_step : float
        方位角搜索步长 (度)。
    consistency_threshold : float
        一致性阈值 (s)，consistency = |dt12 + dt23 − dt13|。
    max_cc_shift : float
        互相关最大允许时移 (s)。

    Returns
    -------
    dict
        time_measure, consistency, ave_CC, azimuth_find, speed_find,
        dt12, dt13, dt23, CC12, CC13, CC23,
        azimuth_mode1, azimuth_mode2, speed_mode1, speed_mode2
        若台站数 ≠ 3，返回 None。
    """
    _require_three_traces(st)
    _require_positive_parameters(
        time_one_slice, time_slide, time_length,
        speed_min, speed_max, speed_step,
        azimuth_step, max_cc_shift,
    )
    _require_sampling_consistency(st)
    _require_sac_headers(st, ref_time_var)

    nwindows, time_offsets = _build_time_windows(time_start, time_slide, time_length)

    # ── 台站几何 ──
    dist_meter_12, az_12, baz_12 = gps2dist_azimuth(
        st[0].stats.sac.stla, st[0].stats.sac.stlo,
        st[1].stats.sac.stla, st[1].stats.sac.stlo)
    dist_meter_13, az_13, baz_13 = gps2dist_azimuth(
        st[0].stats.sac.stla, st[0].stats.sac.stlo,
        st[2].stats.sac.stla, st[2].stats.sac.stlo)
    dist_meter_23, az_23, baz_23 = gps2dist_azimuth(
        st[1].stats.sac.stla, st[1].stats.sac.stlo,
        st[2].stats.sac.stla, st[2].stats.sac.stlo)
    deg2rad = np.pi / 180.0

    # ── 初始化输出数组 ──
    dt12 = np.zeros(nwindows); CC12 = np.zeros(nwindows); warn12 = np.zeros(nwindows, dtype=int)
    dt13 = np.zeros(nwindows); CC13 = np.zeros(nwindows); warn13 = np.zeros(nwindows, dtype=int)
    dt23 = np.zeros(nwindows); CC23 = np.zeros(nwindows); warn23 = np.zeros(nwindows, dtype=int)
    time_measure = np.zeros(nwindows)

    t_ref = _find_time_relative_to_b(st[0], ref_time_var)

    # ── 逐窗互相关 ──
    for iwindow in range(nwindows):
        t_pick = st[0].stats.starttime + t_ref + time_offsets[iwindow]
        time_measure[iwindow] = t_ref + time_offsets[iwindow]

        # 3 对台站互相关
        for (ista, jsta, arr_dt, arr_cc, arr_warn) in [
            (0, 1, dt12, CC12, warn12),
            (0, 2, dt13, CC13, warn13),
            (1, 2, dt23, CC23, warn23),
        ]:
            cc_dt, cc_coeff, warn = _xcorr_pick_correction(
                t_pick, st[ista], t_pick, st[jsta],
                0.0, time_one_slice, max_cc_shift,
                plot=False, filename='tmp_CC')
            arr_dt[iwindow] = -cc_dt
            arr_cc[iwindow] = cc_coeff
            arr_warn[iwindow] = warn

    # ── 扫描方位角 & 速度 ──
    azimuths = np.arange(0, 360.0, azimuth_step)
    speeds = np.arange(speed_min, speed_max + 0.5 * speed_step, speed_step)

    consistency = np.zeros(nwindows)
    ave_CC = np.zeros(nwindows)
    misfit = np.full(nwindows, np.nan, dtype=float)
    azimuth_find = np.full(nwindows, -12345.0, dtype=float)
    speed_find = np.full(nwindows, -12345.0, dtype=float)

    cos12 = np.cos((az_12 - azimuths) * deg2rad)
    cos13 = np.cos((az_13 - azimuths) * deg2rad)
    cos23 = np.cos((az_23 - azimuths) * deg2rad)
    inv_speed = 1.0 / speeds[:, None]
    d12_grid = -dist_meter_12 * inv_speed * cos12[None, :]
    d13_grid = -dist_meter_13 * inv_speed * cos13[None, :]
    d23_grid = -dist_meter_23 * inv_speed * cos23[None, :]

    for iwindow in range(nwindows):
        consistency[iwindow] = abs(dt12[iwindow] + dt23[iwindow] - dt13[iwindow])
        ave_CC[iwindow] = (CC12[iwindow] + CC13[iwindow] + CC23[iwindow]) / 3.0

        if consistency[iwindow] >= consistency_threshold or ave_CC[iwindow] < correlation_threshold:
            continue

        mf_grid = (
            (d12_grid - dt12[iwindow]) ** 2
            + (d13_grid - dt13[iwindow]) ** 2
            + (d23_grid - dt23[iwindow]) ** 2
        )
        idx = np.unravel_index(np.argmin(mf_grid), mf_grid.shape)
        i_speed, i_az = idx
        misfit[iwindow] = mf_grid[idx]
        azimuth_find[iwindow] = (azimuths[i_az] + 180.0) % 360.0
        speed_find[iwindow] = speeds[i_speed]

    # ── 统计众数 ──
    valid_mask = (azimuth_find > -1e4) & (speed_find > -1e4)
    if np.any(valid_mask):
        az_mode2, az_mode1 = _find_second_largest(azimuth_find[valid_mask])
        sp_mode2, sp_mode1 = _find_second_largest(speed_find[valid_mask])
    else:
        az_mode1 = az_mode2 = -12345
        sp_mode1 = sp_mode2 = -12345

    all_warns = np.concatenate([warn12, warn13, warn23])
    uniq_warn, cnt_warn = np.unique(all_warns, return_counts=True)
    warning_summary = {int(k): int(v) for k, v in zip(uniq_warn, cnt_warn)}

    return {
        'time_measure': time_measure,
        'consistency': consistency,
        'ave_CC': ave_CC,
        'misfit': misfit,
        'azimuth_find': azimuth_find,
        'speed_find': speed_find,
        'dt12': dt12, 'dt13': dt13, 'dt23': dt23,
        'CC12': CC12, 'CC13': CC13, 'CC23': CC23,
        'warn12': warn12, 'warn13': warn13, 'warn23': warn23,
        'valid_mask': valid_mask,
        'azimuth_mode1': az_mode1,
        'azimuth_mode2': az_mode2,
        'speed_mode1': sp_mode1,
        'speed_mode2': sp_mode2,
        'meta': {
            'ref_time_var': ref_time_var,
            'consistency_threshold': consistency_threshold,
            'correlation_threshold': correlation_threshold,
            'max_cc_shift': max_cc_shift,
            'speed_min': speed_min,
            'speed_max': speed_max,
            'speed_step': speed_step,
            'azimuth_step': azimuth_step,
            'valid_window_ratio': float(np.mean(valid_mask)),
            'warning_summary': warning_summary,
            'station_geometry': {
                'dist_meter_12': dist_meter_12, 'az_12': az_12,
                'dist_meter_13': dist_meter_13, 'az_13': az_13,
                'dist_meter_23': dist_meter_23, 'az_23': az_23,
            },
        },
    }


def plot_pmcc_diagnostics(result, top_window_index=None, cmap_name="viridis"):
    """绘制 PMCC 诊断图：QC散点、方位角/速度时序、代表窗misfit、延迟拟合对比。"""
    import matplotlib.pyplot as plt

    t = result["time_measure"]
    consistency = result["consistency"]
    ave_cc = result["ave_CC"]
    az = result["azimuth_find"]
    sp = result["speed_find"]
    dt12 = result["dt12"]; dt13 = result["dt13"]; dt23 = result["dt23"]
    valid = result["valid_mask"]
    meta = result["meta"]

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=False)

    # 图1：QC 双指标
    axs[0].scatter(consistency[~valid], ave_cc[~valid], s=12, alpha=0.5, label="invalid")
    axs[0].scatter(consistency[valid], ave_cc[valid], s=14, alpha=0.8, label="valid")
    axs[0].axvline(meta["consistency_threshold"], color="tab:red", ls="--", lw=1.0)
    axs[0].axhline(meta["correlation_threshold"], color="tab:green", ls="--", lw=1.0)
    axs[0].set_xlabel("consistency (s)")
    axs[0].set_ylabel("average CC")
    axs[0].set_title("PMCC QC")
    axs[0].legend(loc="best")

    # 图2：方位角/速度时序
    axs[1].plot(t[valid], az[valid], ".", ms=4, label="azimuth (valid)")
    axs[1].axhline(result["azimuth_mode1"], color="tab:red", ls="--", lw=1.0, label="mode1")
    axs[1].set_ylabel("Azimuth (deg)")
    axs[1].set_ylim(0.0, 360.0)
    axs[1].legend(loc="best")

    ax1b = axs[1].twinx()
    ax1b.plot(t[valid], sp[valid], ".", ms=4, color="tab:orange", label="speed (valid)")
    ax1b.axhline(result["speed_mode1"], color="tab:orange", ls="--", lw=1.0)
    ax1b.set_ylabel("Speed (m/s)")

    # 图3：代表窗 misfit(azimuth, speed)
    if np.any(valid):
        valid_idx = np.where(valid)[0]
        if top_window_index is None:
            top_window_index = int(valid_idx[np.argmax(ave_cc[valid_idx])])
        else:
            top_window_index = int(top_window_index)

        geo = meta["station_geometry"]
        azimuth_step = meta["azimuth_step"]
        speed_step = meta["speed_step"]
        azimuths = np.arange(0, 360.0, azimuth_step)
        speeds = np.arange(meta["speed_min"], meta["speed_max"] + 0.5 * speed_step, speed_step)
        deg2rad = np.pi / 180.0
        inv_speed = 1.0 / speeds[:, None]
        m12 = -geo["dist_meter_12"] * inv_speed * np.cos((geo["az_12"] - azimuths)[None, :] * deg2rad)
        m13 = -geo["dist_meter_13"] * inv_speed * np.cos((geo["az_13"] - azimuths)[None, :] * deg2rad)
        m23 = -geo["dist_meter_23"] * inv_speed * np.cos((geo["az_23"] - azimuths)[None, :] * deg2rad)
        mf = (m12 - dt12[top_window_index]) ** 2 + (m13 - dt13[top_window_index]) ** 2 + (m23 - dt23[top_window_index]) ** 2

        im = axs[2].contourf(azimuths, speeds, mf, levels=40, cmap=cmap_name)
        plt.colorbar(im, ax=axs[2], pad=0.01, label="misfit")
        axs[2].set_xlabel("Azimuth (deg)")
        axs[2].set_ylabel("Speed (m/s)")
        axs[2].set_title(f"Misfit surface @ window {top_window_index}")
    else:
        axs[2].text(0.5, 0.5, "No valid window for misfit map", ha="center", va="center")
        axs[2].set_axis_off()

    # 图4：延迟拟合对比（用每窗最佳 az/speed 生成预测）
    pred12 = np.full_like(dt12, np.nan)
    pred13 = np.full_like(dt13, np.nan)
    pred23 = np.full_like(dt23, np.nan)
    geo = meta["station_geometry"]
    deg2rad = np.pi / 180.0
    pred12[valid] = -geo["dist_meter_12"] * np.cos((geo["az_12"] - ((az[valid] - 180.0) % 360.0)) * deg2rad) / sp[valid]
    pred13[valid] = -geo["dist_meter_13"] * np.cos((geo["az_13"] - ((az[valid] - 180.0) % 360.0)) * deg2rad) / sp[valid]
    pred23[valid] = -geo["dist_meter_23"] * np.cos((geo["az_23"] - ((az[valid] - 180.0) % 360.0)) * deg2rad) / sp[valid]

    axs[3].plot(t, dt12, color="tab:blue", lw=0.8, label="obs dt12")
    axs[3].plot(t, pred12, color="tab:blue", ls="--", lw=1.0, label="pred dt12")
    axs[3].plot(t, dt13, color="tab:orange", lw=0.8, label="obs dt13")
    axs[3].plot(t, pred13, color="tab:orange", ls="--", lw=1.0, label="pred dt13")
    axs[3].plot(t, dt23, color="tab:green", lw=0.8, label="obs dt23")
    axs[3].plot(t, pred23, color="tab:green", ls="--", lw=1.0, label="pred dt23")
    axs[3].set_xlabel("Time (s, relative to reference)")
    axs[3].set_ylabel("Delay (s)")
    axs[3].set_title("Observed vs Predicted Delays")
    axs[3].legend(loc="best", ncol=2, fontsize=8)

    fig.tight_layout()
    return fig, axs
