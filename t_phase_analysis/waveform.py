#!/usr/bin/env python3
"""
波形 + 时频谱绘制函数。

从 plot_close_stations.py 提取，供其他脚本直接调用。

函数
----
plot_waveform_panel  — 在给定 axes 上绘制滤波波形、包络、理论到时标注
plot_spectrogram_panel — 在给定 axes 上绘制时频谱 + 理论到时线
compute_spectrogram  — 计算时频谱数据

调用示例
--------
```python
from t_phase_analysis import compute_spectrogram

f, t, sxx_db = compute_spectrogram(data_raw=trace.data, sr=trace.stats.sampling_rate)
print(f.shape, t.shape, sxx_db.shape)
```
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal

# ── 色标 ──────────────────────────────────────────────────────────────────
WHITE_RED = LinearSegmentedColormap.from_list('white_red', [
    '#ffffff', '#fff7bc', '#fee391', '#fec44f',
    '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506',
])


# ══════════════════════════════════════════════════════════════════════════════
# 时频谱计算
# ══════════════════════════════════════════════════════════════════════════════

def compute_spectrogram(data_raw, sr, nperseg=None, noverlap=None, nfft=None):
    """
    计算时频谱 (dB)。

    Parameters
    ----------
    data_raw : ndarray
        原始波形数据（未滤波）。
    sr : float
        采样率 (Hz)。
    nperseg : int, optional
        STFT 每段采样点数。默认 ``min(int(sr * 2.56), max(64, npts // 4))``。
    noverlap : int, optional
        重叠采样点数。默认 ``int(nperseg * 0.9)``。
    nfft : int, optional
        FFT 点数。默认 ``int(sr * 4)``。

    Returns
    -------
    f : ndarray
        频率轴 (Hz)。
    t_spec_arr : ndarray
        时频谱时间轴（相对于数据起点，秒）。
    Sxx_db : ndarray
        功率谱密度 (dB, shape=(len(f), len(t_spec_arr)))。
    """
    npts = len(data_raw)

    if nperseg is None:
        nperseg = min(int(sr * 2.56), max(64, npts // 4))
    if noverlap is None:
        noverlap = int(nperseg * 0.9)
    if nfft is None:
        nfft = int(sr * 4)

    f, t_spec_arr, Sxx = signal.spectrogram(
        data_raw, sr, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
    )
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-20))
    return f, t_spec_arr, Sxx_db


# ══════════════════════════════════════════════════════════════════════════════
# 波形面板
# ══════════════════════════════════════════════════════════════════════════════

def plot_waveform_panel(
    ax,
    t_rel,
    data_bp,
    env,
    tt_p,
    tt_s,
    tt_t=None,
    has_twave=True,
    t_spec_rel_start=None,
    label="",
    origin_str="",
    ylim_mult=1.3,
    arrival_labels=True,
    legend_loc='upper right',
    legend_bbox=(0.80, 0.98),
    waveform_label='Waveform filtered 2-8 Hz',
    envelope_label='Envelope',
):
    """
    在给定 axes 上绘制波形面板：滤波波形、包络、理论到时标注。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标 axes。
    t_rel : ndarray
        相对发震时刻的时间轴 (s)。
    data_bp : ndarray
        带通滤波后的数据。
    env : ndarray
        包络。
    tt_p, tt_s : float
        P、S 波理论到时 (s)。
    tt_t : float, optional
        T 波理论到时 (s)。has_twave=True 时必须提供。
    has_twave : bool
        是否有 T 波路径。
    t_spec_rel_start : float, optional
        时频谱起始时间，用于 axvspan 标记 P 波前噪声窗口。
    label : str
        台站标签，用于标题。
    origin_str : str
        发震时刻字符串，用于标题。
    ylim_mult : float
        y 轴范围倍数（相对于 max|data_bp|）。
    arrival_labels : bool
        是否在竖线旁标注到时文字。
    legend_loc : str
        legend 位置。
    legend_bbox : tuple
        legend bbox_to_anchor。
    waveform_label, envelope_label : str
        legend 中的标签文字。
    """
    ax.plot(t_rel, data_bp, color='black', linewidth=0.4, label=waveform_label)
    ax.plot(t_rel, env, color='#C05040', linewidth=0.8, alpha=0.7, label=envelope_label)

    ax.axvline(x=tt_p, color='green', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.axvline(x=tt_s, color='orange', linestyle=':', linewidth=1.0, alpha=0.6)
    if has_twave:
        ax.axvline(x=tt_t, color='blue', linestyle='--', linewidth=1.2)
    else:
        ax.text(0.02, 0.92, 'Great-circle path: all-land\n(no direct T-wave)',
                transform=ax.transAxes, fontsize=9, color='#555555', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#aaaaaa', alpha=0.8))

    if t_spec_rel_start is not None:
        ax.axvspan(t_spec_rel_start, tt_p, color='orange', alpha=0.06)

    ax.set_ylabel('Velocity (m/s)', fontsize=10)
    title = label
    if origin_str:
        title += f'  |  t=0 = origin ({origin_str})'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(t_rel[0], t_rel[-1])

    ylim = np.max(np.abs(data_bp)) * ylim_mult
    ax.set_ylim(-ylim, ylim)

    # ── 理论到时标注 ──
    if arrival_labels:
        label_y = ylim * 0.88
        for x_val, phase_name, color in [
            (tt_p, 'P', 'green'),
            (tt_s, 'S', 'orange'),
        ]:
            ax.text(x_val, label_y, f'{phase_name} ({x_val:.0f}s)',
                    rotation=0, va='bottom', ha='center',
                    color=color, fontsize=7.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, alpha=0.8))
        if has_twave:
            ax.text(tt_t, label_y, f'T ({tt_t:.0f}s)',
                    rotation=0, va='bottom', ha='center',
                    color='blue', fontsize=7.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='blue', alpha=0.8))

    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox,
              fontsize=7.5, framealpha=0.8)
    ax.tick_params(labelbottom=False)


# ══════════════════════════════════════════════════════════════════════════════
# 时频谱面板
# ══════════════════════════════════════════════════════════════════════════════

def plot_spectrogram_panel(
    ax,
    t_spec_rel,
    f,
    Sxx_db,
    tt_p,
    tt_s,
    tt_t=None,
    has_twave=True,
    freq_max=10,
    cmap=None,
    db_range=50,
    db_percentile=98,
    colorbar_label='dB',
):
    """
    在给定 axes 上绘制时频谱。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标 axes。
    t_spec_rel, f, Sxx_db : ndarray
        时频谱数据（compute_spectrogram 的输出 + t_offset）。
    tt_p, tt_s : float
        P、S 波理论到时 (s)。
    tt_t : float, optional
        T 波理论到时。
    has_twave : bool
        是否绘制 T 波线。
    freq_max : float
        频率轴上限 (Hz)。
    cmap : matplotlib colormap, optional
        色标，默认 WHITE_RED。
    db_range : float
        dB 动态范围（vmin = vmax - db_range）。
    db_percentile : float
        vmax 取 Sxx_db 的该百分位数。
    colorbar_label : str
        colorbar 标签。
    """
    if cmap is None:
        cmap = WHITE_RED

    vmax = np.percentile(Sxx_db, db_percentile)
    vmin = vmax - db_range

    pcm = ax.pcolormesh(t_spec_rel, f, Sxx_db, shading='gouraud',
                        cmap=cmap, vmin=vmin, vmax=vmax)

    ax.axvline(x=tt_p, color='green', linestyle=':', linewidth=1.0, alpha=0.5)
    ax.axvline(x=tt_s, color='orange', linestyle=':', linewidth=1.0, alpha=0.5)
    if has_twave:
        ax.axvline(x=tt_t, color='cyan', linestyle='--', linewidth=1.5)

    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.set_xlabel('Time from origin (s)', fontsize=10)
    ax.set_ylim(0, freq_max)
    ax.set_xlim(t_spec_rel[0], t_spec_rel[-1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.12)
    cb = ax.figure.colorbar(pcm, cax=cax, label=colorbar_label)
    cb.ax.tick_params(labelsize=8)
