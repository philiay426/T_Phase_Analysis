"""
T_Phase_Analysis — T 波（T-phase）分析 Python 包。

功能模块
--------
traveltime_profile  — 震源–台站路径剖面提取、T 波走时计算
waveform            — 波形面板与时频谱绘图 (Matplotlib)
tpef                — T 相能量因子 (TPEF) 计算
snr                 — 信噪比 (SNR) 计算
pmcc                — PMCC 方位角/视速度反演
fdetector           — F-statistic 相干检测
denoise             — 参数无关 SS-CWT 域降噪
pwf                 — 平面波拟合 (PWF) 方位角/速度反演与流程化处理

安装
----
pip install -e ~/workspace/T_Phase_Analysis

依赖
----
numpy, scipy, obspy, pygmt, matplotlib, pandas

作者: Yong Zhou (zhouyong@scsio.ac.cn)

调用示例
--------
```python
from t_phase_analysis import tt_t, snr, tpef, pmcc, f_statistic, pwf
```
"""

import sys, os

# 确保 PMCC 内部模块可导入
_pmcc_dir = os.path.join(os.path.dirname(__file__), 'internals', 'PMCC')
if _pmcc_dir not in sys.path:
    sys.path.insert(0, _pmcc_dir)

# ── traveltime_profile ──
from .traveltime_profile import (
    auto_resolution,
    get_track,
    tt_t,
    get_track_reflection,
    tt_t_reflection,
    ELEVATION_MIN_M,
    ELEVATION_MAX_M,
    DEFAULT_REFERENCE_ELEVATION_M,
    DEFAULT_TAUP_MODEL,
    DEFAULT_WATER_SOUND_SPEED_KMPS,
    DEFAULT_CRUST_P_SPEED_KMPS,
    DEFAULT_CRUST_S_SPEED_KMPS,
    DEFAULT_SOURCE_DEPTH_KM,
)

# ── tpef ──
from .tpef import tpef

# ── snr ──
from .snr import snr

# ── waveform ──
from .waveform import (
    compute_spectrogram,
    plot_waveform_panel,
    plot_spectrogram_panel,
    WHITE_RED,
)

# ── pmcc ──
from .pmcc import pmcc, plot_pmcc_diagnostics

# ── f-detector ──
from .fdetector import f_statistic, plot_fdetector_diagnostics

# ── denoise ──
from .denoise import para_free_denoise

# ── pwf ──
from .pwf import (
    latlon_to_utm,
    cross_corr_max_lag,
    plane_wave_fitting,
    calculate_azimuth_error,
    preprocess_three_component_data,
    build_station_coords_utm,
    analyze_pwf_windows,
    plot_pwf_overview,
    save_pwf_txt,
    pwf,
)

__all__ = [
    "auto_resolution",
    "get_track",
    "tt_t",
    "get_track_reflection",
    "tt_t_reflection",
    "tpef",
    "snr",
    "compute_spectrogram",
    "plot_waveform_panel",
    "plot_spectrogram_panel",
    "WHITE_RED",
    "pmcc",
    "plot_pmcc_diagnostics",
    "f_statistic",
    "plot_fdetector_diagnostics",
    "para_free_denoise",
    "latlon_to_utm",
    "cross_corr_max_lag",
    "plane_wave_fitting",
    "calculate_azimuth_error",
    "preprocess_three_component_data",
    "build_station_coords_utm",
    "analyze_pwf_windows",
    "plot_pwf_overview",
    "save_pwf_txt",
    "pwf",
    "ELEVATION_MIN_M",
    "ELEVATION_MAX_M",
    "DEFAULT_REFERENCE_ELEVATION_M",
    "DEFAULT_TAUP_MODEL",
    "DEFAULT_WATER_SOUND_SPEED_KMPS",
    "DEFAULT_CRUST_P_SPEED_KMPS",
    "DEFAULT_CRUST_S_SPEED_KMPS",
    "DEFAULT_SOURCE_DEPTH_KM",
]
