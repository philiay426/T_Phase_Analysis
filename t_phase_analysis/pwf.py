"""
PWF (Plane Wave Fitting) 工具函数。

整合原 pwf_deepseek.py 与 pwd.py，提供：
- latlon_to_utm
- cross_corr_max_lag
- plane_wave_fitting
- calculate_azimuth_error
- preprocess_three_component_data
- build_station_coords_utm
- analyze_pwf_windows
- plot_pwf_overview
- save_pwf_txt
- pwf

调用示例
--------
```python
from t_phase_analysis.pwf import pwf

result = pwf(
    st=st,  # obspy.Stream，恰好 3 道
    # station_coords_ll=[(lat1, lon1), (lat2, lon2), (lat3, lon3)],  # 可选
    figure_output_path="./pwf_txt/example_waveform_pwf.png",  # 可选
    mat_output_dir="./pwf_txt",  # 可选
)

print("可靠窗数量:", result["meta"]["n_reliable_windows"])
print("方位角样本:", result["back_azi"][:5])
```

```python
import numpy as np
from t_phase_analysis.pwf import plane_wave_fitting, calculate_azimuth_error

# 示例：3 台站短窗数据（每台 2000 样点）
fs = 250.0
stations_data = [
    np.random.randn(2000),
    np.random.randn(2000),
    np.random.randn(2000),
]

# 台站 UTM 坐标（米），shape=(3, 2)
stations_coords = np.array([
    [0.0, 0.0],
    [1500.0, 200.0],
    [600.0, 1700.0],
])

p, v, az, cov_p, p_std, rms, t_lags, pairs = plane_wave_fitting(
    stations_data=stations_data,
    stations_coords=stations_coords,
    fs=fs,
    max_lag_samples=200,
)
az_std, az_interval = calculate_azimuth_error(p, cov_p)

print(f"v={v:.1f} m/s, az={az:.1f} deg, az_std={az_std:.2f} deg")
print("pairs:", pairs, "lags(s):", t_lags)
```
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import obspy
from obspy import Trace
from pyproj import Proj
from scipy.signal import correlate, detrend


DEFAULT_MAX_LAG_SAMPLES = 100
DEFAULT_BANDPASS = (2.0, 8.0)
DEFAULT_SAMPLING_RATE_HZ = 250.0
DEFAULT_WINDOW_STEP_S = 10.0
DEFAULT_WINDOW_LENGTH_S = 20.0
DEFAULT_MAX_LAG_AT_25HZ = 200
DEFAULT_RELIABLE_VELO_MIN = 1450.0
DEFAULT_RELIABLE_VELO_MAX = 1550.0


def _extract_station_coords_ll_from_stream(st):
    if len(st) != 3:
        raise ValueError("st must contain exactly 3 traces.")
    coords = []
    for i, tr in enumerate(st):
        if not hasattr(tr.stats, "sac"):
            raise ValueError(f"Trace {i} has no SAC header; provide station_coords_ll explicitly.")
        if "stla" not in tr.stats.sac or "stlo" not in tr.stats.sac:
            raise ValueError(
                f"Trace {i} SAC header missing stla/stlo; provide station_coords_ll explicitly."
            )
        coords.append((float(tr.stats.sac.stla), float(tr.stats.sac.stlo)))
    return coords


def _extract_data_and_fs_from_stream(st):
    if len(st) != 3:
        raise ValueError("st must contain exactly 3 traces.")
    srs = np.array([tr.stats.sampling_rate for tr in st], dtype=float)
    if not np.allclose(srs, srs[0], rtol=0.0, atol=1e-6):
        raise ValueError(f"Sampling rates are inconsistent: {srs}")
    min_npts = min(tr.stats.npts for tr in st)
    if min_npts < 2:
        raise ValueError("Each trace must contain at least 2 samples.")
    data = np.column_stack([np.asarray(tr.data[:min_npts], dtype=float) for tr in st])
    return data, float(srs[0])


def _starttime_prefix_from_stream(st):
    starttime = st[0].stats.starttime
    starttime_str = str(starttime)
    # 文件名安全化：替换 ":"，其余保留 ISO 风格
    return starttime_str.replace(":", "-")


def latlon_to_utm(lat, lon):
    """
    将经纬度转换为 UTM 坐标（米）。

    Returns
    -------
    tuple
        (x, y, utm_zone, is_northern)
    """
    utm_zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    proj_utm = Proj(
        proj="utm",
        zone=utm_zone,
        ellps="WGS84",
        datum="WGS84",
        south=not is_northern,
    )
    x, y = proj_utm(lon, lat)
    return x, y, utm_zone, is_northern


def cross_corr_max_lag(signal1, signal2, fs, max_lag_samples=DEFAULT_MAX_LAG_SAMPLES):
    """
    计算两个信号互相关的最大滞后时间（秒）。

    Parameters
    ----------
    signal1, signal2 : array-like
        输入信号。
    fs : float
        采样率 (Hz)。
    max_lag_samples : int or None
        最大滞后样点；None 表示使用全长。
    """
    if fs <= 0:
        raise ValueError("fs must be positive.")

    signal1 = np.asarray(signal1, dtype=float)
    signal2 = np.asarray(signal2, dtype=float)
    if signal1.size < 2 or signal2.size < 2:
        raise ValueError("signal length must be at least 2 samples.")

    signal1_detrended = detrend(signal1)
    signal2_detrended = detrend(signal2)

    corr = correlate(signal1_detrended, signal2_detrended, mode="full")
    center = len(signal2_detrended) - 1

    if max_lag_samples is not None:
        if max_lag_samples < 1:
            raise ValueError("max_lag_samples must be >= 1 or None.")
        max_lag = min(max_lag_samples, len(signal1) - 1, len(signal2) - 1)
        start = center - max_lag
        end = center + max_lag + 1
        corr_limited = corr[start:end]
        lags = np.arange(-max_lag, max_lag + 1, dtype=float) / fs
    else:
        corr_limited = corr
        lags = np.arange(-len(signal2_detrended) + 1, len(signal1_detrended), dtype=float) / fs

    max_idx = int(np.argmax(np.abs(corr_limited)))
    t_lag = float(lags[max_idx])
    return t_lag, corr_limited, lags


def plane_wave_fitting(stations_data, stations_coords, fs, max_lag_samples=DEFAULT_MAX_LAG_SAMPLES):
    """
    执行平面波拟合。

    Parameters
    ----------
    stations_data : sequence of array-like
        各台站波形，长度 >= 3。
    stations_coords : ndarray, shape (N, 2)
        各台站坐标（米）。
    fs : float
        采样率 (Hz)。

    Returns
    -------
    tuple
        p, v, azimuth, cov_p, p_std, rms_residual, t_lags_list, pair_indices
    """
    if fs <= 0:
        raise ValueError("fs must be positive.")

    N = len(stations_data)
    if N < 3:
        raise ValueError("At least 3 stations are required for plane wave fitting.")

    stations_coords = np.asarray(stations_coords, dtype=float)
    if stations_coords.shape != (N, 2):
        raise ValueError(f"stations_coords shape must be ({N}, 2).")

    M = N * (N - 1) // 2
    t_vec = np.zeros(M, dtype=float)
    Ax_mat = np.zeros((M, 2), dtype=float)
    t_lags_list = []
    pair_indices = []

    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            t_lag, corr, lags = cross_corr_max_lag(
                stations_data[i],
                stations_data[j],
                fs=fs,
                max_lag_samples=max_lag_samples,
            )
            t_vec[k] = t_lag
            Ax_mat[k, :] = stations_coords[i] - stations_coords[j]
            pair_indices.append((i, j))
            t_lags_list.append(t_lag)
            k += 1

    try:
        p, residuals, rank, s = np.linalg.lstsq(Ax_mat, t_vec, rcond=None)
    except np.linalg.LinAlgError:
        p = np.linalg.pinv(Ax_mat) @ t_vec

    slowness_mag = float(np.linalg.norm(p))
    if slowness_mag > 0:
        v = 1.0 / slowness_mag
        azimuth = np.degrees(np.arctan2(p[0], p[1]))
        if azimuth < 0:
            azimuth += 360.0
    else:
        v = np.inf
        azimuth = 0.0

    t_pred = Ax_mat @ p
    residuals = t_vec - t_pred
    rms_residual = float(np.sqrt(np.mean(residuals ** 2)))

    sigma_t = 1.0 / fs
    cov_t = np.eye(M) * (sigma_t ** 2)
    try:
        Ax_inv = np.linalg.pinv(Ax_mat)
        cov_p = Ax_inv @ cov_t @ Ax_inv.T
    except np.linalg.LinAlgError:
        cov_p = np.eye(2) * np.nan

    if not np.any(np.isnan(cov_p)):
        p_std = np.sqrt(np.diag(cov_p))
    else:
        p_std = np.array([np.nan, np.nan])

    return p, v, azimuth, cov_p, p_std, rms_residual, t_lags_list, pair_indices


def calculate_azimuth_error(p, cov_p):
    """
    基于误差传播计算方位角标准差（度）与区间。
    """
    p_x, p_y = p
    denom = p_x ** 2 + p_y ** 2
    if denom == 0:
        return np.nan, (np.nan, np.nan)

    J = np.array([p_y / denom, -p_x / denom], dtype=float)
    var_phi_rad = float(J @ cov_p @ J.T)
    std_phi_rad = np.sqrt(abs(var_phi_rad))
    std_phi_deg = float(np.degrees(std_phi_rad))

    phi = np.degrees(np.arctan2(p_x, p_y))
    if phi < 0:
        phi += 360.0

    phi_lower = (phi - std_phi_deg) % 360.0
    phi_upper = (phi + std_phi_deg) % 360.0
    return std_phi_deg, (phi_lower, phi_upper)


def _normalize_bandpassed_trace(data_1d, fs, bandpass):
    tr = Trace(data=np.asarray(data_1d, dtype=float))
    tr = tr.taper(0.005, type="hann", max_length=None, side="both")
    tr_f = obspy.signal.filter.bandpass(
        tr, bandpass[0], bandpass[1], corners=4, zerophase=True, df=fs
    )
    max_abs = np.max(np.abs(tr_f))
    if max_abs <= 0:
        return np.asarray(tr_f, dtype=float)
    return np.asarray(tr_f / max_abs, dtype=float)


def preprocess_three_component_data(data, fs=DEFAULT_SAMPLING_RATE_HZ, bandpass=DEFAULT_BANDPASS):
    """对前三通道做带通+归一化，返回 shape=(3, npts) 数据。"""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("data must be 2D array with at least 3 channels in axis=1.")
    if fs <= 0:
        raise ValueError("fs must be positive.")
    return np.vstack([_normalize_bandpassed_trace(data[:, i], fs, bandpass) for i in range(3)])


def build_station_coords_utm(station_coords_ll, latlon_to_utm_fn=latlon_to_utm):
    """经纬度台站坐标转 UTM 坐标（米），并尽量保持同一 UTM 带。"""
    if len(station_coords_ll) != 3:
        raise ValueError("station_coords_ll must contain exactly 3 stations.")

    coords_utm = []
    first_utm_zone = None
    for lat, lon in station_coords_ll:
        x, y, utm_zone, is_northern = latlon_to_utm_fn(lat, lon)

        if first_utm_zone is None:
            first_utm_zone = utm_zone
        elif utm_zone != first_utm_zone:
            proj_utm = Proj(
                proj="utm",
                zone=first_utm_zone,
                ellps="WGS84",
                datum="WGS84",
                south=not (lat >= 0),
            )
            x, y = proj_utm(lon, lat)

        coords_utm.append([x, y])
    return np.asarray(coords_utm, dtype=float)


def analyze_pwf_windows(
    data_f1,
    stations_coords_utm,
    plane_wave_fitting_fn=plane_wave_fitting,
    calculate_azimuth_error_fn=calculate_azimuth_error,
    fs=DEFAULT_SAMPLING_RATE_HZ,
    window_step_s=DEFAULT_WINDOW_STEP_S,
    window_length_s=DEFAULT_WINDOW_LENGTH_S,
    max_lag_at_25hz=DEFAULT_MAX_LAG_AT_25HZ,
    reliable_velo_min=DEFAULT_RELIABLE_VELO_MIN,
    reliable_velo_max=DEFAULT_RELIABLE_VELO_MAX,
):
    """滑窗执行平面波拟合，返回方位角/速度及可靠窗。"""
    data_f1 = np.asarray(data_f1, dtype=float)
    if data_f1.shape[0] != 3:
        raise ValueError("data_f1 must have shape (3, npts).")
    if stations_coords_utm.shape != (3, 2):
        raise ValueError("stations_coords_utm must have shape (3, 2).")
    if fs <= 0:
        raise ValueError("fs must be positive.")

    total_duration_s = data_f1.shape[1] / fs
    win_start = np.arange(0.0, total_duration_s, window_step_s)

    azimuth = np.full(win_start.shape, np.nan, dtype=float)
    velo = np.full(win_start.shape, np.nan, dtype=float)
    azi_std = np.full(win_start.shape, np.nan, dtype=float)
    misfit_like = np.full(win_start.shape, np.nan, dtype=float)

    max_lag_samples = int(max_lag_at_25hz * (fs / 25.0))

    for i, t0 in enumerate(win_start):
        start_sample = int(t0 * fs)
        end_sample = int((t0 + window_length_s) * fs)
        if end_sample <= start_sample or end_sample > data_f1.shape[1]:
            continue

        stations_data = [d[start_sample:end_sample] for d in data_f1]
        p, v, az_tmp, cov_p, p_std, rms_residual, t_lags, pair_indices = plane_wave_fitting_fn(
            stations_data, stations_coords_utm, fs, max_lag_samples
        )
        az_std, az_interval = calculate_azimuth_error_fn(p, cov_p)

        azi_std[i] = float(az_std)
        azimuth[i] = (float(az_tmp) + 180.0) % 360.0
        velo[i] = float(v)
        misfit_like[i] = float(rms_residual)

    reliable_mask = (velo > reliable_velo_min) & (velo < reliable_velo_max)
    t_obs = win_start[reliable_mask]
    back_azi = azimuth[reliable_mask]
    back_azi_err = azi_std[reliable_mask]

    return {
        "win_start": win_start,
        "azimuth": azimuth,
        "velocity": velo,
        "azimuth_std": azi_std,
        "reliable_mask": reliable_mask,
        "T_obs": t_obs,
        "back_azi": back_azi,
        "back_azi_err": back_azi_err,
        "rms_residual": misfit_like,
        "meta": {
            "fs": fs,
            "window_step_s": window_step_s,
            "window_length_s": window_length_s,
            "max_lag_samples": max_lag_samples,
            "reliable_velo_min": reliable_velo_min,
            "reliable_velo_max": reliable_velo_max,
        },
    }


def plot_pwf_overview(
    data_f1,
    fs,
    win_start,
    velocity,
    azimuth,
    azimuth_std,
    reliable_mask,
    fig_path=None,
):
    """绘制两联图：上图单道波形，下图方位角/速度及方位角误差。"""
    t = np.linspace(0, data_f1.shape[1], data_f1.shape[1]) / fs
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.08},
    )

    # 上图：仅画一道波形（第一道）
    ax_top.plot(t, data_f1[0, :], color="black", lw=0.9)
    ax_top.set_ylabel("Waveform")
    ax_top.set_title("PWF Overview")

    # 下图：左轴方位角（误差用色标表示）；右轴速度
    ax_bottom.set_ylabel("Azimuth (deg)", color="tab:blue", fontsize=12)
    ax_bottom.set_ylim(-10, 370)
    ax_bottom.set_xlim(t[0], t[-1])
    ax_bottom.tick_params(axis="y", labelcolor="tab:blue")

    ax_vel = ax_bottom.twinx()
    ax_vel.set_ylabel("Velocity (m/s)", color="tab:red", fontsize=12)
    ax_vel.set_ylim(1000, 2500)
    ax_vel.tick_params(axis="y", labelcolor="tab:red")

    if np.any(reliable_mask):
        x = win_start[reliable_mask]
        az = azimuth[reliable_mask]
        az_err = azimuth_std[reliable_mask]
        vel = velocity[reliable_mask]

        # 方位角点：误差通过色标表达
        cmin = float(np.nanmin(az_err))
        cmax = float(np.nanmax(az_err))
        sc = ax_bottom.scatter(
            x, az,
            s=26,
            c=az_err,
            cmap="jet",
            vmin=cmin,
            vmax=cmax if cmax > cmin else cmin + 1e-6,
            alpha=0.9,
            label="Azimuth",
        )
        # 独立色标轴：放在下图右侧外部，不压缩上下主图宽度
        pos = ax_bottom.get_position()
        cax = fig.add_axes([pos.x1 + 0.07, pos.y0, 0.014, pos.height])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("Azimuth error (deg)")

        # 速度散点
        ax_vel.plot(x, vel, "o", ms=3.5, color="tab:red", alpha=0.85, label="Velocity")

        lines1, labels1 = ax_bottom.get_legend_handles_labels()
        lines2, labels2 = ax_vel.get_legend_handles_labels()
        ax_bottom.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax_bottom.set_xlabel("Time (s)", fontsize=12)

    if fig_path is not None:
        fig_path = Path(fig_path)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=90, bbox_inches="tight")
    return fig, (ax_top, ax_bottom, ax_vel)


def save_pwf_txt(file_prefix, t_obs, back_azi, back_azi_err, output_dir="."):
    """保存 PWF 结果到 .txt 文件。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / f"{file_prefix}_Tobs.txt", np.asarray(t_obs, dtype=float), fmt="%.8f")
    np.savetxt(out_dir / f"{file_prefix}_back_azi.txt", np.asarray(back_azi, dtype=float), fmt="%.8f")
    np.savetxt(
        out_dir / f"{file_prefix}_back_azi_err.txt",
        np.asarray(back_azi_err, dtype=float),
        fmt="%.8f",
    )


def pwf(
    st,
    station_coords_ll=None,
    latlon_to_utm_fn=latlon_to_utm,
    plane_wave_fitting_fn=plane_wave_fitting,
    calculate_azimuth_error_fn=calculate_azimuth_error,
    fs=None,
    bandpass=DEFAULT_BANDPASS,
    window_step_s=DEFAULT_WINDOW_STEP_S,
    window_length_s=DEFAULT_WINDOW_LENGTH_S,
    max_lag_at_25hz=DEFAULT_MAX_LAG_AT_25HZ,
    reliable_velo_min=DEFAULT_RELIABLE_VELO_MIN,
    reliable_velo_max=DEFAULT_RELIABLE_VELO_MAX,
    figure_output_path=None,
    mat_output_dir=None,
):
    """
    PWF 一站式处理入口：仅需 3 台站波形。

    Parameters
    ----------
    st : obspy.Stream
        恰好 3 道波形。
    station_coords_ll : list[tuple] or None
        三个台站的 (lat, lon)。为 None 时从 st[*].stats.sac.stla/stlo 读取。
    fs : float or None
        采样率。为 None 时使用 st 的采样率；若指定则覆盖。
    figure_output_path : str or None
        若不为 None，保存概览图到该路径。
    mat_output_dir : str or None
        若不为 None，保存 .txt 结果到该目录。
    """
    data, fs_stream = _extract_data_and_fs_from_stream(st)
    fs_use = fs_stream if fs is None else float(fs)
    if fs_use <= 0:
        raise ValueError("fs must be positive.")

    data_f1 = preprocess_three_component_data(data, fs=fs_use, bandpass=bandpass)
    if station_coords_ll is None:
        station_coords_ll = _extract_station_coords_ll_from_stream(st)
    stations_coords_utm = build_station_coords_utm(station_coords_ll, latlon_to_utm_fn)

    result = analyze_pwf_windows(
        data_f1=data_f1,
        stations_coords_utm=stations_coords_utm,
        plane_wave_fitting_fn=plane_wave_fitting_fn,
        calculate_azimuth_error_fn=calculate_azimuth_error_fn,
        fs=fs_use,
        window_step_s=window_step_s,
        window_length_s=window_length_s,
        max_lag_at_25hz=max_lag_at_25hz,
        reliable_velo_min=reliable_velo_min,
        reliable_velo_max=reliable_velo_max,
    )

    if figure_output_path is not None:
        plot_pwf_overview(
            data_f1=data_f1,
            fs=fs_use,
            win_start=result["win_start"],
            velocity=result["velocity"],
            azimuth=result["azimuth"],
            azimuth_std=result["azimuth_std"],
            reliable_mask=result["reliable_mask"],
            fig_path=figure_output_path,
        )
        plt.close()

    if mat_output_dir is not None:
        file_prefix = _starttime_prefix_from_stream(st)
        save_pwf_txt(
            file_prefix=file_prefix,
            t_obs=result["T_obs"],
            back_azi=result["back_azi"],
            back_azi_err=result["back_azi_err"],
            output_dir=mat_output_dir,
        )

    result["meta"].update(
        {
            "file_prefix": _starttime_prefix_from_stream(st),
            "fs_input": fs,
            "fs_used": fs_use,
            "station_coords_ll": station_coords_ll,
            "n_reliable_windows": int(np.sum(result["reliable_mask"])),
            "figure_output_path": figure_output_path,
            "mat_output_dir": mat_output_dir,
        }
    )
    return result

