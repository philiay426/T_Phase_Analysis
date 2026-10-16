"""
三分量波形 PCA 偏振分析。

基于滑动时间窗对 3 分量协方差矩阵做特征分解，提取：
- 主偏振方向（principal_vector）
- 方位角（back_azimuth）
- 入射角（incidence）
- 偏振指标（linearity, planarity, sphericity）
- 时窗长度：过短会使协方差矩阵估计不稳定，过长会平滑掉偏振变化。通常取信号主周期的 1.5~3 倍。

协方差估计：
- ``real``（默认）：实值三分量协方差（Jurkevics, 1988）
- ``analytic``：Hilbert 解析信号复协方差（Vidale, 1986），窄带信号特征值更稳健

参考文献：
- Vidale, J. E. (1986). Complex polarization analysis of particle motion.
  Bulletin of the Seismological Society of America, 76(5), 1393-1405.
- Park, J., Vernon, F. L., & Lindberg, C. R. (1987). Frequency dependent
  polarization analysis of high-frequency seismograms. Journal of Geophysical
  Research, 92(B12), 12664-12674.
- Jurkevics, A. (1988). Polarization analysis of three-component array data.
  Bulletin of the Seismological Society of America, 78(5), 1725-1743.
- Koper, K. D., & Hawley, V. L. (2010). Frequency dependent polarization
  analysis of ambient seismic noise recorded at a broadband seismometer in the
  central United States. Earthquake Science, 23(5), 439-447.

调用示例
--------
```python
from t_phase_analysis import pca, plot_pca_diagnostics, rotate_stream_to_zne

# 标准 ZNE（实值协方差，默认）
res = pca(st, window_length_s=10.0, window_step_s=2.0)

# 解析信号协方差（窄带信号推荐）
res = pca(st, covariance_method="analytic", window_length_s=10.0, window_step_s=2.0)

# 非标准三分量：显式 dip/azimuth（SEED/ObsPy 约定）→ ZNE
st_zne = rotate_stream_to_zne(st, orientations=[(-90, 0), (0, 0), (0, 90)])

# 有反方位角时 → ZRT 再分析
res = pca(st, back_azimuth=120.0, window_length_s=10.0, window_step_s=2.0)

fig, axs = plot_pca_diagnostics(res, z_trace=st[0])
```
"""

import numpy as np
from scipy.signal import hilbert


DEFAULT_PCA_WINDOW_LENGTH_S = 10.0
DEFAULT_PCA_WINDOW_STEP_S = 2.0
DEFAULT_PCA_EPS = 1e-20
COVARIANCE_METHOD_REAL = "real"
COVARIANCE_METHOD_ANALYTIC = "analytic"
DEFAULT_COVARIANCE_METHOD = COVARIANCE_METHOD_REAL

# Okabe–Ito 三色：色盲友好，适合正刊印刷（蓝 / 朱红 / 蓝绿）
PCA_SERIES_COLORS = ("#0072B2", "#D55E00", "#009E73")


def _dip_azimuth_to_zne_vector(dip, azimuth):
    """SEED/ObsPy dip-azimuth → ZNE 单位向量（Z 向下为正）。"""
    try:
        from obspy.signal.rotate import _dip_and_azimuth2zne_vector
        return np.asarray(_dip_and_azimuth2zne_vector(dip, azimuth), dtype=float)
    except ImportError:
        dip_r = np.deg2rad(dip)
        az_r = np.deg2rad(azimuth)
        vec = np.array([
            np.sin(dip_r),
            -np.cos(dip_r) * np.cos(az_r),
            -np.cos(dip_r) * np.sin(az_r),
        ], dtype=float)
        vec /= np.linalg.norm(vec)
        return vec


def _get_trace_component_code(trace):
    if getattr(trace.stats, "component", None):
        return trace.stats.component.upper()
    if trace.stats.channel:
        return trace.stats.channel[-1].upper()
    return "?"


def _standard_zne_sac_angles(component):
    if component == "Z":
        return 0.0, 0.0
    if component == "N":
        return 90.0, 0.0
    if component == "E":
        return 90.0, 90.0
    return None


def _is_standard_zne_stream(st, atol=1.0):
    comps = [_get_trace_component_code(tr) for tr in st]
    if set(comps) != {"Z", "N", "E"}:
        return False
    for tr in st:
        comp = _get_trace_component_code(tr)
        expected = _standard_zne_sac_angles(comp)
        if expected is None:
            return False
        sac = getattr(tr.stats, "sac", None)
        if sac is None or not hasattr(sac, "cmpinc"):
            continue
        inc = float(sac.cmpinc)
        az = float(getattr(sac, "cmpaz", 0.0))
        exp_inc, exp_az = expected
        if abs(inc - exp_inc) > atol:
            return False
        if abs((az - exp_az + 180.0) % 360.0 - 180.0) > atol:
            return False
    return True


def _sac_to_dip_azimuth(trace):
    sac = trace.stats.sac
    if not hasattr(sac, "cmpinc"):
        raise ValueError(f"Trace {trace.id} missing SAC cmpinc for orientation.")
    inc = float(sac.cmpinc)
    az = float(getattr(sac, "cmpaz", 0.0))
    return inc - 90.0, az


def _resolve_orientations(st, orientations=None):
    if orientations is not None:
        if len(orientations) != 3:
            raise ValueError("orientations must contain exactly 3 (dip, azimuth) pairs.")
        return [(float(d), float(a)) for d, a in orientations]
    return [_sac_to_dip_azimuth(tr) for tr in st]


def _rotate_sensor_data_to_zne(data, orientations):
    """sensor 坐标 [d1,d2,d3] → ZNE，orientations 与 data 列顺序一致。"""
    axes = [_dip_azimuth_to_zne_vector(dip, az) for dip, az in orientations]
    rot = np.column_stack(axes)
    return (rot @ np.asarray(data, dtype=float).T).T


def _sort_component_stream(st, component_order):
    order_map = {comp: idx for idx, comp in enumerate(component_order)}
    traces = sorted(
        st,
        key=lambda tr: order_map.get(_get_trace_component_code(tr), 99),
    )
    from obspy import Stream
    return Stream(traces)


def _time_alignment_tolerance(fs):
    """允许的起止时间差异（秒），取半采样间隔。"""
    return 0.5 / float(fs)


def traces_are_time_aligned(st, atol_s=None):
    """
    检查 3 道波形是否已在共同时间段且样本数一致。

    Parameters
    ----------
    st : obspy.Stream
        3 道波形。
    atol_s : float, optional
        起止时间允许偏差（秒）；默认半采样间隔。

    Returns
    -------
    bool
    """
    if len(st) != 3:
        return False

    srs = np.array([tr.stats.sampling_rate for tr in st], dtype=float)
    if not np.allclose(srs, srs[0], rtol=0.0, atol=1e-6):
        return False
    fs = float(srs[0])
    if atol_s is None:
        atol_s = _time_alignment_tolerance(fs)

    starts = np.array([float(tr.stats.starttime) for tr in st], dtype=float)
    ends = np.array([float(tr.stats.endtime) for tr in st], dtype=float)
    npts = [tr.stats.npts for tr in st]

    if (starts.max() - starts.min()) > atol_s:
        return False
    if (ends.max() - ends.min()) > atol_s:
        return False
    if len(set(npts)) != 1:
        return False
    return True


def align_three_component_stream(st):
    """
    将 3 道波形对齐到共同时间段与统一样本网格。

    取各道重叠区间 [max(start), min(end)]，裁剪至相同 ``starttime`` 与 ``npts``。

    Parameters
    ----------
    st : obspy.Stream
        恰好 3 道、采样率一致的波形。

    Returns
    -------
    obspy.Stream
        时间对齐后的 3 道波形（新对象，不修改输入）。
    """
    from obspy import Stream

    if len(st) != 3:
        raise ValueError("Three-component stream requires exactly 3 traces.")

    srs = np.array([tr.stats.sampling_rate for tr in st], dtype=float)
    if not np.allclose(srs, srs[0], rtol=0.0, atol=1e-6):
        raise ValueError(f"Sampling rates are inconsistent: {srs}")

    common_start = max(tr.stats.starttime for tr in st)
    common_end = min(tr.stats.endtime for tr in st)
    if common_end <= common_start:
        raise ValueError("No common time range among the three traces.")

    sliced = Stream(tr.slice(common_start, common_end) for tr in st)
    min_npts = min(tr.stats.npts for tr in sliced)
    if min_npts < 3:
        raise ValueError("Common time window too short for alignment.")

    aligned = Stream()
    for tr in sliced:
        tr_out = tr.copy()
        tr_out.data = np.asarray(tr_out.data[:min_npts], dtype=tr_out.data.dtype)
        tr_out.stats.starttime = common_start
        tr_out.stats.npts = min_npts
        aligned.append(tr_out)
    return aligned


def _sync_three_traces(st):
    aligned = align_three_component_stream(st)
    data = np.column_stack([np.asarray(tr.data, dtype=float) for tr in aligned])
    return list(aligned), data, aligned[0].stats.starttime


def _build_stream_from_data(tr_templates, data, components):
    from obspy import Stream
    st_out = Stream()
    for i, comp in enumerate(components):
        tr = tr_templates[i].copy()
        tr.data = np.asarray(data[:, i], dtype=tr.data.dtype)
        if tr.stats.channel:
            tr.stats.channel = tr.stats.channel[:-1] + comp
        tr.stats.component = comp
        st_out.append(tr)
    return st_out


def rotate_stream_to_zne(st, orientations=None):
    """
    将任意三分量旋转至 Z、N、E。

    优先使用 ``orientations``（SEED/ObsPy 约定 dip、azimuth，与 ``st`` 道序一致）；
    否则从 SAC ``cmpinc``/``cmpaz`` 读取。若已是标准 ZNE 则原样返回（仅排序）。
    """
    from obspy import Stream

    if orientations is None and _is_standard_zne_stream(st):
        st_aligned = align_three_component_stream(Stream([tr.copy() for tr in st]))
        return _sort_component_stream(st_aligned, ("Z", "N", "E"))

    ori = _resolve_orientations(st, orientations=orientations)
    tr_sync, data, _ = _sync_three_traces(Stream([tr.copy() for tr in st]))
    data_zne = _rotate_sensor_data_to_zne(data, ori)
    return _build_stream_from_data(tr_sync, data_zne, ("Z", "N", "E"))


def rotate_stream_to_zrt(st, back_azimuth):
    """
    将 ZNE 三分量旋转至 Z、R、T。

    Parameters
    ----------
    st : obspy.Stream
        标准 ZNE 三分量。
    back_azimuth : float
        反方位角（度，自北顺时针，ObsPy 约定）。
    """
    from obspy import Stream

    st_zne = _sort_component_stream(
        align_three_component_stream(Stream([tr.copy() for tr in st])),
        ("Z", "N", "E"),
    )
    # 统一用 obspy rotate_ne_rt 约定（R 背离震源，与 process_image/lib.rotation 一致）
    # 2026-09-02: 原 fallback 手写公式 r=n·cos+e·sin 与 obspy 差 180° 反相；
    #             obspy 无 rotate_zne_rt，这里显式用 rotate_ne_rt。
    from obspy.signal.rotate import rotate_ne_rt
    tr_sync, data, _ = _sync_three_traces(st_zne)
    n_data, e_data = data[:, 1], data[:, 2]
    r_data, t_data = rotate_ne_rt(n_data, e_data, float(back_azimuth))
    data_zrt = np.column_stack([data[:, 0], r_data, t_data])
    return _build_stream_from_data(tr_sync, data_zrt, ("Z", "R", "T"))


def prepare_three_component_stream(st, orientations=None, back_azimuth=None):
    """
    预处理三分量：必要时旋转至 ZNE，若给定反方位角则进一步旋转至 ZRT。

    Returns
    -------
    st_out : obspy.Stream
    component_order : str
        ``'ZNE'`` 或 ``'ZRT'``。
    back_azimuth_used : float or None
    """
    need_zne = orientations is not None or not _is_standard_zne_stream(st)
    from obspy import Stream

    st_aligned = align_three_component_stream(Stream([tr.copy() for tr in st]))
    if need_zne:
        st_out = rotate_stream_to_zne(st_aligned, orientations=orientations)
    else:
        st_out = _sort_component_stream(st_aligned, ("Z", "N", "E"))

    component_order = "ZNE"
    baz_used = None
    if back_azimuth is not None:
        st_out = rotate_stream_to_zrt(st_out, back_azimuth)
        component_order = "ZRT"
        baz_used = float(back_azimuth) % 360.0
    return st_out, component_order, baz_used


def _horizontal_back_azimuth(u_horiz_1, u_horiz_2, component_order, back_azimuth_used=None):
    """水平偏振方向 → 自北顺时针反方位角（度）。

    2026-09-02: 旋转约定已统一为 obspy（R 背离震源，r = -n·cos - e·sin）。
    ZRT 逆变换：u_n = -u_r·cos(baz) - u_t·sin(baz)? 需与 rotate_ne_rt 一致：
      r = -e·sin - n·cos;  t = -e·cos + n·sin
    解出 n,e：由 r,t 反求 (n,e)：
      n = -r·cos(baz) + t·sin(baz)
      e = -r·sin(baz) - t·cos(baz)
    """
    if component_order == "ZRT" and back_azimuth_used is not None:
        baz_r = np.deg2rad(back_azimuth_used)
        u_n = -u_horiz_1 * np.cos(baz_r) + u_horiz_2 * np.sin(baz_r)
        u_e = -u_horiz_1 * np.sin(baz_r) - u_horiz_2 * np.cos(baz_r)
        return np.degrees(np.arctan2(u_e, u_n)) % 360.0
    return np.degrees(np.arctan2(u_horiz_2, u_horiz_1)) % 360.0


def _validate_covariance_method(covariance_method):
    if covariance_method not in (COVARIANCE_METHOD_REAL, COVARIANCE_METHOD_ANALYTIC):
        raise ValueError(
            "covariance_method must be 'real' or 'analytic', "
            f"got {covariance_method!r}."
        )


def _complex_polarization_vector_to_real(u, eps):
    """复特征向量 → 实单位偏振方向（Vidale 相位对齐）。"""
    u = np.asarray(u, dtype=np.complex128)
    norm = np.linalg.norm(u)
    if norm <= eps:
        return np.full(3, np.nan, dtype=float)
    u = u / norm
    ref = int(np.argmax(np.abs(u)))
    u_aligned = u * np.exp(-1j * np.angle(u[ref]))
    u_real = np.real(u_aligned)
    real_norm = np.linalg.norm(u_real)
    if real_norm > eps:
        return u_real / real_norm
    return np.real(u) / (np.linalg.norm(np.real(u)) + eps)


def _covariance_eigen_real(x):
    """实值三分量协方差特征分解。"""
    x0 = x - np.mean(x, axis=0, keepdims=True)
    cov = np.cov(x0, rowvar=False, bias=False)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    lam = np.maximum(vals[idx], 0.0)
    u1 = vecs[:, idx[0]]
    return lam, u1


def _covariance_eigen_analytic(x):
    """
    解析信号复协方差特征分解（Vidale, 1986）。

    各分量 Hilbert 变换构成解析信号 z = x + i H[x]，
    复 Hermitian 协方差 C = <z z^H> 的特征值对窄带信号更稳健。
    """
    x0 = x - np.mean(x, axis=0, keepdims=True)
    z = hilbert(x0, axis=0)
    n = z.shape[0]
    cov = (z.conj().T @ z) / max(n - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(np.real(vals))[::-1]
    lam = np.maximum(np.real(vals[idx]), 0.0)
    u1 = _complex_polarization_vector_to_real(vecs[:, idx[0]], DEFAULT_PCA_EPS)
    return lam, u1


def _extract_synced_three_components(st):
    """从 Stream 提取三分量；若未对齐则先对齐。"""
    if len(st) != 3:
        raise ValueError("pca() requires exactly 3 traces.")

    aligned = align_three_component_stream(st)
    fs = float(aligned[0].stats.sampling_rate)
    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")

    data = np.column_stack([np.asarray(tr.data, dtype=float) for tr in aligned])
    return data, fs, aligned[0].stats.starttime


def pca(
    st,
    window_length_s=DEFAULT_PCA_WINDOW_LENGTH_S,
    window_step_s=DEFAULT_PCA_WINDOW_STEP_S,
    eps=DEFAULT_PCA_EPS,
    orientations=None,
    back_azimuth=None,
    covariance_method=DEFAULT_COVARIANCE_METHOD,
):
    """
    三分量 PCA 偏振分析（滑窗）。

    Parameters
    ----------
    st : obspy.Stream
        恰好 3 道波形。可为任意三分量；若非标准 ZNE，需提供 ``orientations``
        或 SAC ``cmpinc``/``cmpaz``。分析前按 [Z, N, E]（或给定 ``back_azimuth``
        时的 [Z, R, T]）排列。
    window_length_s : float
        滑动窗长度（秒）。
    window_step_s : float
        滑动步长（秒）。
    eps : float
        数值稳定项。
    orientations : list of (dip, azimuth), optional
        各道 SEED/ObsPy 取向（度），与 ``st`` 道序一致；用于旋转至 ZNE。
    back_azimuth : float, optional
        反方位角（度）；若提供，在 ZNE 基础上旋转至 ZRT 再分析。
    covariance_method : {'real', 'analytic'}
        协方差估计方式。``'real'``（默认）为实值协方差；``'analytic'`` 使用
        Hilbert 解析信号复协方差（Vidale, 1986），
        窄带信号特征值更稳健。

    Returns
    -------
    dict
        time_windows_s, time_windows_utc, eigenvalues, principal_vector,
        back_azimuth, incidence, linearity, planarity, sphericity, valid_mask, meta

        偏振指标（λ₁ ≥ λ₂ ≥ λ₃）：
        - linearity (L) = (λ₁ − λ₂) / λ₁
        - planarity (P) = (λ₂ − λ₃) / λ₁
        - sphericity (S) = λ₃ / λ₁
    """
    if window_length_s <= 0 or window_step_s <= 0:
        raise ValueError("window_length_s and window_step_s must be positive.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if len(st) != 3:
        raise ValueError("pca() requires exactly 3 traces.")
    _validate_covariance_method(covariance_method)

    try:
        st_prep, component_order, baz_used = prepare_three_component_stream(
            st, orientations=orientations, back_azimuth=back_azimuth,
        )
    except ValueError as exc:
        if orientations is None and not _is_standard_zne_stream(st):
            raise ValueError(
                "Non-ZNE components require orientations=(dip, azimuth) for each "
                "trace, or SAC cmpinc/cmpaz in trace headers."
            ) from exc
        raise

    time_aligned_input = traces_are_time_aligned(st)
    data, fs, common_start = _extract_synced_three_components(st_prep)
    npts = data.shape[0]

    win_n = int(round(window_length_s * fs))
    step_n = int(round(window_step_s * fs))
    if win_n < 3:
        raise ValueError("window_length_s is too short (must contain >= 3 samples).")
    if step_n < 1:
        raise ValueError("window_step_s is too short.")
    if npts < win_n:
        raise ValueError("Data is shorter than one analysis window.")

    nwindows = 1 + (npts - win_n) // step_n
    starts = np.arange(nwindows, dtype=int) * step_n
    time_windows_s = starts / fs

    eigenvalues = np.full((nwindows, 3), np.nan, dtype=float)
    principal_vector = np.full((nwindows, 3), np.nan, dtype=float)
    back_azimuth = np.full(nwindows, np.nan, dtype=float)
    incidence = np.full(nwindows, np.nan, dtype=float)
    linearity = np.full(nwindows, np.nan, dtype=float)
    planarity = np.full(nwindows, np.nan, dtype=float)
    sphericity = np.full(nwindows, np.nan, dtype=float)
    valid_mask = np.zeros(nwindows, dtype=bool)

    for i, s0 in enumerate(starts):
        x = data[s0:s0 + win_n, :]

        if covariance_method == COVARIANCE_METHOD_ANALYTIC:
            lam, u1 = _covariance_eigen_analytic(x)
        else:
            lam, u1 = _covariance_eigen_real(x)

        lam1, lam2, lam3 = lam
        lam1_d = lam1 + eps

        # 偏振主轴方向：分量顺序 [Z, horiz1, horiz2] = ZNE 或 ZRT
        az = _horizontal_back_azimuth(u1[1], u1[2], component_order, baz_used)
        inc = np.degrees(np.arctan2(np.sqrt(u1[1] ** 2 + u1[2] ** 2), abs(u1[0])))

        l_val = (lam1 - lam2) / lam1_d
        p_val = (lam2 - lam3) / lam1_d
        s_val = lam3 / lam1_d

        eigenvalues[i, :] = lam
        principal_vector[i, :] = u1
        back_azimuth[i] = az
        incidence[i] = inc
        linearity[i] = np.clip(l_val, 0.0, 1.0)
        planarity[i] = np.clip(p_val, 0.0, 1.0)
        sphericity[i] = np.clip(s_val, 0.0, 1.0)
        valid_mask[i] = (
            np.isfinite(lam1) and lam1 > eps and np.all(np.isfinite(u1))
        )

    time_windows_utc = [common_start + float(t) for t in time_windows_s]

    return {
        "time_windows_s": time_windows_s,
        "time_windows_utc": time_windows_utc,
        "eigenvalues": eigenvalues,
        "principal_vector": principal_vector,
        "back_azimuth": back_azimuth,
        "incidence": incidence,
        "linearity": linearity,
        "planarity": planarity,
        "sphericity": sphericity,
        "valid_mask": valid_mask,
        "meta": {
            "fs": fs,
            "window_length_s": window_length_s,
            "window_step_s": window_step_s,
            "window_samples": win_n,
            "step_samples": step_n,
            "nwindows": nwindows,
            "common_start": str(common_start),
            "component_order": component_order,
            "back_azimuth_rotation": baz_used,
            "covariance_method": covariance_method,
            "time_aligned_on_input": time_aligned_input,
        },
    }


def plot_pca_diagnostics(result, z_trace=None, stacked=False):
    """
    绘制 PCA 偏振诊断图（共享 X 轴）。

    Parameters
    ----------
    result : dict
        `pca()` 的返回结果。
    z_trace : obspy.Trace, optional
        Z 分量波形；若提供则绘制在第一行，时间轴与滑窗结果对齐。
    stacked : bool
        若为 True，L/P/S 以堆叠面积图绘制（总高度恒为 1）；
        若为 False（默认），三条曲线以不同颜色绘制在同一坐标轴上。

    Returns
    -------
    fig, axs
        Matplotlib 图形与子图数组。
    """
    import matplotlib.pyplot as plt
    from obspy import UTCDateTime

    t = np.asarray(result["time_windows_s"], dtype=float)
    l_val = np.asarray(result["linearity"], dtype=float)
    p_val = np.asarray(result["planarity"], dtype=float)
    s_val = np.asarray(result["sphericity"], dtype=float)
    valid = np.asarray(result["valid_mask"], dtype=bool)
    eig = np.asarray(result["eigenvalues"], dtype=float)
    meta = result["meta"]
    fs = float(meta["fs"])

    c1, c2, c3 = PCA_SERIES_COLORS
    colors = {"L": c1, "P": c2, "S": c3}
    lam_colors = {"λ1": c1, "λ2": c2, "λ3": c3}

    fig, axs = plt.subplots(
        5,
        1,
        figsize=(12, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0, 1.0, 1.2], "hspace": 0.08},
    )

    # 第 1 行：Z 分量波形
    if z_trace is not None:
        common_start = UTCDateTime(meta["common_start"])
        win_end_s = float(meta["window_length_s"])
        plot_end = common_start + (float(t[-1]) + win_end_s if len(t) else z_trace.stats.endtime - common_start)
        tr_z = z_trace.slice(common_start, plot_end)
        t_wave = np.arange(tr_z.stats.npts, dtype=float) / fs
        axs[0].plot(t_wave, tr_z.data, color="black", lw=0.7)
        axs[0].set_ylabel("Z")
    else:
        axs[0].text(
            0.5,
            0.5,
            "No z_trace provided",
            transform=axs[0].transAxes,
            ha="center",
            va="center",
        )
        axs[0].set_ylabel("Z")
    axs[0].set_title("PCA Polarization Diagnostics")
    axs[0].grid(alpha=0.2, ls="--")

    # 第 2 行：特征值 λ1, λ2, λ3
    axs[1].plot(t, eig[:, 0], "-", lw=1.0, color=lam_colors["λ1"], label="λ₁")
    axs[1].plot(t, eig[:, 1], "-", lw=1.0, color=lam_colors["λ2"], label="λ₂")
    axs[1].plot(t, eig[:, 2], "-", lw=1.0, color=lam_colors["λ3"], label="λ₃")
    axs[1].set_ylabel("Eigenvalue")
    axs[1].set_title("Covariance eigenvalues")
    axs[1].grid(alpha=0.2, ls="--")
    axs[1].legend(loc="upper right", fontsize=8)

    axs[2].plot(t[valid], result["back_azimuth"][valid], ".", ms=3, color="tab:blue")
    axs[2].set_ylabel("Back-azimuth (deg)")
    axs[2].set_ylim(0, 360)
    axs[2].grid(alpha=0.2, ls="--")

    axs[3].plot(t[valid], result["incidence"][valid], ".", ms=3, color="tab:orange")
    axs[3].set_ylabel("Incidence (deg)")
    axs[3].grid(alpha=0.2, ls="--")

    if stacked:
        l_plot = np.where(valid, l_val, 0.0)
        p_plot = np.where(valid, p_val, 0.0)
        s_plot = np.where(valid, s_val, 0.0)
        axs[4].stackplot(
            t,
            s_plot,
            p_plot,
            l_plot,
            labels=["S (sphericity)", "P (planarity)", "L (linearity)"],
            colors=[colors["S"], colors["P"], colors["L"]],
            alpha=0.85,
        )
        axs[4].set_ylabel("Fraction (L + P + S = 1)")
        axs[4].set_title("Polarization indices (stacked)")
    else:
        axs[4].plot(t, l_val, "-", lw=1.0, color=colors["L"], label="L (linearity)")
        axs[4].plot(t, p_val, "-", lw=1.0, color=colors["P"], label="P (planarity)")
        axs[4].plot(t, s_val, "-", lw=1.0, color=colors["S"], label="S (sphericity)")
        axs[4].set_ylabel("Index (L + P + S = 1)")
        axs[4].set_title("Polarization indices")

    axs[4].set_ylim(0.0, 1.0)
    axs[4].grid(alpha=0.2, ls="--")
    axs[4].legend(loc="upper right", fontsize=8)
    axs[4].set_xlabel("Time since common start (s)")

    fig.tight_layout()
    return fig, axs

