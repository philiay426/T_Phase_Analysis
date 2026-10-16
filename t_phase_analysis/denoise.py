"""
参数无关 SS-CWT 域降噪 (Parameter-Free Denoising in SS-CWT Domain)。

基于 Zeng Zhiyi (SUSTech) 的 MATLAB 实现改写为 Python。

算法步骤：
1. SS-CWT 时频变换 (Synchrosqueezed Continuous Wavelet Transform)
2. ROV (Ratio of Variance) 估计纯背景噪声范围
3. GCV (Generalized Cross-Validation) 逐频段优化阈值
4. 像素连通性滤波去除孤立噪声残留
5. 逆 SS-CWT 重建降噪波形

依赖: numpy, scipy

作者: Yong Zhou (zhouyong@scsio.ac.cn)

调用示例
--------
```python
from t_phase_analysis import para_free_denoise

result = para_free_denoise(trace.data, dt=trace.stats.delta)
print(result["denoised"].shape, result["frequencies"].shape)
```
"""

import numpy as np
from scipy import fftpack
from scipy.ndimage import label


# ═══════════════════════════════════════════════════════════════════════════
# Morlet 小波 & CWT & SS-CWT
# ═══════════════════════════════════════════════════════════════════════════

def _morlet_ft(omega, w0=6.0):
    """Analytic Morlet wavelet in frequency domain (omega in rad/s).
    Normalization matched to MATLAB 'amor' wavelet (empirical factor 2.054)."""
    H = np.where(omega > 0, 1.0, 0.0)
    return 2.054 * np.pi**(-0.25) * H * np.exp(-0.5 * (omega - w0)**2)


def _cwt_forward(data, dt, nv=32, w0=6.0):
    """L1-normalized continuous wavelet transform via FFT."""
    n = len(data)
    x_fft = fftpack.fft(data)
    f_nyq = 0.5 / dt
    f_min = max(1.0 / (n * dt), f_nyq / n)
    n_octaves = np.log2(f_nyq / f_min)
    na = max(2, int(np.floor(n_octaves)) * nv)
    freqs = np.logspace(np.log10(f_min), np.log10(f_nyq), na)
    # scale in SECONDS (NOT samples)
    scales = w0 / (2.0 * np.pi * freqs)
    omega_grid = 2.0 * np.pi * fftpack.fftfreq(n, dt)
    W = np.zeros((na, n), dtype=np.complex128)
    for i, a in enumerate(scales):
        psi_hat = _morlet_ft(a * omega_grid, w0)
        W[i, :] = fftpack.ifft(x_fft * np.conj(psi_hat))
    return W, scales, freqs


# Admissibility constant (cached)
_PSI_ADMISSIBILITY = None


def _psi_admissibility(w0=6.0):
    """C_psi = 0.5 * ∫ conj(psi_hat(ω)) / ω dω."""
    global _PSI_ADMISSIBILITY
    if _PSI_ADMISSIBILITY is None:
        ot = np.linspace(0.01, 60, 200000)
        _PSI_ADMISSIBILITY = 0.5 * np.trapezoid(_morlet_ft(ot, w0) / ot, ot).real
    return _PSI_ADMISSIBILITY


def _ss_cwt_forward(data, dt, nv=32, w0=6.0):
    """Synchrosqueezed CWT forward — delegates to MATLAB wsst via Engine."""
    _ensure_matlab_engine()
    import matlab
    fs = 1.0 / dt
    data_mat = matlab.double(data.tolist())
    fs_mat = matlab.double(float(fs))
    Tx_cell, f_cell = _MATLAB_ENG.wsst(data_mat, fs_mat, nargout=2)
    Tx = np.array(Tx_cell, dtype=np.complex128)
    freqs = np.array(f_cell).ravel()
    return Tx, freqs


def _ss_cwt_inverse(Tx, freqs, freq_range=None):
    """Inverse SS-CWT — delegates to MATLAB iwsst via Engine."""
    _ensure_matlab_engine()
    import matlab, scipy.io as sio, os, tempfile
    # Save Tx to temp .mat file, load in MATLAB, run iwsst
    tmpf = os.path.join(tempfile.gettempdir(), '_hermes_iwsst.mat')
    sio.savemat(tmpf, {'Tx_py': Tx, 'f_py': freqs})
    _MATLAB_ENG.eval(f"load('{tmpf}'); dn = iwsst(Tx_py, f_py, [f_py(1) f_py(end)]);", nargout=0)
    x_rec = np.array(_MATLAB_ENG.workspace['dn']).ravel()
    return x_rec


# ── MATLAB Engine singleton ──
_MATLAB_ENG = None


def _ensure_matlab_engine():
    """Lazily start MATLAB Engine if not already running."""
    global _MATLAB_ENG
    if _MATLAB_ENG is None:
        import matlab.engine
        _MATLAB_ENG = matlab.engine.start_matlab()



# ═══════════════════════════════════════════════════════════════════════════
# ROV — Ratio of Variance 噪声估计
# ═══════════════════════════════════════════════════════════════════════════

def _rov_estimate(data, nrs=10):
    """
    通过方差比 (ROV) 估计纯背景噪声范围。
    
    ROV(i) = var(data[0:i]) / var(data[i:end])
    最小值位置 → 信号起始点，之前为纯噪声。
    
    Parameters
    ----------
    data : ndarray (n,)
        输入信号。
    nrs : int
        ROV 曲线起始样本的除数因子。默认 10。
    
    Returns
    -------
    noise_end_idx : int
        纯噪声结束样本索引 (0 ~ noise_end_idx 为噪声范围)。
    rov : ndarray
        ROV 曲线 (诊断用)。
    """
    n = len(data)
    start_s = max(1, n // nrs)
    end_s = start_s
    
    rov = np.zeros(n)
    
    for i in range(start_s + 1, n - end_s):
        rov[i] = np.var(data[:i]) / (np.var(data[i:]) + 1e-20)
    
    # 在有效范围内找最小值
    valid_range = slice(start_s + 1, n - end_s)
    noise_t = np.argmin(rov[valid_range])
    noise_end_idx = noise_t + start_s + 1
    
    return noise_end_idx, rov


# ═══════════════════════════════════════════════════════════════════════════
# GCV — Generalized Cross-Validation 阈值
# ═══════════════════════════════════════════════════════════════════════════

def _gcv_score(coeffs, threshold):
    """
    GCV 评分函数。
    
    对系数施加软阈值后，计算 GCV 指标。
    
    Parameters
    ----------
    coeffs : ndarray
        小波系数。
    threshold : float
        阈值。
    
    Returns
    -------
    gcv : float
        GCV 评分（越小越好）。
    """
    N = len(coeffs)
    res = np.abs(coeffs) - threshold
    N0 = np.sum(res <= 0)
    
    if N0 == 0:
        return 0.0
    
    # 软阈值
    y = np.sign(coeffs) * res * (res >= 0)
    
    gcv = N * (np.linalg.norm(y - coeffs) / N0)**2
    return gcv


def _fibonacci_search(coeffs, interval, n_iter=20):
    """
    Fibonacci 搜索：在 [a, b] 内找使 GCV 最小的阈值。
    
    Parameters
    ----------
    coeffs : ndarray
        小波系数。
    interval : tuple (a, b)
        搜索区间。
    n_iter : int
        Fibonacci 数列长度。默认 20。
    
    Returns
    -------
    best_threshold : float
        最优阈值。
    """
    a, b = float(interval[0]), float(interval[1])
    
    if b <= a:
        return (a + b) / 2.0
    
    # Fibonacci 数列
    F = [1, 1]
    for _ in range(n_iter - 2):
        F.append(F[-1] + F[-2])
    F = F[:n_iter]
    n = len(F) - 1
    
    x1 = a + (F[n - 2] / F[n]) * (b - a)
    x2 = a + (F[n - 1] / F[n]) * (b - a)
    f1 = _gcv_score(coeffs, x1)
    f2 = _gcv_score(coeffs, x2)
    
    while n > 3:
        n -= 1
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (F[n - 1] / F[n]) * (b - a)
            f2 = _gcv_score(coeffs, x2)
        elif f1 == f2:
            a = x1
            b = x2
            x1 = a + (F[n - 2] / F[n]) * (b - a)
            x2 = a + (F[n - 1] / F[n]) * (b - a)
            f1 = _gcv_score(coeffs, x1)
            f2 = _gcv_score(coeffs, x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (F[n - 2] / F[n]) * (b - a)
            f1 = _gcv_score(coeffs, x1)
    
    return (a + b) / 2.0


# ═══════════════════════════════════════════════════════════════════════════
# 阈值处理
# ═══════════════════════════════════════════════════════════════════════════

def _hard_threshold(coeffs, threshold):
    """
    硬阈值：低于阈值的系数设为零，高于的保留。
    
    res = |y| - t
    x = (sign(res) + 1) * y / 2
    """
    res = np.abs(coeffs) - threshold
    return (np.sign(res) + 1) * coeffs / 2.0


# ═══════════════════════════════════════════════════════════════════════════
# 像素连通性滤波
# ═══════════════════════════════════════════════════════════════════════════

def _pixel_connectivity_filter(Tx, connectivity=8):
    """
    像素连通性滤波：去除时频谱中面积过小的孤立区域。

    将非零系数转为二值图，用连通域分析标记各区域，
    面积低于阈值的区域视为噪声残留 → 设为零。
    与 MATLAB pixe_pro() 一致：纯二值掩码，不加面积权重。

    Parameters
    ----------
    Tx : ndarray (n_freqs, n_times)
        阈值化后的时频谱。
    connectivity : int
        连通性：4 或 8。默认 8。

    Returns
    -------
    filtered_Tx : ndarray
        滤波后时频谱。
    """
    binary = (np.abs(Tx) > 0).astype(np.int32)

    # 连通域标记
    structure = np.ones((3, 3)) if connectivity == 8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, n_features = label(binary, structure=structure)

    if n_features == 0:
        return Tx

    # 计算各区域面积
    areas = np.bincount(labeled.ravel())[1:]  # 去掉背景 (label=0)

    if len(areas) == 0:
        return Tx

    # 面积阈值：基于面积的 MAD（与 MATLAB 一致）
    area_threshold = np.sqrt(2.0 * np.log(len(areas))) * _mad(areas) / 0.6748

    # 找出面积足够大的区域
    valid_labels = np.where(areas >= area_threshold)[0] + 1  # +1 因为 label 从 1 开始

    if len(valid_labels) == 0:
        # 保留最大区域
        valid_labels = np.array([np.argmax(areas) + 1])

    # MATLAB 风格：纯二值掩码 wk .* BW2，不加面积权重
    mask = np.isin(labeled, valid_labels)
    filtered = Tx * mask

    return filtered


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def _mad(x):
    """均值绝对偏差 (Mean Absolute Deviation) — 与 MATLAB mad(X) 一致。"""
    return np.mean(np.abs(x - np.mean(x)))


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def para_free_denoise(data, dt, f_s=0.0, f_e=None, nrs=20, bwconn=8,
                      pixel_passes=1, variant=5):
    """
    参数无关 SS-CWT 域降噪。
    
    Parameters
    ----------
    data : ndarray (n,)
        含噪信号（去除 NaN 后的一维数组）。
    dt : float
        采样间隔 (s)。
    f_s, f_e : float
        降噪频率范围 (Hz)。f_e=None 时使用 Nyquist 频率。
    nrs : int
        ROV 起始样本除数。默认 20。
    bwconn : int
        像素连通性 (4 或 8)。默认 8。
    pixel_passes : int
        像素连通性滤波次数。默认 1。
    variant : int
        算法变体 (0, 3, 5)。
        - 0: 基础版，噪声窗估计阈值，无归一化
        - 3: 全数据估计阈值，mad/0.6745 归一化
        - 5: 噪声窗 + mad/0.67 + 均值修正 + 2次像素滤波 (推荐)
    
    Returns
    -------
    result : dict
        ``denoised``: 降噪后波形
        ``denoised_tf``: 降噪后时频谱
        ``original_tf``: 原始时频谱
        ``frequencies``: 频率轴 (Hz)
        ``gcv_thresholds``: 各频段 GCV 阈值
        ``noise_end_idx``: ROV 噪声结束索引
        ``rov``: ROV 曲线
    """
    n = len(data)
    nyquist = 1.0 / (2.0 * dt)
    if f_e is None:
        f_e = nyquist
    
    # ── 1. SS-CWT ──
    Tx, freqs = _ss_cwt_forward(data, dt)
    n_freqs, n_times = Tx.shape
    
    # ── 2. ROV 噪声估计 ──
    noise_end_idx, rov = _rov_estimate(data, nrs=nrs)
    
    # ── 3. 频率范围 ──
    fi1 = np.argmin(np.abs(freqs - f_s))
    fi2 = np.argmin(np.abs(freqs - f_e))
    
    # ── 4. GCV 逐频阈值 ──
    Tx_denoised = np.zeros_like(Tx)
    thresholds = np.zeros(n_freqs)
    
    for i in range(fi1, fi2 + 1):
        wk = Tx[i, :]
        
        if variant == 0:
            # 变体 0: 仅用噪声窗，不除 0.6745
            noise_wk = np.abs(wk[:noise_end_idx])
            thre1 = np.sqrt(2.0 * np.log(noise_end_idx)) * _mad(noise_wk)
            thre_a = thre1
        elif variant == 3:
            # 变体 3: 全数据，除 0.6745
            noise_wk = np.abs(wk)
            thre1 = np.sqrt(2.0 * np.log(noise_end_idx)) * _mad(noise_wk) / 0.6745
            thre_a = thre1
        else:  # variant == 5 (推荐)
            # 变体 5: 噪声窗 + /0.67 + 如超阈值取均值
            noise_wk = np.abs(wk[:noise_end_idx])
            thre1 = np.sqrt(2.0 * np.log(noise_end_idx)) * _mad(noise_wk) / 0.67
            above = noise_wk[noise_wk >= thre1]
            if len(above) > 0:
                thre_a = np.mean(above)
            else:
                thre_a = thre1
        
        thre_b = np.max(np.abs(wk))
        interval = (thre_a, thre_b)
        
        gcv_thre = _fibonacci_search(wk, interval)
        Tx_denoised[i, :] = _hard_threshold(wk, gcv_thre)
        thresholds[i] = gcv_thre
    
    # ── 5. 像素连通性滤波 ──
    for _ in range(pixel_passes):
        Tx_denoised = _pixel_connectivity_filter(Tx_denoised, connectivity=bwconn)
    
    # ── 6. 逆变换 ──
    denoised = _ss_cwt_inverse(Tx_denoised, freqs)
    
    # 长度对齐
    denoised = denoised[:n] if len(denoised) > n else np.pad(denoised, (0, max(0, n - len(denoised))))
    
    return {
        'denoised': denoised,
        'denoised_tf': Tx_denoised,
        'original_tf': Tx,
        'frequencies': freqs,
        'gcv_thresholds': thresholds,
        'noise_end_idx': noise_end_idx,
        'rov': rov,
    }
