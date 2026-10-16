# Python `denoise.py` vs MATLAB `para_free_denoising5` 对比

## 版本定义

| 版本 | wsst | iwsst | ROV | GCV | 像素滤波 | 说明 |
|------|------|-------|-----|-----|----------|------|
| **m1** | Python 手写 SS-CWT | Python 手写 iSS-CWT | Python | Python | Python | 纯 Python，无 MATLAB 依赖 |
| **m4** | MATLAB Engine | MATLAB Engine | Python | Python | Python | 混合方案（仅 wsst/iwsst 调 MATLAB） |
| **m5** | MATLAB 原生 | MATLAB 原生 | MATLAB | MATLAB | MATLAB | 纯 MATLAB，作为 ground truth |

## 测试数据

- **syn1**: `data_noise_new7.mat` + `original_synthetic_data.mat`
- 参数: sr=6000 Hz, 2676 samples, variant=5, f=[30,3000] Hz, nrs=20, bwconn=8, pixel_passes=2

## 定量对比

| 指标 | m1 Python | m4 Hybrid | m5 MATLAB |
|------|:--:|:--:|:--:|
| CC vs clean | 0.9675 | 0.9673 | 0.9673 |
| 输出范围 | [-0.647, 0.731] | [-0.674, 0.735] | [-0.674, 0.735] |
| vs m5 CC | 0.9950 | **0.99995** | 1.00000 |
| vs m5 NRMSE | 0.97% | **0.09%** | 0% |
| vs m5 max\|Δ\| | 0.103 | **0.014** | 0 |

## 修复历史

| # | 问题 | 修复 | 影响版本 |
|---|------|------|:--:|
| 1 | `_mad` 用中位数偏差，MATLAB `mad()` 用均值偏差 | 改为 `mean(abs(x-mean(x)))` | m1/m4 |
| 2 | 像素滤波加面积权重 `(area/max_area+0.1)`，MATLAB 用纯二值掩码 | 改为 `np.isin(labeled, valid_labels)` | m1/m4 |
| 3 | CWT 用 L2 归一化 (`*np.sqrt(a)`)，MATLAB 用 L1 | 去掉 `*np.sqrt(a)` | m1/m4 |
| 4 | 频率网格 `int(n_octaves*nv)`，MATLAB 用 `floor(n_octaves)*nv` | 改为 `int(floor(n_octaves))*nv` | m1/m4 |
| 5 | SS-CWT squeezing 算法与 MATLAB 本质不同，无法通过调参对齐 | 用 MATLAB Engine 替换 wsst/iwsst | m4 |

## 结论

- **m4 与 m5 等价**：仅剩 0.09% NRMSE，来自 `numpy.var()` vs MATLAB `var()` 的浮点精度差异（ROV 噪声窗差 2 样本），非算法错误
- **m1 残差 ~1%**：手写 SS-CWT 与 MATLAB wsst 内部算法差异无法完全消除
- **生产推荐 m4**：wsst/iwsst 走 MATLAB Engine，ROV/GCV/像素滤波保留纯 Python

## 运行环境

- **m4/m5 依赖**: MATLAB R2023b + Python 3.11 (matlab_py311 conda env)
- **m1 依赖**: numpy, scipy（纯 Python，无 MATLAB）
