# T_Phase_Analysis 修改日志

## 2026-05-22 — 更名为 T_Phase_Analysis，建正式 Python 包骨架

- 目录: `~/workspace/T_Phase_Analysis/`
- 安装: `pip install -e ~/workspace/T_Phase_Analysis`
- 导入: `from t_phase_analysis import get_track, tt_t, pmcc, f_statistic`
- 结构:
  ```
  t_phase_analysis/
  ├── __init__.py          # 统一导出
  ├── analysis.py          # get_track, tt_t, tpef, snr (原 T_phase_analysis.py)
  ├── waveform.py          # plot_waveform_panel, plot_spectrogram_panel, compute_spectrogram
  ├── pmcc_detector.py     # pmcc, f_statistic (原 pmcc_wrapper.py)
  └── internals/
      ├── PMCC/            # 原 PMCC 子包
      └── F-detector/      # 原 F-detector 子包
  ```
- 旧路径 `sys.path.insert` 方式继续兼容。

## 2026-05-22 — PMCC/F-detector 可调用接口 + xcorr 函数改造

### 背景

项目 `taiwan_eq_20240402/plot_close_stations.py` 需要调用 T 波分析工具计算理论到时。此前已将 `T_phase_analysis.py` 中的 `get_track()` 和 `tt_t()` 集成到波形图中。本次进一步分析 `F-detector/` 和 `PMCC/` 两个子包，将其改造为可直接调用的函数。

### 分析

#### 1. 可调用性

| 模块 | 原有形态 | 问题 |
|------|---------|------|
| `F_statistic.py` | `compute_F_statistic(st, ...)` | ✓ 已是可调用函数 |
| `azimuth_speed_array.py` | `main()` 通过 `sys.argv` 读取参数 | ✗ 仅命令行可用 |

`azimuth_speed_array.py` 的核心逻辑（互相关→延迟时间→扫描方位角/速度→反演最优解）被嵌入在 `main()` 中，与文件 I/O 和绘图混在一起，无法被其他脚本导入使用。

#### 2. `xcorr_pick_correction` 版本选择

PMCC 依赖互相关函数 `xcorr_pick_correction`。存在两个版本：

| | ObsPy 官方 | WENBO 修改版 (`cross_correlation_WENBO.py`) |
|---|---|---|
| 返回值 | `(dt, cc)` | `(dt, cc, warning_tag)` |
| `num_samples < 3` | 抛异常 | warning_tag=6 |
| `CC_max < 0` 处理 | 打印警告 | 静默修正 |
| `CC_max < 0.8` | 打印警告 | 注释掉 |
| 截断时间修正 | 无 | `cut_time_truncation`（~半采样间隔） |
| 依赖 | 纯 ObsPy | 额外依赖 `future`（**Py3.14 不可用**） |

**决策**：以 ObsPy 官方版为内核，包装一层加入 WENBO 的 3 个关键改进：

- `warning_tag` 编码 — 批量运行时事后质控
- 不抛异常 — 自动化友好，不因个别坏窗中断
- 静默处理 — 避免 `stderr` 刷屏

WENBO 的 `cut_time_truncation` 修正（约 0.025s @ 20Hz）对 PMCC 方位角反演影响可忽略，不纳入移植。

### 具体改动

#### 新建 `pmcc_wrapper.py`

路径：`T_phase_tools/pmcc_wrapper.py`

提供两个可直接调用的函数：

**`pmcc(st, time_start, time_one_slice, time_slide, time_length, ...)`**
```
从 azimuth_speed_array.py 的 main() 中提取的核心逻辑：
1. 逐窗对 3 台站做两两互相关 → dt12/13/23, CC12/13/23
2. consistency = |dt12 + dt23 - dt13| < 阈值 ?
3. 扫描方位角(0–360°) × 速度(1200–1800 m/s)
4. misfit = Σ(dt_pred - dt_obs)² → 最小 misfit 对应最佳解
5. 统计众数作为最终方位角/速度估计
返回：time_measure, consistency, ave_CC, azimuth_find, speed_find, ...
```

**`f_statistic(st, time_start, time_one_slice, time_slide, time_length, ...)`**
```
从 F_statistic.py 的 compute_F_statistic() 整理：
1. 对每个方位角/时间窗组合计算 F 统计量
2. F = (2/3) × (Σtr)² / (3·Σ var(tr))
返回：time_windows, azimuths, F_statistic (n_azimuth × n_windows)
```

#### 内置 `_xcorr_pick_correction()` 包装函数

```python
def _xcorr_pick_correction(pick1, trace1, pick2, trace2, ...):
    """
    WENBO 风格包装：在 ObsPy xcorr_pick_correction 基础上加入 warning_tag。
    - 调用 obspy.signal.cross_correlation.xcorr_pick_correction（官方内核）
    - 用 warnings.catch_warnings 拦截 ObsPy 警告 → 转为 warning_tag 编码
    - num_samples < 3 时捕获异常，设 warning_tag=6，不中断
    - 返回 (dt, cc, warning_tag)
    """
```

warning_tag 编码：0=正常, 1=trace id 不匹配, 2/3=滤波边界, 4=CC 负值, 5=拟合在最大 lag 边界, 6=拟合样本<5, 7=抛物线开口向上, 8=拟合残差大。

#### 消除的依赖链

原 PMCC 导入链：`azimuth_speed_array.py` → `PMCC_auxiliary.py` → `global_parameters_TOT.py` → `cross_correlation_WENBO.py` → `future`

新 `pmcc_wrapper.py`：零内部依赖，仅需 `numpy`, `obspy`。`find_time_relative_to_b` 和 `find_second_largest` 两个简单辅助函数直接内联实现。

### 使用方法

```python
# 方式一：pip install -e 后
from t_phase_analysis import pmcc, f_statistic

# 方式二：旧路径兼容
# import sys; sys.path.insert(0, '/Users/stardust/workspace/T_Phase_Analysis')
# from pmcc_wrapper import pmcc, f_statistic

# 需要恰好 3 道波形
result = pmcc(st, time_start=-50, time_one_slice=10.0,
              time_slide=2.0, time_length=200.0)
print(result['azimuth_mode1'], result['speed_mode1'])
```

### 限制

- PMCC / F-detector 均需要 **3 台站**。当前 `plot_close_stations.py` 仅有 IU.TATO 和 JP.YOJ 两个台站，尚未调用。
- 需在 `pygmt` conda 环境下运行（`obspy` 环境有 PyGMT sqlite3 符号冲突）。

## 2026-05-23 — PWF 接口简化 + 统一示例文档 + 模块清理

### 1) PWF 模块重构与统一

- 新建并保留统一模块：`t_phase_analysis/pwf.py`，整合原 `pwf_deepseek.py` 与流程化调用能力。
- 删除冗余文件：
  - `t_phase_analysis/pwf_deepseek.py`
  - `t_phase_analysis/pwd.py`
  - `t_phase_analysis/do_pwf_deepseek.py`
- `__init__.py` 已统一导出 PWF 相关函数（核心算法 + 流程接口）。

### 2) PWF 入口改为最小必需输入

- `pwf()` 现仅要求 **3 台站波形**：
  - 必需：`st`（`obspy.Stream`，恰好 3 道）
  - 可选：`station_coords_ll=[(lat, lon), ...]`
- 台站坐标来源规则：
  - 若传入 `station_coords_ll`，优先使用显式坐标；
  - 否则从 `st[i].stats.sac.stla/stlo` 自动读取。
- 去除旧版文件读取依赖参数（不再强制 `read_station_list_fn/read_all_h5_fn` 等外部 I/O 注入）。

### 3) 输出与默认行为优化

- `eqname` 参数已删除。
- 当设置 `mat_output_dir` 时，`.mat` 文件前缀改为输入波形起始时间（`st[0].stats.starttime`，并做文件名安全化）。
- 输出改为“按需保存”：
  - `figure_output_path=None` 时不保存图；
  - `mat_output_dir=None` 时不保存 `.mat`。
- `result["meta"]` 同步更新为 `file_prefix/fs_used/station_coords_ll` 等更贴近运行事实的字段。

### 4) PMCC / F-detector / SNR 改进（本轮累计）

- `pmcc.py`
  - 增加输入校验、采样率一致性检查；
  - 加入 `correlation_threshold` 与 `consistency_threshold` 联合筛选；
  - 网格搜索向量化，过滤无效窗后统计众数；
  - 新增 `plot_pmcc_diagnostics()` 诊断绘图。
- `fdetector.py`
  - 默认参数集中常量化；
  - 增加窗口有效样本与几何/采样校验；
  - 新增 `F_max/azimuth_peak/valid_window_mask/meta` 输出；
  - 新增 `plot_fdetector_diagnostics()` 诊断绘图。
- `snr.py`
  - 增加边界与输入校验；
  - `max` 方法改为绝对峰值比；
  - 增加分母近零保护与 `trace_index` 可选参数。

### 5) 文档与排版统一

- 在 `t_phase_analysis` 顶层主要模块中补充“调用示例”区块：
  - `traveltime_profile.py`, `waveform.py`, `denoise.py`, `tpef.py`, `snr.py`, `pmcc.py`, `fdetector.py`, `pwf.py`, `__init__.py`
- 统一 `__init__.py` 顶部“功能模块”对齐与模块说明，加入 `pwf` 描述。
