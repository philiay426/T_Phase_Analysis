

# T_Phase_Analysis

T_Phase_Analysis 是一个用于地震波相分析与信号处理的开源 Python 包。该项目提供了一系列用于分析地震三分量记录的函数，包括 P 波初至检测、平面波拟合、去噪、信噪比计算等功能。

## 功能特性

- **PMCC (Progressive Multi-Channel Correlation)**：基于互相关的 P 波初至检测方法
- **F-detector**：F 统计检验检测器，用于地震相位识别
- **PWF (Plane Wave Fitting)**：平面波拟合反演，获取back-azimuth和慢度
- **SNR (Signal-to-Noise Ratio)**：信噪比计算
- **TPEF (Time-Window Phase Energy Feature)**：时窗相位能量特征分析
- **去噪 (Denoise)**：基于小波变换的自由参数去噪方法
- **波形可视化**：支持波形面板和频谱图绘制

## 安装

```bash
pip install -e .
```

或者将项目路径添加到 Python 搜索路径：

```python
import sys
sys.path.insert(0, '/path/to/T_Phase_Analysis')
```

## 依赖

主要依赖包括：
- NumPy
- SciPy
- ObsPy
- matplotlib

## 使用示例

### PMCC 初至检测

```python
from t_phase_analysis import pmcc

result = pmcc(st, time_start, time_one_slice, time_slide, time_length)
```

### F 统计检测器

```python
from t_phase_analysis import f_statistic

result = f_statistic(st, time_start, time_one_slice, time_slide, time_length)
```

### 平面波拟合

```python
from t_phase_analysis import pwf

result = pwf(st, station_coords_ll=coords)
```

### 信噪比计算

```python
from t_phase_analysis import snr

snr_value = snr(st, t_signal, t_noise)
```

## 模块结构

- `pmcc.py`：PMCC 初至检测模块
- `fdetector.py`：F 检测器模块
- `pwf.py`：平面波拟合模块
- `snr.py`：信噪比计算模块
- `tpef.py`：时窗相位能量特征模块
- `denoise.py`：小波去噪模块
- `waveform.py`：波形可视化模块
- `traveltime_profile.py`：走时剖面计算模块

## 开发相关

- 修改日志见 `CHANGELOG.md`
- 详细文档见 `docs/` 目录

## 许可证

本项目仅供研究使用。