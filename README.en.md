# T_Phase_Analysis

T_Phase_Analysis is an open-source Python package for seismic wave phase analysis and signal processing. This project provides a suite of functions for analyzing three-component seismic recordings, including P-wave arrival detection, plane wave fitting, denoising, and signal-to-noise ratio calculation.

## Features

- **PMCC (Progressive Multi-Channel Correlation)**: P-wave arrival detection based on cross-correlation
- **F-detector**: F-statistic detector for seismic phase identification
- **PWF (Plane Wave Fitting)**: Plane wave fitting inversion to retrieve back-azimuth and slowness
- **SNR (Signal-to-Noise Ratio)**: Signal-to-noise ratio calculation
- **TPEF (Time-Window Phase Energy Feature)**: Time-window phase energy feature analysis
- **Denoise**: Parameter-free denoising method based on wavelet transform
- **Waveform Visualization**: Supports plotting of waveform panels and spectrograms

## Installation

```bash
pip install -e .
```

Alternatively, add the project path to the Python search path:

```python
import sys
sys.path.insert(0, '/path/to/T_Phase_Analysis')
```

## Dependencies

Primary dependencies include:
- NumPy
- SciPy
- ObsPy
- matplotlib

## Usage Examples

### PMCC Arrival Detection

```python
from t_phase_analysis import pmcc

result = pmcc(st, time_start, time_one_slice, time_slide, time_length)
```

### F-Statistic Detector

```python
from t_phase_analysis import f_statistic

result = f_statistic(st, time_start, time_one_slice, time_slide, time_length)
```

### Plane Wave Fitting

```python
from t_phase_analysis import pwf

result = pwf(st, station_coords_ll=coords)
```

### Signal-to-Noise Ratio Calculation

```python
from t_phase_analysis import snr

snr_value = snr(st, t_signal, t_noise)
```

## Module Structure

- `pmcc.py`: PMCC arrival detection module
- `fdetector.py`: F-detector module
- `pwf.py`: Plane wave fitting module
- `snr.py`: SNR calculation module
- `tpef.py`: Time-window phase energy feature module
- `denoise.py`: Wavelet denoising module
- `waveform.py`: Waveform visualization module
- `traveltime_profile.py`: Traveltime profile calculation module

## Development

- See `CHANGELOG.md` for revision history
- Refer to the `docs/` directory for detailed documentation

## License

This project is intended for research use only.