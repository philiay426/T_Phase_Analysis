"""
T 相分析工具模块。

主要功能：
- 基于 PyGMT 提取震源–台站（及反射点）大圆路径与高程剖面；
- 按海水/陆地分段估算 T 相走时。

作者: Yong Zhou (zhouyong@scsio.ac.cn)
版本: 2024-04-07 初版；2024-09-02 增加 snr；2026-05-22 注释与变量名整理，改进到时计算。

调用示例
--------
```python
from t_phase_analysis import get_track, tt_t

track = get_track(evlo=121.0, evla=23.5, stlo=122.1, stla=24.2)
res = tt_t(track)
print(res["tt_total_s"])
```
"""

import warnings

from obspy.geodetics.base import degrees2kilometers
import obspy.geodetics.base as geodetics
import pandas as pd
import pygmt
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.filter import envelope
from obspy.signal.util import smooth
from obspy.taup import TauPyModel

# 剖面图默认高程范围 (m)
ELEVATION_MIN_M = -11000
ELEVATION_MAX_M = 9000
# 海水–陆地分界参考高程 (m)，低于此视为海洋路径
DEFAULT_REFERENCE_ELEVATION_M = -1000
DEFAULT_TAUP_MODEL = 'iasp91'
# 默认传播速度（适用于短距离地壳传播场景）
DEFAULT_WATER_SOUND_SPEED_KMPS = 1.48
DEFAULT_CRUST_P_SPEED_KMPS = 6.0
DEFAULT_CRUST_S_SPEED_KMPS = 3.6
DEFAULT_SOURCE_DEPTH_KM = 10.0
# 深度语义约定（重要）：
# - 本模块中 source_depth_km / station_depth_m 都“相对于固体地球上边界”定义；
# - 固体地球上边界：陆地为地表，海洋为海底（需结合 track 的海底高程判断）；
# - 深度为正：位于固体地球内部；深度为负：位于边界之上
#   （陆地上方为空气；海洋上方可能在海水中或空气中）；
# - TauP 仅支持固体地球内传播，边界之上的值会在调用前裁剪到 0。
# 海洋路径中“微小陆地段”过滤阈值：用于小岛穿越场景
IGNORE_LAND_PATH_MAX_KM = 2.0
IGNORE_LAND_TO_OCEAN_RATIO_MAX = 1.0 / 100.0
_TAUP_MODEL_CACHE = {}


def auto_resolution(distance_deg):
    """
    根据大圆距离 (度) 自动选择 ETOPO 网格分辨率字符串。

    距离越长，分辨率越粗，以平衡下载量与精度。
    """
    # 以 200° 为参考尺度的分段阈值
    distance_scale_deg = 200.0

    if distance_deg >= distance_scale_deg * 1.0:
        grid_resolution = '01d'
    elif distance_deg >= distance_scale_deg * 30 / 60:
        grid_resolution = '30m'
    elif distance_deg >= distance_scale_deg * 15 / 60:
        grid_resolution = '15m'
    elif distance_deg >= distance_scale_deg * 3 / 60:
        grid_resolution = '03m'
    elif distance_deg >= distance_scale_deg * 1 / 60:
        grid_resolution = '01m'
    elif distance_deg >= distance_scale_deg * 30 / 3600:
        grid_resolution = '30s'
    elif distance_deg >= distance_scale_deg * 15 / 3600:
        grid_resolution = '15s'
    elif distance_deg >= distance_scale_deg * 3 / 3600:
        grid_resolution = '03s'
    else:
        grid_resolution = '01s'

    return grid_resolution


def _region_with_margin(longitudes, latitudes, margin_fraction=0.1):
    """
    为 PyGMT 地图构建 [west, east, south, north] 区域（带比例边距）。

    使用跨度加边距，避免负经度/纬度时 min*0.9 导致范围颠倒。
    """
    lon_min, lon_max = min(longitudes), max(longitudes)
    lat_min, lat_max = min(latitudes), max(latitudes)
    lon_padding = max((lon_max - lon_min) * margin_fraction, 0.5)
    lat_padding = max((lat_max - lat_min) * margin_fraction, 0.5)
    return [
        lon_min - lon_padding,
        lon_max + lon_padding,
        lat_min - lat_padding,
        lat_max + lat_padding,
    ]


def _elevation_m_to_depth_km(elevation_m):
    """
    将高程(m)转换为 TauP 使用的深度(km, 向下为正)。
    """
    return max(-float(elevation_m) / 1000.0, 0.0)


def _clip_to_solid_earth_depth_km(depth_km):
    """
    将“相对固体边界深度”裁剪为 TauP 可接受的深度（>=0）。
    """
    return max(float(depth_km), 0.0)


def _get_first_taup_arrival_time(
    distance_deg,
    source_depth_km,
    phase,
    model_name=DEFAULT_TAUP_MODEL,
    receiver_depth_km=0.0,
):
    """
    用 TauP 计算指定震相首波到时（秒）。

    若无可用到时，返回 ``np.nan`` 并给出警告。
    """
    taup_model = _TAUP_MODEL_CACHE.get(model_name)
    if taup_model is None:
        taup_model = TauPyModel(model=model_name)
        _TAUP_MODEL_CACHE[model_name] = taup_model
    source_depth_km = _clip_to_solid_earth_depth_km(source_depth_km)
    receiver_depth_km = _clip_to_solid_earth_depth_km(receiver_depth_km)
    if distance_deg <= 0:
        return 0.0

    try:
        arrivals = taup_model.get_travel_times(
            source_depth_in_km=source_depth_km,
            distance_in_degree=distance_deg,
            phase_list=[phase],
            receiver_depth_in_km=receiver_depth_km,
        )
    except TypeError:
        # 兼容不支持 receiver_depth_in_km 的旧版接口
        arrivals = taup_model.get_travel_times(
            source_depth_in_km=source_depth_km,
            distance_in_degree=distance_deg,
            phase_list=[phase],
        )
    if not arrivals:
        warnings.warn(
            f'No TauP arrival for phase {phase} at distance={distance_deg:.3f} deg, '
            f'source_depth={source_depth_km:.3f} km, receiver_depth={receiver_depth_km:.3f} km.',
            UserWarning,
        )
        return np.nan
    return float(arrivals[0].time)


def _build_propagation_segments(track, ref_el):
    """
    基于剖面构建连续传播段列表（海洋/陆地），单位为度。

    Returns
    -------
    list[dict]
        每段包含 ``medium`` (ocean/land), ``start_deg``, ``end_deg``, ``length_deg``。
    """
    segments = []

    def _append_segment(medium, start_deg, end_deg):
        length_deg = end_deg - start_deg
        if length_deg <= 0:
            return
        if segments and segments[-1]['medium'] == medium and abs(segments[-1]['end_deg'] - start_deg) < 1e-10:
            segments[-1]['end_deg'] = end_deg
            segments[-1]['length_deg'] += length_deg
        else:
            segments.append({
                'medium': medium,
                'start_deg': start_deg,
                'end_deg': end_deg,
                'length_deg': length_deg,
            })

    for idx in range(len(track) - 1):
        p0 = float(track.p.iloc[idx])
        p1 = float(track.p.iloc[idx + 1])
        z0 = float(track.el.iloc[idx]) - ref_el
        z1 = float(track.el.iloc[idx + 1]) - ref_el
        if p1 <= p0:
            continue

        if z0 <= 0 and z1 <= 0:
            _append_segment('ocean', p0, p1)
            continue
        if z0 > 0 and z1 > 0:
            _append_segment('land', p0, p1)
            continue

        crossing_fraction = -z0 / (z1 - z0)
        crossing_fraction = min(max(crossing_fraction, 0.0), 1.0)
        p_cross = p0 + (p1 - p0) * crossing_fraction
        if z0 <= 0 < z1:
            _append_segment('ocean', p0, p_cross)
            _append_segment('land', p_cross, p1)
        else:
            _append_segment('land', p0, p_cross)
            _append_segment('ocean', p_cross, p1)

    return segments


def _ignore_small_island_land_segments(segments):
    """
    忽略被海洋段夹住的微小陆地段（小岛）。

    判定条件（逐段）：
    1) 该陆地段长度 < ``IGNORE_LAND_PATH_MAX_KM``；
    2) 该陆地段长度 / (前海洋段 + 后海洋段长度) < ``IGNORE_LAND_TO_OCEAN_RATIO_MAX``。
    """
    filtered = []
    idx = 0
    while idx < len(segments):
        if (
            idx + 2 < len(segments)
            and segments[idx]['medium'] == 'ocean'
            and segments[idx + 1]['medium'] == 'land'
            and segments[idx + 2]['medium'] == 'ocean'
        ):
            left_ocean_deg = segments[idx]['length_deg']
            island_land_deg = segments[idx + 1]['length_deg']
            right_ocean_deg = segments[idx + 2]['length_deg']
            island_land_km = degrees2kilometers(island_land_deg)
            neighboring_ocean_km = degrees2kilometers(left_ocean_deg + right_ocean_deg)
            if (
                neighboring_ocean_km > 0
                and island_land_km < IGNORE_LAND_PATH_MAX_KM
                and island_land_km / neighboring_ocean_km < IGNORE_LAND_TO_OCEAN_RATIO_MAX
            ):
                filtered.append({
                    'medium': 'ocean',
                    'start_deg': segments[idx]['start_deg'],
                    'end_deg': segments[idx + 2]['end_deg'],
                    'length_deg': left_ocean_deg + island_land_deg + right_ocean_deg,
                })
                idx += 3
                continue

        filtered.append(segments[idx])
        idx += 1

    # 再做一次合并，避免前一步拼接后出现相邻同类型段
    merged = []
    for seg in filtered:
        if (
            merged
            and merged[-1]['medium'] == seg['medium']
            and abs(merged[-1]['end_deg'] - seg['start_deg']) < 1e-10
        ):
            merged[-1]['end_deg'] = seg['end_deg']
            merged[-1]['length_deg'] += seg['length_deg']
        else:
            merged.append(seg.copy())

    return merged


def get_track(evlo, evla, stlo, stla, dist_int, ref_el=DEFAULT_REFERENCE_ELEVATION_M,
              plot_map=True, plot_elevation=True):
    """
    提取震源到台站的大圆路径及沿路径高程。

    Parameters
    ----------
    evlo, evla : float
        震源经度、纬度 (度)。
    stlo, stla : float
        台站经度、纬度 (度)。
    dist_int : float
        沿大圆路径的采样间隔 (度)。
    ref_el : float
        海水–陆地分界参考高程 (m)，默认 -1000。
    plot_map, plot_elevation : bool
        是否绘制地图与高程剖面。

    Returns
    -------
    pandas.DataFrame
        列含路径距离 ``p`` (度)、高程 ``el`` (m) 等（PyGMT grdtrack 输出）。
    """
    epicentral_distance_deg = geodetics.locations2degrees(evla, evlo, stla, stlo)

    map_region = _region_with_margin([evlo, stlo], [evla, stla])
    relief_grid = pygmt.datasets.load_earth_relief(
        region=map_region,
        resolution=auto_resolution(epicentral_distance_deg),
    )

    great_circle_points = pygmt.project(
        center=[evlo, evla],
        endpoint=[stlo, stla],
        generate=str(dist_int),
    )
    path_profile = pygmt.grdtrack(
        relief_grid,
        points=great_circle_points,
        newcolname='el',
    )

    if plot_map:
        figure = pygmt.Figure()
        figure.basemap(region=map_region, projection='M10c', frame='af')
        figure.grdimage(grid=relief_grid, cmap='terra')
        figure.plot(x=[evlo, stlo], y=[evla, stla], pen='2p,black,solid')
        figure.plot(x=evlo, y=evla, style='c0.2c', pen='1p,red', fill='red')
        figure.text(x=evlo, y=evla, text='Source', offset='0.8c/0.2c', font='12p')
        figure.plot(x=stlo, y=stla, style='c0.2c', pen='1p,red', fill='red')
        figure.text(x=stlo, y=stla, text='Station', offset='0.8c/0.2c', font='12p')
        figure.show()

    if plot_elevation:
        bathymetry_region = [0, epicentral_distance_deg, ELEVATION_MIN_M, ELEVATION_MAX_M]
        figure = pygmt.Figure()
        figure.basemap(
            region=bathymetry_region,
            projection='X10c/5c',
            frame=['WSen', 'xa4f2+lDistance/°', 'ya4000f2000+lElevation/m'],
        )
        figure.plot(
            x=[0, epicentral_distance_deg], y=[0, 0],
            fill='lightblue', close=f'+y{ELEVATION_MIN_M}',
        )
        figure.plot(
            x=[0, epicentral_distance_deg], y=[ref_el, ref_el],
            pen='1p,black,--',
        )
        figure.plot(
            x=path_profile.p, y=path_profile.el,
            fill='gray', pen='1p,black,solid', close=f'+y{ELEVATION_MIN_M}',
        )
        figure.show()

    return path_profile


def tt_t(
    track,
    ref_el = DEFAULT_REFERENCE_ELEVATION_M,
    vel_w = DEFAULT_WATER_SOUND_SPEED_KMPS,
    vel = DEFAULT_CRUST_P_SPEED_KMPS,
    vel_s = DEFAULT_CRUST_S_SPEED_KMPS,
    source_depth_km = DEFAULT_SOURCE_DEPTH_KM,
    station_depth_m = 0.0,
    ignore_small_islands=False,
    taup_model_name=DEFAULT_TAUP_MODEL,
):
    """
    根据路径高程剖面估算 T 相总走时（海水段 + 陆地段）。

    Parameters
    ----------
    track : pandas.DataFrame
        ``get_track`` 或 ``get_track_reflection`` 返回的剖面；``p`` 为度，``el`` 为 m。
    ref_el : float
        低于此高程视为海洋路径 (m)。
    vel_w : float
        海水段等效速度 (km/s)，默认 1.48 km/s。
    vel : float
        陆地 P 波速度 (km/s)，默认 6.0 km/s；当 TauP P 到时不可用时回退。
    vel_s : float
        陆地 S 波速度 (km/s)，默认 3.6 km/s；当 TauP S 到时不可用时回退。
    source_depth_km : float
        震源深度 (km)，默认 10 km。
        该值相对“固体地球上边界”定义：陆地为地表、海洋为海底（结合高程判断）。
        正值表示在固体地球内部；负值表示在边界上方（空气/海水），TauP 计算时按 0 处理。
    station_depth_m : float
        台站深度 (m)，默认 0。仅作用于最后一个陆地段的接收端深度。
        与 source_depth_km 同语义：相对固体边界定义，负值表示在边界上方。
        内部会转换为 TauP 需要的接收端深度 (km, >=0)。
    ignore_small_islands : bool
        是否忽略海洋中的微小岛屿陆地段。默认 False（不忽略）。
    taup_model_name : str
        TauP 速度模型名，默认 ``iasp91``。
    说明
    ----
    先按 track 识别连续海洋段/陆地段；当 ``ignore_small_islands=True`` 时，
    小岛过滤按“海洋-陆地-海洋”三段结构逐段判断：
    1) 中间陆地段长度 < ``IGNORE_LAND_PATH_MAX_KM``；
    2) 中间陆地段 / (前海洋段 + 后海洋段) < ``IGNORE_LAND_TO_OCEAN_RATIO_MAX``。
    陆地段走时按段计算，深度规则（均相对固体边界）：
    - 第一个陆地段：source_depth = ``source_depth_km``；
    - 最后一个陆地段：receiver_depth 由 ``station_depth_m`` 给定（默认 0）；
    - 其他陆地段：source/receiver 都取 ``DEFAULT_REFERENCE_ELEVATION_M``。

    Returns
    -------
    dict or None
        键 ``tt_t``, ``tt_ocean``, ``tt_land``, ``tt_land_p``, ``tt_land_s``,
        ``dist_ocean``, ``dist_land``, ``num_ocean_segments``, ``num_land_segments``,
        ``is_all_ocean``, ``is_all_land``；
        采样点过少时返回 None。
    """
    if len(track) < 3:
        warnings.warn(
            'track is too small! Please use smaller dist_int.',
            UserWarning,
        )
        return None

    propagation_segments = _build_propagation_segments(track, ref_el=ref_el)
    if ignore_small_islands:
        propagation_segments = _ignore_small_island_land_segments(propagation_segments)
    ocean_segments = [seg for seg in propagation_segments if seg['medium'] == 'ocean']
    land_segments = [seg for seg in propagation_segments if seg['medium'] == 'land']
    num_ocean_segments = len(ocean_segments)
    num_land_segments = len(land_segments)
    ocean_path_deg = sum(seg['length_deg'] for seg in ocean_segments)
    land_path_deg = sum(seg['length_deg'] for seg in land_segments)
    reference_depth_km = _elevation_m_to_depth_km(DEFAULT_REFERENCE_ELEVATION_M)
    station_depth_km = _elevation_m_to_depth_km(station_depth_m)

    travel_time_ocean_s = degrees2kilometers(ocean_path_deg) / vel_w
    if land_path_deg <= 0:
        travel_time_land_p_s = 0.0
        travel_time_land_s_s = 0.0
    else:
        travel_time_land_p_s = 0.0
        travel_time_land_s_s = 0.0
        for land_index, land_segment in enumerate(land_segments):
            segment_distance_deg = land_segment['length_deg']
            segment_distance_km = degrees2kilometers(segment_distance_deg)
            is_first_land_segment = land_index == 0
            is_last_land_segment = land_index == len(land_segments) - 1
            segment_source_depth_km = source_depth_km if is_first_land_segment else reference_depth_km
            segment_receiver_depth_km = station_depth_km if is_last_land_segment else reference_depth_km

            segment_time_p_s = _get_first_taup_arrival_time(
                distance_deg=segment_distance_deg,
                source_depth_km=segment_source_depth_km,
                phase='P',
                model_name=taup_model_name,
                receiver_depth_km=segment_receiver_depth_km,
            )
            if np.isnan(segment_time_p_s):
                segment_time_p_s = segment_distance_km / vel
            travel_time_land_p_s += segment_time_p_s

            segment_time_s_s = _get_first_taup_arrival_time(
                distance_deg=segment_distance_deg,
                source_depth_km=segment_source_depth_km,
                phase='S',
                model_name=taup_model_name,
                receiver_depth_km=segment_receiver_depth_km,
            )
            if np.isnan(segment_time_s_s):
                segment_time_s_s = segment_distance_km / vel_s
            travel_time_land_s_s += segment_time_s_s

    # 兼容旧接口：tt_land 仍保留，代表陆地段 P 波到时总和
    travel_time_land_p_for_total_s = travel_time_land_p_s
    travel_time_total_s = travel_time_ocean_s + travel_time_land_p_for_total_s

    return {
        'tt_t': travel_time_total_s,
        'tt_ocean': travel_time_ocean_s,
        'dist_ocean': ocean_path_deg,
        'tt_land': travel_time_land_p_for_total_s,
        'tt_land_p': travel_time_land_p_s,
        'tt_land_s': travel_time_land_s_s,
        'dist_land': land_path_deg,
        'num_ocean_segments': num_ocean_segments,
        'num_land_segments': num_land_segments,
        'is_all_ocean': num_land_segments == 0 and num_ocean_segments > 0,
        'is_all_land': num_ocean_segments == 0 and num_land_segments > 0,
    }


def get_track_reflection(evlo, evla, stlo, stla, reflo, refla, dist_int,
                         ref_el=DEFAULT_REFERENCE_ELEVATION_M, plot_map=True, plot_elevation=True):
    """
    经反射点提取震源–反射点–台站两段路径及高程。

    Parameters
    ----------
    reflo, refla : float
        反射点经度、纬度 (度)。
    其余参数同 ``get_track``。

    Returns
    -------
    track_source_to_reflection, track_reflection_to_station : pandas.DataFrame
        两段路径剖面。
    """
    source_to_reflection_deg = geodetics.locations2degrees(evla, evlo, refla, reflo)
    reflection_to_station_deg = geodetics.locations2degrees(refla, reflo, stla, stlo)
    total_path_distance_deg = source_to_reflection_deg + reflection_to_station_deg

    map_region = _region_with_margin(
        [evlo, stlo, reflo], [evla, stla, refla],
    )
    relief_grid = pygmt.datasets.load_earth_relief(
        region=map_region,
        resolution=auto_resolution(total_path_distance_deg),
    )

    points_source_to_reflection = pygmt.project(
        center=[evlo, evla], endpoint=[reflo, refla], generate=str(dist_int),
    )
    track_source_to_reflection = pygmt.grdtrack(
        relief_grid, points=points_source_to_reflection, newcolname='el',
    )

    points_reflection_to_station = pygmt.project(
        center=[reflo, refla], endpoint=[stlo, stla], generate=str(dist_int),
    )
    track_reflection_to_station = pygmt.grdtrack(
        relief_grid, points=points_reflection_to_station, newcolname='el',
    )

    if plot_map:
        figure = pygmt.Figure()
        figure.basemap(region=map_region, projection='M10c', frame='af')
        figure.grdimage(grid=relief_grid, cmap='terra')
        figure.plot(x=[evlo, reflo], y=[evla, refla], pen='2p,black,solid')
        figure.plot(x=[reflo, stlo], y=[refla, stla], pen='2p,black,solid')
        figure.plot(x=evlo, y=evla, style='c0.2c', pen='1p,red', fill='red')
        figure.text(x=evlo, y=evla, text='Source', offset='0.8c/0.2c', font='12p')
        figure.plot(x=reflo, y=refla, style='c0.2c', pen='1p,red', fill='red')
        figure.text(x=reflo, y=refla, text='Reflection Point', offset='-0.2c/0.4c', font='12p')
        figure.plot(x=stlo, y=stla, style='c0.2c', pen='1p,red', fill='red')
        figure.text(x=stlo, y=stla, text='Station', offset='0.8c/0.2c', font='12p')
        figure.show()

    if plot_elevation:
        # 第二段距离坐标平移到第一段末端之后，再首尾拼接
        track_second_offset = track_reflection_to_station.copy()
        track_second_offset['p'] = (
            track_second_offset['p'] + track_source_to_reflection['p'].iloc[-1]
        )
        combined_profile = pd.concat(
            [track_source_to_reflection, track_second_offset.iloc[1:]],
            ignore_index=True,
        )

        bathymetry_region = [0, total_path_distance_deg, ELEVATION_MIN_M, ELEVATION_MAX_M]
        figure = pygmt.Figure()
        figure.basemap(
            region=bathymetry_region,
            projection='X10c/5c',
            frame=['WSen', 'xa4f2+lDistance/°', 'ya4000f2000+lElevation/m'],
        )
        figure.plot(
            x=[0, total_path_distance_deg], y=[0, 0],
            fill='lightblue', close=f'+y{ELEVATION_MIN_M}',
        )
        figure.plot(
            x=[0, total_path_distance_deg], y=[ref_el, ref_el],
            pen='1p,black,--',
        )
        figure.plot(
            x=combined_profile.p, y=combined_profile.el,
            fill='gray', pen='1p,black,solid', close=f'+y{ELEVATION_MIN_M}',
        )
        figure.plot(
            x=[source_to_reflection_deg, source_to_reflection_deg],
            y=[ELEVATION_MIN_M, ELEVATION_MAX_M],
            pen='1p,black,--',
        )
        figure.text(x=source_to_reflection_deg, y=ELEVATION_MAX_M * 0.8, text='reflection point')
        figure.show()

    return track_source_to_reflection, track_reflection_to_station


def tt_t_reflection(
    track1,
    track2,
    ref_el=DEFAULT_REFERENCE_ELEVATION_M,
    vel_w=DEFAULT_WATER_SOUND_SPEED_KMPS,
    vel=DEFAULT_CRUST_P_SPEED_KMPS,
    vel_s=DEFAULT_CRUST_S_SPEED_KMPS,
    source_depth_km=DEFAULT_SOURCE_DEPTH_KM,
    station_depth_m=0.0,
    ignore_small_islands=False,
    taup_model_name=DEFAULT_TAUP_MODEL,
):
    """
    反射路径走时：分别为两段剖面调用 ``tt_t`` 后相加。
    depth 参数语义与 ``tt_t`` 一致：相对固体边界定义，负值表示边界之上。

    Returns
    -------
    dict or None
        与 ``tt_t`` 相同键名，值为两段之和。
    """
    # station_depth 只传给整条反射路径中“最后一个实际存在的陆地段”
    segments_track2 = _build_propagation_segments(track2, ref_el=ref_el)
    if ignore_small_islands:
        segments_track2 = _ignore_small_island_land_segments(segments_track2)
    track2_has_land = any(seg['medium'] == 'land' for seg in segments_track2)
    station_depth_for_track1 = DEFAULT_REFERENCE_ELEVATION_M if track2_has_land else station_depth_m
    station_depth_for_track2 = station_depth_m if track2_has_land else DEFAULT_REFERENCE_ELEVATION_M

    travel_times_segment1 = tt_t(
        track1,
        ref_el=ref_el,
        vel_w=vel_w,
        vel=vel,
        vel_s=vel_s,
        source_depth_km=source_depth_km,
        station_depth_m=station_depth_for_track1,
        ignore_small_islands=ignore_small_islands,
        taup_model_name=taup_model_name,
    )
    travel_times_segment2 = tt_t(
        track2,
        ref_el=ref_el,
        vel_w=vel_w,
        vel=vel,
        vel_s=vel_s,
        source_depth_km=_elevation_m_to_depth_km(ref_el),
        station_depth_m=station_depth_for_track2,
        ignore_small_islands=ignore_small_islands,
        taup_model_name=taup_model_name,
    )
    if travel_times_segment1 is None or travel_times_segment2 is None:
        return None

    return {
        'tt_t': travel_times_segment1['tt_t'] + travel_times_segment2['tt_t'],
        'tt_ocean': travel_times_segment1['tt_ocean'] + travel_times_segment2['tt_ocean'],
        'dist_ocean': travel_times_segment1['dist_ocean'] + travel_times_segment2['dist_ocean'],
        'tt_land': travel_times_segment1['tt_land'] + travel_times_segment2['tt_land'],
        'tt_land_p': travel_times_segment1['tt_land_p'] + travel_times_segment2['tt_land_p'],
        'tt_land_s': travel_times_segment1['tt_land_s'] + travel_times_segment2['tt_land_s'],
        'dist_land': travel_times_segment1['dist_land'] + travel_times_segment2['dist_land'],
        'num_ocean_segments': travel_times_segment1['num_ocean_segments'] + travel_times_segment2['num_ocean_segments'],
        'num_land_segments': travel_times_segment1['num_land_segments'] + travel_times_segment2['num_land_segments'],
        'is_all_ocean': travel_times_segment1['is_all_ocean'] and travel_times_segment2['is_all_ocean'],
        'is_all_land': travel_times_segment1['is_all_land'] and travel_times_segment2['is_all_land'],
    }


