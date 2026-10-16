import numpy as np
# from obspy.core import Stream
# from obspy.signal.util import util_geo_km
from obspy.geodetics.base import gps2dist_azimuth
from matplotlib import pyplot as plt
import matplotlib.cm as cm
# from obspy.signal.array_analysis import array_processing, get_timeshift#, get_geometry


def compute_F_statistic(st, time_start, time_one_slice, time_slide, time_length):
    """
    计算F统计量

    参数:
        st: ObsPy Stream对象
        time_start: 开始时间
        time_one_slice: 窗口大小
        time_slide: 窗口滑动步长
        time_length: 总时长

    返回:
        time_windows: 时间窗口
        azimuths: 方位角
        F_statistic: F统计量
    """
    if len(st) != 3:
        print('Only 3 seismograms is accepted!')
        return

    nwindows = int(time_length / time_slide)

    # the azimuth and distance between stations 1, 2 and 3.
    [dist_meter_12, az_12, baz_12] = \
        gps2dist_azimuth(st[0].stats.sac.stla, st[0].stats.sac.stlo,
                         st[1].stats.sac.stla, st[1].stats.sac.stlo)

    [dist_meter_13, az_13, baz_13] = \
        gps2dist_azimuth(st[0].stats.sac.stla, st[0].stats.sac.stlo,
                         st[2].stats.sac.stla, st[2].stats.sac.stlo)
    # [dist_meter_23, az_23, baz_23] = \
    #         gps2dist_azimuth(st[1].stats.sac.stla,st[1].stats.sac.stlo,st[2].stats.sac.stla,st[2].stats.sac.stlo)
    deg2rad = np.pi / 180.0

    # Coherent Spectrum (F statistics) with time and azimuth. add by yzhou.
    # azimuths = np.arange (0,360.0,global_var.dazimuth_search)
    azimuths = np.arange(0, 360.0, 1)  # back azimuth
    nazimuths = len(azimuths)
    time_windows = np.arange(time_start, time_start + time_length, time_slide)
    F_statistic = np.zeros((nazimuths, nwindows), dtype=float)
    speed = 1480.0  # m/s

    for iazimuth in range(nazimuths):
        dt1_2_predict = -dist_meter_12 * \
            np.cos((az_12 - azimuths[iazimuth]) * deg2rad) / speed  # at2-at1
        dt1_3_predict = -dist_meter_13 * \
            np.cos((az_13 - azimuths[iazimuth]) * deg2rad) / speed  # at3-at1
        # dt2_3_predict=-dist_meter_23*cos((az_23-azimuths[iazimuth])*deg2rad)/speed

        for iwindow in range(nwindows):
            # The sliding time of this time window, relative to the arrival time of T-wave.
            time_slide_this_window = iwindow * time_slide
            # The starting time of CC.
            t_pick = st[0].stats.starttime + \
                time_start + time_slide_this_window

            tr_0 = st[0].slice(t_pick, t_pick + time_one_slice)
            tr_1 = st[1].slice(t_pick + dt1_2_predict,
                               t_pick + time_one_slice + dt1_2_predict)
            tr_2 = st[2].slice(t_pick + dt1_3_predict,
                               t_pick + time_one_slice + dt1_3_predict)

            # 处理可能的长度不一致问题
            try:
                tr_data = np.vstack(([tr.data for tr in [tr_0, tr_1, tr_2]]))
            except ValueError:
                tr_data_min_length = min(
                    tr_0.data.size, tr_1.data.size, tr_2.data.size)
                tr_data = np.vstack([tr.data[:tr_data_min_length]
                                    for tr in [tr_0, tr_1, tr_2]])

            # 计算F统计量
            F_denominator = np.sum(3.0 * np.var(tr_data, axis=0))
            F_numerator = np.sum(np.sum(tr_data, axis=0)**2)
            F_statistic[iazimuth, iwindow] = (
                2.0/3.0)*(F_numerator / F_denominator)

    return (time_windows, azimuths, F_statistic)


def plot_F_statistic(st, time_start, time_one_slice=5.0, time_slide=5.0, time_length=5000.0, vmin=0.5, cmap_name='jet'):
    """
    绘制F统计量图

    参数:
        st: ObsPy Stream对象
        time_start: 起始时间
        time_one_slice: 时间窗口大小
        time_slide: 滑动窗口步长
        time_length: 总时长
        vmin: 颜色映射最小值
        cmap_name: 颜色映射名称

    返回:
        fig: matplotlib Figure对象
    """
    time_windows, azimuths, F_statistic = compute_F_statistic(
        st, time_start, time_one_slice, time_slide, time_length)

    fig, axs = plt.subplots(1)

    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=np.max(
        F_statistic))  # 创建规格化对象，使颜色条的范围与我们的数据相对应
    color = cmap(norm(F_statistic))  # 使用 colormap 来获取每一层的颜色

    t0 = st[0].stats.sac['t0']
    contour_plot = axs.contourf(
        time_windows-t0, azimuths, F_statistic, vmin=vmin, cmap=cmap)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    plt.colorbar(sm, ax=axs, label='F-statistic', pad=0.01)

    axs.set(xlabel='Time after origin $(s)$', ylabel='BackAzimuth $(^\circ)$')
    # axs.set_xlim([43000, 47000])
    # axs.invert_yaxis()

    return fig, time_windows, azimuths, F_statistic


def get_st_center(st):
    """
    计算台站位置的中心坐标

    参数:
        st: ObsPy Stream对象

    返回:
        stlo_center: 平均经度
        stla_center: 平均纬度
    """
    # 使用列表推导式一次性提取所有经纬度
    stlo = np.array([tr.stats.sac['stlo'] for tr in st])
    stla = np.array([tr.stats.sac['stla'] for tr in st])

    # 直接计算平均值
    return np.mean(stlo), np.mean(stla)


def get_direct_T(evlo, evla, st, T_v=1.48):
    """
    计算理论T波到时和反方位角

    参数:
        evlo: 事件经度
        evla: 事件纬度
        st: ObsPy Stream对象
        T_v: T波速度(km/s)，默认1.48 km/s

    返回:
        tt: 理论到时(秒)
        baz: 反方位角(度)
    """
    # 获取台站中心位置
    stlo, stla = get_st_center(st)

    # 计算距离和反方位角
    dist_meter, _, baz = gps2dist_azimuth(evla, evlo, stla, stlo)

    # 计算理论到时(将米转换为公里)
    tt = dist_meter / 1000.0 / T_v

    return tt, baz


def plot_all_times(st, sac_times, stream_name, tt=None, baz=None,
                   time_one_slice=20.0, time_slide=5.0, time_length=5000.0, vmin=0.5,
                   keys_to_plot=None, cmap_name='jet'):
    """
    循环遍历时间值并为每个Stream绘图，在T波理论到时处添加标识

    参数:
        st: ObsPy Stream对象
        sac_times: 包含时间标记的命名元组
        stream_name: 数据流名称，用于标题
        tt: T波理论到时(秒)，可选
        baz: 理论反方位角(度)，可选
        time_one_slice: 时间窗口大小
        time_slide: 滑动窗口步长
        time_length: 总时长
        vmin: 颜色映射最小值
        keys_to_plot: 要绘制的键列表，如果为None则绘制所有键

    返回:
        figures: 包含所有生成图形的字典
    """
    # 创建一个图形和轴的字典，用于后续引用
    figures = {}

    # 确定要绘制的键
    if keys_to_plot is None:
        keys_to_plot = sac_times._fields

    t0 = st[0].stats.sac['t0']
    # 循环遍历要绘制的键
    for key_name in keys_to_plot:
        # 获取相应的时间值
        t = getattr(sac_times, key_name)

        # 跳过None值
        if t is None:
            continue

        # 对t0特殊处理，加1000秒
        time_start = t + 2000 if key_name == 't0' else t

        # 绘制图形
        fig, time_windows, azimuths, F_statistic = plot_F_statistic(st, time_start, time_one_slice,
                               time_slide, time_length, vmin, cmap_name=cmap_name)

        # 设置标题，包括详细信息
        title = f'{stream_name} - {key_name} (time_start = {time_start - t0:.2f}s)'

        # 对t0特殊处理，添加T波到时标识
        if key_name == 't0' and tt is not None and baz is not None:
            # 计算相对于time_start的理论T波到时
            t_wave_time = t + tt  # t0加了1000后，需要减去1000再加tt

            # 在理论T波初至处添加标识
            plt.axvline(x=t_wave_time - t0, color='white',
                        linestyle='--', alpha=0.7, label='Theoretical direct T-wave')
            plt.plot(t_wave_time - t0, baz, marker='s', markeredgewidth=1.0,
                     color='white', fillstyle='none',
                     markersize=7, label='T-wave arrival')

            # 添加到标题中
            title += f' | T-wave: {t_wave_time - t0:.2f}s at BAZ: {baz:.1f}°'

            # 添加图例
            plt.legend(loc='best',
                       facecolor=(0.2, 0.2, 0.2, 0.7),  # 深灰色半透明背景
                       labelcolor='white',               # 白色文字
                       framealpha=0.4                    # 整体透明度
                       )

        plt.title(title)
        plt.tight_layout()

        # 保存图形引用
        figures[key_name] = fig

        # 显示图形
        plt.show()

    return figures, time_windows, azimuths, F_statistic

# def get_geometry(stream, coordsys='lonlat', return_center=False,
#                  verbose=False):
#     """
#     Method to calculate the array geometry and the center coordinates in km

#     :param stream: Stream object, the trace.stats dict like class must
#         contain an :class:`~obspy.core.util.attribdict.AttribDict` with
#         'latitude', 'longitude' (in degrees) and 'elevation' (in km), or 'x',
#         'y', 'elevation' (in km) items/attributes. See param ``coordsys``
#     :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
#         attributes to use for coordinates
#     :param return_center: Returns the center coordinates as extra tuple
#     :return: Returns the geometry of the stations as 2d :class:`numpy.ndarray`
#             The first dimension are the station indexes with the same order
#             as the traces in the stream object. The second index are the
#             values of [lat, lon, elev] in km
#             last index contains center [lat, lon, elev] in degrees and km if
#             return_center is true
#     """
#     nstat = len(stream)
#     center_lat = 0.
#     center_lon = 0.
#     # center_h = 0.
#     geometry = np.empty((nstat, 2))

#     if isinstance(stream, Stream):
#         for i, tr in enumerate(stream):
#             if coordsys == 'lonlat':
#                 geometry[i, 0] = tr.stats.sac.stlo
#                 geometry[i, 1] = tr.stats.sac.stla
#                 # geometry[i, 2] = tr.stats.coordinates.elevation
#             elif coordsys == 'xy':
#                 geometry[i, 0] = tr.stats.coordinates.x
#                 geometry[i, 1] = tr.stats.coordinates.y
#                 # geometry[i, 2] = tr.stats.coordinates.elevation
#     elif isinstance(stream, np.ndarray):
#         geometry = stream.copy()
#     else:
#         raise TypeError('only Stream or numpy.ndarray allowed')

#     if verbose:
#         print("coordsys = " + coordsys)

#     if coordsys == 'lonlat':
#         center_lon = geometry[:, 0].mean()
#         center_lat = geometry[:, 1].mean()
#         # center_h = geometry[:, 2].mean()
#         for i in np.arange(nstat):
#             x, y = util_geo_km(center_lon, center_lat, geometry[i, 0],
#                                geometry[i, 1])
#             geometry[i, 0] = x
#             geometry[i, 1] = y
#             # geometry[i, 2] -= center_h
#     elif coordsys == 'xy':
#         geometry[:, 0] -= geometry[:, 0].mean()
#         geometry[:, 1] -= geometry[:, 1].mean()
#         # geometry[:, 2] -= geometry[:, 2].mean()
#     else:
#         raise ValueError("Coordsys must be one of 'lonlat', 'xy'")

#     # if return_center:
#     #     return np.c_[geometry.T,
#     #                  np.array((center_lon, center_lat, center_h))].T
#     # else:
#     #     return geometry
#     return geometry

# def simplified_get_geometry(st):
#     """
#     简化版的台站几何计算函数，适用于只有少量台站的情况

#     参数:
#         st: ObsPy Stream对象

#     返回:
#         geometry: 台站几何数组，相对于中心台站的坐标(km)
#     """
#     # 获取每个台站的经纬度
#     stlo = np.array([tr.stats.sac['stlo'] for tr in st])
#     stla = np.array([tr.stats.sac['stla'] for tr in st])

#     # 计算中心点
#     center_lon = np.mean(stlo)
#     center_lat = np.mean(stla)

#     # 计算每个台站相对于中心点的位置(km)
#     nstat = len(st)
#     geometry = np.empty((nstat, 2))

#     for i in range(nstat):
#         # 使用obspy的util_geo_km计算相对位置(km)
#         x, y = util_geo_km(center_lon, center_lat, stlo[i], stla[i])
#         geometry[i, 0] = x
#         geometry[i, 1] = y

#     return geometry

# def compute_delay(stream, backazimuth, velocity=1.5):
#     geometry = get_geometry(stream)

#     # 将反方位角转换为弧度
#     backazimuth_rad = np.radians(backazimuth)

#     # 计算反方位角方向的单位向量并交换顺序
#     direction_vector = np.array(
#         [np.sin(backazimuth_rad), np.cos(backazimuth_rad)]).T

#     # 分别将geometry中的元素与direction_vector对应元素相除，再统一除以速度
#     delays = np.dot(geometry, direction_vector) / velocity

#     return delays

# def simplified_compute_delay(st, backazimuth, velocity=1.5):
#     """
#     简化版的延迟时间计算函数，适用于只有少量台站的情况

#     参数:
#         st: ObsPy Stream对象
#         backazimuth: 反方位角(度)
#         velocity: 波速(km/s)，默认1.5 km/s

#     返回:
#         delays: 每个台站相对到时延迟(s)
#     """
#     # 获取台站几何
#     geometry = simplified_get_geometry(st)

#     # 将反方位角转换为弧度
#     backazimuth_rad = np.radians(backazimuth)

#     # 计算反方位角方向的单位向量
#     # 注意：反方位角是波的到来方向，与波前进方向相反
#     # sin和cos的顺序与原函数保持一致，确保结果的一致性
#     direction_vector = np.array([np.sin(backazimuth_rad), np.cos(backazimuth_rad)])

#     # 计算每个台站在波前进方向上的投影距离，再除以速度得到时间延迟
#     # 注意：这里直接用点积计算投影
#     delays = np.dot(geometry, direction_vector) / velocity

#     return delays
