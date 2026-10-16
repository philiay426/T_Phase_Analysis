"""
基于 ObsPy 的去仪器响应工具（输出 DEF）。

适用场景：
- 输入原始波形（MiniSEED）+ 台站响应（StationXML）
- 使用 ObsPy 官方 remove_response 进行去响应
- 输出选择 DEF（仪器定义量；对水听器通常对应声压）
"""

from pathlib import Path

from obspy import read, read_inventory


DEFAULT_PREFILT = (0.001, 0.005, 45.0, 50.0)
DEFAULT_WATER_LEVEL = 60.0


def remove_response_def(
    st,
    inventory,
    pre_filt=DEFAULT_PREFILT,
    water_level=DEFAULT_WATER_LEVEL,
    zero_mean=True,
    taper=True,
    taper_fraction=0.05,
):
    """
    使用 ObsPy remove_response 去仪器响应，输出 DEF。

    Parameters
    ----------
    st : obspy.Stream
        原始波形。
    inventory : obspy.Inventory
        台站响应信息（StationXML 读取结果）。
    pre_filt : tuple(float, float, float, float)
        预滤波频带 (f1, f2, f3, f4)，用于稳定反褶积。
    water_level : float
        水位参数。
    zero_mean : bool
        去均值。
    taper : bool
        是否加 taper。
    taper_fraction : float
        taper 比例。

    Returns
    -------
    obspy.Stream
        去响应后的新 Stream（不修改原始 st）。
    """
    st_out = st.copy()
    st_out.remove_response(
        inventory=inventory,
        output="DEF",
        pre_filt=pre_filt,
        water_level=water_level,
        zero_mean=zero_mean,
        taper=taper,
        taper_fraction=taper_fraction,
    )
    return st_out


def load_and_remove_response_def(
    mseed_path,
    stationxml_path,
    pre_filt=DEFAULT_PREFILT,
    water_level=DEFAULT_WATER_LEVEL,
    zero_mean=True,
    taper=True,
    taper_fraction=0.05,
):
    """
    一站式读取 mseed + xml，并输出 DEF 去响应结果。

    Parameters
    ----------
    mseed_path : str
        MiniSEED 文件路径。
    stationxml_path : str
        StationXML 文件路径。

    Returns
    -------
    tuple(obspy.Stream, obspy.Inventory)
        (去响应后的 stream, inventory)
    """
    st = read(mseed_path)
    inv = read_inventory(stationxml_path)
    st_def = remove_response_def(
        st=st,
        inventory=inv,
        pre_filt=pre_filt,
        water_level=water_level,
        zero_mean=zero_mean,
        taper=taper,
        taper_fraction=taper_fraction,
    )
    return st_def, inv


def batch_remove_response_def(
    data_dir,
    output_dir=None,
    stationxml_path=None,
    mseed_glob="*.mseed",
    pre_filt=DEFAULT_PREFILT,
    water_level=DEFAULT_WATER_LEVEL,
    zero_mean=True,
    taper=True,
    taper_fraction=0.05,
):
    """
    批量读取目录中的 mseed 并去响应到 DEF。

    匹配规则：
    1) 若给定 stationxml_path，则所有 mseed 共用该响应文件；
    2) 否则尝试在 data_dir 下查找唯一的 *.xml 并复用；
    3) 若 data_dir 下有多个 xml 且未显式指定 stationxml_path，则报错。

    Parameters
    ----------
    data_dir : str or Path
        包含 mseed/xml 的目录。
    output_dir : str or Path or None
        去响应后 mseed 输出目录。None 时默认 data_dir/def_out。
    stationxml_path : str or Path or None
        显式 StationXML 路径。
    mseed_glob : str
        mseed 匹配模式，默认 ``*.mseed``。

    Returns
    -------
    list[dict]
        每个 mseed 的处理结果摘要：
        file_in, file_out, stationxml, n_traces
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"data_dir does not exist: {data_dir}")

    mseed_files = sorted(data_dir.glob(mseed_glob))
    if not mseed_files:
        raise ValueError(f"No mseed files matched '{mseed_glob}' in {data_dir}")

    if output_dir is None:
        output_dir = data_dir / "def_out"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stationxml_path is not None:
        stationxml_path = Path(stationxml_path)
        if not stationxml_path.exists():
            raise ValueError(f"stationxml_path does not exist: {stationxml_path}")
    else:
        xml_files = sorted(data_dir.glob("*.xml"))
        if len(xml_files) == 1:
            stationxml_path = xml_files[0]
        elif len(xml_files) == 0:
            raise ValueError("No StationXML found in data_dir; please provide stationxml_path.")
        else:
            raise ValueError(
                "Multiple StationXML files found in data_dir; please specify stationxml_path."
            )

    inv = read_inventory(str(stationxml_path))
    results = []

    failed_files = []
    for mseed_file in mseed_files:
        try:
            st = read(str(mseed_file))
            st_def = remove_response_def(
                st=st,
                inventory=inv,
                pre_filt=pre_filt,
                water_level=water_level,
                zero_mean=zero_mean,
                taper=taper,
                taper_fraction=taper_fraction,
            )
            out_file = output_dir / f"{mseed_file.stem}_DEF.mseed"
            st_def.write(str(out_file), format="MSEED")

            results.append(
                {
                    "file_in": str(mseed_file),
                    "file_out": str(out_file),
                    "stationxml": str(stationxml_path),
                    "n_traces": len(st_def),
                }
            )
        except Exception as exc:
            failed_files.append({"file": str(mseed_file), "error": str(exc)})
            import logging
            logging.warning(f"Skipping {mseed_file.name}: {exc}")

    if failed_files:
        import logging
        logging.warning(
            f"batch_remove_response_def: {len(failed_files)}/{len(mseed_files)} files failed"
        )

    return results, failed_files

