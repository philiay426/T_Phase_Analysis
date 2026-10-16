#!/usr/bin/env python3
"""通用 FK 分析 (PMCC) — 三步法：准备SAC → 跑PMCC → 汇总结果"""
import sys, os, glob, argparse

sys.path.insert(0, '/home/sy/codex/t_phase_analysis')

from obspy import read
from t_phase_analysis.internals.PMCC import azimuth_speed_array

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proc-dir', required=True, help='sacprocess 目录')
    p.add_argument('--out-dir', required=True, help='输出目录')
    p.add_argument('--stations', required=True, help='逗号分隔的三站名')
    p.add_argument('--t0-times', required=True, help='逗号分隔的三站T-P到时 (s)')
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--slice', type=float, default=20)
    p.add_argument('--slide', type=float, default=10)
    p.add_argument('--total', type=float, default=100)
    args = p.parse_args()
    
    stas = [s.strip() for s in args.stations.split(',')]
    tps  = [float(t) for t in args.t0_times.split(',')]
    assert len(stas) == 3 and len(tps) == 3
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    sac_files = []
    for sta, tp in zip(stas, tps):
        pattern = os.path.join(args.proc_dir, f'*{sta}*BHZ*proc.sac')
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"找不到 {sta}"); return
        tr = read(matches[0])[0]
        tr.stats.sac.t0 = tp
        # 统一重采样到40Hz
        tr.resample(40.0)
        dst = os.path.join(args.out_dir, f'{sta}.BHZ.sac')
        tr.write(dst, format='SAC')
        sac_files.append(dst)
        print(f'  {sta}: t0={tp}s')
    
    list_file = os.path.join(args.out_dir, 'file_list.txt')
    with open(list_file, 'w') as f:
        for fn in sac_files: f.write(fn + '\n')
    
    cwd = os.getcwd()
    os.chdir(args.out_dir)
    sys.argv = ['pmcc', list_file,
                str(args.start), str(args.slice),
                str(args.slide), str(args.total)]
    azimuth_speed_array.main()
    os.chdir(cwd)
    
    # 移动结果图
    for pat in ['n_Consistency*.png', 'n_ave_CC*.png', 'n_Azimuth*.png', 'n_Speed*.png']:
        for fn in glob.glob(os.path.join(args.out_dir, pat)):
            pass  # already in out_dir
    print(f'✅ {args.out_dir}')

if __name__ == '__main__':
    main()
