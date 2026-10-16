#!/usr/bin/env bash
# ============================================================
# Codespaces 云端容器初始化脚本
# 【注】本脚本只在云端 Codespaces 容器内由 devcontainer.json 的
#       postCreateCommand 调用，不影响本地 conda 环境。
# 【注】本地真实分析请继续用 /home/sy/miniforge3/envs/obspy/bin/python
# ============================================================
set -e

echo "[1/3] 安装系统级依赖 GMT (pygmt 的底层库)..."
sudo apt-get update -qq
sudo apt-get install -y -qq gmt gmt-dcw gmt-gshhg \
  || sudo apt-get install -y -qq gmt gshhg-gmt-nc4 \
  || echo "[warn] GMT 安装失败, pygmt 功能不可用, 其余功能不受影响"

echo "[2/3] 安装 Python 依赖 (editable 方式安装本包)..."
python -m pip install --upgrade pip
python -m pip install -e ".[pygmt]"

echo "[3/3] 自检..."
python - <<'PY'
import sys
mods = ["numpy", "scipy", "obspy", "matplotlib", "pandas"]
for m in mods:
    try:
        __import__(m)
        print(f"  [OK]  {m}")
    except ImportError as e:
        print(f"  [MISS] {m}: {e}")
try:
    import pygmt
    print("  [OK]  pygmt")
except Exception as e:
    print(f"  [warn] pygmt 不可用: {e}")
print(f"Python: {sys.version}")
PY

echo "[DONE] 云端环境就绪"
