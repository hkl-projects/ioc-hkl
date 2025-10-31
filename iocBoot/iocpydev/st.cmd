#!/usr/bin/env bash
set -euo pipefail

# start in project root (where pixi.toml lives)
cd "$(dirname "$0")/../.."

# activate pixi enviroment
eval "$(pixi shell-hook -s bash)"

# locate pixi environment directory (default path)
PIXI_ENV="$PWD/.pixi/envs/default"

# detect python X.Y dir inside the pixi env
PY_VER_DIR="$(ls -d "$PIXI_ENV"/lib/python3.* | head -n1)"
PY_VER="$(basename "$PY_VER_DIR" | sed 's/python//')"

# make the IOC embed and load pixi's python runtime
export PYTHONHOME="$PIXI_ENV"
export LD_LIBRARY_PATH="$PIXI_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/python:$PIXI_ENV/lib/python${PY_VER}/site-packages"

# set GI typelib path
if [ -d "$PIXI_ENV/lib/girepository-1.0" ]; then
  export GI_TYPELIB_PATH="$PIXI_ENV/lib/girepository-1.0${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"
fi

which python || true
python - <<'PY'
import sys, importlib.util
print("launcher python:", sys.executable)
print("sys.version:", sys.version.split()[0])
print("has pandas?:", importlib.util.find_spec("pandas") is not None)
print("has gi?:", importlib.util.find_spec("gi") is not None)
PY

# enter ioc boot dir and launch app
cd iocBoot/iocpydev
exec ../../bin/linux-x86_64/hklApp st_base.cmd

