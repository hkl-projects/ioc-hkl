#!/bin/bash

echo "Choose an option:"
echo "1) pixi python environment (recommended)"
echo "2) system python (for building hkl from source)"
echo -n "Enter choice [1/2]: "
read choice

case $choice in
    1)
        echo "Setting up pixi environment"
        rm pixi.toml
        rm pixi.lock
        pixi init
        pixi add python=3.12 numpy=2.3.4 pandas scipy matplotlib tqdm pygobject hkl
        rm configure/CONFIG_PYENV
        touch configure/CONFIG_PYENV
        echo "PYTHON_CONFIG=/epics/iocs/ioc-hkl/.pixi/envs/default/bin/python3.12-config" > configure/CONFIG_PYENV
        ;;
    2)
        echo "Setting up system python"
        echo "PYTHON_CONFIG=/usr/bin/python3-config" > configure/CONFIG_PYENV 
        ;;
esac

echo "Done."

make clean
make -j4
