#!/bin/bash

top=$(pwd)

echo "Be sure to have the EPICS base installed, and point to its path in configure/RELEASE"

# check if cif2hkl installed in /usr/bin
if [[ ! -x /usr/bin/cif2hkl ]]; then
    echo ""
    echo " cif2hkl is not installed at /usr/bin/cif2hkl"
    echo ""
    echo "You can install cif2hkl from:"
    echo " https://gitlab.com/soleil-data-treatment/soleil-software-projects/cif2hkl "
    echo ""
    echo "On Debian/Ubuntu, you may also install via apt:"
    echo "    sudo apt install cif2hkl"
    echo ""
    exit 1
else
echo "cif2hkl is installed, continuing..."

echo "Choose an option:"
echo "1) pixi python environment (recommended)"
echo "2) system python (for building hkl from source)"
echo -n "Enter choice [1/2]: "
read choice

case $choice in
    1) 
        echo "Checking for pixi installation on system"

        if ! command -v pixi >/dev/null 2>&1; then
            echo "Error: pixi is not installed."
            echo "Check documentation/installation.md"
            exit 1
        fi

echo "pixi is installed, continuing..."
    
        echo "Setting up pixi environment"
        rm pixi.toml
        rm pixi.lock
        pixi init
        pixi add python=3.12 numpy=2.3.4 pandas scipy matplotlib tqdm pygobject hkl
        rm configure/RELEASE.local
        touch configure/RELEASE.local
        #rm configure/CONFIG_PYENV
        #touch configure/CONFIG_PYENV
        #echo "PYTHON_CONFIG=/epics/iocs/ioc-hkl/.pixi/envs/default/bin/python3.12-config" > configure/CONFIG_PYENV
        echo "PYTHON_CONFIG=${top}/.pixi/envs/default/bin/python3.12-config" > configure/RELEASE.local
        ;;
    2)
        echo "Setting up system python"
        echo "PYTHON_CONFIG=/usr/bin/python3-config" > configure/CONFIG_PYENV 
        echo "Install hkl https://repo.or.cz/hkl.git"
        echo "configure with flags --enable-introspection --disable-binoculars"
        echo "check hkl shared libraries are in /usr/local/lib"
        ;;
esac

echo "Done."

make clean
make -j4
