#!/bin/bash
set -euo pipefail

top=$(pwd)
RELEASE_LOCAL="${top}/configure/RELEASE.local"
PIXI_PYTHON="${top}/.pixi/envs/default/bin/python"
PIXI_PYCONFIG_FS="${top}/.pixi/envs/default/bin/python3.12-config"
PIXI_PYCONFIG_MAKE='$(TOP)/.pixi/envs/default/bin/python3.12-config'

echo "Be sure to have the EPICS base installed, and point to its path in configure/RELEASE"
echo "Platform guides: documentation/install/README.md"

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
    echo "See documentation/install/ for other platforms (e.g. RHEL 9)."
    echo ""
    exit 1
else
    echo "cif2hkl is installed, continuing..."
fi

set_python_config() {
    local pyconf="$1"
    touch "${RELEASE_LOCAL}"
    if grep -q '^PYTHON_CONFIG=' "${RELEASE_LOCAL}" 2>/dev/null; then
        sed -i "s|^PYTHON_CONFIG=.*|PYTHON_CONFIG=${pyconf}|" "${RELEASE_LOCAL}"
    else
        echo "PYTHON_CONFIG=${pyconf}" >> "${RELEASE_LOCAL}"
    fi
}

setup_pixi_env() {
    if ! command -v pixi >/dev/null 2>&1; then
        echo "Error: pixi is not installed."
        echo "See documentation/install/README.md or https://pixi.sh/latest/installation/"
        exit 1
    fi

    echo "Setting up pixi environment..."
    if [[ -f pixi.toml ]]; then
        pixi install
    else
        echo "Warning: pixi.toml missing; creating environment (prefer cloning repo with pixi.toml committed)."
        pixi init
        pixi add python=3.12 numpy=2.3.4 pandas scipy matplotlib tqdm pygobject hkl
    fi

    if [[ ! -x "${PIXI_PYTHON}" ]]; then
        echo "Error: Pixi Python not found at ${PIXI_PYTHON}"
        echo "Run: pixi install"
        echo "See documentation/install/ubuntu-22.04.md"
        exit 1
    fi

    if [[ ! -x "${PIXI_PYCONFIG_FS}" ]]; then
        echo "Error: ${PIXI_PYCONFIG_FS} not found."
        exit 1
    fi

    echo "Using Pixi Python: $("${PIXI_PYTHON}" --version)"
    set_python_config "${PIXI_PYCONFIG_MAKE}"
}

echo "Choose an option:"
echo "1) pixi python environment (recommended)"
echo "2) system python (for building hkl from source)"
echo -n "Enter choice [1/2]: "
read -r choice

case $choice in
    1)
        setup_pixi_env
        ;;
    2)
        echo "Setting up system python"
        if [[ -f configure/CONFIG_PYENV ]]; then
            echo "PYTHON_CONFIG=/usr/bin/python3-config" > configure/CONFIG_PYENV
        fi
        set_python_config "/usr/bin/python3-config"
        echo "Install hkl https://repo.or.cz/hkl.git"
        echo "configure with flags --enable-introspection --disable-binoculars"
        echo "check hkl shared libraries are in /usr/local/lib"
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

echo "Done configuring Python."

echo "Choose an option:"
echo "1) build with only channel access"
echo "2) build with channel access and pvaccess (must have pvxs EPICS module)"
echo -n "Enter choice [1/2]: "
read -r choice

case $choice in
    1)
        echo "Building with only channel access"
        ;;
    2)
        echo "Building with pvaccess"
        touch "${RELEASE_LOCAL}"
        if ! grep -q '^PVXS=' "${RELEASE_LOCAL}" 2>/dev/null; then
            echo "PVXS=\$(MODULES)/pvxs" >> "${RELEASE_LOCAL}"
        fi
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

make clean
make -j4

echo "Build complete. Run the IOC with: cd iocBoot/iocpydev && ./st_pixi.cmd"
