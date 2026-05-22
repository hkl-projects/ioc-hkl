# Installing ioc-hkl on Ubuntu 24.04

This is the **primary reference** environment for ioc-hkl. Ubuntu 24.04 provides **Python 3.12** as system `python3`, but the IOC still uses **Pixi** for a reproducible environment with pinned `hkl`, NumPy, and PyGObject.

## Prerequisites

```bash
sudo apt update
sudo apt install -y build-essential git curl cif2hkl
```

- **EPICS base** — e.g. `/epics/base`; set `EPICS_BASE` in `configure/RELEASE`.
- **cif2hkl** — `/usr/bin/cif2hkl` via apt, or build from [cif2hkl](https://gitlab.com/soleil-data-treatment/soleil-software-projects/cif2hkl).

## Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL
```

## Clone and set up

```bash
cd /epics/iocs
git clone https://github.com/hkl-projects/ioc-hkl.git
cd ioc-hkl
pixi install
```

Verify:

```bash
.pixi/envs/default/bin/python --version
.pixi/envs/default/bin/python -c \
  "import gi; gi.require_version('Hkl', '5.0'); from gi.repository import Hkl; print('Hkl OK')"
```

## Build and run

```bash
./run_HKL.sh
# Option 1: Pixi Python
# Option 1 or 2: CA only vs PVAccess

cd iocBoot/iocpydev
./st_pixi.cmd
```

Manual build (equivalent):

```bash
echo "PYTHON_CONFIG=$(pwd)/.pixi/envs/default/bin/python3.12-config" >> configure/RELEASE.local
make clean && make -j4
```

## Notes specific to 24.04

- You may use `sudo apt install python3-numpy` for **system** tools; the IOC build should still use **Pixi** `PYTHON_CONFIG`.
- Venv/`gi` path issues documented in [../ioc_installation.md](../ioc_installation.md) apply to venv setups, not the recommended Pixi + `st_pixi.cmd` path.

## See also

- [ubuntu-22.04.md](ubuntu-22.04.md) — same Pixi workflow; explicit 22.04 / Python 3.10 caveats
- [../ioc_quickstart.md](../ioc_quickstart.md)
