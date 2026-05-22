
## Overview

This repository provides an EPICS Input/Output Controller (IOC) that performs real-time crystallographic HKL calculations for diffractometers and scattering instruments. It integrates the Python `hkl` library with EPICS via PyDevice, exposing HKL calculations and diffractometer geometry transformations as EPICS process variables. This allows control systems to convert between motor positions and reciprocal-space coordinates, configure diffractometer geometries, and drive scans directly in HKL space.

## Installation

**Platform guides** (recommended): [documentation/install/README.md](documentation/install/README.md)

| OS | Guide |
|----|--------|
| Ubuntu 22.04 | [documentation/install/ubuntu-22.04.md](documentation/install/ubuntu-22.04.md) |
| Ubuntu 24.04 | [documentation/install/ubuntu-24.04.md](documentation/install/ubuntu-24.04.md) |
| RHEL 9 | [documentation/install/rhel-9.md](documentation/install/rhel-9.md) (draft) |

**Quick start:** [documentation/ioc_quickstart.md](documentation/ioc_quickstart.md)

### Dependencies

* EPICS — https://epics.anl.gov/
* PyDevice — bundled in this repo
* Python 3.12 + `hkl` — via Pixi (`pixi.toml` in repo)
* cif2hkl — https://gitlab.com/soleil-data-treatment/soleil-software-projects/cif2hkl

### Minimal steps

1. Install EPICS base; set `EPICS_BASE` in `configure/RELEASE`.
2. Install Pixi: https://pixi.sh/latest/installation/
3. Clone into `/epics/iocs/`, install `cif2hkl`, then:

```bash
cd /epics/iocs/ioc-hkl
pixi install
./run_HKL.sh
```

4. Run: `cd iocBoot/iocpydev && ./st_pixi.cmd`

On **Ubuntu 22.04**, system `python3` is 3.10; the IOC uses **Pixi Python 3.12** — see the 22.04 guide if `.pixi/envs/default/bin/python` is missing.

### hkl from source (optional)

Only needed for new diffractometer geometries — https://repo.or.cz/hkl.git

```bash
cd /epics/support
git clone https://repo.or.cz/hkl.git
cd hkl
sudo apt install gtk-doc-tools autoconf libgtkmm-3.0-dev libyaml-dev gettext autopoint gobject-introspection libtool autoconf-archive debhelper gnuplot-nox gobject-introspection gtk-doc-tools libbullet-dev libg3d-dev libg3d-plugins libgirepository1.0-dev libgl-dev libgsl-dev libgtk-3-dev libgtkglext1-dev libhdf5-dev python3-gi python3-pip elpa-htmlize dvipng libhdf5-dev povray asymptote libhdf5-dev libcglm-dev libinih-dev
./autogen
./configure --enable-introspection --disable-binoculars
make && sudo make install
```

Use `./run_HKL.sh` option **2** for system Python when building this way.

### Test

In EPICS shell: `pydev("hklApp.test()")`  
In bash: `caget TAS:hb3:in:pseudoaxesh`
