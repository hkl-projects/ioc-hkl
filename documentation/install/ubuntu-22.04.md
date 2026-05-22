# Installing ioc-hkl on Ubuntu 22.04

Ubuntu 22.04 ships **Python 3.10** as `/usr/bin/python3`. This IOC is built and run against **Python 3.12** from Pixi (conda-forge), which also supplies `hkl` and PyGObject. Do not point the EPICS build at system Python unless you are following the optional [source hkl](#optional-build-hkl-from-source) path.

## Prerequisites

```bash
sudo apt update
sudo apt install -y build-essential git curl cif2hkl
```

- **EPICS base** — build and install under e.g. `/epics/base`. Set `EPICS_BASE` in `configure/RELEASE` (absolute path).
- **cif2hkl** — must be on `PATH` at `/usr/bin/cif2hkl` (package above) or equivalent.

## Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL
pixi --version
```

## Clone the IOC

```bash
cd /epics/iocs
git clone https://github.com/hkl-projects/ioc-hkl.git
cd ioc-hkl
```

## Python environment (Pixi, recommended)

The repo includes `pixi.toml` and `pixi.lock`. Install the locked environment:

```bash
pixi install
```

Verify:

```bash
.pixi/envs/default/bin/python --version
# Python 3.12.x

.pixi/envs/default/bin/python -c \
  "import gi; gi.require_version('Hkl', '5.0'); from gi.repository import Hkl; print('Hkl OK')"
```

Point the EPICS build at Pixi’s Python (or use `./run_HKL.sh`, which sets this in `configure/RELEASE.local`):

```bash
echo "PYTHON_CONFIG=$(pwd)/.pixi/envs/default/bin/python3.12-config" >> configure/RELEASE.local
```

If `configure/RELEASE.local` already contains `PYTHON_CONFIG=...`, update that line instead of duplicating it.

## Build the IOC

```bash
make clean
make -j4
```

Or interactively:

```bash
./run_HKL.sh
```

Choose **1** for the Pixi Python environment, then **1** or **2** for Channel Access only vs PVAccess (requires PVXS).

## Run the IOC

```bash
cd iocBoot/iocpydev
./st_pixi.cmd
```

Use `st_pixi.cmd`, not `st.cmd`, so the launcher picks up Pixi’s `PYTHONHOME`, libraries, and `gi` typelibs.

## Troubleshooting

### `No such file or directory` for `.pixi/envs/default/bin/python`

The Pixi environment was never created. From the repo root:

```bash
pixi install
```

Then rebuild with `make clean && make -j4`.

### `RELEASE.local` points at Pixi but build fails

`configure/RELEASE.local` may have been copied from another machine. Ensure `PYTHON_CONFIG` matches an existing file:

```bash
ls -l .pixi/envs/default/bin/python3.12-config
```

### System `python3` is 3.10 — is that a problem?

Only if you build against `/usr/bin/python3-config`. For the recommended path, ignore system Python for the IOC; use Pixi 3.12 only.

### Import errors for `gi` or `Hkl` at runtime

Run with `./st_pixi.cmd`. If problems persist, check:

```bash
.pixi/envs/default/bin/python -c "import gi; from gi.repository import Hkl"
```

## Optional: build hkl from source

Use this when adding new diffractometer geometries, not for a normal beamline deploy. See [../ioc_installation.md](../ioc_installation.md) (hkl from source) and README “hkl installation (from source)”.

On Ubuntu 22.04, `./run_HKL.sh` option **2** sets `PYTHON_CONFIG=/usr/bin/python3-config` and expects you to build and install `hkl` into `/usr/local`.

## See also

- [ubuntu-24.04.md](ubuntu-24.04.md) — same Pixi flow; system Python is also 3.12 on 24.04
- [../ioc_quickstart.md](../ioc_quickstart.md)
