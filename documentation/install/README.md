# Platform installation guides

These guides describe how to build and run **ioc-hkl** on specific Linux distributions. The recommended path uses **Pixi** to provide Python 3.12 and the `hkl` bindings, independent of the system Python version.

| Platform | Guide | Status |
|----------|-------|--------|
| Ubuntu 24.04 | [ubuntu-24.04.md](ubuntu-24.04.md) | Primary reference environment |
| Ubuntu 22.04 | [ubuntu-22.04.md](ubuntu-22.04.md) | Supported (system Python 3.10; use Pixi 3.12) |
| RHEL 9 / Rocky 9 | [rhel-9.md](rhel-9.md) | Draft — validate on a real host |

## Quick path (any supported Linux)

1. Install [EPICS base](https://epics.anl.gov/) and set `EPICS_BASE` in `configure/RELEASE`.
2. Install [Pixi](https://pixi.sh/latest/installation/).
3. Install `cif2hkl` (see your platform guide).
4. From the repo root: `./run_HKL.sh` (Pixi env + build), then run `./iocBoot/iocpydev/st_pixi.cmd`.

See also [../ioc_quickstart.md](../ioc_quickstart.md) for a minimal end-to-end checklist.

## EPICS `RELEASE` vs these docs

- **`configure/RELEASE`** — EPICS module paths (e.g. `EPICS_BASE`). Edit for your site; do not confuse with this folder.
- **`configure/RELEASE.local`** — Local overrides (e.g. `PYTHON_CONFIG` for Pixi). Created/updated by `run_HKL.sh` or manually per platform guide.

## Legacy / extended notes

[../ioc_installation.md](../ioc_installation.md) contains older lab notes (motorSim coupling, Phoebus, venv/`gi` troubleshooting on Ubuntu 24.04). Prefer the platform guides above for a fresh IOC install.
