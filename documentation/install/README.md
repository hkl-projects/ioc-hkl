# Platform installation guides

These guides describe how to build and run **ioc-hkl** on specific Linux distributions. The recommended path uses **Pixi** to provide Python 3.12 and the `hkl` bindings, independent of the system Python version.

| Platform | Guide | Status |
|----------|-------|--------|
| Ubuntu 24.04 | [ubuntu-24.04.md](ubuntu-24.04.md) | Primary reference environment |
| Ubuntu 22.04 | [ubuntu-22.04.md](ubuntu-22.04.md) | Supported (system Python 3.10; use Pixi 3.12) |
| RHEL 9 / Rocky 9 | [rhel-9.md](rhel-9.md) | SNS beamline (proxy, Pixi, NFS cache notes) |

## Quick path (any supported Linux)

1. Install [EPICS base](https://epics.anl.gov/) and set `EPICS_BASE` in `configure/RELEASE`.
2. Install [Pixi](https://pixi.sh/latest/installation/).
3. Install `cif2hkl` (see your platform guide).
4. From the repo root: `./run_HKL.sh` (Pixi env + build), then run `./iocBoot/iocpydev/st_pixi.cmd`.

See also [../ioc_quickstart.md](../ioc_quickstart.md) for a minimal end-to-end checklist.

## EPICS `RELEASE` vs these docs

- **`configure/RELEASE`** — EPICS module paths (e.g. `EPICS_BASE`). Edit for your site; do not confuse with this folder.
- **`configure/RELEASE`** — Facility defaults include `PYTHON_CONFIG` for Pixi (after `pixi install`).
- **`configure/RELEASE.local`** — Local overrides (not in git; copy from `configure/RELEASE.local.example`). Set `MODULES`, `EPICS_BASE`, optional `PVXS` on non-facility machines. `run_HKL.sh` can update `PYTHON_CONFIG` / `PVXS` in `RELEASE.local`.

## Legacy / extended notes

[../ioc_installation.md](../ioc_installation.md) contains older lab notes (motorSim coupling, Phoebus, venv/`gi` troubleshooting on Ubuntu 24.04). Prefer the platform guides above for a fresh IOC install.
