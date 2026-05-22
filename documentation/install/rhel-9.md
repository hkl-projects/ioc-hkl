# Installing ioc-hkl on RHEL 9 / Rocky Linux 9

> **Status:** Draft. Steps below are expected deltas from Ubuntu guides; validate on a real RHEL 9 host and update this file with confirmed package names and fixes.

## Expected differences from Ubuntu

| Topic | Ubuntu (22.04 / 24.04) | RHEL 9 |
|-------|------------------------|--------|
| Packages | `apt` | `dnf` |
| System Python | 3.10 (22.04) / 3.12 (24.04) | 3.9 typical |
| IOC Python | Pixi 3.12 (recommended) | Same — use Pixi |
| `cif2hkl` | `apt install cif2hkl` | Build from source or third-party RPM; no standard `dnf` package |

## Prerequisites (draft)

```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y git curl
```

Install EPICS base per your site; set `EPICS_BASE` in `configure/RELEASE`.

## Pixi (same as Ubuntu)

```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL
cd /path/to/ioc-hkl
pixi install
.pixi/envs/default/bin/python --version   # expect 3.12.x
```

## cif2hkl

Build and install from [cif2hkl](https://gitlab.com/soleil-data-treatment/soleil-software-projects/cif2hkl) so `cif2hkl` is on `PATH` (e.g. `/usr/local/bin/cif2hkl`). `run_HKL.sh` checks `/usr/bin/cif2hkl` by default; adjust the path or symlink after install.

## Build and run

```bash
./run_HKL.sh
cd iocBoot/iocpydev
./st_pixi.cmd
```

Set `PYTHON_CONFIG` in `configure/RELEASE.local` to:

```text
$(pwd)/.pixi/envs/default/bin/python3.12-config
```

## hkl / GObject (watch when testing)

- Ensure `GI_TYPELIB_PATH` includes Pixi’s `lib/girepository-1.0` when running (handled by `st_pixi.cmd`).
- If linking fails, install GObject/introspection development packages via `dnf` (exact names TBD on test system).

## Contributing

After a successful install on RHEL 9, please update this document with:

- Confirmed `dnf install ...` package list
- `cif2hkl` install path and any `run_HKL.sh` check adjustments
- SELinux or firewall notes if relevant

## See also

- [ubuntu-22.04.md](ubuntu-22.04.md) — detailed Pixi troubleshooting
- [README.md](README.md)
