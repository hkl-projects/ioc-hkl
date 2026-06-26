# Installing ioc-hkl on RHEL 9 / Rocky Linux 9

> **Status:** Validated on RHEL 9.x (`linux-x86_64`) at a facility EPICS deployment. Site-specific hostnames, proxy URLs, and filesystem paths belong in **local** runbooks (see [../local.example/README.md](../local.example/README.md)), not in this public guide.

## Expected differences from Ubuntu

| Topic | Ubuntu (22.04 / 24.04) | RHEL 9 (facility) |
|-------|------------------------|-------------------|
| Packages | `apt` | `dnf` |
| System Python | 3.10 / 3.12 | **3.9** — leave alone; IOC uses Pixi **3.12** |
| IOC Python | Pixi 3.12 | Same |
| `cif2hkl` | `apt install cif2hkl` | Site-specific; must be on `PATH` at `/usr/bin/cif2hkl` for `run_HKL.sh` |
| Pixi install script | Usually works | May be blocked by corporate HTTP proxy — see below |

## Prerequisites

```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y git curl perl gcc gcc-c++ make readline-devel
```

EPICS base and support modules: set in **`configure/RELEASE.local`** (copy from `configure/RELEASE.local.example`). Typical variables:

| Variable | Example layout |
|----------|----------------|
| `MODULES` | Parent directory of facility EPICS support modules |
| `EPICS_BASE` | `$(MODULES)/base/main` or your site’s base path |
| `PVXS` | `$(MODULES)/pvxs/main` (optional; PVAccess in `hklApp`) |

`configure/RELEASE` sets Pixi Python for builds:

```makefile
PYTHON_CONFIG = $(TOP)/.pixi/envs/default/bin/python3.12-config
```

---

## Installing Pixi behind a corporate proxy

Many facility networks block `https://pixi.sh/install.sh` or GitHub release assets while allowing **conda-forge**. Typical pattern:

1. Set `http_proxy` / `https_proxy` per your site (ask local IT; do **not** commit proxy hostnames to git).
2. Install the **`pixi` binary** on a machine with open access, or via an approved internal mirror, then copy to the build host.
3. Run **`pixi install`** in the IOC tree (conda-forge packages often work through the proxy).

### Step 1 — Install the `pixi` executable

On a host that can reach GitHub:

```bash
curl -fsSL -o /tmp/pixi.tar.gz \
  https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl.tar.gz
tar -xzf /tmp/pixi.tar.gz -C /tmp
mkdir -p ~/.pixi/bin
install -m 755 /tmp/pixi ~/.pixi/bin/pixi
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

Copy the binary to the facility build host with `scp` or your site’s software distribution process.

### Step 2 — Pixi environment in the IOC tree

If home directories are on **NFS**, point Pixi caches at **local disk** to avoid corrupted packages (`EOF while parsing paths.json`):

```bash
export PIXI_CACHE_DIR=/tmp/pixi-cache-${USER}
mkdir -p "$PIXI_CACHE_DIR"

cd /path/to/ioc-hkl
pixi install
```

If a previous attempt failed, clear partial state and retry:

```bash
rm -rf .pixi/envs/default
rm -rf "${HOME}/.cache/rattler" "$PIXI_CACHE_DIR"
pixi cache clean --yes 2>/dev/null || true
pixi install
```

### Step 3 — Verify Python and `hkl`

```bash
.pixi/envs/default/bin/python --version          # Python 3.12.x
.pixi/envs/default/bin/python -c \
  "import gi; gi.require_version('Hkl', '5.0'); from gi.repository import Hkl; print('Hkl OK')"
test -x .pixi/envs/default/bin/python3.12-config && echo PYTHON_CONFIG OK
```

---

## cif2hkl

Ensure `cif2hkl` is executable at **`/usr/bin/cif2hkl`** (required by `run_HKL.sh`), or symlink after installing elsewhere:

```bash
test -x /usr/bin/cif2hkl && echo OK
```

Build from [cif2hkl](https://gitlab.com/soleil-data-treatment/soleil-software-projects/cif2hkl) if your site does not provide a package.

---

## Build the IOC

```bash
cd /path/to/ioc-hkl
pixi install
make -C configure clean_pydevice
make clean
make -sj
```

Or use `./run_HKL.sh` (option **1** = Pixi; **1** or **2** = CA / PVAccess).

If you see `make: --help: No such file or directory` or `Python.h: No such file or directory`:

- Run **`pixi install`** first.
- Confirm `PYTHON_CONFIG` is set (in `configure/RELEASE` or `RELEASE.local`).

---

## Run the IOC

**Development** (from the build tree):

```bash
cd iocBoot/iocpydev
./st_pixi.cmd
```

Use **`st_pixi.cmd`**, not a bare EPICS `st.cmd`, so the launcher sets `PYTHONHOME`, `LD_LIBRARY_PATH`, and `GI_TYPELIB_PATH` from the Pixi env.

**Production (procServ):** use a beamline `st.cmd` with explicit `IOC_TOP` and `BOOT_DIR` — see [../deploy/procServ-pixi.md](../deploy/procServ-pixi.md).

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `curl pixi.sh` → 403 | Copy `pixi` binary from another host; set proxy per site IT |
| `failed to link ncurses` / `EOF` in `paths.json` | `export PIXI_CACHE_DIR=/tmp/...`; remove `.pixi/envs/default` and rattler cache; `pixi install` again |
| `make: --help: No such file or directory` | `PYTHON_CONFIG` unset — check `configure/RELEASE` and `pixi install` |
| `Python.h: No such file or directory` | Same as above; `make -C configure clean_pydevice && make` |
| System `python3` is 3.9 | Expected on RHEL 9; ignore for this IOC |

---

## See also

- [ubuntu-22.04.md](ubuntu-22.04.md) — Pixi on Ubuntu (open network)
- [README.md](README.md) — guide index
- [../deploy/procServ-pixi.md](../deploy/procServ-pixi.md) — procServ + Pixi launcher pattern
- [../local.example/README.md](../local.example/README.md) — site-specific runbooks (not in public git)
