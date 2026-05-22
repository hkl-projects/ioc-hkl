# Installing ioc-hkl on RHEL 9 / Rocky Linux 9

> **Status:** Validated on SNS beamline host **hb2c-dassrv1** (RHEL 9.7, `linux-x86_64`) with facility paths under `/home/controls/common`. Pixi install notes reflect SNS proxy and NFS constraints.

## Expected differences from Ubuntu

| Topic | Ubuntu (22.04 / 24.04) | RHEL 9 (beamline) |
|-------|------------------------|-------------------|
| Packages | `apt` | `dnf` |
| System Python | 3.10 / 3.12 | **3.9** — leave alone; IOC uses Pixi **3.12** |
| IOC Python | Pixi 3.12 | Same |
| `cif2hkl` | `apt install cif2hkl` | Site-specific; must be on `PATH` at `/usr/bin/cif2hkl` for `run_HKL.sh` |
| Pixi install script | Usually works | **`pixi.sh` often 403** through SNS proxy — see below |

## Prerequisites

```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y git curl perl gcc gcc-c++ make readline-devel
```

EPICS base and support modules should match `configure/RELEASE` (facility example):

| Variable | Example (SNS) |
|----------|----------------|
| `MODULES` | `/home/controls/common` |
| `EPICS_BASE` | `$(MODULES)/base/main` |
| `PVXS` | `$(MODULES)/pvxs/main` (optional; for PVAccess in `hklApp`) |

`configure/RELEASE` also sets:

```makefile
PYTHON_CONFIG = $(TOP)/.pixi/envs/default/bin/python3.12-config
```

Laptop or non-facility paths: copy `configure/RELEASE.local.example` to `configure/RELEASE.local` and override `MODULES` / `EPICS_BASE` only.

---

## Installing Pixi at the beamline (RHEL 9, SNS proxy)

On SNS hosts, outbound HTTP often uses **`bl-proxy1.sns.gov:3128`**. Set the proxy with a **full URL** (scheme required):

```bash
export http_proxy=http://bl-proxy1.sns.gov:3128
export https_proxy=http://bl-proxy1.sns.gov:3128
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy=localhost,127.0.0.1,.sns.gov,.ornl.gov
```

Add those lines to `~/.bashrc` if you use Pixi often on that host.

### What works and what is blocked (typical SNS)

| URL | Typical result |
|-----|----------------|
| `https://pixi.sh/install.sh` | **403** (Squid `ERR_ACCESS_DENIED`) |
| `https://github.com/.../pixi-*.tar.gz` | Redirect OK; **release asset download 403** |
| `https://conda.anaconda.org/conda-forge/...` | **200** — `pixi install` can pull packages |
| `https://repo.prefix.dev/...` | **403** — not required; `pixi.toml` uses **conda-forge** only |

So: **install the `pixi` binary offline or from another machine**; run **`pixi install`** on the beamline (conda-forge through the proxy).

### Step 1 — Install the `pixi` executable

**Option A — Copy from a machine that can reach GitHub** (recommended)

On laptop or build host:

```bash
curl -fsSL -o /tmp/pixi.tar.gz \
  https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl.tar.gz
tar -xzf /tmp/pixi.tar.gz -C /tmp
scp /tmp/pixi kg1@hb2c-dassrv1:~/.pixi/bin/
```

On the beamline:

```bash
mkdir -p ~/.pixi/bin
chmod +x ~/.pixi/bin/pixi
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

**Option B — Try GitHub through the proxy** (if your site allows release assets)

```bash
export http_proxy=http://bl-proxy1.sns.gov:3128
export https_proxy=http://bl-proxy1.sns.gov:3128
curl -fsSL -o /tmp/pixi.tar.gz \
  https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl.tar.gz
# unpack and install as in Option A
```

Do **not** rely on `curl https://pixi.sh/install.sh | bash` on SNS — it is usually blocked even with a correct proxy.

### Step 2 — Pixi environment in the IOC tree

Home directories on **`/SNS/users/...`** are often **NFS**. Point Pixi caches at **local disk** to avoid corrupted packages (`EOF while parsing paths.json`):

```bash
export PIXI_CACHE_DIR=/tmp/pixi-cache-${USER}
mkdir -p "$PIXI_CACHE_DIR"

cd /home/controls/common/ioc-hkl   # or your clone path
pixi install
```

If a previous attempt failed, clear partial state and retry:

```bash
rm -rf .pixi/envs/default
rm -rf /SNS/users/${USER}/.cache/rattler "$PIXI_CACHE_DIR"
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

The lock-file “upgrade to v7” warning is optional; **`pixi install` succeeding is enough**.

---

## cif2hkl

Ensure `cif2hkl` is executable at **`/usr/bin/cif2hkl`** (required by `run_HKL.sh`), or symlink after installing elsewhere:

```bash
test -x /usr/bin/cif2hkl && echo OK
# e.g. sudo ln -s /usr/local/bin/cif2hkl /usr/bin/cif2hkl
```

Build from [cif2hkl](https://gitlab.com/soleil-data-treatment/soleil-software-projects/cif2hkl) if your site does not provide a package.

---

## Build the IOC

With facility `configure/RELEASE` (includes `PYTHON_CONFIG` for Pixi):

```bash
cd /home/controls/common/ioc-hkl
export http_proxy=http://bl-proxy1.sns.gov:3128   # only needed for pixi install, not make
export https_proxy=http://bl-proxy1.sns.gov:3128
export PIXI_CACHE_DIR=/tmp/pixi-cache-${USER}

pixi install
make -C configure clean_pydevice
make clean
make -sj
```

Or use `./run_HKL.sh` (option **1** = Pixi; **1** or **2** = CA / PVAccess).

If you see `make: --help: No such file or directory` or `Python.h: No such file or directory`:

- Run **`pixi install`** first.
- Confirm `configure/RELEASE` contains the `PYTHON_CONFIG = $(TOP)/.pixi/...` line (not only in a missing `RELEASE.local`).

---

## Run the IOC

```bash
cd iocBoot/iocpydev
./st_pixi.cmd
```

Use **`st_pixi.cmd`**, not `st.cmd`, so the launcher sets `PYTHONHOME`, `LD_LIBRARY_PATH`, and `GI_TYPELIB_PATH` from the Pixi env.

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `curl pixi.sh` → 403 | Copy `pixi` binary from another host; use proxy URL with `http://` |
| `failed to link ncurses` / `EOF` in `paths.json` | `export PIXI_CACHE_DIR=/tmp/...`; remove `.pixi/envs/default` and rattler cache; `pixi install` again |
| `make: --help: No such file or directory` | `PYTHON_CONFIG` unset — check `configure/RELEASE` and `pixi install` |
| `Python.h: No such file or directory` | Same as above; regenerate `configure/CONFIG.PyDevice` via `make -C configure clean_pydevice && make` |
| System `python3` is 3.9 | Expected on RHEL 9; ignore for this IOC |

---

## See also

- [ubuntu-22.04.md](ubuntu-22.04.md) — Pixi on Ubuntu (open network)
- [README.md](README.md) — guide index
- [../ioc_quickstart.md](../ioc_quickstart.md)
