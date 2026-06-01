# HB-2C deployment (procServ, hb2c-dassrv1)

Run **ioc-hkl** under **procServ** on SNS beamline host **hb2c-dassrv1** (RHEL 9), with the Pixi Python stack and a facility build tree.

## Directory layout

| Path | Role |
|------|------|
| `/home/controls/common/ioc-hkl` | **Build tree** — `make`, `bin/`, `.pixi/`, `db/`, `python/`, `pixi.toml` |
| `/home/controls/hb2c/applications/hb2c-hkl` | **Beamline boot** — `st.cmd`, `st_base.cmd`, `envPaths` (ops / procServ entry) |

procServ should start **`st.cmd`** in the beamline app directory. That script must **not** rely on `st_pixi.cmd`’s relative `cd ../..` from `iocBoot/iocpydev` (from `hb2c-hkl`, `../..` is not the IOC tree).

For development from the build tree only:

```bash
cd /home/controls/common/ioc-hkl/iocBoot/iocpydev
./st_pixi.cmd
```

## PV prefix (`PREFIX`)

Set in `st_base.cmd` via `epicsEnvSet("PREFIX", "...")`. All records use `$(PREFIX)` from the database templates.

| Choice | Example PV | Notes |
|--------|------------|--------|
| `HB2C:` | `HB2C:wlen` | Short; use only if this IOC owns the whole `HB2C:` namespace |
| **`HB2C:hkl:`** | `HB2C:hkl:wlen` | **Recommended** — beamline + subsystem; room for other HB-2C IOCs |
| `hb2c:hkl:` | `hb2c:hkl:wlen` | Use only if your site standard is lowercase (EPICS PVs are case-sensitive) |

The repo example [`st_base_hb2c.cmd.example`](../../iocBoot/iocpydev/st_base_hb2c.cmd.example) uses **`HB2C:hkl:`**.

After changing `PREFIX`:

1. Update Phoebus / CSS **`$(Sys)`** macro to the same prefix (e.g. `HB2C:hkl:`).
2. Regenerate or edit motor / cross-IOC links that reference hkl PVs.
3. Run `dbl > pvlist.dbl` once from the boot directory after `iocInit` if you maintain a PV list file.

## Pixi: where to install

| Component | Location |
|-----------|----------|
| **`pixi` CLI** | On **`PATH` for the Unix user that runs procServ** (e.g. `~/.pixi/bin` for `kg1`, or `/home/controls/.pixi/bin` if the service runs as `controls`) |
| **Pixi project** (`.pixi/`, `pixi.toml`) | Only under **`/home/controls/common/ioc-hkl`** — matches `PYTHON_CONFIG` in `configure/RELEASE` |
| **Second env in `hb2c-hkl`** | **Not required** |

On NFS home directories, set in `st.cmd`:

```bash
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/tmp/pixi-cache-${USER}}"
```

See [../install/rhel-9.md](../install/rhel-9.md) for proxy, offline `pixi` binary copy, and `pixi install` troubleshooting.

## Install beamline startup files

From the build tree after `make`:

```bash
IOC_TOP=/home/controls/common/ioc-hkl
BOOT=/home/controls/hb2c/applications/hb2c-hkl

cp "$IOC_TOP/iocBoot/iocpydev/st.cmd.example" "$BOOT/st.cmd"
cp "$IOC_TOP/iocBoot/iocpydev/st_base_hb2c.cmd.example" "$BOOT/st_base.cmd"
cp "$IOC_TOP/iocBoot/iocpydev/envPaths" "$BOOT/"

chmod +x "$BOOT/st.cmd"
```

Edit `$BOOT/st.cmd` if paths or the procServ user differ. Edit `$BOOT/st_base.cmd` for wavelength, lattice, κ limits, and (later) motor `dbpf` / links.

Confirm `envPaths` points `TOP` at the build tree:

```bash
grep TOP "$BOOT/envPaths"
# epicsEnvSet("TOP","/home/controls/common/ioc-hkl")
```

## Geometry and WAND²

| `geom` value | Name | HB-2C / WAND² |
|--------------|------|----------------|
| 0 | E4CH | — |
| 1 | E4CV | Default in generic `st_base.cmd` |
| **2** | **K4CV** | **Use for WAND² κ goniometer** |
| 3 | E6C | — |
| 4 | K6C | — |
| 5 | TwoC | — |

`st_base_hb2c.cmd.example` sets `dbpf("$(PREFIX)geom","2")`. Stock hkl K4CV uses **α = 50°**; WAND² needs **45°** (rebuild hkl from source) — see [../wand2_kappa_geometry.md](../wand2_kappa_geometry.md).

## Manual test

```bash
/home/controls/hb2c/applications/hb2c-hkl/st.cmd
```

Expect Python/gi diagnostics on stdout, then IOC init. Check a PV, e.g.:

```bash
caget HB2C:hkl:geom_RBV
```

## procServ

Exact flags depend on your SNS procServ template. Typical pattern:

```bash
procServ -n hb2c-hkl -i <telnet-port> \
  /home/controls/hb2c/applications/hb2c-hkl/st.cmd
```

Checklist:

- `st.cmd` is executable.
- procServ Unix user has **`pixi` on PATH** and read/execute access to `ioc-hkl` and `.pixi`.
- Rebuild in `common/ioc-hkl` after code changes; refresh **`envPaths`** in `hb2c-hkl` when `TOP` or support paths change.

## After each rebuild

```bash
cd /home/controls/common/ioc-hkl
pixi install    # if dependencies changed
make -sj
cp iocBoot/iocpydev/envPaths /home/controls/hb2c/applications/hb2c-hkl/
# merge any local edits into st_base.cmd by hand if needed
```

## See also

- [../install/rhel-9.md](../install/rhel-9.md) — build and Pixi on the beamline
- [../wand2_kappa_geometry.md](../wand2_kappa_geometry.md) — κ = 45°, frame mapping
- [`st.cmd.example`](../../iocBoot/iocpydev/st.cmd.example), [`st_base_hb2c.cmd.example`](../../iocBoot/iocpydev/st_base_hb2c.cmd.example)
