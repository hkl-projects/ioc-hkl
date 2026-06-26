# procServ deployment with Pixi

Run **ioc-hkl** under **procServ** using the Pixi Python stack. This guide uses **placeholders** only. Copy [../local.example/site-procServ.md](../local.example/site-procServ.md) to `documentation/local/` and fill in real hostnames, paths, and PV prefixes for your facility.

## Two-directory layout (typical facility)

| Path | Role |
|------|------|
| `${IOC_TOP}` | **Build tree** — `make`, `bin/`, `.pixi/`, `db/`, `python/`, `pixi.toml` |
| `${BOOT_DIR}` | **Beamline boot** — `st.cmd`, `st_base.cmd`, `envPaths` (procServ entry) |

procServ should start **`st.cmd`** in `${BOOT_DIR}`. That script must **not** rely on `st_pixi.cmd`’s relative `cd ../..` from `iocBoot/iocpydev` (from a separate app directory, `../..` is not the IOC tree).

For development from the build tree only:

```bash
cd ${IOC_TOP}/iocBoot/iocpydev
./st_pixi.cmd
```

## PV prefix (`PREFIX`)

Set in `st_base.cmd` via `epicsEnvSet("PREFIX", "...")`. All records use `$(PREFIX)` from the database templates.

| Pattern | Example PV | Notes |
|---------|------------|--------|
| `BEAMLINE:` | `BEAMLINE:wlen` | Short; only if this IOC owns the whole beamline prefix |
| **`BEAMLINE:hkl:`** | `BEAMLINE:hkl:wlen` | **Recommended** — beamline + subsystem |
| lowercase variant | `beamline:hkl:wlen` | Use only if your site standard is lowercase (EPICS PVs are case-sensitive) |

The repo example [`st_base_site.cmd.example`](../../iocBoot/iocpydev/st_base_site.cmd.example) uses **`BEAMLINE:hkl:`** — replace with your site prefix.

After changing `PREFIX`:

1. Update Phoebus / CSS **`$(Sys)`** macro to match.
2. Update motor / cross-IOC links that reference hkl PVs.
3. Run `dbl > pvlist.dbl` after `iocInit` if you maintain a PV list file.

## Pixi: where to install

| Component | Location |
|-----------|----------|
| **`pixi` CLI** | On **`PATH` for the Unix user that runs procServ** |
| **Pixi project** (`.pixi/`, `pixi.toml`) | Under **`${IOC_TOP}`** — matches `PYTHON_CONFIG` in `configure/RELEASE` |
| **Second env in `${BOOT_DIR}`** | **Not required** |

On NFS home directories, set in `st.cmd`:

```bash
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/tmp/pixi-cache-${USER}}"
```

See [../install/rhel-9.md](../install/rhel-9.md) for generic Pixi install notes on RHEL.

## Install beamline startup files

From the build tree after `make`:

```bash
IOC_TOP=/path/to/ioc-hkl
BOOT_DIR=/path/to/beamline-app

cp "$IOC_TOP/iocBoot/iocpydev/st.cmd.example" "$BOOT_DIR/st.cmd"
cp "$IOC_TOP/iocBoot/iocpydev/st_base_site.cmd.example" "$BOOT_DIR/st_base.cmd"
cp "$IOC_TOP/iocBoot/iocpydev/envPaths" "$BOOT_DIR/"

chmod +x "$BOOT_DIR/st.cmd"
```

Edit `$BOOT_DIR/st.cmd` — set `IOC_TOP` and `BOOT_DIR` (or export them before procServ starts). Edit `st_base.cmd` for wavelength, lattice, geometry, and motor links.

Confirm `envPaths` points `TOP` at the build tree:

```bash
grep TOP "$BOOT_DIR/envPaths"
```

## Geometry selection

| `geom` value | Name | Typical use |
|--------------|------|-------------|
| 0 | E4CH | — |
| 1 | E4CV | Four-circle vertical |
| **2** | **K4CV** | **Kappa four-circle vertical** |
| 3 | E6C | Six-circle Euler |
| 4 | K6C | Kappa six-circle |
| 5 | TwoC | Two-circle |

`st_base_site.cmd.example` sets `dbpf("$(PREFIX)geom","2")` for K4CV. See [../wand2_kappa_geometry.md](../wand2_kappa_geometry.md) for κ tilt and frame-mapping notes.

## Manual test

```bash
IOC_TOP=/path/to/ioc-hkl BOOT_DIR=/path/to/beamline-app "$BOOT_DIR/st.cmd"
```

Check a PV (replace prefix):

```bash
caget BEAMLINE:hkl:geom_RBV
```

## procServ

Exact flags depend on your facility template. Typical pattern:

```bash
procServ -n ioc-hkl -i <telnet-port> /path/to/beamline-app/st.cmd
```

Checklist:

- `st.cmd` is executable.
- procServ Unix user has **`pixi` on PATH** and access to `${IOC_TOP}` and `.pixi`.
- After rebuild, refresh **`envPaths`** in `${BOOT_DIR}` when `TOP` or support paths change.

## After each rebuild

```bash
cd ${IOC_TOP}
pixi install    # if dependencies changed
make -sj
cp iocBoot/iocpydev/envPaths ${BOOT_DIR}/
# merge any local edits into st_base.cmd by hand if needed
```

## See also

- [`st.cmd.example`](../../iocBoot/iocpydev/st.cmd.example), [`st_base_site.cmd.example`](../../iocBoot/iocpydev/st_base_site.cmd.example)
- [../local.example/README.md](../local.example/README.md) — site runbook templates
