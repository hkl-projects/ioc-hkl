# Site procServ runbook (template)

Copy to `documentation/local/` and replace every `REPLACE_*` token. **Do not commit the filled-in file to public git.**

## Directory layout

| Path | Role |
|------|------|
| `REPLACE_IOC_TOP` | Build tree (`make`, `bin/`, `.pixi/`, `pixi.toml`) |
| `REPLACE_BOOT_DIR` | Beamline boot (`st.cmd`, `st_base.cmd`, `envPaths`) |

## PV prefix

| Choice | Example PV |
|--------|------------|
| `REPLACE_PREFIX` | e.g. `BEAMLINE:hkl:` |

Set in `st_base.cmd`: `epicsEnvSet("PREFIX", "REPLACE_PREFIX")`.

## Pixi

| Component | Location |
|-----------|----------|
| `pixi` CLI | `REPLACE_PIXI_PATH` (on PATH for procServ user) |
| `.pixi` env | `REPLACE_IOC_TOP/.pixi` |

### Corporate proxy (if applicable)

```bash
export http_proxy=REPLACE_HTTP_PROXY
export https_proxy=REPLACE_HTTPS_PROXY
export no_proxy=REPLACE_NO_PROXY
export PIXI_CACHE_DIR=/tmp/pixi-cache-${USER}
```

## Startup files

```bash
IOC_TOP=REPLACE_IOC_TOP
BOOT=REPLACE_BOOT_DIR

cp "$IOC_TOP/iocBoot/iocpydev/st.cmd.example" "$BOOT/st.cmd"
cp "$IOC_TOP/iocBoot/iocpydev/st_base_site.cmd.example" "$BOOT/st_base.cmd"
cp "$IOC_TOP/iocBoot/iocpydev/envPaths" "$BOOT/"
chmod +x "$BOOT/st.cmd"
```

Edit `st.cmd` defaults for `IOC_TOP` and `BOOT_DIR`. Edit `st_base.cmd` for PREFIX, geometry, motors.

## procServ

```bash
procServ -n REPLACE_IOC_NAME -i REPLACE_TELNET_PORT "$BOOT/st.cmd"
```

## RELEASE.local

```makefile
MODULES = REPLACE_MODULES
EPICS_BASE = REPLACE_EPICS_BASE
PVXS = REPLACE_PVXS
```

## Rebuild sync

```bash
cd REPLACE_IOC_TOP
pixi install && make -sj
cp iocBoot/iocpydev/envPaths REPLACE_BOOT_DIR/
```
