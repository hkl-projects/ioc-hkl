# Deployment guides

| Topic | Document |
|-------|----------|
| procServ + Pixi (generic) | [procServ-pixi.md](procServ-pixi.md) |
| Site-specific runbooks | [../local.example/README.md](../local.example/README.md) (templates; real hosts/paths stay local) |

Example launch files:

- [`../../iocBoot/iocpydev/st.cmd.example`](../../iocBoot/iocpydev/st.cmd.example) — procServ entry (Pixi + `IOC_TOP` / `BOOT_DIR`)
- [`../../iocBoot/iocpydev/st_base_site.cmd.example`](../../iocBoot/iocpydev/st_base_site.cmd.example) — EPICS startup with `BEAMLINE:hkl:` and K4CV defaults
