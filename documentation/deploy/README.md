# Deployment guides

| Site / host | Document |
|-------------|----------|
| HB-2C, **hb2c-dassrv1** (procServ + Pixi) | [hb2c-procServ.md](hb2c-procServ.md) |

Build and Pixi setup: [../install/rhel-9.md](../install/rhel-9.md). WAND² κ geometry: [../wand2_kappa_geometry.md](../wand2_kappa_geometry.md).

Example launch files (copy to the beamline app directory):

- [`../../iocBoot/iocpydev/st.cmd.example`](../../iocBoot/iocpydev/st.cmd.example) — procServ entry (Pixi + fixed `IOC_TOP`)
- [`../../iocBoot/iocpydev/st_base_hb2c.cmd.example`](../../iocBoot/iocpydev/st_base_hb2c.cmd.example) — EPICS startup with `HB2C:hkl:` and K4CV defaults
