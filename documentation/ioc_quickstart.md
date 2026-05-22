### Quickstart Guide

1. **Clone this repo:**
   [https://github.com/hkl-projects/ioc-hkl](https://github.com/hkl-projects/ioc-hkl)

2. **Follow a platform install guide** if needed (Ubuntu 22.04 vs 24.04 differ mainly in system Python; both use Pixi 3.12 for the IOC):
   [install/README.md](install/README.md)

3. **Install Pixi** and the locked Python environment:
   [https://pixi.sh/latest/installation/](https://pixi.sh/latest/installation/)

   ```bash
   cd /epics/iocs/ioc-hkl
   pixi install
   ```

4. **Install EPICS Base** and set its path in `configure/RELEASE` (absolute path to `EPICS_BASE`).

5. **Install `cif2hkl`** — on Ubuntu: `sudo apt install cif2hkl`. Other platforms: [install/README.md](install/README.md).

6. **Build the IOC:** run `./run_HKL.sh` (option 1: Pixi; then CA or PVAccess). This sets `PYTHON_CONFIG` in `configure/RELEASE.local` and runs `make`.

7. **Run the IOC:**

   ```bash
   cd iocBoot/iocpydev/
   ./st_pixi.cmd
   ```

   If you see **160.191477991** as the last output, communications are working.

8. **Open Phoebus** (optional): [Phoebus](https://controlssoftware.sns.ornl.gov/css_phoebus/) — screen `hklApp/op/bob/hkl_main.bob`.

9. **Initialize a sample** in the *Initials* tab (lattice parameters or `.cif`).

10. **Generate a trajectory** in the *Trajectory* tab.
