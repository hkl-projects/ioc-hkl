### Quickstart Guide

1. **Clone this repo:**
   [https://github.com/hkl-projects/ioc-hkl](https://github.com/hkl-projects/ioc-hkl)

2. **Install Pixi** to set up the Python environment. It is a quick installation, and once you have it, everything else will install automatically:
   [https://pixi.sh/latest/installation/](https://pixi.sh/latest/installation/)

3. **Install EPICS Base** and specify its path in:
   `ioc-hkl/configure/RELEASE`

4. **Install Phoebus:**
   [https://controlssoftware.sns.ornl.gov/css_phoebus/](https://controlssoftware.sns.ornl.gov/css_phoebus/)

5. **Install `cif2hkl`** for structure-factor calculations:
   [https://gitlab.com/soleil-data-treatment/soleil-software-projects/cif2hkl](https://gitlab.com/soleil-data-treatment/soleil-software-projects/cif2hkl)
   On Ubuntu, it is available via apt:

   ```
   sudo apt install cif2hkl
   ```

   Otherwise, follow the installation instructions provided in the link.

6. **Build the IOC:**
   Run `./run_HKL.sh`, which will check your installations of the prerequisites above and build the IOC.

7. **Run the IOC:**

   ```
   cd iocBoot/iocpydev/
   ./st_pixi.cmd
   ```

   If you see the value **160.191477991** as the last output when running the IOC, all communications are working.

8. **Open Phoebus:**
   Run `./phoebus.sh` and open the CSS screen:
   `ioc-hkl/hklApp/op/bob/hkl_main.bob`
   This is the main IOC screen.

9. **Initialize a sample:**
   In the *Initials* tab, select a sample either by entering lattice parameters or by importing a `.cif` file at the top right.

10. **Generate a trajectory:**
    Navigate to the *Trajectory* tab and define the start and end values of hkl along with a step size, then press **Compute Trajectory**.
