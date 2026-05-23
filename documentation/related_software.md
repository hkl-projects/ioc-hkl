# Related software and HKL / orientation engines

This page lists tools and ecosystems related to **ioc-hkl**. It is not an exhaustive survey; it groups what this project builds on versus established commercial stacks and other open-source efforts.

**This IOC uses:** the **[hkl](https://repo.or.cz/hkl.git)** C library (GObject introspection → Python `gi.repository.Hkl`) via **[PyDevice](https://github.com/klemenv/PyDevice)**, inside an EPICS soft IOC. See [hkl_architecture.md](hkl_architecture.md) and [ioc_comparison_table.md](ioc_comparison_table.md).

---

## 1. Primary engine: `hkl` (Picca / Soleil lineage)

The crystallographic core used by **ioc-hkl** is the **`hkl`** library: diffractometer geometries, UB matrix, pseudo-axes, forward/inverse HKL ↔ motor calculations, and related refinement.

| Resource | Notes |
|----------|--------|
| [hkl source repository](https://repo.or.cz/hkl.git) | Canonical upstream; build with `--enable-introspection` for Python/GI |
| [hkl documentation (Picca)](https://people.debian.org/~picca/hkl/hkl.html) | User/developer docs for current `hkl` releases (geometries, pseudo-axes, bindings) |
| Debian / conda packaging | `hkl` on [conda-forge](https://conda.anaconda.org/conda-forge/) (used by this repo’s `pixi.toml`) |

Related upstream components (same ecosystem) include **Ghkl** (GUI), **Binoculars** (volume visualization), and tools to add new diffractometer definitions—see the Picca documentation table of contents.

**ioc-hkl role:** EPICS PV layer and beamline integration around this engine—not a replacement for `hkl` itself.

---

## 2. EPICS integration

| Project | Role |
|---------|------|
| [PyDevice](https://github.com/klemenv/PyDevice) | Embeds Python in a C++ EPICS IOC; **bundled** in this repository under `src/` |
| [EPICS Base](https://epics.anl.gov/) | Control system framework |
| [Phoebus](https://controlssoftware.sns.ornl.gov/css_phoebus/) | Operator displays (`.bob` screens in `hklApp/op/bob/`) |
| [PVXS](https://github.com/mdavidsaver/pvxs) | Optional PVAccess support when `PVXS` is set in `configure/RELEASE` |

---

## 3. Established beamline software (commercial)

| Product | Vendor | Notes |
|---------|--------|--------|
| [SPEC](https://www.certif.com/) | [Certified Scientific Software](https://www.certif.com/) | Widely used **commercial** instrument control and data acquisition for X-ray/neutron diffractometry; macro language, mature diffractometer support. **Not open source** (closed, licensed). |
| C-PLOT | CSS | Graphics/analysis; often used with SPEC |

**ioc-hkl** command and PV concepts are partly aligned with SPEC-style workflows; see [ioc_comparison_table.md](ioc_comparison_table.md) for SPEC vs `hkl_ioc` vs `hkl_picca` vs Bluesky.

---

## 4. Other open-source HKL / orientation efforts

These are **not** drop-in replacements for the `hkl` library inside this IOC, but they address overlapping problems (orientation, reciprocal space, or Laue).

| Project | Focus | Link |
|---------|--------|------|
| **hklpy / Bluesky** | Python diffractometer abstraction in the Bluesky ecosystem | [bluesky/hklpy](https://github.com/bluesky/hklpy) (see comparison table) |
| **subhkl** | ORNL: crystal orientation from **2D Laue** diffraction images (JAX, `uv`, Docker); GPL-3.0 | [neutrons/subhkl](https://github.com/neutrons/subhkl) |
| **cif2hkl** | Structure factors / `.cif` → HKL (used by this IOC for intensity-related workflows) | [cif2hkl](https://gitlab.com/soleil-data-treatment/soleil-software-projects/cif2hkl) |

**subhkl** targets Laue image analysis and orientation refinement, not real-time EPICS motor/HKL coordination like **ioc-hkl**.

---

## 5. Neutron facilities (inelastic, TAS, Laue imaging)

At institutes such as the **Institut Laue-Langevin (ILL)**, software for **inelastic neutron scattering** and **triple-axis spectrometers (TAS)** solves a related but distinct problem: instrument configuration and **Q-space** for spectroscopy, not typically the same single-crystal four-circle **HKL ↔ motor** loop as `hkl`.

| Tool | Typical use | Link |
|------|-------------|------|
| **vTAS** | Virtual TAS; reciprocal-space and instrument-space displays | [ILL scientific tools](https://www.ill.eu/en/for-ill-users/support-infrastructures/software-scientific-tools/) |
| **TAS-Paths** | Automatic pathfinding for TAS scans | same catalogue |
| **LAMP** | ILL data treatment (IDL-based) for neutron experiments | same catalogue |
| **Restrax** | TAS neutron optics simulation | ILL spectroscopy software pages |
| **Orient-Express / Cyclops** | **Laue** neutron cameras for rapid crystal orientation (ILL + Photonic Science) | [ILL Laue systems overview](https://photonicscience.com/neutron-laue-diffraction-system-ill-grenoble/) |
| **DAVE** | NIST: reduction/visualization/analysis for **inelastic** neutron spectroscopy | [NIST DAVE](https://www.ncnr.nist.gov/dave/) |

For a broader catalogue of neutron/photon analysis packages, see **PaNdata** (linked from [ILL software tools](https://www.ill.eu/en/for-ill-users/support-infrastructures/software-scientific-tools/)).

---

## 6. How to read this list

```text
Commercial control (SPEC)          ← established, closed source
        │
        ▼
Open engine (hkl / Picca)          ← this IOC’s math core
        │
        ├── EPICS IOC (ioc-hkl)    ← real-time PVs, PyDevice
        ├── Python/Bluesky (hklpy)
        ├── Laue imaging (subhkl)
        └── Neutron TAS/Laue facility tools (ILL, NIST, …) ← related physics, different stack
```

If you know of other open **single-crystal diffractometer** HKL engines worth listing, please open an issue or PR on [hkl-projects/ioc-hkl](https://github.com/hkl-projects/ioc-hkl).

---

## See also

- [ioc_comparison_table.md](ioc_comparison_table.md) — SPEC / hklpy / `hkl_picca` / `hkl_ioc` commands
- [hkl_user_guide.md](hkl_user_guide.md) — using HKL through this IOC
- [install/README.md](install/README.md) — platform installation
