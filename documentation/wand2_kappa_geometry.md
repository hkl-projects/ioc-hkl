# WAND² (HB-2C) Kappa geometry notes

Notes for testing the **three-circle Kappa goniometer** on WAND² with **ioc-hkl** and the [hkl](https://repo.or.cz/hkl.git) library ([Picca documentation](https://people.debian.org/~picca/hkl/hkl.html)).

## Hardware summary

| Item | WAND² / beamline staff | hkl |
|------|------------------------|-----|
| Sample axes | ω, κ, φ (3 rotations) | **K4CV** (`komega`, `kappa`, `kphi`) |
| Detector axis | 2θ (if driven) | `tth` |
| Kappa tilt | **45°** from vertical ω (Mikhail) | Stock **K4CV** uses **α = 50°** |
| Instrument frame | X → detector, Y ↑, Z → beam downstream (Matthias) | Beam along **+X**, sample axes in Y–Z plane |
| Mount | Goniometer **pointing down** on beamline | hkl geometries defined at **all axes = 0** |

**Use K4CV, not K6C.** The stage has three sample circles (ω, κ, φ). K6C adds μ, γ, δ and is for a different layout.

---

## Kappa angle α in hkl

In K4CV, **α** is the angle between the κ rotation axis and **+Y**. The κ axis vector is `(0, -cos α, -sin α)`.

| α | κ axis | Notes |
|---|--------|--------|
| 50° | `[0, -0.643, -0.766]` | Default [K4CV in hkl docs](https://people.debian.org/~picca/hkl/hkl.html) |
| **45°** | `[0, -0.707, -0.707]` | **WAND²** (confirmed by beamline staff) |
| 54.74° | ≈ arccos(1/√3) | Common on some commercial κ stages |
| 60° | (custom) | Example only — not WAND² |

Changing α requires editing **`hkl-engine-k4c.c`** (all κ-dependent mode math, not just the axis vector) and **rebuilding hkl from source**. The conda-forge / Pixi `hkl` package does not include a custom α.

### Recommended hkl change (WAND²)

- Patch **`hkl-engine-k4c.c`** with **α = 45°** (see beamline copy under `k4cv_kappa45deg/`).
- Prefer `#define KAPPA_ALPHA (45.0 * HKL_DEGTORAD)` at the top of the file instead of scattered literals.
- **WAND-only build:** keep registered name **`K4CV`** so existing IOC screens/PVs (`*_k4c`) work unchanged.
- **Multi-site repo:** register a separate geometry (e.g. `K4CV_kappa45`) in `Makefile.am` and select it in the IOC; keep stock K4CV at 50°.

Build after patching hkl:

```bash
cd /epics/support/hkl/hkl
# apply patched hkl-engine-k4c.c
cd ..
make clean
./configure --enable-introspection --disable-binoculars
make && sudo make install
```

Rebuild the IOC with **system / locally built hkl** (`run_HKL.sh` option **2**, or point `PYTHON_CONFIG` at the GI that loads the new `libhkl`).

---

## Frame mapping checklist

Complete **before** trusting HKL scans. A correct κ angle alone does not fix frame or sign errors.

### Instrument coordinates (WAND²)

Right-handed, from Matthias:

- **X** — perpendicular to beam, toward detector  
- **Y** — vertical up  
- **Z** — along beam, downstream  

Goniometer mount (inverted):

- ω axis along **−Y** (effective **CCW positive** when mounted down)  
- κ axis in **Z, −Y** plane; **45°** from vertical ω  
- φ axis along **−Y** when κ = 0 (**CCW positive** at κ = 0)  
- κ remains **CW positive**

### hkl K4CV conventions

From [Picca K4CV](https://people.debian.org/~picca/hkl/hkl.html):

- X-ray / beam along **+X**  
- **komega**, **kphi**, **tth** — rotation about **−Y** `(0, -1, 0)`  
- **kappa** — rotation about `(0, -cos α, -sin α)`  
- Angles — **right-hand rule** about each axis vector  

### Mapping table (verify on hardware)

| WAND motor / axis | hkl K4CV | Verified? |
|-------------------|----------|-----------|
| ω (vertical) | `komega` | ☐ |
| κ (tilted) | `kappa` | ☐ |
| φ | `kphi` | ☐ |
| 2θ (if used) | `tth` | ☐ |
| Beam +Z (downstream) | hkl +X beam | ☐ |
| Y up | hkl Y (check sign) | ☐ |

**Open items to resolve with beamline staff**

- [ ] Confirm physical **κ = 45°** (photo + vendor drawing).  
- [ ] Confirm **beam direction** vs hkl +X (may need a fixed rotation of the lab frame, not only α).  
- [ ] Resolve **rotation sense** (Matthias: all CW in hardware; Jens: mixed on some axes) — use motor **DIR**, negated readbacks, or offsets after geometry is correct.  
- [ ] Confirm **inverted mount** effects on ω and φ sign.  
- [ ] Map EPICS motor names → `komega` / `kappa` / `kphi` / `tth` PVs (`*_k4c` in this IOC).

---

## Validation steps (before beamtime)

Run with IOC geometry **K4CV** (or `K4CV_kappa45` if registered) and **patched hkl (α = 45°)**.

### 1. Environment check

```bash
python -c "import gi; gi.require_version('Hkl','5.0'); from gi.repository import Hkl; print(Hkl.factories()['K4CV'])"
```

Confirm the factory loads and the IOC uses the rebuilt library (not an older Pixi `hkl`).

### 2. Two-reflection UB (indexing)

1. Center a strong reflection; record **(h, k, l)** and motor positions **(ω, κ, φ)**.  
2. Repeat for a second reflection (large angular separation).  
3. Enter both in the IOC **Initials / reflections** flow (K4CV / `*_k4c` PVs).  
4. Compute UB; check that listed **pseudo-axes** and **UB matrix** are reasonable.

**Pass:** Both reflections index with small residual; UB stable when re-entered.

### 3. Forward calculation (HKL → motors)

1. Choose a third reflection in range.  
2. Run **forward** (compute motors for target hkl).  
3. Move motors to computed positions; confirm Bragg condition on detector (or monitor count).

**Pass:** Peak found without large manual tweaks (< ~0.5° per axis is a reasonable first target; tighten with staff).

### 4. Backward check (motors → HKL)

1. Move to a known reflection manually.  
2. Read motor positions into the IOC; run **backward** / read computed hkl.  
3. Compare to expected (h, k, l).

**Pass:** Computed hkl matches within tolerance.

### 5. Single-axis sign test

For each axis **ω, κ, φ** separately (others fixed):

1. Apply a small **+1°** hardware move (CW positive per Matthias, noting inverted mount).  
2. Observe change in computed **Q** / hkl or monitor intensity.  
3. If direction is wrong, fix **one** layer: EPICS motor sign, axis offset in hkl, or geometry — do not mix fixes blindly.

**Pass:** Each +1° move changes Q in the expected direction.

### 6. Document final conventions

Record in beamline log or `RELEASE.local` notes:

- Final α used (45°)  
- Motor ↔ hkl axis map  
- Any sign flips or zero offsets  
- Wavelength and sample–detector distance if relevant for `tth`

---

## IOC configuration (this repo)

| Setting | WAND² typical value |
|---------|---------------------|
| Geometry | `K4CV` (`geom_name` in Python; `TWST` in DB) |
| PV suffix | `*_k4c` (axes, reflections, solutions) |
| Screens | `hklApp/op/bob/reflections/*k4c*`, `*wand2*` |
| Visualization | May still assume Euler 6-circle — treat as approximate until updated for K4CV |

Select K4CV in the IOC (e.g. `switch_geom()` / geometry index **2** in `python/hkl.py`).

---

## Related documents

- [related_software.md](related_software.md) — hkl, SPEC, other engines  
- [ioc_comparison_table.md](ioc_comparison_table.md) — SPEC vs hkl_ioc commands  
- [hkl_architecture.md](hkl_architecture.md) — how this IOC uses hkl  
- [install/rhel-9.md](install/rhel-9.md) — beamline build (Pixi, proxy)

## References

- [hkl source](https://repo.or.cz/hkl.git)  
- [Picca hkl documentation — K4CV](https://people.debian.org/~picca/hkl/hkl.html)  
- WAND² beamline correspondence (May 2026): κ = 45°; instrument frame X, Y, Z; inverted mount rotation senses
