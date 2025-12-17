# HKL Library User Guide

## Introduction

The HKL library provides comprehensive tools for crystallographic calculations in X-ray diffraction experiments. This guide covers practical usage of the library's main components and applications.

### Basic Setup

```c
#include <hkl.h>
#include <stdio.h>

int main() {
    GError *error = NULL;
    
    // Create a factory for E4CV geometry
    HklFactory *factory = hkl_factory_get_by_name("E4CV", &error);
    if (error) {
        fprintf(stderr, "Error: %s\n", error->message);
        g_error_free(error);
        return 1;
    }
    
    // Create geometry and set wavelength
    HklGeometry *geometry = hkl_factory_create_new_geometry(factory);
    hkl_geometry_wavelength_set(geometry, 1.0, HKL_UNIT_DEFAULT, &error);
    
    // Create detector and sample
    HklDetector *detector = hkl_detector_factory_new(HKL_DETECTOR_TYPE_0D);
    HklSample *sample = hkl_sample_new("test_crystal");
    
    // Create engine list
    HklEngineList *engines = hkl_factory_create_new_engine_list(factory);
    hkl_engine_list_init(engines, geometry, detector, sample);
    
    // Cleanup
    hkl_geometry_free(geometry);
    hkl_detector_free(detector);
    hkl_sample_free(sample);
    hkl_engine_list_free(engines);
    
    return 0;
}
```

### Compilation

```bash
# Compile with HKL library
gcc -o example example.c `pkg-config --cflags --libs hkl`
```

## Core Concepts

### 1. Geometry

A **geometry** represents a specific diffractometer configuration with:
- **Axes**: Physical rotation axes (omega, chi, phi, etc.)
- **Wavelength**: X-ray wavelength
- **Detector**: Detector position and configuration

### 2. Sample

A **sample** contains crystal information:
- **Lattice**: Unit cell parameters (a, b, c, α, β, γ)
- **UB Matrix**: Orientation matrix linking crystal and laboratory coordinates
- **Reflections**: Known reflections for calibration

### 3. Engine

An **engine** performs calculations between different coordinate systems:
- **HKL Engine**: Miller indices (H, K, L)
- **Q Engine**: Q-space coordinates
- **Psi Engine**: Sample rotation around Q

### 4. Factory

A **factory** creates geometries and engines for specific diffractometer types.

## Geometry Setup

### Available Geometries

```c
// List available geometries
HklFactory **factories;
size_t n_factories;
factories = hkl_factory_get_all(&n_factories);

for (size_t i = 0; i < n_factories; i++) {
    printf("Geometry: %s\n", hkl_factory_name_get(factories[i]));
}
```

### Common Geometries

#### E4CV (Eulerian 4-Circle Vertical)
```c
HklFactory *factory = hkl_factory_get_by_name("E4CV", &error);
HklGeometry *geometry = hkl_factory_create_new_geometry(factory);

// Set wavelength (1.0 Å)
hkl_geometry_wavelength_set(geometry, 1.0, HKL_UNIT_DEFAULT, &error);

// Get axis names
const darray_string *axis_names = hkl_geometry_axis_names_get(geometry);
for (size_t i = 0; i < axis_names->size; i++) {
    printf("Axis: %s\n", axis_names->item[i]);
}
```

#### K6C (Kappa 6-Circle)
```c
HklFactory *factory = hkl_factory_get_by_name("K6C", &error);
HklGeometry *geometry = hkl_factory_create_new_geometry(factory);
```

### Axis Configuration

```c
// Set axis values
double axis_values[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
hkl_geometry_axis_values_set(geometry, axis_values, 6, HKL_UNIT_DEFAULT, &error);

// Get axis values
double current_values[6];
hkl_geometry_axis_values_get(geometry, current_values, 6, HKL_UNIT_DEFAULT);
```

## Sample Management

### Creating a Sample

```c
// Create sample with name
HklSample *sample = hkl_sample_new("my_crystal");

// Create lattice (cubic, a=10.0 Å)
HklLattice *lattice = hkl_lattice_new(10.0, 10.0, 10.0, 90.0, 90.0, 90.0, &error);
hkl_sample_lattice_set(sample, lattice);
```

### UB Matrix Calculation

```c
// Create UB matrix manually
HklMatrix *UB = hkl_matrix_new_full(
    1.0, 0.0, 0.0,  // First row
    0.0, 1.0, 0.0,  // Second row
    0.0, 0.0, 1.0   // Third row
);
hkl_sample_UB_set(sample, UB, &error);

// Calculate UB from reflections (Busing-Levy method)
HklSampleReflection *r1 = hkl_sample_reflection_new(geometry, detector, 1, 0, 0, &error);
HklSampleReflection *r2 = hkl_sample_reflection_new(geometry, detector, 0, 1, 0, &error);

hkl_sample_compute_UB_busing_levy(sample, r1, r2, &error);
```

### Working with Reflections

```c
// Add reflection
HklSampleReflection *reflection = hkl_sample_reflection_new(
    geometry, detector, 1, 1, 0, &error);
hkl_sample_add_reflection(sample, reflection);

// Iterate over reflections
HklSampleReflection *item;
HKL_SAMPLE_REFLECTIONS_FOREACH(item, sample) {
    double h, k, l;
    hkl_sample_reflection_hkl_get(item, &h, &k, &l);
    printf("Reflection: %.1f %.1f %.1f\n", h, k, l);
}
```

## Engine Usage

### HKL Engine

```c
// Get HKL engine
HklEngine *hkl_engine = hkl_engine_list_engine_get_by_name(engines, "hkl", &error);

// Set HKL values and get solutions
double hkl_values[] = {1.0, 0.0, 0.0};
HklGeometryList *solutions = hkl_engine_pseudo_axis_values_set(
    hkl_engine, hkl_values, 3, HKL_UNIT_DEFAULT, &error);

// Iterate over solutions
HklGeometryListItem *item;
HKL_GEOMETRY_LIST_FOREACH(item, solutions) {
    const HklGeometry *solution = hkl_geometry_list_item_geometry_get(item);
    
    // Get axis values for this solution
    double axis_values[6];
    hkl_geometry_axis_values_get(solution, axis_values, 6, HKL_UNIT_DEFAULT);
    
    printf("Solution: omega=%.2f, chi=%.2f, phi=%.2f, 2theta=%.2f\n",
           axis_values[0], axis_values[1], axis_values[2], axis_values[3]);
}
```

### Q Engine

```c
// Get Q engine
HklEngine *q_engine = hkl_engine_list_engine_get_by_name(engines, "q", &error);

// Set Q values
double q_values[] = {1.0, 0.0, 0.0};  // Qx, Qy, Qz
HklGeometryList *q_solutions = q_engine_pseudo_axis_values_set(
    q_engine, q_values, 3, HKL_UNIT_DEFAULT, &error);
```

### Engine Modes

```c
// Get available modes
const darray_string *modes = hkl_engine_modes_names_get(hkl_engine);
for (size_t i = 0; i < modes->size; i++) {
    printf("Mode: %s\n", modes->item[i]);
}

// Set mode
hkl_engine_current_mode_set(hkl_engine, "bissector", &error);
```

## Applications

### Command Line Interface

#### Basic hkl Tool
```bash
# List available geometries
hkl --list-geometries

# Calculate HKL to angles
hkl --geometry E4CV --hkl 1 0 0 --wavelength 1.0
```

#### Binoculars-NG

```bash
# Process 2D detector data
binoculars-ng process config.ini

# Create new configuration
binoculars-ng cfg-new qxqyqz --output my_config.ini

# Merge processed files
binoculars-ng merge --output merged.h5 file1.h5 file2.h5
```

### GUI Application (ghkl)

```bash
# Launch GUI
ghkl

# Load configuration
ghkl --config my_config.ini
```

## Configuration

### Configuration Files

#### Geometry Configuration
```ini
[geometry]
name = E4CV
wavelength = 1.0

[axes]
omega = 0.0
chi = 0.0
phi = 0.0
2theta = 0.0
```

#### Sample Configuration
```ini
[sample]
name = my_crystal
lattice_a = 10.0
lattice_b = 10.0
lattice_c = 10.0
lattice_alpha = 90.0
lattice_beta = 90.0
lattice_gamma = 90.0

[ub_matrix]
u11 = 1.0
u12 = 0.0
u13 = 0.0
u21 = 0.0
u22 = 1.0
u23 = 0.0
u31 = 0.0
u32 = 0.0
u33 = 1.0
```

#### Binoculars Configuration
```ini
[detector]
type = Pilatus1M
pixel_size_x = 0.172
pixel_size_y = 0.172
distance = 200.0

[projection]
type = qxqyqz
binning = 100

[data]
source = /path/to/data.h5
output = /path/to/output.h5
```

## Examples

### Complete Example: HKL to Angles

```c
#include <hkl.h>
#include <stdio.h>

int main() {
    GError *error = NULL;
    
    // Setup
    HklFactory *factory = hkl_factory_get_by_name("E4CV", &error);
    HklGeometry *geometry = hkl_factory_create_new_geometry(factory);
    HklDetector *detector = hkl_detector_factory_new(HKL_DETECTOR_TYPE_0D);
    HklSample *sample = hkl_sample_new("test");
    
    // Set wavelength
    hkl_geometry_wavelength_set(geometry, 1.0, HKL_UNIT_DEFAULT, &error);
    
    // Create sample with cubic lattice
    HklLattice *lattice = hkl_lattice_new(10.0, 10.0, 10.0, 90.0, 90.0, 90.0, &error);
    hkl_sample_lattice_set(sample, lattice);
    
    // Create engine list
    HklEngineList *engines = hkl_factory_create_new_engine_list(factory);
    hkl_engine_list_init(engines, geometry, detector, sample);
    
    // Get HKL engine
    HklEngine *hkl_engine = hkl_engine_list_engine_get_by_name(engines, "hkl", &error);
    
    // Calculate for (1,0,0) reflection
    double hkl_values[] = {1.0, 0.0, 0.0};
    HklGeometryList *solutions = hkl_engine_pseudo_axis_values_set(
        hkl_engine, hkl_values, 3, HKL_UNIT_DEFAULT, &error);
    
    // Print solutions
    printf("Solutions for (1,0,0) reflection:\n");
    HklGeometryListItem *item;
    HKL_GEOMETRY_LIST_FOREACH(item, solutions) {
        const HklGeometry *solution = hkl_geometry_list_item_geometry_get(item);
        double axis_values[4];
        hkl_geometry_axis_values_get(solution, axis_values, 4, HKL_UNIT_DEFAULT);
        printf("  omega=%.2f°, chi=%.2f°, phi=%.2f°, 2theta=%.2f°\n",
               axis_values[0], axis_values[1], axis_values[2], axis_values[3]);
    }
    
    // Cleanup
    hkl_geometry_list_free(solutions);
    hkl_engine_list_free(engines);
    hkl_sample_free(sample);
    hkl_detector_free(detector);
    hkl_geometry_free(geometry);
    
    return 0;
}
```

### Example: Q-Space Calculation

```c
// Calculate Q vector for given geometry
HklVector ki = hkl_geometry_ki_get(geometry);
HklVector kf = hkl_geometry_kf_get(geometry, detector);

// Q = kf - ki
HklVector q = {
    .data = {
        kf.data[0] - ki.data[0],
        kf.data[1] - ki.data[1],
        kf.data[2] - ki.data[2]
    }
};

printf("Q vector: [%.3f, %.3f, %.3f]\n", q.data[0], q.data[1], q.data[2]);
```

### Example: Sample Orientation

```c
// Create UB matrix from lattice and orientation
HklMatrix *U = hkl_matrix_new_euler(0.0, 0.0, 0.0);  // Identity rotation
HklMatrix *B;
hkl_lattice_get_B(lattice, B);

// UB = U * B
HklMatrix *UB = hkl_matrix_new_copy(U);
hkl_matrix_times_matrix(UB, B);

hkl_sample_UB_set(sample, UB, &error);
```

This user guide provides practical examples for using the HKL library. For more advanced usage, refer to the API documentation and example applications included with the library.
