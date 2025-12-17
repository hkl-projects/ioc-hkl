# HKL Library API Reference

## Overview

This document provides a reference for the HKL library's public API. The API is designed to be stable across the 5.x series and provides access to all major functionality for crystallographic calculations.

## Header Files

### Primary Header
```c
#include <hkl.h>
```
Single header for the public API

## Core Data Types

### Basic Types

#### HklVector
3D vector for crystallographic calculations.

```c
typedef struct _HklVector HklVector;

struct _HklVector {
    double data[3];
};
```

**Constants:**
```c
#define HKL_VECTOR_X {{1, 0, 0}}
#define HKL_VECTOR_Y {{0, 1, 0}}
#define HKL_VECTOR_Z {{0, 0, 1}}
```

**Functions:**
```c
void hkl_vector_init(HklVector *self, double x, double y, double z);
```

#### HklMatrix
3x3 transformation matrix.

```c
typedef struct _HklMatrix HklMatrix;
```

**Functions:**
```c
HklMatrix *hkl_matrix_new(void);
HklMatrix *hkl_matrix_new_full(double m11, double m12, double m13,
                              double m21, double m22, double m23,
                              double m31, double m32, double m33);
HklMatrix *hkl_matrix_new_euler(double euler_x, double euler_y, double euler_z);

double hkl_matrix_get(const HklMatrix *self, unsigned int i, unsigned int j);
void hkl_matrix_free(HklMatrix *self);
void hkl_matrix_init(HklMatrix *self,
                    double m11, double m12, double m13,
                    double m21, double m22, double m23,
                    double m31, double m32, double m33);
int hkl_matrix_cmp(const HklMatrix *self, const HklMatrix *m);
void hkl_matrix_times_matrix(HklMatrix *self, const HklMatrix *m);
```

#### HklQuaternion
Quaternion for rotation representation.

```c
typedef struct _HklQuaternion HklQuaternion;

struct _HklQuaternion {
    double data[4];
};
```

**Functions:**
```c
void hkl_quaternion_to_matrix(const HklQuaternion *self, HklMatrix *m);
```

### Parameter System

#### HklParameter
Configurable parameter with units and constraints.

```c
typedef struct _HklParameter HklParameter;
```

**Functions:**
```c
HklParameter *hkl_parameter_new_copy(const HklParameter *self);
void hkl_parameter_free(HklParameter *self);

const char *hkl_parameter_name_get(const HklParameter *self);
const char *hkl_parameter_default_unit_get(const HklParameter *self);
const char *hkl_parameter_user_unit_get(const HklParameter *self);

double hkl_parameter_value_get(const HklParameter *self, HklUnitEnum unit_type);
int hkl_parameter_value_set(HklParameter *self, double value,
                           HklUnitEnum unit_type, GError **error);

void hkl_parameter_min_max_get(const HklParameter *self, double *min, double *max,
                              HklUnitEnum unit_type);
int hkl_parameter_min_max_set(HklParameter *self, double min, double max,
                             HklUnitEnum unit_type, GError **error);

int hkl_parameter_fit_get(const HklParameter *self);
void hkl_parameter_fit_set(HklParameter *self, int fit);
void hkl_parameter_randomize(HklParameter *self);
```

**Specialized Accessors:**
```c
const HklVector *hkl_parameter_axis_v_get(const HklParameter *self);
const HklQuaternion *hkl_parameter_quaternion_get(const HklParameter *self);
const char *hkl_parameter_description_get(const HklParameter *self);
void hkl_parameter_fprintf(FILE *f, const HklParameter *self);
```

#### Unit System

```c
typedef struct _HklUnitDimension HklUnitDimension;

struct _HklUnitDimension {
    int l;  /* Length */
    int m;  /* Mass */
    int t;  /* Time */
    int i;  /* Electric current */
    int th; /* Thermodynamic temperature */
    int n;  /* Amount of substance */
    int j;  /* Luminous intensity */
};

typedef enum _HklUnitEnum {
    HKL_UNIT_DEFAULT,
    HKL_UNIT_USER
} HklUnitEnum;

typedef struct _HklUnit HklUnit;

struct _HklUnit {
    HklUnitDimension dimension;
    double factor;
    char const *name;
    char const *repr;
};
```

**Predefined Units:**
```c
extern const HklUnit hkl_unit_angle_deg;
extern const HklUnit hkl_unit_angle_rad;
extern const HklUnit hkl_unit_length_nm;
extern const HklUnit hkl_unit_angle_mrad;
extern const HklUnit hkl_unit_length_mm;
extern const HklUnit hkl_unit_length_meter;
```

## Geometry System

### HklGeometry
Central component for diffractometer geometry management.

```c
typedef struct _HklGeometry HklGeometry;
```

**Creation and Management:**
```c
HklGeometry *hkl_geometry_new_copy(const HklGeometry *self);
void hkl_geometry_free(HklGeometry *self);
int hkl_geometry_set(HklGeometry *self, const HklGeometry *src);
```

**Axis Management:**
```c
const darray_string *hkl_geometry_axis_names_get(const HklGeometry *self);
const HklParameter *hkl_geometry_axis_get(const HklGeometry *self, const char *name,
                                         GError **error);
int hkl_geometry_axis_set(HklGeometry *self, const char *name,
                         const HklParameter *axis, GError **error);

void hkl_geometry_axis_values_get(const HklGeometry *self,
                                 double values[], size_t n_values,
                                 HklUnitEnum unit_type);
int hkl_geometry_axis_values_set(HklGeometry *self,
                                double values[], size_t n_values,
                                HklUnitEnum unit_type, GError **error);
```

**Geometry Properties:**
```c
const char *hkl_geometry_name_get(const HklGeometry *self);
double hkl_geometry_wavelength_get(const HklGeometry *self, HklUnitEnum unit_type);
int hkl_geometry_wavelength_set(HklGeometry *self, double wavelength,
                               HklUnitEnum unit_type, GError **error);
```

**Vector Calculations:**
```c
HklVector hkl_geometry_ki_get(const HklGeometry *self);
HklVector hkl_geometry_ki_abc_get(const HklGeometry *self, const HklSample *sample);
HklVector hkl_geometry_kf_get(const HklGeometry *self, const HklDetector *detector);
HklVector hkl_geometry_kf_abc_get(const HklGeometry *self,
                                 const HklDetector *detector,
                                 const HklSample *sample);
```

**Rotation Quaternions:**
```c
HklQuaternion hkl_geometry_sample_rotation_get(const HklGeometry *self,
                                              const HklSample *sample);
HklQuaternion hkl_geometry_detector_rotation_get(const HklGeometry *self,
                                                const HklDetector *detector);
```

**Utility Functions:**
```c
void hkl_geometry_randomize(HklGeometry *self);
int hkl_geometry_set_values_v(HklGeometry *self, HklUnitEnum unit_type,
                             GError **error, ...);
void hkl_geometry_fprintf(FILE *file, const HklGeometry *self);
```

### HklGeometryList
Container for multiple geometry solutions.

```c
typedef struct _HklGeometryList HklGeometryList;
typedef struct _HklGeometryListItem HklGeometryListItem;

#define HKL_GEOMETRY_LIST_FOREACH(item, list) for((item)=hkl_geometry_list_items_first_get((list)); \
                                                  (item); \
                                                  (item)=hkl_geometry_list_items_next_get((list), (item)))
```

**Functions:**
```c
void hkl_geometry_list_free(HklGeometryList *self);
size_t hkl_geometry_list_n_items_get(const HklGeometryList *self);

const HklGeometryListItem *hkl_geometry_list_items_first_get(const HklGeometryList *self);
const HklGeometryListItem *hkl_geometry_list_items_next_get(const HklGeometryList *self,
                                                           const HklGeometryListItem *item);

const HklGeometry *hkl_geometry_list_item_geometry_get(const HklGeometryListItem *self);
```

## Detector System

### HklDetector
Detector configuration and management.

```c
typedef struct _HklDetector HklDetector;

typedef enum _HklDetectorType {
    HKL_DETECTOR_TYPE_0D
} HklDetectorType;
```

**Functions:**
```c
HklDetector *hkl_detector_factory_new(HklDetectorType type);
HklDetector *hkl_detector_new_copy(const HklDetector *src);
void hkl_detector_free(HklDetector *self);
void hkl_detector_fprintf(FILE *f, const HklDetector *self);
```

## Sample System

### HklSample
Crystal sample management.

```c
typedef struct _HklSample HklSample;
```

**Creation and Management:**
```c
HklSample *hkl_sample_new(const char *name);
HklSample *hkl_sample_new_copy(const HklSample *self);
void hkl_sample_free(HklSample *self);
```

**Basic Properties:**
```c
const char *hkl_sample_name_get(const HklSample *self);
void hkl_sample_name_set(HklSample *self, const char *name);
```

**Lattice Management:**
```c
const HklLattice *hkl_sample_lattice_get(HklSample *self);
void hkl_sample_lattice_set(HklSample *self, const HklLattice *lattice);
```

**Orientation Vectors:**
```c
const HklParameter *hkl_sample_ux_get(const HklSample *self);
int hkl_sample_ux_set(HklSample *self, const HklParameter *ux, GError **error);
const HklParameter *hkl_sample_uy_get(const HklSample *self);
int hkl_sample_uy_set(HklSample *self, const HklParameter *uy, GError **error);
const HklParameter *hkl_sample_uz_get(const HklSample *self);
int hkl_sample_uz_set(HklSample *self, const HklParameter *uz, GError **error);
```

**Orientation Matrices:**
```c
const HklMatrix *hkl_sample_U_get(const HklSample *self);
void hkl_sample_U_set(HklSample *self, const HklMatrix *U, GError **error);
const HklMatrix *hkl_sample_UB_get(const HklSample *self);
int hkl_sample_UB_set(HklSample *self, const HklMatrix *UB, GError **error);
```

**Reflection Management:**
```c
size_t hkl_sample_n_reflections_get(const HklSample *self);

#define HKL_SAMPLE_REFLECTIONS_FOREACH(_item, _list) for((_item)=hkl_sample_reflections_first_get((_list)); \
                                                         (_item); \
                                                         (_item)=hkl_sample_reflections_next_get((_list), (_item)))

HklSampleReflection *hkl_sample_reflections_first_get(HklSample *self);
HklSampleReflection *hkl_sample_reflections_next_get(HklSample *self,
                                                    HklSampleReflection *reflection);
void hkl_sample_del_reflection(HklSample *self, HklSampleReflection *reflection);
void hkl_sample_add_reflection(HklSample *self, HklSampleReflection *reflection);
```

**UB Matrix Calculations:**
```c
int hkl_sample_compute_UB_busing_levy(HklSample *self,
                                     const HklSampleReflection *r1,
                                     const HklSampleReflection *r2,
                                     GError **error);
double hkl_sample_get_reflection_measured_angle(const HklSample *self,
                                               const HklSampleReflection *r1,
                                               const HklSampleReflection *r2);
double hkl_sample_get_reflection_theoretical_angle(const HklSample *self,
                                                  const HklSampleReflection *r1,
                                                  const HklSampleReflection *r2);
int hkl_sample_affine(HklSample *self, GError **error);
```

### HklLattice
Crystal lattice parameters.

```c
typedef struct _HklLattice HklLattice;
```

**Creation:**
```c
HklLattice *hkl_lattice_new(double a, double b, double c,
                           double alpha, double beta, double gamma,
                           GError **error);
HklLattice *hkl_lattice_new_copy(const HklLattice *self);
HklLattice *hkl_lattice_new_default(void);
void hkl_lattice_free(HklLattice *self);
```

**Lattice Parameters:**
```c
const HklParameter *hkl_lattice_a_get(const HklLattice *self);
int hkl_lattice_a_set(HklLattice *self, const HklParameter *parameter, GError **error);
const HklParameter *hkl_lattice_b_get(const HklLattice *self);
int hkl_lattice_b_set(HklLattice *self, const HklParameter *parameter, GError **error);
const HklParameter *hkl_lattice_c_get(const HklLattice *self);
int hkl_lattice_c_set(HklLattice *self, const HklParameter *parameter, GError **error);
const HklParameter *hkl_lattice_alpha_get(const HklLattice *self);
int hkl_lattice_alpha_set(HklLattice *self, const HklParameter *parameter, GError **error);
const HklParameter *hkl_lattice_beta_get(const HklLattice *self);
int hkl_lattice_beta_set(HklLattice *self, const HklParameter *parameter, GError **error);
const HklParameter *hkl_lattice_gamma_get(const HklLattice *self);
int hkl_lattice_gamma_set(HklLattice *self, const HklParameter *parameter, GError **error);
const HklParameter *hkl_lattice_volume_get(const HklLattice *self);
```

**Bulk Operations:**
```c
int hkl_lattice_set(HklLattice *self,
                   double a, double b, double c,
                   double alpha, double beta, double gamma,
                   HklUnitEnum unit_type, GError **error);
void hkl_lattice_get(const HklLattice *self,
                    double *a, double *b, double *c,
                    double *alpha, double *beta, double *gamma,
                    HklUnitEnum unit_type);
```

**Matrix Operations:**
```c
int hkl_lattice_get_B(const HklLattice *self, HklMatrix *B);
int hkl_lattice_get_1_B(const HklLattice *self, HklMatrix *B);
int hkl_lattice_reciprocal(const HklLattice *self, HklLattice *reciprocal);
```

### HklSampleReflection
Individual reflection data.

```c
typedef struct _HklSampleReflection HklSampleReflection;
```

**Creation:**
```c
HklSampleReflection *hkl_sample_reflection_new(const HklGeometry *geometry,
                                              const HklDetector *detector,
                                              double h, double k, double l,
                                              GError **error);
```

**HKL Indices:**
```c
void hkl_sample_reflection_hkl_get(const HklSampleReflection *self,
                                  double *h, double *k, double *l);
int hkl_sample_reflection_hkl_set(HklSampleReflection *self,
                                 double h, double k, double l,
                                 GError **error);
```

**Properties:**
```c
int hkl_sample_reflection_flag_get(const HklSampleReflection *self);
void hkl_sample_reflection_flag_set(HklSampleReflection *self, int flag);
const HklGeometry *hkl_sample_reflection_geometry_get(HklSampleReflection *self);
void hkl_sample_reflection_geometry_set(HklSampleReflection *self,
                                       const HklGeometry *geometry);
```

## Engine System

### HklEngine
Abstract base class for pseudo-axis calculations.

```c
typedef struct _HklEngine HklEngine;
```

**Basic Properties:**
```c
const char *hkl_engine_name_get(const HklEngine *self);
unsigned int hkl_engine_len(const HklEngine *self);
```

**Pseudo-Axis Management:**
```c
const darray_string *hkl_engine_pseudo_axis_names_get(HklEngine *self);
int hkl_engine_pseudo_axis_values_get(HklEngine *self,
                                     double values[], size_t n_values,
                                     HklUnitEnum unit_type, GError **error);
HklGeometryList *hkl_engine_pseudo_axis_values_set(HklEngine *self,
                                                  double values[], size_t n_values,
                                                  HklUnitEnum unit_type, GError **error);
const HklParameter *hkl_engine_pseudo_axis_get(const HklEngine *self,
                                              const char *name,
                                              GError **error);
```

**Engine Capabilities:**
```c
typedef enum _HklEngineCapabilities {
    HKL_ENGINE_CAPABILITIES_READABLE = 1u << 0,
    HKL_ENGINE_CAPABILITIES_WRITABLE = 1u << 1,
    HKL_ENGINE_CAPABILITIES_INITIALIZABLE = 1u << 2,
} HklEngineCapabilities;

unsigned int hkl_engine_capabilities_get(const HklEngine *self);
int hkl_engine_initialized_get(const HklEngine *self);
int hkl_engine_initialized_set(HklEngine *self, int initialized, GError **error);
```

**Mode Management:**
```c
const darray_string *hkl_engine_modes_names_get(const HklEngine *self);
const char *hkl_engine_current_mode_get(const HklEngine *self);
int hkl_engine_current_mode_set(HklEngine *self, const char *name, GError **error);
```

**Parameter Management:**
```c
typedef enum _HklEngineAxisNamesGet {
    HKL_ENGINE_AXIS_NAMES_GET_READ,
    HKL_ENGINE_AXIS_NAMES_GET_WRITE,
} HklEngineAxisNamesGet;

const darray_string *hkl_engine_axis_names_get(const HklEngine *self,
                                              HklEngineAxisNamesGet mode);
const darray_string *hkl_engine_parameters_names_get(const HklEngine *self);
const HklParameter *hkl_engine_parameter_get(const HklEngine *self, const char *name,
                                            GError **error);
int hkl_engine_parameter_set(HklEngine *self,
                            const char *name, const HklParameter *parameter,
                            GError **error);
void hkl_engine_parameters_values_get(const HklEngine *self,
                                     double values[], size_t n_values,
                                     HklUnitEnum unit_type);
int hkl_engine_parameters_values_set(HklEngine *self,
                                    double values[], size_t n_values,
                                    HklUnitEnum unit_type, GError **error);
```

**Dependencies:**
```c
typedef enum _HklEngineDependencies {
    HKL_ENGINE_DEPENDENCIES_AXES = 1u << 0,
    HKL_ENGINE_DEPENDENCIES_ENERGY = 1u << 1,
    HKL_ENGINE_DEPENDENCIES_SAMPLE = 1u << 2,
} HklEngineDependencies;

unsigned int hkl_engine_dependencies_get(const HklEngine *self);
```

**Utility Functions:**
```c
void hkl_engine_fprintf(FILE *f, const HklEngine *self);
```

### HklEngineList
Container for multiple engines.

```c
typedef struct _HklEngineList HklEngineList;
typedef darray(HklEngine *) darray_engine;
```

**Management:**
```c
void hkl_engine_list_free(HklEngineList *self);
darray_engine *hkl_engine_list_engines_get(HklEngineList *self);
```

**Geometry Association:**
```c
HklGeometry *hkl_engine_list_geometry_get(HklEngineList *self);
int hkl_engine_list_geometry_set(HklEngineList *self, const HklGeometry *geometry);
int hkl_engine_list_select_solution(HklEngineList *self,
                                   const HklGeometryListItem *item);
```

**Engine Access:**
```c
HklEngine *hkl_engine_list_engine_get_by_name(HklEngineList *self,
                                             const char *name,
                                             GError **error);
```

**Initialization:**
```c
void hkl_engine_list_init(HklEngineList *self,
                         HklGeometry *geometry,
                         HklDetector *detector,
                         HklSample *sample);
int hkl_engine_list_get(HklEngineList *self);
```

**Parameter Management:**
```c
const darray_string *hkl_engine_list_parameters_names_get(const HklEngineList *self);
const HklParameter *hkl_engine_list_parameter_get(const HklEngineList *self, const char *name,
                                                 GError **error);
int hkl_engine_list_parameter_set(HklEngineList *self,
                                 const char *name, const HklParameter *parameter,
                                 GError **error);
void hkl_engine_list_parameters_values_get(const HklEngineList *self,
                                          double values[], size_t n_values,
                                          HklUnitEnum unit_type);
int hkl_engine_list_parameters_values_set(HklEngineList *self,
                                         double values[], size_t n_values,
                                         HklUnitEnum unit_type, GError **error);
```

**Utility Functions:**
```c
void hkl_engine_list_fprintf(FILE *f, const HklEngineList *self);
```

## Factory System

### HklFactory
Factory for creating geometries and engines.

```c
typedef struct _HklFactory HklFactory;
```

**Factory Discovery:**
```c
HklFactory **hkl_factory_get_all(size_t *n);
HklFactory *hkl_factory_get_by_name(const char *name, GError **error);
```

**Factory Properties:**
```c
const char *hkl_factory_name_get(const HklFactory *self);
```

**Object Creation:**
```c
HklGeometry *hkl_factory_create_new_geometry(const HklFactory *self);
HklEngineList *hkl_factory_create_new_engine_list(const HklFactory *self);
```

## Constants and Macros

### Mathematical Constants
```c
#define HKL_TINY 1e-7
#define HKL_EPSILON 1e-6
#define HKL_DEGTORAD (M_PI/180.)
#define HKL_RADTODEG (180./M_PI)
#define HKL_TAU (2. * M_PI)
```

### Version Information
```c
#define HKL_VERSION "@PACKAGE_VERSION@"
```

### Function Attributes
```c
#define HKLAPI __attribute__ ((visibility("default")))
#define HKL_ARG_NONNULL(...) __attribute__ ((__nonnull__(__VA_ARGS__)))
#define HKL_WARN_UNUSED_RESULT __attribute__ ((__warn_unused_result__))
```

### Deprecation Macros
```c
#define HKL_DEPRECATED __attribute__((__deprecated__))
#define HKL_DEPRECATED_FOR(f) __attribute__((__deprecated__("Use '" #f "' instead")))
```

## Error Handling

### Error Propagation
All functions that can fail take a `GError **error` parameter:

```c
GError *error = NULL;
HklGeometry *geom = some_function(&error);
if (error != NULL) {
    // Handle error
    g_error_free(error);
    return;
}
```

### Error Domains
Errors are organized into domains for better handling:

- Geometry errors
- Engine errors  
- Sample errors
- Parameter errors
- Factory errors

## Usage Examples

### Basic Geometry Setup
```c
#include <hkl.h>

int main() {
    GError *error = NULL;
    
    // Create factory and geometry
    HklFactory *factory = hkl_factory_get_by_name("E4CV", &error);
    if (error) {
        g_error_free(error);
        return 1;
    }
    
    HklGeometry *geometry = hkl_factory_create_new_geometry(factory);
    
    // Set wavelength
    hkl_geometry_wavelength_set(geometry, 1.0, HKL_UNIT_DEFAULT, &error);
    
    // Cleanup
    hkl_geometry_free(geometry);
    
    return 0;
}
```

### Sample and UB Matrix
```c
// Create sample with lattice
HklSample *sample = hkl_sample_new("my_crystal");
HklLattice *lattice = hkl_lattice_new(10.0, 10.0, 10.0, 90.0, 90.0, 90.0, &error);
hkl_sample_lattice_set(sample, lattice);

// Create UB matrix
HklMatrix *UB = hkl_matrix_new_full(/* UB matrix elements */);
hkl_sample_UB_set(sample, UB, &error);
```

### Engine Calculations
```c
// Create engine list
HklEngineList *engines = hkl_factory_create_new_engine_list(factory);
hkl_engine_list_init(engines, geometry, detector, sample);

// Get HKL engine
HklEngine *hkl_engine = hkl_engine_list_engine_get_by_name(engines, "hkl", &error);

// Set HKL values
double hkl_values[] = {1.0, 0.0, 0.0};
HklGeometryList *solutions = hkl_engine_pseudo_axis_values_set(hkl_engine, 
                                                              hkl_values, 3,
                                                              HKL_UNIT_DEFAULT, &error);
```

This API reference provides coverage of the HKL library's public interface. For more detailed examples and usage patterns, refer to the test suite and examples included with the library.
