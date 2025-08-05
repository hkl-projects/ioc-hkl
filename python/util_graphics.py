import numpy as np
import pandas as pd
from tqdm import tqdm
from util import sci2dec
import math
import gi
from gi.repository import GLib
gi.require_version('Hkl', '5.0')
from gi.repository import Hkl
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import subprocess
import os.path
# in shell: source /epics/iocs/ioc-hkl/iochkl/bin/activate
# export GI_TYPELIB_PATH=/usr/local/lib/girepository-1.0


# Detector peak positions
def real2det_curved_e6c(gamma_axis, delta_axis, s_gamma, s_delta, R, cyl_center, ray_origin):
    #TODO use gamma/delta axes instead of manually flipping
    delta = np.deg2rad(s_delta)
    z_hit = R*np.tan(delta)
    return [s_gamma, z_hit]

def real2det_flat_e6c(gamma_axis, delta_axis, s_gamma, s_delta, R, cyl_center, ray_origin):
    #x_center = -120 # 60 degrees from minumum rotation -180
    #s_delta = s_delta+120 # 0 point is at -120 for left side of detector starting at -180 and having width 120
    gamma = np.deg2rad(s_delta)
    z_hit = R*np.tan(delta)
    delta = np.deg2rad(s_gamma)
    x_hit = R*np.tan(gamma)
    return [-x_hit, z_hit]



# parse hkl file, format into dataframe
def hkl2dfhkl(hkl_path):
    lattice = {}
    with open(hkl_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                continue
            tokens = line.strip().split()
            if len(tokens) < 3:
                continue
            key = tokens[1]
            if key == 'lattice_a':
                lattice['a'] = float(tokens[2])
            elif key == 'lattice_b':
                lattice['b'] = float(tokens[2])
            elif key == 'lattice_c':
                lattice['c'] = float(tokens[2])
            elif key == 'lattice_aa':
                lattice['alpha'] = float(tokens[2])
            elif key == 'lattice_bb':
                lattice['beta'] = float(tokens[2])
            elif key == 'lattice_cc':
                lattice['gamma'] = float(tokens[2])
    with open(hkl_path, "r") as f:
        lines = f.readlines()
    intensity_lines = []
    found_data_start = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# H") and "|Fc|^2" in line:
            found_data_start = True
            continue
        if not found_data_start or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 6:
            h, k, l, mult, d, intensity_sci = parts[:6]
            intensity_dec = sci2dec(intensity_sci)
            intensity_lines.append("%3s %3s %3s %12s %8s" % (h, k, l, d, intensity_dec))
    rows = [line.split() for line in intensity_lines]
    df = pd.DataFrame(rows, columns=['h', 'k', 'l', 'd', 'intensity'])
    df = df.astype({
        'h': int,
        'k': int,
        'l': int,
        'd': float,
        'intensity': float
    })
    return lattice, df

# save diffractometer peak positions to df
def dfhkl2dfhklaxes_e6c(df, min_intensity, factory, geometry, detector, sample, user):
    rows = []
    new_df = pd.DataFrame(columns=['h', 'k', 'l', 'mu', 'omega', 'chi', 'phi', 'gamma','delta', 'd', 'intensity']) 
    engines = factory.create_new_engine_list()
    engines.init(geometry, detector, sample)
    engines.get()
    engine_hkl = engines.engine_get_by_name("hkl")
    engine_hkl.current_mode_set('lifting_detector_omega') # TODO CHECK THIS
    #engine_hkl.current_mode_set('lifting_detector_mu') # TODO CHECK THIS
    axes = geometry.axis_names_get()
    #for axis in axes:
    #    tmp = geometry.axis_get(axis)
    #    #if (axis=='mu') or (axis=='chi') or (axis=='phi'):
    #    if (axis=='omega') or (axis=='chi') or (axis=='phi'):
    #        tmp.min_max_set(-0.01, 0.01, user)
    #        geometry.axis_set(axis, tmp)
    found = 0
    not_found = 0
    total_num_refl = len(df)
    df = df[df['intensity']>min_intensity]
    num_refl = len(df)
    print(f'total reflections: {total_num_refl}\nreflections filtered by intensity: {num_refl}')
    print(f"Searching through {num_refl} reflections...")
    #TODO get tqdm progress bar into CSS, like IOC error messages
    for refl in tqdm(df.itertuples(index=False), total=num_refl):
        h = refl.h
        k = refl.k
        l = refl.l
        d = refl.d
        inten = refl.intensity
        try:
            solutions = engine_hkl.pseudo_axis_values_set([h,k,l], user)
            # similar to apply_axes_solns in hkl.py
            for i, item in enumerate(solutions.items()):
                read = item.geometry_get().axis_values_get(user)
                if read is not None:
                    rows.append({'h':h, \
                                 'k':k, \
                                 'l':l, \
                                 'd':d, \
                                 'intensity':inten, \
                                 'mu':read[0], \
                                 'omega':read[1], \
                                 'chi':read[2], \
                                 'phi':read[3], \
                                 'gamma':read[4], \
                                 'delta':read[5]})
                    found += 1
        except Exception as e:
            #print(f"Exception for hkl=({h},{k},{l}): {e}")
            not_found += 1
    new_df = pd.DataFrame(rows, columns=['h', 'k', 'l', 'mu', 'omega', 'chi', 'phi', 'gamma', 'delta', 'd', 'intensity'])
    foundrefl = num_refl-not_found
    print(f"found {found} motor positions in {foundrefl} reflections. Did not find positions for {not_found} reflections.")
    print("Completed dfhkl2dfhklaxes. Output DataFrame has %d rows", len(new_df))
    #print(f'{new_df}')
    #new_df.to_csv('test.csv')
    if new_df is not None:
        return new_df
    else:
        print("empty dataframe, something went wrong")
        return

# search for diffractometer peak positions
def intensities2detint_e6c(cif_path, hkl_path, wavelength, UB, min_intensity, R, geom, zmin, zmax, det_shape):
    #print(f"det shape: {det_shape}")
    cyl_center = (0,0)
    ray_origin = np.array([0,0,0])
    gamma_axis = [0,0,-1]
    delta_axis = [0,-1,0]
    lst = []
    #generate hkl file with given cif file, wavelength
    #TODO check if hkl file exists before generating
    if not os.path.isfile(hkl_path):
        cif2hkl_bin = '/usr/bin/cif2hkl'
        cmd = [cif2hkl_bin, '--mode', 'NUC', '--out', hkl_path, '--lambda', str(wavelength), '--xtal', cif_path]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()

    # go from hkl file output by cif2hkl to a dataframe of reflections/intensities
    latt, df = hkl2dfhkl(hkl_path)
    #print(latt)

    a = latt['a']
    b = latt['b']
    c = latt['c']
    alpha = latt['alpha']
    beta = latt['beta']
    gamma = latt['gamma']

    user = Hkl.UnitEnum.USER
    detector = Hkl.Detector.factory_new(Hkl.DetectorType(0))
    factory  = Hkl.factories()[geom]
    geometry = factory.create_new_geometry()
    geometry.wavelength_set(wavelength, Hkl.UnitEnum.USER)
    sample = Hkl.Sample.new("toto") # sample. tab to check attributes

    alpha = math.radians(alpha)
    beta  = math.radians(beta)
    gamma = math.radians(gamma)
    lattice = Hkl.Lattice.new(a,b,c,alpha,beta,gamma)
    
    sample.lattice_set(lattice)
    #UB_temp = sample.UB_get()
    #Hkl.Matrix.init(UB_temp, UB)
    #Hkl.Matrix.init(UB_temp, *UB.ravel())
    try:
        sample.UB_set(UB)
    except Exception as e:
        print(f"UB set error: {e}")

    # add columns for real axes motor positions to reflection df
    df2 = dfhkl2dfhklaxes_e6c(df, min_intensity, factory, geometry, detector, sample, user)
    #print(f"DF2 {df2}")
    theta, z, intensities = [], [], []
    #df2.to_csv('refls2.csv')
    for idx, refl in df2.iterrows():
        mu = refl['mu']
        omega = refl['omega']
        chi = refl['chi']
        phi = refl['phi']
        gamma = refl['gamma']
        delta = refl['delta']
        h = refl['h']
        k = refl['k']
        l = refl['l']
        inten = refl['intensity']
        if det_shape == 0: # curved
            detthetaz = real2det_curved_e6c(gamma_axis, delta_axis, gamma, delta, R, cyl_center, ray_origin)
        elif det_shape == 1: # flat
            detthetaz = real2det_flat_e6c(gamma_axis, delta_axis, gamma, delta, R, cyl_center, ray_origin)
        else:
            print("non valid detector shape")
            return
        #print(f"detthetaz: {detthetaz}")
        if (detthetaz is not None):
            theta = float(detthetaz[0])
            z = float(detthetaz[1])
            if (z<zmax) and (z>zmin):
                lst.append((theta, z, inten, h, k, l, mu, omega, chi, phi, gamma, delta))
    if lst is not []:
        #print(f"LST: {lst}")
        return lst
    else:
        print("no points found")
        return None

