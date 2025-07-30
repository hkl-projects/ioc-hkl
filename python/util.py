import numpy as np
import math
import subprocess
import re

e = 1.6021766300e-19 # [C]
h = 6.6260701500e-34 # [m^2*kg/s]
c = 299792458 # [m/s^2]
eV2J = 1.602176634e-19 # 1eV = ... J
m2A = 1e10 # meter to angstrom
m_neutron = 1.6749274710e-27 #[kg]
m_proton = 1.6726219200e-27 #[kg]
m_electron = 9.1093837e-27 #[kg]
velocity = 0. # sqrt(2E/m) #TODO

def sci2dec(sci):
    dec = np.round(np.float64(sci), 1)
    return dec

def energy2wavelength_neutron(energy):
    # E = p^2/2m = h^2/(2m*lambda^2) 
    # lambda = sqrt(h^2/2*E*m_neutron) = h*sqrt(1/2*E*m_neutron) * m2A #converting to A
    if energy > 0:
        energy = eV2J*energy/1000 # input is meV not eV
        coeff = h*m2A
        wavelength = coeff*math.sqrt(1/(2*energy*m_neutron))
    else:
        wavelength = 0
    return wavelength

def energy2wavelength_xrays(energy):
    # E [kev] = hc/lambda [m^2kgs^-2]
    coeff = h*c #12.39841987 # hc [kev*A]
    if energy > 0:
        wavelength = coeff/energy # hc/energy
    else:
        wavelength = 0 
    return wavelength


def parse_cif_lattice_params(cif_path):
    lattice = {}
    pattern = re.compile(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)(?:\(\d+\))?")
    with open(cif_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('_cell_length_a'):
                match = pattern.search(line)
                if match:
                    lattice['a'] = float(match.group(1))
            elif line.startswith('_cell_length_b'):
                match = pattern.search(line)
                if match:
                    lattice['b'] = float(match.group(1))
            elif line.startswith('_cell_length_c'):
                match = pattern.search(line)
                if match:
                    lattice['c'] = float(match.group(1))
            elif line.startswith('_cell_angle_alpha'):
                match = pattern.search(line)
                if match:
                    lattice['alpha'] = float(match.group(1))
            elif line.startswith('_cell_angle_beta'):
                match = pattern.search(line)
                if match:
                    lattice['beta'] = float(match.group(1))
            elif line.startswith('_cell_angle_gamma'):
                match = pattern.search(line)
                if match:
                    lattice['gamma'] = float(match.group(1))
    return lattice


def intensity_calc(wavelength, cif_path):
    hkl_path = cif_path.replace(".cif", ".hkl")
    try:
        lattice = parse_cif_lattice_params(cif_path)
    except Exception as e:
        lattice = {}
        output = f'error: {e}'
        return output, lattice
    ##### scattering intensities calculation #####
    cif2hkl_bin = '/usr/bin/cif2hkl'
    cmd = [cif2hkl_bin, '--mode', 'NUC', '--out', hkl_path, '--lambda', str(wavelength), '--xtal', cif_path]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
    except Exception as e:
        print("run_cif (cif2hkl) error:", e)
        output = e
        return output, lattice
    try:
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
            if len(intensity_lines) >= 100:
                intensity_lines.append("Truncated, check generated .hkl file")
                break
    except Exception as e:
        intensity_lines = ["Error: " + str(e)]
    output = "\n".join(intensity_lines)
    return hkl_path, output, lattice

