import math
import numpy as np
#import pandas as pd
import gi
from gi.repository import GLib
gi.require_version('Hkl', '5.0')
from gi.repository import Hkl
import sys

class hklCalculator():
    def __init__(self, num_axes_solns=30, num_reflections = 10, geom=2, geom_name = 'E4CV'):
        # initials
        self.wavelength = 0.
        self.geom_name = geom_name
        self.geom = geom
        self.geometry = np.nan # hkl object placeholder
        self.detector = np.nan # hkl object placeholder
        self.factory = np.nan
        self.sample = np.nan # hkl object placeholder
        self.engines = np.nan # hkl object placeholder
        self.engine_hkl = np.nan # hkl object placeholder
        self.mode_4c = 0 # bissector, constant omega...
        self.mode_6c = 0 # bissector_vertical, constant_omega_vertical
        self.latt = [0., 0., 0., 0., 0., 0.] 
        # ^ [a1, a2, a3, alpha, beta, gamma], angstroms and radians
        self.lattice = np.nan 
        self.lattice_vol = 0.
        
        self.errors = 'test string'
        
        self.particle_type = 0
        self.neutron = 0.
        self.velocity = 0. # sqrt(2E/m) 
        self.e = 1.6021766300e-19 # [C]
        self.h = 6.6260701500e-34 # [m^2*kg/s]
        self.c = 299792458 # [m/s^2]
        self.m_neutron = 1.6749274710e-27 #[kg]
        self.m_proton = 1.6726219200e-27 #[kg]
        self.m_electron = 9.1093837e-27 #[kg]

        # sample orientation
        # initial 2 reflections
        self.num_reflections = num_reflections
        self.refl1_input_e4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl2_input_e4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl1_input_k4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl2_input_k4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl1_input_e6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.refl2_input_e6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.refl1_input_k6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.refl2_input_k6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.refl1 = np.nan
        self.refl2 = np.nan
        
        self.curr_num_refls = 0
 
        # refine with reflections
        self.refl_refine_input_e4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl_refine_input_k4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl_refine_input_e6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.refl_refine_input_k6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.refl_refine_input_list_e4c = []
        self.refl_refine_input_list_k4c = []
        self.refl_refine_input_list_e6c = []
        self.refl_refine_input_list_k6c = []

        for i in range(self.num_reflections):
            self.refl_refine_input_list_e4c.append([0., 0., 0., 0., 0., 0., 0.])
            self.refl_refine_input_list_k4c.append([0., 0., 0., 0., 0., 0., 0.])
            self.refl_refine_input_list_e6c.append([0., 0., 0., 0., 0., 0., 0., 0., 0.])
            self.refl_refine_input_list_k6c.append([0., 0., 0., 0., 0., 0., 0., 0., 0.])

        self.refl_refine_e4c = np.nan
        self.refl_refine_k4c = np.nan
        self.refl_refine_e6c = np.nan
        self.refl_refine_k6c = np.nan

        self.refl_refine_list_e4c = []
        self.refl_refine_list_k4c = []
        self.refl_refine_list_e6c = []
        self.refl_refine_list_k6c = []

        self.selected_refl = [] # used for deleting reflection from list
        self.latt_refine = [0., 0., 0., 0., 0., 0.]
        # UB
        self.UB_matrix = np.zeros((3,3), dtype=float)
        self.UB_matrix_input = np.zeros((3,3), dtype=float)

        #self.sample_rot_matrix = np.zeros((8,8), dtype=float)
        self.u_matrix = np.zeros((3,3), dtype=float)
        
        # U vector
        self.ux = 0.
        self.uy = 0.
        self.uz = 0.

        # UB busing levy
        self.UB_matrix_bl = np.zeros((3,3), dtype=float)

        # UB simplex
        self.UB_matrix_simplex = np.zeros((3,3), dtype=float)
        
        ### axes
        self.num_axes_solns = num_axes_solns

        # Eulerian 4-circle (omega, chi, phi, tth)
        self.axes_e4c = [0.,0.,0.,0.]

        # Kappa 4-circle (komega, kappa, kphi, tth)
        self.axes_k4c = [0.,0.,0.,0.]

        # Eulerian 6-circle (mu, omega, chi, phi, gamma, delta)
        self.axes_e6c = [0.,0.,0.,0.,0.,0.]

        # Kappa 6-circle (mu, komega, kappa, kphi, gamma, delta)
        self.axes_k4c = [0.,0.,0.,0.,0.,0.]

        #TODO move to arrays for each geom
        ### axes for UB calculation - only used internally - avoids setting on calculation
        self.axes_UB_e4c = [0., 0., 0., 0.]
        self.axes_UB_k4c = [0., 0., 0., 0.]
        self.axes_UB_e6c = [0., 0., 0., 0., 0., 0.]
        self.axes_UB_k6c = [0., 0., 0., 0., 0., 0.]

        ### pseduoaxes 
        self.pseudoaxes_h = 0.
        self.pseudoaxes_k = 0.
        self.pseudoaxes_l = 0.
        self.pseudoaxes_psi = 0.
        self.pseudoaxes_q = 0.
        self.pseudoaxes_incidence = 0.
        self.pseudoaxes_azimuth1 = 0.
        self.pseudoaxes_emergence = 0.
        self.pseudoaxes_azimuth2 = 0.
        self.pseudoaxes_omega = 0.
        self.pseudoaxes_chi = 0.
        self.pseudoaxes_phi = 0.
       
        ### axes solutions 
        # Eulerian 4-circle
        self.axes_solns_omega_e4c = []
        self.axes_solns_chi_e4c = []
        self.axes_solns_phi_e4c = []
        self.axes_solns_tth_e4c = []

        # Kappa 4-circle
        self.axes_solns_komega_k4c = []
        self.axes_solns_kappa_k4c = []
        self.axes_solns_kphi_k4c = []
        self.axes_solns_tth_k4c = []

        # Eulerian 6-circle
        self.axes_solns_mu_e6c = []
        self.axes_solns_omega_e6c = []
        self.axes_solns_chi_e6c = []
        self.axes_solns_phi_e6c = []
        self.axes_solns_gamma_e6c = []
        self.axes_solns_delta_e6c = []

        # Kappa 6-circle
        self.axes_solns_mu_k6c = []
        self.axes_solns_komega_k6c = []
        self.axes_solns_kappa_k6c = []
        self.axes_solns_kphi_k6c = []
        self.axes_solns_gamma_k6c = []
        self.axes_solns_delta_k6c = []

        for _ in range(self.num_axes_solns):
            # Eulerian 4-circle
            self.axes_solns_omega_e4c.append(0)
            self.axes_solns_chi_e4c.append(0)
            self.axes_solns_phi_e4c.append(0)
            self.axes_solns_tth_e4c.append(0)
            # Kappa 4-circle
            self.axes_solns_komega_k4c.append(0)
            self.axes_solns_kappa_k4c.append(0)
            self.axes_solns_kphi_k4c.append(0)
            self.axes_solns_tth_k4c.append(0)
            # Eulerian 6-circle
            self.axes_solns_mu_e6c.append(0)
            self.axes_solns_omega_e6c.append(0)
            self.axes_solns_chi_e6c.append(0)
            self.axes_solns_phi_e6c.append(0)
            self.axes_solns_gamma_e6c.append(0)
            self.axes_solns_delta_e6c.append(0)
            # Kappa 6-circle
            self.axes_solns_mu_k6c.append(0)
            self.axes_solns_komega_k6c.append(0)
            self.axes_solns_kappa_k6c.append(0)
            self.axes_solns_kphi_k6c.append(0)
            self.axes_solns_gamma_k6c.append(0)
            self.axes_solns_delta_k6c.append(0)
        
        # pseudoaxes solutions
        self.pseudoaxes_solns_h = 0.
        self.pseudoaxes_solns_k = 0.
        self.pseudoaxes_solns_l = 0.
        self.pseudoaxes_solns_psi = 0.
        self.pseudoaxes_solns_q = 0.
        self.pseudoaxes_solns_incidence = 0.
        self.pseudoaxes_solns_azimuth1 = 0.
        self.pseudoaxes_solns_emergence = 0.
        self.pseudoaxes_solns_azimuth2 = 0.
        self.pseudoaxes_solns_omega = 0.
        self.pseudoaxes_solns_chi = 0.
        self.pseudoaxes_solns_phi = 0.
        self.pseudoaxes_solns_tth = 0.
        self.pseudoaxes_solns_alpha = 0.
        self.pseudoaxes_solns_alpha2 = 0.
        self.pseudoaxes_solns_qper = 0.
        self.pseudoaxes_solns_qpar = 0.
 
    def start(self):
        self.detector = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        self.factory  = Hkl.factories()[self.geom_name]
        self.geometry = self.factory.create_new_geometry()
        self.geometry.wavelength_set(self.wavelength, Hkl.UnitEnum.USER)
        
        self.sample = Hkl.Sample.new("toto")
        a,b,c,alpha,beta,gamma = [i for i in self.latt]
        alpha=math.radians(alpha)
        beta=math.radians(beta)
        gamma=math.radians(gamma)
        self.lattice = Hkl.Lattice.new(a,b,c,alpha,beta,gamma)
        self.sample.lattice_set(self.lattice)             

        self.engines = self.factory.create_new_engine_list()
        self.engines.init(self.geometry, self.detector, self.sample)
        self.engines.get()
        self.engine_hkl = self.engines.engine_get_by_name("hkl")

        self.get_UB_matrix()    
        self.get_latt_vol()
        self.errors = f'{self.geom_name} started\n {self.get_info()}'

    def energy_to_wavelength_neutron(self):
        # from milli_ev to Angstrom
        if (self.particle_type == 0) and (self.energy > 0): # Xray
            # E [kev] = hc/lambda [m^2kgs^-2]
            coeff = 12.39841987 # hc [kev*A]
            self.wavelength = coeff/self.energy # hc/energy
            
        elif (self.particle_type == 1) and (self.energy > 0): # neutron
            # E [mev] = p^2/2m = h^2/(2m*lambda^2) [...]
            coeff = 81.804211883 # h^2/2m_neutron [mevA^2] for neutrons
            self.wavelength = math.sqrt(coeff/self.energy)

    def switch_geom(self):
        if self.geom == 0:
            print("switching to TwoC")
            self.geom_name = "TwoC"

        if self.geom == 1:
            print("switching to E4CH")
            self.geom_name = "E4CH"
            self.start()

        if self.geom == 2:
            print("switching to E4CV")
            self.geom_name = "E4CV"
            self.start()

        if self.geom == 3:
            print("switching to K4CV")
            self.geom_name = "K4CV"
            self.start()

        if self.geom == 4:
            print("switching to E6C")
            self.geom_name = "E6C"
            self.start()

        if self.geom == 5:
            print("switching to K6C")
            self.geom_name = "K6C"
            self.start()

        if self.geom == 6:
            print("switching to ZAXIS")
            self.geom_name = "ZAXIS"
            self.start()

    def forward(self):
        print("Forward function start")
        self.reset_pseudoaxes_solns()
        print(f"geom: {self.geom}")
        print(f"geom name: {self.geom_name}")
        if (self.geom_name == 'E4CH') or (self.geom_name == 'E4CV'):
            print("aaaaaaaaaaa")
            values_w = [float(self.axes_e4c[0]), \
                        float(self.axes_e4c[1]), \
                        float(self.axes_e4c[2]), \
                        float(self.axes_e4c[3])] 
            print(values_w)
        if self.geom_name == "K4CV":
            print("bbbbbbbbbbb")
            values_w = [float(self.axes_k4c[0]), \
                        float(self.axes_k4c[1]), \
                        float(self.axes_k4c[2]), \
                        float(self.axes_k4c[3])] 
            print(values_w)
        elif self.geom_name == "E6C":
            print("ccccccccccc")
            values_w = [float(self.axes_e6c[0]), \
                        float(self.axes_e6c[1]), \
                        float(self.axes_e6c[2]), \
                        float(self.axes_e6c[3]), \
                        float(self.axes_e6c[4]), \
                        float(self.axes_e6c[5])] 
            print(values_w)
        elif self.geom_name == "K6C":
            print("ddddddddddd")
            values_w = [float(self.axes_k6c[0]), \
                        float(self.axes_k6c[1]), \
                        float(self.axes_k6c[2]), \
                        float(self.axes_k6c[3]), \
                        float(self.axes_k6c[4]), \
                        float(self.axes_k6c[5])] 
            print(values_w)

        try:
            self.geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
        except:
            print("invalid axes values")
            #TODO catch different types of errors
            return

        self.engines.init(self.geometry, self.detector, self.sample)
        self.engines.get()
        self.engine_hkl = self.engines.engine_get_by_name("hkl")
        print(self.engine_hkl)
 
        values_hkl = self.engine_hkl.pseudo_axis_values_get(Hkl.UnitEnum.USER)
        print(values_hkl)
        self.pseudoaxes_solns_h, self.pseudoaxes_solns_k, self.pseudoaxes_solns_l = values_hkl
        self.get_UB_matrix()

    def forward_UB(self):
        print("Forward UB function start")
        if self.geom == 1 or 2:
            values_w = [float(self.axes_UB_e4c[0]), \
                        float(self.axes_UB_e4c[1]), \
                        float(self.axes_UB_e4c[2]), \
                        float(self.axes_UB_e4c[3])] 
        elif self.geom == 3:
            values_w = [float(self.axes_UB_k4c[0]), \
                        float(self.axes_UB_k4c[1]), \
                        float(self.axes_UB_k4c[2]), \
                        float(self.axes_UB_k4c[3])] 
        elif self.geom == 4:
            values_w = [float(self.axes_UB_e6c[0]), \
                        float(self.axes_UB_e6c[1]), \
                        float(self.axes_UB_e6c[2]), \
                        float(self.axes_UB_e6c[3]), \
                        float(self.axes_UB_e6c[4]), \
                        float(self.axes_UB_e6c[5])] 
        elif self.geom == 5:
            values_w = [float(self.axes_UB_k6c[0]), \
                        float(self.axes_UB_k6c[1]), \
                        float(self.axes_UB_k6c[2]), \
                        float(self.axes_UB_k6c[3]), \
                        float(self.axes_UB_k6c[4]), \
                        float(self.axes_UB_k6c[5])] 
        try:
            self.geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
        except:
            print("invalid axes values")
            return

    def backward(self):
        print("Backward function start")
        self.reset_axes_solns()
        # set mode
        if self.geom == (1 or 2 or 3):
            if self.mode_4c == 0:
                self.engine_hkl.current_mode_set('bissector')
            elif self.mode_4c == 1:
                self.engine_hkl.current_mode_set('constant_omega')
            elif self.mode_4c == 2:
                self.engine_hkl.current_mode_set('constant_chi')
            elif self.mode_4c == 3:
                self.engine_hkl.current_mode_set('constant_phi')
        elif self.geom == (4 or 5):
            if self.mode_6c == 0: 
                self.engine_hkl.current_mode_set('bissector_vertical')
            if self.mode_6c == 1: 
                self.engine_hkl.current_mode_set('constant_omega_vertical')
            if self.mode_6c == 2: 
                self.engine_hkl.current_mode_set('constant_chi_vertical')
            if self.mode_6c == 3: 
                self.engine_hkl.current_mode_set('constant_phi_vertical')

        values_hkl = [float(self.pseudoaxes_h), \
                      float(self.pseudoaxes_k), \
                      float(self.pseudoaxes_l)]

        solutions = self.engine_hkl.pseudo_axis_values_set(values_hkl, Hkl.UnitEnum.USER)
        values_w_all = []
        len_solns = len(solutions.items())
        for i, item in enumerate(solutions.items()):
            read = item.geometry_get().axis_values_get(Hkl.UnitEnum.USER)
            values_w_all.append(read)
        if len_solns > self.num_axes_solns: # truncate if above max available soln slots
            len_solns = self.num_axes_solns
        if self.geom == 1 or 2:
            for i in range(len_solns): 
                self.axes_solns_omega_e4c[i], \
                self.axes_solns_chi_e4c[i], \
                self.axes_solns_phi_e4c[i], \
                self.axes_solns_tth_e4c[i] = values_w_all[i]         
        elif self.geom == 3:
            for i in range(len_solns): 
                self.axes_solns_komega_k4c[i], \
                self.axes_solns_kappa_k4c[i], \
                self.axes_solns_kphi_k4c[i], \
                self.axes_solns_tth_k4c[i] = values_w_all[i]         
        elif self.geom == 4:
            for i in range(len_solns): 
                self.axes_solns_mu_e6c[i], \
                self.axes_solns_omega_e6c[i], \
                self.axes_solns_chi_e6c[i], \
                self.axes_solns_phi_e6c[i], \
                self.axes_solns_gamma_e6c[i], \
                self.axes_solns_delta_e6c[i] = values_w_all[i]       
        elif self.geom==5:  
             for i in range(len_solns): 
                self.axes_solns_mu_k6c[i], \
                self.axes_solns_komega_k6c[i], \
                self.axes_solns_kappa_k6c[i], \
                self.axes_solns_kphi_k6c[i], \
                self.axes_solns_gamma_k6c[i], \
                self.axes_solns_delta_k6c[i] = values_w_all[i]         
     
    def get_axes(self):
        if self.geom == 1 or 2:
            axes = (self.axes_e4c[0], \
                    self.axes_e4c[1], \
                    self.axes_e4c[2], \
                    self.axes_e4c[3])
        if self.geom == 3:
            axes = (self.axes_k4c[0], \
                    self.axes_k4c[1], \
                    self.axes_k4c[2], \
                    self.axes_k4c[3])
        if self.geom == 4:
            axes = (self.axes_e6c[0], \
                    self.axes_e6c[1], \
                    self.axes_e6c[2], \
                    self.axes_e6c[3], \
                    self.axes_e6c[4], \
                    self.axes_e6c[5])
        if self.geom == 5:
            axes = (self.axes_k6c[0], \
                    self.axes_k6c[1], \
                    self.axes_k6c[2], \
                    self.axes_k6c[3], \
                    self.axes_k6c[4], \
                    self.axes_k6c[5])
        print(axes)
        return axes 

    def get_pseudoaxes(self):
        pseudoaxes = (self.pseudoaxes_h, \
                      self.pseudoaxes_k, \
                      self.pseudoaxes_l)
        return pseudoaxes

    def get_axes_solns(self):
        axes = {}
        if self.geom == 1 or 2:
            axes['omega'] = self.axes_solns_omega_e4c
            axes['chi']   = self.axes_solns_chi_e4c
            axes['phi']   = self.axes_solns_phi_e4c
            axes['tth']   = self.axes_solns_tth_e4c
        elif self.geom == 3:
            axes['komega'] = self.axes_solns_komega_k4c
            axes['kappa']  = self.axes_solns_kappa_k4c
            axes['kphi']   = self.axes_solns_kphi_k4c
            axes['tth']    = self.axes_solns_tth_k4c
        elif self.geom == 4:
            axes['mu']     = self.axes_solns_mu_e6c
            axes['omega']  = self.axes_solns_omega_e6c
            axes['chi']    = self.axes_solns_chi_e6c
            axes['phi']    = self.axes_solns_phi_e6c
            axes['gamma']  = self.axes_solns_gamma_e6c
            axes['delta']  = self.axes_solns_delta_e6c
        elif self.geom == 5:
            axes['mu']     = self.axes_solns_mu_k6c
            axes['komega'] = self.axes_solns_komega_k6c
            axes['kappa']  = self.axes_solns_kappa_k6c
            axes['kphi']   = self.axes_solns_kphi_k6c
            axes['gamma']  = self.axes_solns_gamma_k6c
            axes['delta']  = self.axes_solns_delta_k6c
        return axes

    def get_pseudoaxes_solns(self):
        pseudoaxes_solns = (self.pseudoaxes_solns_h, \
                      self.pseudoaxes_solns_k, \
                      self.pseudoaxes_solns_l)
        return pseudoaxes_solns

    def reset_pseudoaxes_solns(self):
        self.pseudoaxes_solns_h = 0
        self.pseudoaxes_solns_k = 0
        self.pseudoaxes_solns_l = 0
        self.pseudoaxes_solns_psi = 0
        self.pseudoaxes_solns_q = 0
        self.pseudoaxes_solns_incidence = 0
        self.pseudoaxes_solns_azimuth1 = 0
        self.pseudoaxes_solns_emergence = 0
        self.pseudoaxes_solns_azimuth2 = 0

    def reset_axes_solns(self):
        if self.geom == 1 or 2:
            self.axes_solns_omega_e4c = []
            self.axes_solns_chi_e4c = []
            self.axes_solns_phi_e4c = []
            self.axes_solns_tth_e4c = []
            for _ in range(self.num_axes_solns):
                self.axes_solns_omega_e4c.append(0)
                self.axes_solns_chi_e4c.append(0)
                self.axes_solns_phi_e4c.append(0)
                self.axes_solns_tth_e4c.append(0)
        elif self.geom == 3:
            self.axes_solns_komega_k4c = []
            self.axes_solns_kappa_k4c = []
            self.axes_solns_kphi_k4c = []
            self.axes_solns_tth_k4c = []
            for _ in range(self.num_axes_solns):
                self.axes_solns_komega_k4c.append(0)
                self.axes_solns_kappa_k4c.append(0)
                self.axes_solns_kphi_k4c.append(0)
                self.axes_solns_tth_k4c.append(0)
        elif self.geom == 4:
            self.axes_solns_mu_e6c = []
            self.axes_solns_omega_e6c = []
            self.axes_solns_chi_e6c = []
            self.axes_solns_phi_e6c = []
            self.axes_solns_gamma_e6c = []
            self.axes_solns_delta_e6c = []
            for _ in range(self.num_axes_solns):
                self.axes_solns_mu_e6c.append(0)
                self.axes_solns_omega_e6c.append(0)
                self.axes_solns_chi_e6c.append(0)
                self.axes_solns_phi_e6c.append(0)
                self.axes_solns_gamma_e6c.append(0)
                self.axes_solns_delta_e6c.append(0)
        elif self.geom == 5:
            self.axes_solns_mu_k6c = []
            self.axes_solns_komega_k6c = []
            self.axes_solns_kappa_k6c = []
            self.axes_solns_kphi_k6c = []
            self.axes_solns_gamma_k6c = []
            self.axes_solns_delta_k6c = []
            for _ in range(self.num_axes_solns):
                self.axes_solns_mu_k6c.append(0)
                self.axes_solns_komega_k6c.append(0)
                self.axes_solns_kappa_k6c.append(0)
                self.axes_solns_kphi_k6c.append(0)
                self.axes_solns_gamma_k6c.append(0)
                self.axes_solns_delta_k6c.append(0)


    def add_reflection1(self):
        '''
        adds reflection #1 to sample for busing-levy calculation
        '''
        if self.geom == 1 or 2:
            self.axes_UB_e4c[0] = self.refl1_input_e4c[3]
            self.axes_UB_e4c[1] = self.refl1_input_e4c[4]
            self.axes_UB_e4c[2] = self.refl1_input_e4c[5]
            self.axes_UB_e4c[3] = self.refl1_input_e4c[6]
        elif self.geom == 3:
            self.axes_UB_k4c[0] = self.refl1_input_k4c[3]
            self.axes_UB_k4c[1] = self.refl1_input_k4c[4]
            self.axes_UB_k4c[2] = self.refl1_input_k4c[5]
            self.axes_UB_k4c[3] = self.refl1_input_k4c[6]
        elif self.geom == 4:
            self.axes_UB_e6c[0] = self.refl1_input_e6c[3]
            self.axes_UB_e6c[1] = self.refl1_input_e6c[4]
            self.axes_UB_e6c[2] = self.refl1_input_e6c[5]
            self.axes_UB_e6c[3] = self.refl1_input_e6c[6]
            self.axes_UB_e6c[4] = self.refl1_input_e6c[7]
            self.axes_UB_e6c[5] = self.refl1_input_e6c[8]
        elif self.geom == 5:
            self.axes_UB_k6c[0] = self.refl1_input_k6c[3]
            self.axes_UB_k6c[1] = self.refl1_input_k6c[4]
            self.axes_UB_k6c[2] = self.refl1_input_k6c[5]
            self.axes_UB_k6c[3] = self.refl1_input_k6c[6]
            self.axes_UB_k6c[4] = self.refl1_input_k6c[7]
            self.axes_UB_k6c[5] = self.refl1_input_k6c[8]
        self.forward_UB()
        if self.geom == 1 or 2:
            self.refl1 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl1_input_e4c[0], \
                                                    self.refl1_input_e4c[1], \
                                                    self.refl1_input_e4c[2])
        if self.geom == 3:
            self.refl1 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl1_input_k4c[0], \
                                                    self.refl1_input_k4c[1], \
                                                    self.refl1_input_k4c[2])
        if self.geom == 4:
            self.refl1 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl1_input_e6c[0], \
                                                    self.refl1_input_e6c[1], \
                                                    self.refl1_input_e6c[2])
        if self.geom == 5:
            self.refl1 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl1_input_k6c[0], \
                                                    self.refl1_input_k6c[1], \
                                                    self.refl1_input_k6c[2])

    def add_reflection2(self):
        '''
        adds reflection #2 to sample for busing levy calculation
        '''
        if self.geom == 1 or 2:
            self.axes_UB_e4c[0] = self.refl2_input_e4c[3]
            self.axes_UB_e4c[1] = self.refl2_input_e4c[4]
            self.axes_UB_e4c[2] = self.refl2_input_e4c[5]
            self.axes_UB_e4c[3] = self.refl2_input_e4c[6]
        elif self.geom == 3:
            self.axes_UB_k4c[0] = self.refl2_input_k4c[3]
            self.axes_UB_k4c[1] = self.refl2_input_k4c[4]
            self.axes_UB_k4c[2] = self.refl2_input_k4c[5]
            self.axes_UB_k4c[3] = self.refl2_input_k4c[6]
        elif self.geom == 4:
            self.axes_UB_e6c[0] = self.refl2_input_e6c[3]
            self.axes_UB_e6c[1] = self.refl2_input_e6c[4]
            self.axes_UB_e6c[2] = self.refl2_input_e6c[5]
            self.axes_UB_e6c[3] = self.refl2_input_e6c[6]
            self.axes_UB_e6c[4] = self.refl2_input_e6c[7]
            self.axes_UB_e6c[5] = self.refl2_input_e6c[8]
        elif self.geom == 5:
            self.axes_UB_k6c[0] = self.refl2_input_k6c[3]
            self.axes_UB_k6c[1] = self.refl2_input_k6c[4]
            self.axes_UB_k6c[2] = self.refl2_input_k6c[5]
            self.axes_UB_k6c[3] = self.refl2_input_k6c[6]
            self.axes_UB_k6c[4] = self.refl2_input_k6c[7]
            self.axes_UB_k6c[5] = self.refl2_input_k6c[8]
        self.forward_UB()
        # Hkl.SampleReflection(self.geometry, self.detector, h, k, l)
        if self.geom == 1 or 2:
            self.refl2 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl2_input_e4c[0], \
                                                    self.refl2_input_e4c[1], \
                                                    self.refl2_input_e4c[2])
        elif self.geom == 3:
            self.refl2 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl2_input_k4c[0], \
                                                    self.refl2_input_k4c[1], \
                                                    self.refl2_input_k4c[2])
        elif self.geom == 4:
            self.refl2 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl2_input_e6c[0], \
                                                    self.refl2_input_e6c[1], \
                                                    self.refl2_input_e6c[2])
        elif self.geom == 5:
            self.refl2 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl2_input_k6c[0], \
                                                    self.refl2_input_k6c[1], \
                                                    self.refl2_input_k6c[2])
 

    def compute_UB_matrix(self):
        '''
        busing-levy UB calculation
        '''
        print("Computing UB matrix")
        self.add_reflection1()
        self.add_reflection2()
        self.sample.compute_UB_busing_levy(self.refl1, self.refl2)
        UB = self.sample.UB_get()
        for i in range(3):
            for j in range(3):
                self.UB_matrix_bl[i,j] = UB.get(i,j)
        self.start() # Reinitializes sample with lattice parameters

    def compute_set_UB_matrix(self):
        '''
        same thing as compute_UB_matrix, but without start()
        '''
        self.add_reflection1()
        self.add_reflection2()
        self.sample.compute_UB_busing_levy(self.refl1, self.refl2)
        UB = self.sample.UB_get()
        for i in range(3):
            for j in range(3):
                self.UB_matrix_bl[i,j] = UB.get(i,j)
                self.UB_matrix[i,j] = UB.get(i,j)

    def get_UB_matrix(self):
        UB = self.sample.UB_get()
        for i in range(3):
            for j in range(3):
                self.UB_matrix[i,j] = UB.get(i,j)

    def set_input_UB(self):
        UB_temp = self.sample.UB_get()
        for i in range(3):
            for j in range(3):
                print(UB_temp.get(i,j))
        self.UB_matrix = self.UB_matrix_input    
        Hkl.Matrix.init(UB_temp, *self.UB_matrix.ravel())         
        for i in range(3):
            for j in range(3):
                print(UB_temp.get(i,j))
        self.sample.UB_set(UB_temp)
        self.get_UB_matrix()
        self.get_info()        

    def affine(self):
        '''
        takes in >2 reflections to refine lattice parameters and UB matrix
        '''
        try:
            self.sample.affine()
        except RuntimeError as e:
            self.errors = traceback.print_exc() # credit to Alex S
            
        self.start()
        UB = self.sample.UB_get()
        for i in range(3):
            for j in range(3):
                self.UB_matrix_simplex[i,j] = UB.get(i,j)
        self.latt_refine[0], self.latt_refine[1], self.latt_refine[2], \
            self.latt_refine[3], self.latt_refine[4], self.latt_refine[5] = \
            self.lattice.get(Hkl.UnitEnum.USER)        
        self.start()

    def affine_set(self):
        '''
        takes in >2 reflections to refine lattice parameters and UB matrix
        '''
        self.sample.affine()
        UB = self.sample.UB_get()
        for i in range(3):
            for j in range(3):
                self.UB_matrix_simplex[i,j] = UB.get(i,j)
                self.UB_matrix[i,j] = UB.get(i,j)
        self.latt[0], self.latt[1], self.latt[2], self.latt[3], self.latt[4], \
            self.latt[5] = self.lattice.get(Hkl.UnitEnum.USER)
        self.latt_refine[0], self.latt_refine[1], self.latt_refine[2], \
            self.latt_refine[3], self.latt_refine[4], self.latt_refine[5] = \
            self.lattice.get(Hkl.UnitEnum.USER)             
                
    def add_refl_refine(self):
        if self.geom == 1 or 2:
            self.axes_UB_e4c[0] = self.refl_refine_input_e4c[3]
            self.axes_UB_e4c[1] = self.refl_refine_input_e4c[4]
            self.axes_UB_e4c[2] = self.refl_refine_input_e4c[5]
            self.axes_UB_e4c[3] = self.refl_refine_input_e4c[6]
        elif self.geom == 3:
            self.axes_UB_k4c[0] = self.refl_refine_input_k4c[3]
            self.axes_UB_k4c[1] = self.refl_refine_input_k4c[4]
            self.axes_UB_k4c[2] = self.refl_refine_input_k4c[5]
            self.axes_UB_k4c[3] = self.refl_refine_input_k4c[6]
        elif self.geom == 4:
            self.axes_UB_e6c[0] = self.refl_refine_input_e6c[3]
            self.axes_UB_e6c[1] = self.refl_refine_input_e6c[4]
            self.axes_UB_e6c[2] = self.refl_refine_input_e6c[5]
            self.axes_UB_e6c[3] = self.refl_refine_input_e6c[6]
            self.axes_UB_e6c[4] = self.refl_refine_input_e6c[7]
            self.axes_UB_e6c[5] = self.refl_refine_input_e6c[8]
        elif self.geom == 5:
            self.axes_UB_k6c[0] = self.refl_refine_input_e6c[3]
            self.axes_UB_k6c[1] = self.refl_refine_input_e6c[4]
            self.axes_UB_k6c[2] = self.refl_refine_input_e6c[5]
            self.axes_UB_k6c[3] = self.refl_refine_input_e6c[6]
            self.axes_UB_k6c[4] = self.refl_refine_input_e6c[7]
            self.axes_UB_k6c[5] = self.refl_refine_input_e6c[8]
        self.forward_UB()
        if self.geom == 1 or 2:
            self.refl_refine = self.sample.add_reflection(self.geometry, \
                self.detector, self.refl_refine_input_e4c[0], \
                self.refl_refine_input_e4c[1], self.refl_refine_input_e4c[2])   
            #self.refl_refine_list.append(self.refl_refine)
            self.refl_refine_input_list_e4c[self.curr_num_refls] = self.refl_refine_input_e4c.copy()
        if self.geom == 3:
            self.refl_refine = self.sample.add_reflection(self.geometry, \
                self.detector, self.refl_refine_input_k4c[0], \
                self.refl_refine_input_k4c[1], self.refl_refine_input_k4c[2])   
            self.refl_refine_input_list_k4c[self.curr_num_refls] = self.refl_refine_input_k4c.copy()
        if self.geom == 4:
            self.refl_refine = self.sample.add_reflection(self.geometry, \
                self.detector, self.refl_refine_input_e6c[0], \
                self.refl_refine_input_e6c[1], self.refl_refine_input_e6c[2])   
            self.refl_refine_input_list_e6c[self.curr_num_refls] = self.refl_refine_input_e6c.copy()
        if self.geom == 5:
            self.refl_refine = self.sample.add_reflection(self.geometry, \
                self.detector, self.refl_refine_input_k6c[0], \
                self.refl_refine_input_k6c[1], self.refl_refine_input_k6c[2])   
            self.refl_refine_input_list_k6c[self.curr_num_refls] = self.refl_refine_input_k6c.copy()
        self.curr_num_refls += 1

    #def del_refl_refine(self):
    #    self.selected_refl
    #    self.refl_refine = self.sample.del_reflection(self.geometry, \
    #                self.detector, self.refl_refine_input[0], \
    #                self.refl_refine_input[1], self.refl_refine[2])   
    #    self.refl_refine_list.append(self.refl.refine)

    def get_sample_rotation(self):
        rot = self.geometry.sample_rotation_get(self.sample).to_matrix()
        dim = len(self.sample_rot_matrix)
        for i in range(dim):
            for j in range(dim):
                self.sample_rot_matrix[i,j] = rot.get(i,j)     
        return self.sample_rot_matrix  

    def get_u_matrix(self):
        rot = self.sample.U_get()
        dim = len(self.u_matrix)
        for i in range(dim):
            for j in range(dim):
                self.u_matrix[i,j] = rot.get(i,j)     
        return self.u_matrix

    def get_u_xyz(self):
        self.ux = self.sample.ux_get().value_get(0)
        self.uy = self.sample.uy_get().value_get(0)
        self.uz = self.sample.uz_get().value_get(0)
        return self.ux, self.uy, self.uz

    def set_wavelength(self, wlen):
        self.wavelength = wlen

    def get_latt_vol(self):
        self.lattice_vol = self.lattice.volume_get().value_get(0)
        print(self.lattice_vol)

    def get_info(self):
        print(f'self.geom: {self.geom}')
        diff_geom = self.factory.name_get()
        print(diff_geom)
        samp = self.sample.lattice_get()
        a,b,c,alpha,beta,gamma = samp.get(Hkl.UnitEnum.USER)
        print(f'a: {a}, b: {b}, c: {c}, alpha: {alpha}, beta: {beta}, gamma: {gamma}')
        print(f'UB: {self.UB_matrix}')
        print(f'latt vol: {self.lattice_vol}')
        #x = self.engine_hkl.axis_names_get(Hkl.UnitEnum.USER)
        #x = self.engine_hkl.axis_names_get(Hkl.EngineAxisNamesGet.WRITE) ???
        # https://repo.or.cz/hkl.git/blob/5a20e852030a90f6920f4040fc437395bec3b532:/tests/bindings/python.py
        x = self.engine_hkl.axis_names_get(Hkl.EngineAxisNamesGet.READ)
        print(x)
