import math
import numpy as np
import datetime
import gi
from gi.repository import GLib
gi.require_version('Hkl', '5.0')
from gi.repository import Hkl
from util import energy2wavelength_neutron, intensity_calc
from util_graphics import intensities2detint_e6c
from scipy.ndimage import gaussian_filter

class hklCalculator():
    def __init__(self, num_axes_solns=30, num_reflections = 10, geom=1, geom_name = 'E4CV'):
        # initials
        self.wavelength = 0.
        self.geom_name = geom_name
        self.geom = geom
        self.geometry = np.nan # hkl object placeholder
        self.detector = np.nan # hkl object placeholder
        self.factory = np.nan # hkl object placeholder
        self.sample = np.nan # hkl object placeholder
        self.engines = np.nan # hkl object placeholder
        self.engine_hkl = np.nan # hkl object placeholder
        self.engine_psi = np.nan # hkl object placeholder
        self.engine_q = np.nan # hkl object placeholder
        self.engine_incidence = np.nan # hkl object placeholder
        self.engine_emergence = np.nan # hkl object placeholder
        self.engine_eulerians = np.nan # hkl object placeholder
        self.engine_psi = np.nan # hkl object placeholder
        self.engine_q2 = np.nan # hkl object placeholder
        self.engine_qper_qpar = np.nan # hkl object placeholder
        self.engine_tth2 = np.nan # hkl object placeholder
        self.mode_2c = 0 # bissector, constant omega...
        self.mode_4c = 0 # bissector, constant omega...
        self.mode_e6c = 0 # bissector_vertical, constant_omega_vertical
        self.mode_k6c = 0 # bissector_vertical, constant_omega_vertical
        self.latt = [0., 0., 0., 0., 0., 0.] 
        # ^ [a1, a2, a3, alpha, beta, gamma], angstroms and radians
        self.lattice = np.nan 
        self.lattice_vol = 0.
        
        ct = datetime.datetime.now().isoformat()
        self.errors = [ord(c) for c in str(ct)]
        self.intensities = ''
        self.cif_path = ''        
        self.hkl_path = ''        
        self.visfulllst = ''
        self.det2dvis = ''

        #graphics
        self.min_intensity = 0
        self.zmin = 0
        self.zmax = 0
        self.cur_angle = -120
        self.y_offset = 0

        self.energy = 0.
        self.wavelength_result = 0.
        #self.particle_type = 1 # 0 photon, 1 neutron, ... #TODO
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
        self.businglevyflag = 0
        self.refl1_input_e4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl2_input_e4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl1_input_k4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl2_input_k4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl1_input_e6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.refl2_input_e6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.refl1_input_k6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.refl2_input_k6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.] 
        self.refl1_input_2c = [0., 0., 0., 0., 0.]
        self.refl2_input_2c = [0., 0., 0., 0., 0.]
        self.refl1 = np.nan
        self.refl2 = np.nan
        
        # refine with reflections
        self.refl_refine_input_e4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl_refine_input_k4c = [0., 0., 0., 0., 0., 0., 0.]
        self.refl_refine_input_e6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.refl_refine_input_k6c = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.refl_refine_input_2c = [0., 0., 0., 0., 0.]
        self.refl_refine_input_list_e4c = [[0., 0., 0., 0., 0., 0., 0.] for _ in range(self.num_reflections)]
        self.refl_refine_input_list_k4c = [[0., 0., 0., 0., 0., 0., 0.] for _ in range(self.num_reflections)]
        self.refl_refine_input_list_e6c = [[0., 0., 0., 0., 0., 0., 0., 0., 0.] for _ in range(self.num_reflections)]
        self.refl_refine_input_list_k6c = [[0., 0., 0., 0., 0., 0., 0., 0., 0.] for _ in range(self.num_reflections)]
        self.refl_refine_input_list_2c = [[0., 0., 0., 0., 0.] for _ in range(self.num_reflections)]

        self.refl_list_e4c = [] #TODO maybe delete - depends if the list returned and stored in the_refl_list is in the correct order everytime it is received - store reflection objects, for deleting
        self.refl_list_k4c = [] #TODO maybe delete store reflection objects, for deleting
        self.refl_list_e6c = [] #TODO maybe delete store reflection objects, for deleting
        self.refl_list_k6c = [] #TODO maybe delete store reflection objects, for deleting
        self.refl_list_2c = [] #TODO maybe delete store reflection objects, for deleting
        self.curr_refl_index_e4c = 0
        self.curr_refl_index_k4c = 0
        self.curr_refl_index_e6c = 0
        self.curr_refl_index_k6c = 0
        self.curr_refl_index_2c = 0
 
        self.selected_refl_index = 0 # reflection to delete, index of refl_list
        # UB
        self.UB_matrix = np.zeros((3,3), dtype=float)
        self.UB_matrix_input = np.zeros((3,3), dtype=float)

        #TODO add to screens
        #self.sample_rot_matrix = np.zeros((8,8), dtype=float)
        self.u_matrix = np.zeros((3,3), dtype=float)
        
        #TODO add to screens
        # U vector
        self.ux = 0.
        self.uy = 0.
        self.uz = 0.

        ### axes
        self.num_axes_solns = num_axes_solns

        # Eulerian 4-circle (omega, chi, phi, tth)
        self.axes_e4c = [0.,0.,0.,0.]
        self.axes_e4c_min = [-180.,-180.,-180.,-180.]
        self.axes_e4c_max = [180.,180.,180.,180.]

        # Kappa 4-circle (komega, kappa, kphi, tth)
        self.axes_k4c = [0.,0.,0.,0.]
        self.axes_k4c_min = [-180.,-180.,-180.,-180.]
        self.axes_k4c_max = [180.,180.,180.,180.]
        
        # Eulerian 6-circle (mu, omega, chi, phi, gamma, delta)
        self.axes_e6c = [0.,0.,0.,0.,0.,0.]
        self.axes_e6c_min = [-180.,-180.,-180.,-180.,-180.,-180.]
        self.axes_e6c_max = [180.,180.,180.,180.,180.,180.]

        # Kappa 6-circle (mu, komega, kappa, kphi, gamma, delta)
        self.axes_k6c = [0.,0.,0.,0.,0.,0.]
        self.axes_k6c_min = [-180.,-180.,-180.,-180.,-180.,-180.] 
        self.axes_k6c_max = [180.,180.,180.,180.,180.,180.] 

        # 2-circle (omega, tth)
        self.axes_2c = [0.,0.]
        self.axes_2c_min = [-180.,-180.]
        self.axes_2c_max = [180.,180.]

        ### axes for UB calculation - only used internally - avoids setting on calculation #TODO delete ? 
        self.axes_UB_e4c = [0., 0., 0., 0.]
        self.axes_UB_k4c = [0., 0., 0., 0.]
        self.axes_UB_e6c = [0., 0., 0., 0., 0., 0.]
        self.axes_UB_k6c = [0., 0., 0., 0., 0., 0.]
        self.axes_UB_2c = [0., 0.]

        ### pseduoaxes 
        self.pseudoaxes_h = 0.
        self.pseudoaxes_k = 0.
        self.pseudoaxes_l = 0.
        self.pseudoaxes_dspacing = 0. #TODO get from cif2hkl waveform PV
        self.pseudoaxes_fc2 = 0. #TODO set from cif2hkl waveform PV
        self.pseudoaxes_psi = 0.
        self.pseudoaxes_q = 0.
        self.pseudoaxes_q2 = 0.
        self.pseudoaxes_alpha = 0.
        self.pseudoaxes_qper = 0.
        self.pseudoaxes_qpar = 0.
        self.pseudoaxes_incidence = 0.
        self.pseudoaxes_azimuth1 = 0.
        self.pseudoaxes_emergence = 0.
        self.pseudoaxes_azimuth2 = 0.
        self.pseudoaxes_omega = 0.
        self.pseudoaxes_chi = 0.
        self.pseudoaxes_phi = 0.
        self.pseudoaxes_tth = 0.
        self.pseudoaxes_alpha2 = 0.

        ### pseudoaxes options
        self.pseudoaxes_psi_h2 = 0.
        self.pseudoaxes_psi_k2 = 0.
        self.pseudoaxes_psi_l2 = 0.

        self.pseudoaxes_qperqpar_x = 0.
        self.pseudoaxes_qperqpar_y = 0.
        self.pseudoaxes_qperqpar_z = 0.
        
        self.pseudoaxes_eulerians_solns = 0.

        self.pseudoaxes_incidence_x = 0.
        self.pseudoaxes_incidence_y = 0.
        self.pseudoaxes_incidence_z = 0.

        self.pseudoaxes_emergence_x = 0.
        self.pseudoaxes_emergence_y = 0.
        self.pseudoaxes_emergence_z = 0.

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

        # 2-circle
        self.axes_solns_omega_2c = []
        self.axes_solns_tth_2c = []

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
            # 2-circle
            self.axes_solns_omega_2c.append(0)
            self.axes_solns_tth_2c.append(0)       

        # pseudoaxes solutions
        self.pseudoaxes_solns_h = 0.
        self.pseudoaxes_solns_k = 0.
        self.pseudoaxes_solns_l = 0.
        self.pseudoaxes_solns_dspacing = 0. #TODO set from cif2hkl waveform PV
        self.pseudoaxes_solns_fc2 = 0. #TODO set from cif2hkl waveform PV
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
        print("START")
        self.reset_pseudoaxes_solns()
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

        if (self.geom == 0) or (self.geom == 1):
            self.engine_hkl = self.engines.engine_get_by_name("hkl")
            self.engine_psi = self.engines.engine_get_by_name("psi")
            self.engine_q = self.engines.engine_get_by_name("q")
            self.engine_incidence = self.engines.engine_get_by_name("incidence")
            self.engine_emergence = self.engines.engine_get_by_name("emergence")
        elif (self.geom == 2): 
            self.engine_hkl = self.engines.engine_get_by_name("hkl")
            self.engine_eulerians = self.engines.engine_get_by_name("eulerians")
            self.engine_psi = self.engines.engine_get_by_name("psi")
            self.engine_q = self.engines.engine_get_by_name("q")
            self.engine_incidence = self.engines.engine_get_by_name("incidence")
            self.engine_emergence = self.engines.engine_get_by_name("emergence")
        elif (self.geom == 3): 
            self.engine_hkl = self.engines.engine_get_by_name("hkl")
            self.engine_psi = self.engines.engine_get_by_name("psi")
            self.engine_q2 = self.engines.engine_get_by_name("q2")
            self.engine_qper_qpar = self.engines.engine_get_by_name("qper_qpar")
            self.engine_tth2 = self.engines.engine_get_by_name("tth2")
            self.engine_incidence = self.engines.engine_get_by_name("incidence")
            self.engine_emergence = self.engines.engine_get_by_name("emergence")
        elif (self.geom == 4): 
            self.engine_hkl = self.engines.engine_get_by_name("hkl")
            self.engine_eulerians = self.engines.engine_get_by_name("eulerians")
            self.engine_psi = self.engines.engine_get_by_name("psi")
            self.engine_q2 = self.engines.engine_get_by_name("q2")
            self.engine_qper_qpar = self.engines.engine_get_by_name("qper_qpar")
            self.engine_incidence = self.engines.engine_get_by_name("incidence")
            self.engine_tth2 = self.engines.engine_get_by_name("tth2")
            self.engine_emergence = self.engines.engine_get_by_name("emergence")
        elif (self.geom == 5):
            self.engine_hkl = self.engines.engine_get_by_name("hkl")
            axes = self.geometry.axis_names_get()
            for i, axis in enumerate(axes): 
                if axis in ['chi', 'phi']:
                    tmp = self.geometry.axis_get(axis)
                    tmp.min_max_set(-0.01, 0.01, Hkl.UnitEnum.USER)
                    self.geometry.axis_set(axis, tmp)
            self.engine_psi = self.engines.engine_get_by_name("psi") #TODO
            self.engine_q = self.engines.engine_get_by_name("q") #TODO
            self.engine_incidence = self.engines.engine_get_by_name("incidence") #TODO
            self.engine_emergence = self.engines.engine_get_by_name("emergence") #TODO
            
        self.clear_all_reflections()
        self.businglevyflag=0
        self.get_UB_matrix()    
        self.get_latt_vol()
        status = self.get_info()
        ct = datetime.datetime.now().isoformat()
        status_string = f'initialized geometry {status}'
        self.errors = [ord(c) for c in str(ct + '\n' + status_string)]

    def energy_to_wavelength_neutron(self):
        self.wavelength_result = energy2wavelength_neutron(self.energy)

    def switch_geom(self):
        if self.geom == 0:
            print("switching to E4CH")
            self.geom_name = "E4CH"
            self.start()

        elif self.geom == 1:
            print("switching to E4CV")
            self.geom_name = "E4CV"
            self.start()

        elif self.geom == 2:
            print("switching to K4CV")
            self.geom_name = "K4CV"
            self.start()

        elif self.geom == 3:
            print("switching to E6C")
            self.geom_name = "E6C"
            self.start()

        elif self.geom == 4:
            print("switching to K6C")
            self.geom_name = "K6C"
            self.start()

        elif self.geom == 5:
            print("switching to 2C")
            #self.geom_name = "2C" #TODO
            self.geom_name = "E4CV" #TODO
            self.start()


    def forward(self):
        print("Forward function start")
        self.reset_pseudoaxes_solns()
        self.set_axes_to_sample()
        self.engines.get()
        self.set_engine_parameters()
        ### common to all geoms
        # hkl
        values_hkl = self.engine_hkl.pseudo_axis_values_get(Hkl.UnitEnum.USER)
        self.pseudoaxes_solns_h, self.pseudoaxes_solns_k, self.pseudoaxes_solns_l = values_hkl
        # psi
        values_psi = self.engine_psi.pseudo_axis_values_get(Hkl.UnitEnum.USER)
        self.pseudoaxes_solns_psi = values_psi[0] # [0] required otherwise assigns as list
        # incidence
        values_incidence = self.engine_incidence.pseudo_axis_values_get(Hkl.UnitEnum.USER)
        self.pseudoaxes_solns_incidence, self.pseudoaxes_solns_azimuth1 = values_incidence
        # emergence
        values_emergence = self.engine_emergence.pseudo_axis_values_get(Hkl.UnitEnum.USER)
        self.pseudoaxes_solns_emergence, self.pseudoaxes_solns_azimuth2 = values_emergence
        if (self.geom == 0) or (self.geom == 1):
            # q
            values_q = self.engine_q.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_solns_q = values_q[0]
        elif self.geom == 2:
            # q
            values_q = self.engine_q.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_solns_q = values_q[0]
            # eulerians
            values_eulerians = self.engine_eulerians.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_solns_omega, self.pseudoaxes_solns_chi, self.pseudoaxes_solns_phi = \
                values_eulerians
        elif self.geom == 3:
            # q2
            values_q2 = self.engine_q2.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_solns_q, self.pseudoaxes_solns_alpha = values_q2
            #qper, qpar
            values_qper_qpar = self.engine_qper_qpar.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_solns_qper, self.pseudoaxes_solns_qpar = values_qper_qpar
            # tth2
            values_tth2 = self.engine_tth2.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_solns_tth2, self.pseudoaxes_solns_alpha2 = values_tth2
        elif self.geom == 4:
            # q2
            values_q2 = self.engine_q2.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_solns_q, self.pseudoaxes_solns_alpha = values_q2
            #qper, qpar
            values_qper_qpar = self.engine_qper_qpar.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_solns_qper, self.pseudoaxes_solns_qpar = values_qper_qpar
            # tth2
            values_tth2 = self.engine_tth2.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_solns_tth2, self.pseudoaxes_solns_alpha2 = values_tth2
            #eulerians
            values_eulerians = self.engine_eulerians.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_solns_omega, self.pseudoaxes_solns_chi, self.pseudoaxes_solns_phi = \
                values_eulerians
        elif (self.geom == 5):
            # TODO other pseudo engines for 2c`
            # q
            values_q = self.engine_q.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_solns_q = values_q[0]
            pass
        self.get_UB_matrix()


    def set_axes_to_sample(self):
        #TODO rename? sets diffractometer to ready adding a reflection
        print("Set axes to sample function start")
        self.set_axis_limits()
        if (self.geom==0) or (self.geom==1):
            values_w = [float(self.axes_e4c[0]), \
                        float(self.axes_e4c[1]), \
                        float(self.axes_e4c[2]), \
                        float(self.axes_e4c[3])] 
        elif (self.geom == 2):
            values_w = [float(self.axes_k4c[0]), \
                        float(self.axes_k4c[1]), \
                        float(self.axes_k4c[2]), \
                        float(self.axes_k4c[3])] 
        elif (self.geom == 3):
            values_w = [float(self.axes_e6c[0]), \
                        float(self.axes_e6c[1]), \
                        float(self.axes_e6c[2]), \
                        float(self.axes_e6c[3]), \
                        float(self.axes_e6c[4]), \
                        float(self.axes_e6c[5])] 
        elif (self.geom == 4):
            values_w = [float(self.axes_k6c[0]), \
                        float(self.axes_k6c[1]), \
                        float(self.axes_k6c[2]), \
                        float(self.axes_k6c[3]), \
                        float(self.axes_k6c[4]), \
                        float(self.axes_k6c[5])] 
        elif (self.geom==5):
            values_w = [float(self.axes_2c[0]), \
                        float(0), \
                        float(0), \
                        float(self.axes_2c[1])]
        try:
            print(f'geom {self.geom}')
            print(f'values {values_w}')
            self.geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))]
            print(f'set_axes_to_sample() error: {e}')
            return

    def set_axes_to_sample_UB(self):
        #TODO rename? sets diffractometer to ready adding a reflection
        print("Set axes to sample UB function start")
        if (self.geom==0) or (self.geom==1):
            values_w = [float(self.axes_UB_e4c[0]), \
                        float(self.axes_UB_e4c[1]), \
                        float(self.axes_UB_e4c[2]), \
                        float(self.axes_UB_e4c[3])] 
        elif (self.geom == 2):
            values_w = [float(self.axes_UB_k4c[0]), \
                        float(self.axes_UB_k4c[1]), \
                        float(self.axes_UB_k4c[2]), \
                        float(self.axes_UB_k4c[3])] 
        elif (self.geom == 3):
            values_w = [float(self.axes_UB_e6c[0]), \
                        float(self.axes_UB_e6c[1]), \
                        float(self.axes_UB_e6c[2]), \
                        float(self.axes_UB_e6c[3]), \
                        float(self.axes_UB_e6c[4]), \
                        float(self.axes_UB_e6c[5])] 
        elif (self.geom == 4):
            values_w = [float(self.axes_UB_k6c[0]), \
                        float(self.axes_UB_k6c[1]), \
                        float(self.axes_UB_k6c[2]), \
                        float(self.axes_UB_k6c[3]), \
                        float(self.axes_UB_k6c[4]), \
                        float(self.axes_UB_k6c[5])] 
        elif (self.geom==5):
            values_w = [float(self.axes_UB_2c[0]), \
                        float(0), \
                        float(0), \
                        float(self.axes_UB_2c[1])] 
        try:
            #TODO check if this is how UB calculation is done in hkl package
            # currently need to run this function when adding reflections for some reason
            print(f'geom {self.geom}')
            print(f'values {values_w}')
            self.geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))]
            print(f'set_axes_to_sample_UB() error: {e}')
            return

    def set_constraints(self):
        # set constraint mode
        if (self.geom==0) or (self.geom==1) or (self.geom==2):
            if self.mode_4c == 0:
                self.engine_hkl.current_mode_set('bissector')
            elif self.mode_4c == 1:
                self.engine_hkl.current_mode_set('constant_omega')
            elif self.mode_4c == 2:
                self.engine_hkl.current_mode_set('constant_chi')
            elif self.mode_4c == 3:
                self.engine_hkl.current_mode_set('constant_phi')
            elif self.mode_4c == 4:
                self.engine_hkl.current_mode_set('double_diffraction')
            elif self.mode_4c == 5:
                self.engine_hkl.current_mode_set('psi_constant')

        elif (self.geom==3):
            if self.mode_e6c == 0: 
                self.engine_hkl.current_mode_set('bissector_vertical')
            if self.mode_e6c == 1: 
                self.engine_hkl.current_mode_set('constant_omega_vertical')
            if self.mode_e6c == 2: 
                self.engine_hkl.current_mode_set('constant_chi_vertical')
            if self.mode_e6c == 3: 
                self.engine_hkl.current_mode_set('constant_phi_vertical')
            if self.mode_e6c == 4: 
                self.engine_hkl.current_mode_set('lifting_detector_phi')
            if self.mode_e6c == 5: 
                self.engine_hkl.current_mode_set('lifting_detector_omega')
            if self.mode_e6c == 6: 
                self.engine_hkl.current_mode_set('lifting_detector_mu')
            if self.mode_e6c == 7: 
                self.engine_hkl.current_mode_set('double_diffraction_vertical')
            if self.mode_e6c == 8: 
                self.engine_hkl.current_mode_set('bissector_horizontal')
            if self.mode_e6c == 9:
                self.engine_hkl.current_mode_set('double_diffraction_horizontal')
            if self.mode_e6c == 10: 
                self.engine_hkl.current_mode_set('psi_constant_vertical')
            if self.mode_e6c == 11: 
                self.engine_hkl.current_mode_set('psi_constant_horizontal')
            if self.mode_e6c == 12:
                self.engine_hkl.current_mode_set('constant_mu_horizontal')

        elif (self.geom==4):
            if self.mode_k6c == 0: 
                self.engine_hkl.current_mode_set('bissector_vertical')
            if self.mode_k6c == 1: 
                self.engine_hkl.current_mode_set('constant_omega_vertical')
            if self.mode_k6c == 2: 
                self.engine_hkl.current_mode_set('constant_chi_vertical')
            if self.mode_k6c == 3: 
                self.engine_hkl.current_mode_set('constant_phi_vertical')
            if self.mode_k6c == 4: 
                self.engine_hkl.current_mode_set('lifting_detector_kphi')
            if self.mode_k6c == 5: 
                self.engine_hkl.current_mode_set('lifting_detector_komega')
            if self.mode_k6c == 6: 
                self.engine_hkl.current_mode_set('lifting_detector_mu')
            if self.mode_k6c == 7: 
                self.engine_hkl.current_mode_set('double_diffraction_vertical')
            if self.mode_k6c == 8: 
                self.engine_hkl.current_mode_set('bissector_horizontal')
            if self.mode_k6c == 9:
                self.engine_hkl.current_mode_set('constant_phi_horizontal')
            if self.mode_k6c == 10:
                self.engine_hkl.current_mode_set('constant_kphi_horizontal')
            if self.mode_k6c == 11: 
                self.engine_hkl.current_mode_set('double_diffraction_horizontal')
            if self.mode_k6c == 12: 
                self.engine_hkl.current_mode_set('psi_constant_horizontal')
            if self.mode_k6c == 13:
                self.engine_hkl.current_mode_set("constant_incidence")

        elif (self.geom==5):
            if self.mode_2c == 0:
                self.engine_hkl.current_mode_set('bissector')

    def set_engine_parameters(self):
        if (self.geom==0) or (self.geom==1):
            self.engine_psi.parameters_values_set([self.pseudoaxes_psi_h2, \
                                                  self.pseudoaxes_psi_k2, \
                                                  self.pseudoaxes_psi_l2], \
                                                  Hkl.UnitEnum.USER)
            self.engine_incidence.parameters_values_set([self.pseudoaxes_incidence_x, \
                                                         self.pseudoaxes_incidence_y, \
                                                         self.pseudoaxes_incidence_z], \
                                                         Hkl.UnitEnum.USER)
            self.engine_emergence.parameters_values_set([self.pseudoaxes_emergence_x, \
                                                         self.pseudoaxes_emergence_y, \
                                                         self.pseudoaxes_emergence_z], \
                                                         Hkl.UnitEnum.USER)
        elif (self.geom==2):
            self.engine_eulerians.parameters_values_set([self.pseudoaxes_eulerians_solns], \
                                                         Hkl.UnitEnum.USER)
            self.engine_psi.parameters_values_set([self.pseudoaxes_psi_h2, \
                                                  self.pseudoaxes_psi_k2, \
                                                  self.pseudoaxes_psi_l2], \
                                                  Hkl.UnitEnum.USER)
            self.engine_incidence.parameters_values_set([self.pseudoaxes_incidence_x, \
                                                         self.pseudoaxes_incidence_y, \
                                                         self.pseudoaxes_incidence_z], \
                                                         Hkl.UnitEnum.USER)
            self.engine_emergence.parameters_values_set([self.pseudoaxes_emergence_x, \
                                                         self.pseudoaxes_emergence_y, \
                                                         self.pseudoaxes_emergence_z], \
                                                         Hkl.UnitEnum.USER)
        elif (self.geom==3):
            self.engine_psi.parameters_values_set([self.pseudoaxes_psi_h2, \
                                                  self.pseudoaxes_psi_k2, \
                                                  self.pseudoaxes_psi_l2], \
                                                  Hkl.UnitEnum.USER)
            self.engine_qper_qpar.parameters_values_set([self.pseudoaxes_qperqpar_x, \
                                                        self.pseudoaxes_qperqpar_y, \
                                                        self.pseudoaxes_qperqpar_z], \
                                                        Hkl.UnitEnum.USER)
            self.engine_incidence.parameters_values_set([self.pseudoaxes_incidence_x, \
                                                         self.pseudoaxes_incidence_y, \
                                                         self.pseudoaxes_incidence_z], \
                                                         Hkl.UnitEnum.USER)
            self.engine_emergence.parameters_values_set([self.pseudoaxes_emergence_x, \
                                                         self.pseudoaxes_emergence_y, \
                                                         self.pseudoaxes_emergence_z], \
                                                         Hkl.UnitEnum.USER)
        elif (self.geom==4):
            self.engine_eulerians.parameters_values_set([self.pseudoaxes_eulerians_solns], \
                                                         Hkl.UnitEnum.USER)
            self.engine_psi.parameters_values_set([self.pseudoaxes_psi_h2, \
                                                  self.pseudoaxes_psi_k2, \
                                                  self.pseudoaxes_psi_l2], \
                                                  Hkl.UnitEnum.USER)
            self.engine_qper_qpar.parameters_values_set([self.pseudoaxes_qperqpar_x, \
                                                        self.pseudoaxes_qperqpar_y, \
                                                        self.pseudoaxes_qperqpar_z], \
                                                        Hkl.UnitEnum.USER)
            self.engine_incidence.parameters_values_set([self.pseudoaxes_incidence_x, \
                                                         self.pseudoaxes_incidence_y, \
                                                         self.pseudoaxes_incidence_z], \
                                                         Hkl.UnitEnum.USER)
            self.engine_emergence.parameters_values_set([self.pseudoaxes_emergence_x, \
                                                         self.pseudoaxes_emergence_y, \
                                                         self.pseudoaxes_emergence_z], \
                                                         Hkl.UnitEnum.USER)
        elif (self.geom==5):
            self.engine_psi.parameters_values_set([self.pseudoaxes_psi_h2, \
                                                  self.pseudoaxes_psi_k2, \
                                                  self.pseudoaxes_psi_l2], \
                                                  Hkl.UnitEnum.USER)
            self.engine_incidence.parameters_values_set([self.pseudoaxes_incidence_x, \
                                                         self.pseudoaxes_incidence_y, \
                                                         self.pseudoaxes_incidence_z], \
                                                         Hkl.UnitEnum.USER)
            self.engine_emergence.parameters_values_set([self.pseudoaxes_emergence_x, \
                                                         self.pseudoaxes_emergence_y, \
                                                         self.pseudoaxes_emergence_z], \
                                                         Hkl.UnitEnum.USER)

    def apply_axes_solns(self, solutions):
        values_w_all = []
        len_solns = len(solutions.items())
        #TODO set len sol to a PV, display in backward tab
        for i, item in enumerate(solutions.items()):
            read = item.geometry_get().axis_values_get(Hkl.UnitEnum.USER)
            values_w_all.append(read)
        if len_solns > self.num_axes_solns: # truncate if above max available soln slots
            len_solns = self.num_axes_solns -1
        if (self.geom == 0) or (self.geom==1):
            for i in range(len_solns): 
                self.axes_solns_omega_e4c[i], \
                self.axes_solns_chi_e4c[i], \
                self.axes_solns_phi_e4c[i], \
                self.axes_solns_tth_e4c[i] = values_w_all[i]
        elif self.geom == 2:
            for i in range(len_solns): 
                self.axes_solns_komega_k4c[i], \
                self.axes_solns_kappa_k4c[i], \
                self.axes_solns_kphi_k4c[i], \
                self.axes_solns_tth_k4c[i] = values_w_all[i]         
        elif self.geom == 3:
            for i in range(len_solns): 
                self.axes_solns_mu_e6c[i], \
                self.axes_solns_omega_e6c[i], \
                self.axes_solns_chi_e6c[i], \
                self.axes_solns_phi_e6c[i], \
                self.axes_solns_gamma_e6c[i], \
                self.axes_solns_delta_e6c[i] = values_w_all[i]       
        elif self.geom==4:  
             for i in range(len_solns): 
                self.axes_solns_mu_k6c[i], \
                self.axes_solns_komega_k6c[i], \
                self.axes_solns_kappa_k6c[i], \
                self.axes_solns_kphi_k6c[i], \
                self.axes_solns_gamma_k6c[i], \
                self.axes_solns_delta_k6c[i] = values_w_all[i]
        elif (self.geom == 5):
            for i in range(len_solns): 
                self.axes_solns_omega_2c[i], \
                _, \
                _, \
                self.axes_solns_tth_2c[i] = values_w_all[i]

    def backward_hkl(self):
        print("Backward hkl function start")
        self.reset_axes_solns()
        self.set_constraints()
        self.set_engine_parameters()
        values_hkl = [float(self.pseudoaxes_h), \
                      float(self.pseudoaxes_k), \
                      float(self.pseudoaxes_l)]
        try:
            self.set_axis_limits()
            solutions = self.engine_hkl.pseudo_axis_values_set(values_hkl, Hkl.UnitEnum.USER)
            self.apply_axes_solns(solutions)
            #TODO does running this calculation change other pseudoaxes values?
            # if so, I need to update all other pseudoaxes here (with self.update_pseudoaxes)     
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct)] # success
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))] # failure
            print(f'backward_hkl() error: {e}')

    def backward_psi(self):
        print("Backward psi function start")
        self.reset_axes_solns()
        self.set_engine_parameters()
        values_psi = [float(self.pseudoaxes_psi)]
        try:
            self.set_axis_limits()
            solutions = self.engine_psi.pseudo_axis_values_set(values_psi, Hkl.UnitEnum.USER)
            self.apply_axes_solns(solutions)
            #TODO does running this calculation change other pseudoaxes values?
            # if so, I need to update all other pseudoaxes here (with self.update_pseudoaxes)     
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct)] # success
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))] # failure
            print(f'backward_psi() error: {e}')

    def backward_q(self):
        print("Backward q function start")
        self.reset_axes_solns()
        self.set_engine_parameters()
        values_q = [float(self.pseudoaxes_q)]
        try:
            self.set_axis_limits()
            solutions = self.engine_q.pseudo_axis_values_set(values_q, Hkl.UnitEnum.USER)
            self.apply_axes_solns(solutions)
            #TODO does running this calculation change other pseudoaxes values?
            # if so, I need to update all other pseudoaxes here (with self.update_pseudoaxes)     
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct)] # success
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))] # failure
            print(f'backward_q() error: {e}')

    def backward_q2(self):
        print("Backward q2 function start")
        self.reset_axes_solns()
        self.set_engine_parameters()
        values_q2 = [float(self.pseudoaxes_q2), float(self.pseudoaxes_alpha)]
        try:
            self.set_axis_limits()
            solutions = self.engine_q2.pseudo_axis_values_set(values_q2, Hkl.UnitEnum.USER)
            self.apply_axes_solns(solutions)
            #TODO does running this calculation change other pseudoaxes values?
            # if so, I need to update all other pseudoaxes here (with self.update_pseudoaxes)     
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct)] # success
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))] # failure
            print(f'backward_q2() error: {e}')

    def backward_qperqpar(self):
        print("Backward qperqpar function start")
        self.reset_axes_solns()
        self.set_engine_parameters()
        values_qperqpar = [float(self.pseudoaxes_qper), float(self.pseudoaxes_qpar)]
        try:
            self.set_axis_limits()
            solutions = self.engine_qper_qpar.pseudo_axis_values_set(values_qperqpar, Hkl.UnitEnum.USER)
            self.apply_axes_solns(solutions)
            #TODO does running this calculation change other pseudoaxes values?
            # if so, I need to update all other pseudoaxes here (with self.update_pseudoaxes)     
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct)] # success
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))] # failure
            print(f'backward_qperqpar() error: {e}')

    def backward_tth(self):
        print("Backward tth function start")
        self.reset_axes_solns()
        self.set_engine_parameters()
        values_tth = [float(self.pseudoaxes_tth), float(self.pseudoaxes_alpha2)]
        try:
            self.set_axis_limits()
            solutions = self.engine_tth2.pseudo_axis_values_set(values_tth, Hkl.UnitEnum.USER)
            self.apply_axes_solns(solutions)
            #TODO does running this calculation change other pseudoaxes values?
            # if so, I need to update all other pseudoaxes here (with self.update_pseudoaxes)     
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct)] # success
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))] # failure
            print(f'backward_tth() error: {e}')

    def backward_eulerians(self):
        print("Backward eulerians function start")
        self.reset_axes_solns()
        self.set_engine_parameters()
        values_eulerians = [float(self.pseudoaxes_omega), \
                            float(self.pseudoaxes_chi), \
                            float(self.pseudoaxes_phi)]
        try:
            self.set_axis_limits()
            solutions = self.engine_eulerians.pseudo_axis_values_set(values_eulerians, Hkl.UnitEnum.USER)
            self.apply_axes_solns(solutions)
            #TODO does running this calculation change other pseudoaxes values?
            # if so, I need to update all other pseudoaxes here (with self.update_pseudoaxes)     
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct)] # success
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))] # failure
            print(f'backward_eulerians() error: {e}')


    def update_pseudoaxes(self):
        self.set_engine_parameters()
        #TODO this is temporary, not currently used
        values_hkl = self.engine_hkl.pseudo_axis_values_get(Hkl.UnitEnum.USER)
        self.pseudoaxes_solns_h, self.pseudoaxes_solns_k, self.pseudoaxes_solns_l = values_hkl
        #psi
        values_psi = self.engine_psi.pseudo_axis_values_get(Hkl.UnitEnum.USER)
        self.pseudoaxes_psi = values_psi[0] # [0] required otherwise assigns as list
        # incidence
        values_incidence = self.engine_incidence.pseudo_axis_values_get(Hkl.UnitEnum.USER)
        self.pseudoaxes_incidence, self.pseudoaxes_azimuth1 = values_incidence
        # emergence
        values_emergence = self.engine_emergence.pseudo_axis_values_get(Hkl.UnitEnum.USER)
        self.pseudoaxes_emergence, self.pseudoaxes_azimuth2 = values_emergence
        if (self.geom == 0) or (self.geom == 1):
            # q
            values_q = self.engine_q.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            print(f'values q: {values_q[0]}')
            self.pseudoaxes_q = values_q[0]
        elif self.geom == 2:
            # q
            values_q = self.engine_q.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_q = values_q[0]
            # eulerians #TODO add to screens
            values_eulerians = self.engine_eulerians.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_omega, self.pseudoaxes_chi, self.pseudoaxes_phi = \
                values_eulerians
        elif self.geom == 3:
            # q2
            values_q2 = self.engine_q2.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_q, self.pseudoaxes_alpha = values_q2
            #qper, qpar
            values_qper_qpar = self.engine_qper_qpar.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_qper, self.pseudoaxes_qpar = values_qper_qpar
            # tth2
            values_tth2 = self.engine_tth2.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_tth2, self.pseudoaxes_alpha2 = values_tth2
        elif self.geom == 4:
            # q2
            values_q2 = self.engine_q2.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_q, self.pseudoaxes_alpha = values_q2
            #qper, qpar
            values_qper_qpar = self.engine_qper_qpar.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_qper, self.pseudoaxes_qpar = values_qper_qpar
            # tth2
            values_tth2 = self.engine_tth2.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_tth2, self.pseudoaxes_alpha2 = values_tth2
            #eulerians
            values_eulerians = self.engine_eulerians.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            self.pseudoaxes_omega, self.pseudoaxes_chi, self.pseudoaxes_phi = \
                values_eulerians
        elif (self.geom == 5):
            # q
            #values_q = self.engine_q.pseudo_axis_values_get(Hkl.UnitEnum.USER)
            #print(f'values q: {values_q[0]}')
            #self.pseudoaxes_q = values_q[0]
            #TODO
            pass


    def set_axis_limits(self):
        axes = self.geometry.axis_names_get()
        if (self.geom==0) or (self.geom==1):
            for i, axis in enumerate(axes): 
                tmp = self.geometry.axis_get(axis)
                tmp.min_max_set(self.axes_e4c_min[i], \
                                self.axes_e4c_max[i], \
                                Hkl.UnitEnum.USER)
                self.geometry.axis_set(axis, tmp)
        elif self.geom==2:
            for i, axis in enumerate(axes): 
                tmp = self.geometry.axis_get(axis)
                tmp.min_max_set(self.axes_k4c_min[i], \
                                self.axes_k4c_max[i], \
                                Hkl.UnitEnum.USER)
                self.geometry.axis_set(axis, tmp)
        elif self.geom==3:
            for i, axis in enumerate(axes): 
                tmp = self.geometry.axis_get(axis)
                tmp.min_max_set(self.axes_e6c_min[i], \
                                self.axes_e6c_max[i], \
                                Hkl.UnitEnum.USER)
                self.geometry.axis_set(axis, tmp)
        elif self.geom==4:
            for i, axis in enumerate(axes): 
                tmp = self.geometry.axis_get(axis)
                tmp.min_max_set(self.axes_k6c_min[i], \
                                self.axes_k6c_max[i], \
                                Hkl.UnitEnum.USER)
                self.geometry.axis_set(axis, tmp)
        elif (self.geom==5):
            for i, axis in enumerate(axes):
                if axis in ['chi', 'phi']:
                    tmp = self.geometry.axis_get(axis)
                    tmp.min_max_set(self.axes_e4c_min[i], \
                                    self.axes_e4c_max[i], \
                                    Hkl.UnitEnum.USER)
                    self.geometry.axis_set(axis, tmp)
                else:
                    tmp = self.geometry.axis_get(axis)
                    tmp.min_max_set(self.axes_e4c_min[i], \
                                    self.axes_e4c_max[i], \
                                    Hkl.UnitEnum.USER)
                    self.geometry.axis_set(axis, tmp)

    def reset_pseudoaxes_solns(self):
        self.pseudoaxes_solns_h = 0
        self.pseudoaxes_solns_k = 0
        self.pseudoaxes_solns_l = 0
        self.pseudoaxes_solns_dpsacing = 0
        self.pseudoaxes_solns_fc2 = 0
        self.pseudoaxes_solns_psi = 0
        self.pseudoaxes_solns_q = 0
        self.pseudoaxes_solns_incidence = 0
        self.pseudoaxes_solns_azimuth1 = 0
        self.pseudoaxes_solns_emergence = 0
        self.pseudoaxes_solns_azimuth2 = 0

    def reset_axes_solns(self):
        if (self.geom==0) or (self.geom==1):
            for lst in [self.axes_solns_omega_e4c, \
                        self.axes_solns_chi_e4c, \
                        self.axes_solns_phi_e4c, \
                        self.axes_solns_tth_e4c]:
                lst[:] = [0 for _ in lst]
        elif (self.geom==2):
            for lst in [self.axes_solns_komega_k4c, \
                        self.axes_solns_kappa_k4c, \
                        self.axes_solns_kphi_k4c, \
                        self.axes_solns_tth_k4c ]:
                lst[:] = [0 for _ in lst]
        elif (self.geom==3):
            for lst in [self.axes_solns_mu_e6c, \
                        self.axes_solns_omega_e6c, \
                        self.axes_solns_chi_e6c, \
                        self.axes_solns_phi_e6c, \
                        self.axes_solns_gamma_e6c, \
                        self.axes_solns_delta_e6c]:
                lst[:] = [0 for _ in lst]
        elif (self.geom==4):
            for lst in [self.axes_solns_mu_k6c, \
                        self.axes_solns_komega_k6c, \
                        self.axes_solns_kappa_k6c, \
                        self.axes_solns_kphi_k6c, \
                        self.axes_solns_gamma_k6c, \
                        self.axes_solns_delta_k6c]:
                lst[:] = [0 for _ in lst]
        elif (self.geom==5):
            for lst in [self.axes_solns_omega_2c, \
                        self.axes_solns_tth_2c]:
                lst[:] = [0 for _ in lst]


    def add_reflection1(self):
        '''
        adds reflection #1 to sample for busing-levy calculation
        '''
        print("add BL reflection 1")
        if (self.geom==0) or (self.geom==1):
            for i in range(4):
                self.axes_UB_e4c[i] = self.refl1_input_e4c[(i+3)]
        elif (self.geom==2):
            for i in range(4):
                self.axes_UB_k4c[i] = self.refl1_input_k4c[(i+3)]
        elif (self.geom==3):
            for i in range(6):
                self.axes_UB_e6c[i] = self.refl1_input_e6c[(i+3)]
        elif (self.geom==4):
            for i in range(6):
                self.axes_UB_k6c[i] = self.refl1_input_k6c[(i+3)]
        elif (self.geom==5):
            for i in range(2):
                self.axes_UB_2c[i] = self.refl1_input_2c[(i+3)]
        self.set_axes_to_sample_UB()
        if (self.geom==0) or (self.geom==1):
            self.refl1 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl1_input_e4c[0], \
                                                    self.refl1_input_e4c[1], \
                                                    self.refl1_input_e4c[2])
            self.refl_refine_input_list_e4c[self.curr_refl_index_e4c] = self.refl1_input_e4c.copy()
            #self.refl_list_e4c.append(self.refl1) #TODO maybe delete
            self.curr_refl_index_e4c += 1
        elif (self.geom==2):
            self.refl1 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl1_input_k4c[0], \
                                                    self.refl1_input_k4c[1], \
                                                    self.refl1_input_k4c[2])
            self.refl_refine_input_list_k4c[self.curr_refl_index_k4c] = self.refl1_input_k4c.copy()
            #self.refl_list_k4c.append(self.refl1) #TODO maybe delete
            self.curr_refl_index_k4c += 1
        elif (self.geom==3):
            self.refl1 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl1_input_e6c[0], \
                                                    self.refl1_input_e6c[1], \
                                                    self.refl1_input_e6c[2])
            self.refl_refine_input_list_e6c[self.curr_refl_index_e6c] = self.refl1_input_e6c.copy()
            #self.refl_list_e6c.append(self.refl1) #TODO maybe delete
            self.curr_refl_index_e6c += 1
        elif (self.geom==4):
            self.refl1 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl1_input_k6c[0], \
                                                    self.refl1_input_k6c[1], \
                                                    self.refl1_input_k6c[2])
            self.refl_refine_input_list_k6c[self.curr_refl_index_k6c] = self.refl1_input_k6c.copy()
            #self.refl_list_k6c.append(self.refl1) #TODO maybe delete
            self.curr_refl_index_k6c += 1
        elif (self.geom==5):
            self.refl1 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl1_input_2c[0], \
                                                    self.refl1_input_2c[1], \
                                                    self.refl1_input_2c[2])
            self.refl_refine_input_list_2c[self.curr_refl_index_2c] = self.refl1_input_2c.copy()
            #self.refl_list_e4c.append(self.refl1) #TODO maybe delete
            self.curr_refl_index_2c += 1


    def add_reflection2(self):
        '''
        adds reflection #2 to sample for busing levy calculation
        '''
        print("add BL reflection 2")
        if (self.geom==0) or (self.geom==1):
            for i in range(4):
                self.axes_UB_e4c[i] = self.refl2_input_e4c[(i+3)]
        elif (self.geom==2):
            for i in range(4):
                self.axes_UB_k4c[i] = self.refl2_input_k4c[(i+3)]
        elif (self.geom==3):
            for i in range(6):
                self.axes_UB_e6c[i] = self.refl2_input_e6c[(i+3)]
        elif (self.geom==4):
            for i in range(6):
                self.axes_UB_k6c[i] = self.refl2_input_k6c[(i+3)]
        elif (self.geom==5):
            for i in range(2):
                self.axes_UB_2c[i] = self.refl2_input_2c[(i+3)]
        self.set_axes_to_sample_UB()
        # Hkl.SampleReflection(self.geometry, self.detector, h, k, l)
        if (self.geom==0) or (self.geom==1):
            self.refl2 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl2_input_e4c[0], \
                                                    self.refl2_input_e4c[1], \
                                                    self.refl2_input_e4c[2])
            self.refl_refine_input_list_e4c[self.curr_refl_index_e4c] = self.refl2_input_e4c.copy()
            #self.refl_list_e4c.append(self.refl2) #TODO maybe delete
            self.curr_refl_index_e4c += 1
        elif (self.geom==2):
            self.refl2 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl2_input_k4c[0], \
                                                    self.refl2_input_k4c[1], \
                                                    self.refl2_input_k4c[2])
            self.refl_refine_input_list_k4c[self.curr_refl_index_k4c] = self.refl2_input_k4c.copy()
            #self.refl_list_k4c.append(self.refl2) #TODO maybe delete
            self.curr_refl_index_k4c += 1
        elif (self.geom==3):
            self.refl2 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl2_input_e6c[0], \
                                                    self.refl2_input_e6c[1], \
                                                    self.refl2_input_e6c[2])
            self.refl_refine_input_list_e6c[self.curr_refl_index_e6c] = self.refl2_input_e6c.copy()
            #self.refl_list_e6c.append(self.refl2) #TODO maybe delete
            self.curr_refl_index_e6c += 1
        elif (self.geom==4):
            self.refl2 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl2_input_k6c[0], \
                                                    self.refl2_input_k6c[1], \
                                                    self.refl2_input_k6c[2])
            self.refl_refine_input_list_k6c[self.curr_refl_index_k6c] = self.refl2_input_k6c.copy()
            #self.refl_list_k6c.append(self.refl2) #TODO maybe delete
            self.curr_refl_index_k6c += 1
        elif (self.geom==5):
            self.refl2 = self.sample.add_reflection(self.geometry, \
                                                    self.detector, \
                                                    self.refl2_input_2c[0], \
                                                    self.refl2_input_2c[1], \
                                                    self.refl2_input_2c[2])
            self.refl_refine_input_list_2c[self.curr_refl_index_2c] = self.refl2_input_2c.copy()
            #self.refl_list_e4c.append(self.refl2) #TODO maybe delete
            self.curr_refl_index_2c += 1

    def compute_set_UB_matrix(self):
        #TODO replace by feeding into user input UB
        '''
        same thing as compute_UB_matrix, but without start()
        '''
        if (self.businglevyflag==1):
            # clear all reflections in lists and from sample, redo busing-levy
            self.clear_all_reflections()
            self.businglevyflag = 0
        self.add_reflection1()
        self.add_reflection2()
        try:
            self.sample.compute_UB_busing_levy(self.refl1, self.refl2)
            UB = self.sample.UB_get()
            for i in range(3):
                for j in range(3):
                    self.UB_matrix[i,j] = UB.get(i,j)
            self.businglevyflag = 1
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct)] # success
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))] # failure
            print(f'compute_set_UB_matrix() error: {e}')

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
        try:
            self.sample.UB_set(UB_temp)
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct)] # success
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))] # failure
            print(f'set_input_UB() error: {e}')
        self.get_UB_matrix()

    def affine_set(self):
        '''
        takes in >2 reflections to refine lattice parameters and UB matrix
        '''
        try:
            self.sample.affine()
            #TODO just route PVs/python variables, don't re-run
            UB = self.sample.UB_get()
            for i in range(3):
                for j in range(3):
                    self.UB_matrix[i,j] = UB.get(i,j)
            self.latt[0], self.latt[1], self.latt[2], self.latt[3], self.latt[4], \
                self.latt[5] = self.lattice.get(Hkl.UnitEnum.USER)
            #self.latt = [self.lattice.get(Hkl.UnitEnum.USER)] #TODO test this
            # or maybe starred expression
        except Exception as e:
            ct = datetime.datetime.now().isoformat()
            self.errors = [ord(c) for c in str(ct + '\n' + str(e))]
            print(f'affine_set() error: {e}')
               
    def add_refl_refine(self):
        if (self.geom==0) or (self.geom==1):
            for i in range(4):
                self.axes_UB_e4c[i] = self.refl_refine_input_e4c[(i+3)]
        elif (self.geom==2):
            for i in range(4):
                self.axes_UB_k4c[i] = self.refl_refine_input_k4c[(i+3)]
        elif (self.geom==3):
            for i in range(6):
                self.axes_UB_e6c[i] = self.refl_refine_input_e6c[(i+3)]
        elif (self.geom==4):
            for i in range(6):
                self.axes_UB_k6c[i] = self.refl_refine_input_k6c[(i+3)]
        elif (self.geom==5):
            for i in range(2):
                self.axes_UB_2c[i] = self.refl_refine_input_2c[(i+3)]
        self.set_axes_to_sample_UB()
        if (self.geom==0) or (self.geom==1) and (self.curr_refl_index_e4c<9):
            #self.refl_refine = self.sample.add_reflection(self.geometry, \
            #    self.detector, self.refl_refine_input_e4c[0], \
            #    self.refl_refine_input_e4c[1], self.refl_refine_input_e4c[2])
            self.sample.add_reflection(self.geometry, \
                self.detector, self.refl_refine_input_e4c[0], \
                self.refl_refine_input_e4c[1], self.refl_refine_input_e4c[2])
            self.refl_refine_input_list_e4c[self.curr_refl_index_e4c] = self.refl_refine_input_e4c.copy()
            #self.refl_list_e4c.append(self.refl_refine) #TODO maybe delete
            self.curr_refl_index_e4c += 1
        elif (self.geom==2) and (self.curr_refl_index_k4c<9):
            #self.refl_refine = self.sample.add_reflection(self.geometry, \ #TODO
            self.sample.add_reflection(self.geometry, \
                self.detector, self.refl_refine_input_k4c[0], \
                self.refl_refine_input_k4c[1], self.refl_refine_input_k4c[2]) 
            self.refl_refine_input_list_k4c[self.curr_refl_index_k4c] = self.refl_refine_input_k4c.copy()
            #self.refl_list_k4c.append(self.refl_refine) #TODO maybe delete
            self.curr_refl_index_k4c += 1
        elif (self.geom==3) and (self.curr_refl_index_e6c<9):
            #self.refl_refine = self.sample.add_reflection(self.geometry, \ #TODO
            self.refl_refine = self.sample.add_reflection(self.geometry, \
                self.detector, self.refl_refine_input_e6c[0], \
                self.refl_refine_input_e6c[1], self.refl_refine_input_e6c[2])
            self.refl_refine_input_list_e6c[self.curr_refl_index_e6c] = self.refl_refine_input_e6c.copy()
            #self.refl_list_e6c.append(self.refl_refine) #TODO maybe delete
            self.curr_refl_index_e6c += 1
        elif (self.geom==4) and (self.curr_refl_index_k6c<9):
            #self.refl_refine = self.sample.add_reflection(self.geometry, \ #TODO
            self.sample.add_reflection(self.geometry, \
                self.detector, self.refl_refine_input_k6c[0], \
                self.refl_refine_input_k6c[1], self.refl_refine_input_k6c[2])
            self.refl_refine_input_list_k6c[self.curr_refl_index_k6c] = self.refl_refine_input_k6c.copy()
            #self.refl_list_k6c.append(self.refl_refine) #TODO maybe delete
            self.curr_refl_index_k6c += 1
        elif (self.geom==5) and (self.curr_refl_index_2c<9):
            self.sample.add_reflection(self.geometry, \
                self.detector, self.refl_refine_input_2c[0], \
                self.refl_refine_input_2c[1], self.refl_refine_input_2c[2])
            self.refl_refine_input_list_2c[self.curr_refl_index_2c] = self.refl_refine_input_2c.copy()
            #self.refl_list_e4c.append(self.refl_refine) #TODO maybe delete
            self.curr_refl_index_2c += 1


    def del_refl_refine(self): #TODO verify that the index selected matches the_refl_list
        i = self.selected_refl_index
        the_refl_list = self.sample.reflections_get()
        print(f'\nindex: {i}\n')
        if (self.geom==0) or (self.geom==1) and (i<=self.curr_refl_index_e4c):
            self.sample.del_reflection(the_refl_list[i])
            self.refl_refine_input_list_e4c[i] = [0.,0.,0.,0.,0.,0.,0.]
            for j in range(i, 9):
                self.refl_refine_input_list_e4c[j] = self.refl_refine_input_list_e4c[j+1]
            self.refl_refine_input_list_e4c[9] = [0.,0.,0.,0.,0.,0.,0.]
            self.curr_refl_index_e4c -= 1

        elif (self.geom==2) and (i<=self.curr_refl_index_k4c):
            #self.sample.del_reflection(self.refl_list_k4c[i])
            self.sample.del_reflection(the_refl_list[i])
            self.refl_refine_input_list_k4c[i] = [0.,0.,0.,0.,0.,0.,0.]
            for j in range(i, 9):
                self.refl_refine_input_list_k4c[j] = self.refl_refine_input_list_k4c[j+1]
            self.refl_refine_input_list_k4c[9] = [0.,0.,0.,0.,0.,0.,0.]
            self.curr_refl_index_k4c -= 1

        elif (self.geom==3) and (i<=self.curr_refl_index_e6c):
            #self.sample.del_reflection(self.refl_list_e6c[i])
            self.sample.del_reflection(the_refl_list[i])
            self.refl_refine_input_list_e6c[i] = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
            for j in range(i, 9):
                self.refl_refine_input_list_e6c[j] = self.refl_refine_input_list_e6c[j+1]
            self.refl_refine_input_list_e6c[9] = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
            self.curr_refl_index_e6c -= 1

        elif (self.geom==4) and (i<=self.curr_refl_index_k6c):
            #self.sample.del_reflection(self.refl_list_k6c[i])
            self.sample.del_reflection(the_refl_list)
            self.refl_refine_input_list_k6c[i] = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
            for j in range(i, 9):
                self.refl_refine_input_list_k6c[j] = self.refl_refine_input_list_k6c[j+1]
            self.refl_refine_input_list_k6c[9] = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
            self.curr_refl_index_k6c -= 1

        elif (self.geom==5) and (i<=self.curr_refl_index_2c):
            self.sample.del_reflection(the_refl_list[i])
            self.refl_refine_input_list_2c[i] = [0.,0.,0.,0.,0.]
            for j in range(i, 9):
                self.refl_refine_input_list_2c[j] = self.refl_refine_input_list_2c[j+1]
            self.refl_refine_input_list_e4c[9] = [0.,0.,0.,0.,0.]
            self.curr_refl_index_2c -= 1


    def clear_all_reflections(self):
        '''
        if need to redo busing-levy
        '''
        self.refl_refine_input_list_e4c = []
        self.refl_refine_input_list_k4c = []
        self.refl_refine_input_list_e6c = []
        self.refl_refine_input_list_k6c = []
        self.refl_refine_input_list_2c = []
        for i in range(self.num_reflections):
            self.refl_refine_input_list_e4c.append([0., 0., 0., 0., 0., 0., 0.])
            self.refl_refine_input_list_k4c.append([0., 0., 0., 0., 0., 0., 0.])
            self.refl_refine_input_list_e6c.append([0., 0., 0., 0., 0., 0., 0., 0., 0.])
            self.refl_refine_input_list_k6c.append([0., 0., 0., 0., 0., 0., 0., 0., 0.])
            self.refl_refine_input_list_2c.append([0., 0., 0., 0., 0.])
        the_refl_list = self.sample.reflections_get()
        for item in the_refl_list:
            self.sample.del_reflection(item)
        if (self.geom==0) or (self.geom==1):    
            self.curr_refl_index_e4c = 0
        elif (self.geom==2):
            self.curr_refl_index_k4c = 0
        elif (self.geom==3):
            self.curr_refl_index_e6c = 0
        elif (self.geom==4):
            self.curr_refl_index_k6c = 0
        elif (self.geom==5):
            self.curr_refl_index_2c = 0

    def read_cif(self, givenpath):
        self.cif_path = givenpath

    def run_cif(self):
        #TODO better error handling within intensities function
        try:
            self.hkl_path, temp_intensities, templatt = intensity_calc(self.wavelength, self.cif_path)
        except Exception as e:
            self.errors = f'run_cif error (input file): {e}'
            return
        self.intensities = [ord(c) for c in str(temp_intensities)]
        try:
            self.latt[0] = templatt['a']
            self.latt[1] = templatt['b']
            self.latt[2] = templatt['c']
            self.latt[3] = templatt['alpha']
            self.latt[4] = templatt['beta']
            self.latt[5] = templatt['gamma']
            self.start()
        except Exception as e:
            self.errors = f'run_cif error (lattice parameters): {e}'
            return

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


    def det_vis_lst(self):
        # Assume 6-circle for now #TODO
        # make hardcoded values into PVs #TODO
        R = 70
        geom = 'E6C'

        self.min_intensity = 10
   
        self.wavelength
        angle_deg = 20
        self.zmax = R*np.tan(np.deg2rad(angle_deg))
        self.zmin = -self.zmax

        det_angle_deg = 7.5
        det_zmax = R*np.tan(np.deg2rad(det_angle_deg))
        det_zmin = -det_zmax
        det_height = det_zmax - det_zmin

        self.visfulllst = intensities2detint_e6c(self.cif_path, self.hkl_path, self.wavelength, self.min_intensity, R, geom, self.zmin, self.zmax) #TODO cyl_center, ray_origin both 0's, gamma/delta axis hardcoded
        

    def compute_heatmap(self):
        #TODO
        threshold = 0.5
        darwidth = 100
        gauss_sig = 2
        window_width = 120
        mult = 5
        y_range = 2*self.zmax
        nx, ny = int(mult*window_width), int(mult*y_range)
        theta_grid = np.linspace(0, 360, nx, endpoint=False)
        z_grid = np.linspace(self.zmin, self.zmax, ny)
        if self.visfulllst != []:
            data = np.array(self.visfulllst)
            if data.ndim==2:
                theta, z, intensity, h, k, l, mu, omega, chi, phi, gamma, delta = ( \
                    data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], \
                    data[:, 6], data[:, 7], data[:, 8], data[:, 9], data[:, 10], data[:, 11])
                peaklist = []
                heatmap = np.zeros((ny,nx))
                for t,zz,inten,o,hh,kk,ll,ga,de in zip(theta,z,intensity,omega,h,k,l,gamma,delta):
                    if (inten>threshold) and ((self.cur_angle - darwidth) <= o <= (self.cur_angle + darwidth)):
                        i = int(nx * t /360) % nx
                        j = np.searchsorted(z_grid, zz)
                        if 0 <= j < ny:
                            heatmap[j,i] += inten
                            peaklist.append({
                                'h': hh, 'k': kk, 'l': ll, \
                                'theta': t, 'z': zz, 'intensity': inten, 'omega':o, \
                                'gamma':ga, 'delta':de})
                blurred = gaussian_filter(heatmap, sigma=gauss_sig)
                #blurred = heatmap
                if blurred.max() != 0:
                    blurred /= blurred.max()

                flat = blurred.flatten()
                self.det2dvis = flat.astype(float).tolist()
                # self.peaklist = something #TODO
        else:
            print("NO DATA")
            return




    def get_info(self):
        lines = []
        #lines.append(f'self.geom: {self.geom}')
        diff_geom = self.factory.name_get()
        lines.append(str(diff_geom))
        x = self.engine_hkl.axis_names_get(Hkl.EngineAxisNamesGet.READ)
        lines.append(f'diffractometer axes:\n{x}')
        samp = self.sample.lattice_get()
        a, b, c, alpha, beta, gamma = samp.get(Hkl.UnitEnum.USER)
        lines.append(f'a: {a}, b: {b}, c: {c}, alpha: {alpha}, beta: {beta}, gamma: {gamma}')
        lines.append(f'UB: {self.UB_matrix}')
        lines.append(f'latt vol: {self.lattice_vol}')
        return '\n'.join(lines)
