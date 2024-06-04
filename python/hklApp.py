import math
import numpy as np
import pandas as pd
import gi
from gi.repository import GLib
gi.require_version('Hkl', '5.0')
from gi.repository import Hkl
#from epics import PV
import inspect

class hklCalculator():
    def __init__(self, num_axes_solns=10):
        self.num_axes_solns = num_axes_solns
        #TODO differentiate omega/komega for kappa
        self.wavelength = np.nan
        self.geom = '' #TODO, map to ints
        self.lattice = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] # a1, a2, a3, alpha, beta, gamma
        # self.axes_mu = np.nan
        self.axes_omega = np.nan
        self.axes_chi = np.nan
        self.axes_phi = np.nan
        self.axes_tth = np.nan
        self.pseudoaxes_h = np.nan
        self.pseudoaxes_k = np.nan
        self.pseudoaxes_l = np.nan
        self.pseudoaxes_psi = np.nan
        self.pseudoaxes_q = np.nan
        self.pseudoaxes_incidence = np.nan
        self.pseudoaxes_inc_azimuth = np.nan
        self.pseudoaxes_emergence = np.nan
        self.pseudoaxes_emer_azimuth = np.nan
        self.axes_solns_omega = []
        self.axes_solns_chi = []
        self.axes_solns_phi = []
        self.axes_solns_tth = []
        for _ in range(self.num_axes_solns):
            self.axes_solns_omega.append(np.nan)
            self.axes_solns_chi.append(np.nan)
            self.axes_solns_phi.append(np.nan)
            self.axes_solns_tth.append(np.nan)

    def forward(self, wavelength=None, geom=None, latt=None, values_w=None):
        '''
        forward hkl calculation, real -> reciprocal
        inputs
            wavelength :float:
            lattice :: basis vectors of crystal lattice in radians
            geom :str: instrument specific geometry. Options: E4CV (4-circle) ...
            value_w :list: takes in a list of 6 elements corresponding to ...

        outputs (for E4CV)
            values_hkl :list: (h,k,l)
        '''
        print("Forward function start")
        #TODO dynamically assign input/output variables based on geom
        #self.wavelength = wavelength should maybe put this all in init
        detector   = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        factory    = Hkl.factories()[geom]
        geometry   = factory.create_new_geometry()
        try:
            geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
        except:
            print("invalid axes values")
            #TODO catch different types of errors
        geometry.wavelength_set(wavelength, Hkl.UnitEnum.USER)
        axis_names = geometry.axis_names_get()
        sample     = Hkl.Sample.new("toto")
        lattice    = Hkl.Lattice.new(*latt)
        sample.lattice_set(lattice)
        engines    = factory.create_new_engine_list()
        engines.init(geometry, detector, sample)
        engines.get()
        hkl        = engines.engine_get_by_name("hkl")
        values_hkl = hkl.pseudo_axis_values_get(Hkl.UnitEnum.USER)
        self.pseudoaxes_h, self.pseudoaxes_k, self.pseudoaxes_l = values_hkl
        #return values_hkl # no returns, just get functions

    def backward(self, wavelength, latt, geom, values_hkl):
        '''
        backward hkl calculation, reciprocal -> real
        inputs
            wavelength :float:
            latt :: basis vectors of crystal lattice in radians
            geometry :str: instrument specific geometry. Options: E4CV (4-circle) ...
            value_hkl :list: 

        outputs
            dependent on geometry, for 4-circle: [omega, chi, phi, 2theta]
        '''
        print("Backward function start")
        detector   = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        factory    = Hkl.factories()[geom]
        geometry   = factory.create_new_geometry()
        geometry.wavelength_set(wavelength, Hkl.UnitEnum.USER)
        axis_names = geometry.axis_names_get()
        sample     = Hkl.Sample.new("toto")
        lattice    = Hkl.Lattice.new(*latt)
        sample.lattice_set(lattice)
        # compute all the pseudo axes managed by all engines
        engines = factory.create_new_engine_list()
        engines.init(geometry, detector, sample)
        engines.get()
        # get the hkl engine and do a computation
        hkl = engines.engine_get_by_name("hkl") 
        solutions = hkl.pseudo_axis_values_set(values_hkl, Hkl.UnitEnum.USER)
        print("axis names: ", axis_names)
        values_w_all = []
        for i, item in enumerate(solutions.items()):
            #print("id: ", i) # for kappa 6-circle
            read = item.geometry_get().axis_values_get(Hkl.UnitEnum.USER)
            #print("motor axes solution values: ", read)
            values_w_all.append(read)
        for i in range(self.num_axes_solns):
            self.axes_solns_omega[i], self.axes_solns_chi[i], \
            self.axes_solns_phi[i], self.axes_solns_tth[i] = values_w_all[i]           
    def get_pseudoaxes(self):
        pseudoaxes = (self.pseudoaxes_h, \
                      self.pseudoaxes_k, \
                      self.pseudoaxes_l)
        return pseudoaxes

    def get_axes(self, cleanprint=False):
        '''
        Which rotations are returned depends on geometry, 4-circle currently
        inputs
            cleanprint :bool: 
                if true, dataframe 
                if false, dict
        outputs
            axes :dict: 
                key is rotation name, value is list of solutions up to num_axes_solns
        '''
        axes = {}
        axes['omega'] = self.axes_solns_omega
        axes['chi']   = self.axes_solns_chi
        axes['phi']   = self.axes_solns_phi
        axes['tth']   = self.axes_solns_tth
        if(cleanprint==False):
            return axes
        elif(cleanprint==True):
            axes_df = pd.DataFrame(axes)
            return axes_df

    def update(self, **kwargs):
        #TODO get this working, currently updating in forward/backward
        for key, value in kwargs.items():
            name = f'self.{key}'
            if name in [a for a in dir(self) if not a.startswith('__')]:
                name = value #need to replace string name with variable

        def compute_UB_matrix(self):
            # placeholder
            # could use Hkl.compute_UB_busing_levy, Hkl.hkl_sample_compute_UB_busing_levy, Hkl.hkl_sample_UB_get, ... see "strings /usr/local/lib/girepository-1.0/Hkl-5.0.typelib" 
            pass
        
    def test(self):
        wavelength = 1.54 #Angstrom
        geom = 'E4CV' # 4-circle
        lattice = [1.54, 1.54, 1.54,
                math.radians(90.0),
                math.radians(90.0),
                math.radians(90.)] # cubic
        # forward test
        values_w = [30., 0., 0., 60.0] # [omega, chi, phi, tth]
        print("Running test - Initial conditions:")
        print("##################################")
        print("wavelength: ", wavelength)
        print("geometry: ", geom)
        print("lattice: ", lattice)
        self.forward(wavelength=wavelength, geom=geom, latt=lattice, values_w=values_w) 
        f_results = self.get_pseudoaxes()
        print("input motor values: ", values_w) 
        print("forward function results:\n", f_results)  
        # backward test
        values_hkl = [0.0, 0.0, 1.0] # h, k, l
        self.backward(wavelength=wavelength, geom=geom, latt=lattice, values_hkl=values_hkl)
        b_results = self.get_axes(cleanprint=True)
        print("input hkl values: ", values_hkl)
        print("backward function results:\n", b_results)
                

