import math
#import numpy as np # use to set defaults to np.nan. 0 may be a valid result
import gi
from gi.repository import GLib
gi.require_version('Hkl', '5.0')
from gi.repository import Hkl
#from epics import PV
import inspect

class hklCalculator():
    def __init__(self):
        #TODO differentiate omega/komega for kappa
        self.wavelength = 0
        self.geom = '' #TODO, map to ints
        self.lattice = [0, 0, 0, 0, 0, 0] # a1, a2, a3, alpha, beta, gamma
        self.realaxes_mu = 0
        self.realaxes_omega = 0
        self.realaxes_chi = 0
        self.realaxes_phi = 0
        self.realaxes_gamma = 0
        self.realaxes_delta = 0
        self.pseudoaxes_h = 0
        self.pseudoaxes_k = 0
        self.pseudoaxes_l = 0
        self.pseudoaxes_psi = 0
        self.pseudoaxes_q = 0
        self.pseudoaxes_alpha = 0
        self.pseudoaxes_qper = 0
        self.pseudoaxes_qpar = 0
        self.pseudoaxes_tth = 0
        self.pseudoaxes_incidence = 0
        self.pseudoaxes_azimuth = 0
        self.pseudoaxes_emergence = 0
        #self.pseudoaxes_azimuth 2 azimuth values?
        #self.pseudoaxes_alpha shows up in the gui twice too       

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
            #print("id: ", i) # kappa 6-circle
            read = item.geometry_get().axis_values_get(Hkl.UnitEnum.USER)
            #print("real values: ", read)
            values_w_all.append(read)
        #TODO expand to include all solns, need array PV or something similar
        self.realaxes_omega, self.realaxes_chi, self.realaxes_phi, \
        self.realaxes_tth = values_w_all[0]
        return values_w_all #TODO replace with get_realaxes when handling multiple solns

    def get_pseudoaxes(self):
        pseudoaxes = (self.pseudoaxes_h, \
                      self.pseudoaxes_k, \
                      self.pseudoaxes_l)
        return pseudoaxes

    def get_realaxes(self):
        # which rotations are returned depends on geometry, 4-circle currently
        realaxes = (self.realaxes_omega, \
                    self.realaxes_chi, \
                    self.realaxes_phi, \
                    self.realaxes_tth)
        return realaxes

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
        b_results = self.backward(wavelength=wavelength, geom=geom, latt=lattice, values_hkl=values_hkl)
        #b_results = self.get_realaxes() only returns 1 solution, sort out PV arrays
        print("input hkl values: ", values_hkl)
        print("backward function test results:", *b_results, sep="\n")


