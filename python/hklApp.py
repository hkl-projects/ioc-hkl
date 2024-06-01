import math
#import numpy as np # use to set defaults to np.nan. 0 may be a valid result
import gi
from gi.repository import GLib
gi.require_version('Hkl', '5.0')
from gi.repository import Hkl
#from epics import PV

class hklCalculator():
    def __init__(self):
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
            values_hkl :list: (h,k,l, )
            UB_matrix (?)        
        '''
        #TODO dynamically assign input/output variables based on geom
        # need wavelength, unused arguement for now
        detector = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        factory = Hkl.factories()[geom]
        geometry = factory.create_new_geometry()
        try:
            geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
        except:
            print("invalid axes values")
            #TODO catch only specific error, this may overlap with other issues and mask them
        axis_names = geometry.axis_names_get()
        sample = Hkl.Sample.new("toto")
        lattice = Hkl.Lattice.new(*latt)
        sample.lattice_set(lattice)
        # compute all the pseudo axes managed by all engines
        engines = factory.create_new_engine_list()
        engines.init(geometry, detector, sample)
        engines.get()
        # get the hkl engine and do a computation
        hkl = engines.engine_get_by_name("hkl")
        values_hkl = hkl.pseudo_axis_values_get(Hkl.UnitEnum.USER)
        #TODO compute UB matrix ? maybe its own function, or read in from epics
        # could use Hkl.compute_UB_busing_levy, Hkl.hkl_sample_compute_UB_busing_levy, Hkl.hkl_sample_UB_get, ... see "strings /usr/local/lib/girepository-1.0/Hkl-5.0.typelib" 
        self.pseudoaxes_h, self.pseudoaxes_k, self.pseudoaxes_l = values_hkl
        print(self.pseudoaxes_h)
        return values_hkl

    def backward(self, wavelength, latt, geom, values_hkl):
        '''
        backward hkl calculation, reciprocal -> real
        inputs
            wavelength :float:
            latt :: basis vectors of crystal lattice in radians
            geometry :str: instrument specific geometry. Options: E4CV (4-circle) ...
            value_hkl :list: (?)

        outputs
            2theta, ... rotations
            UB_matrix (?)        
        '''
        # placeholder
        pass

    def get_pseudoaxes(self):
        pseudoaxes = (self.pseudoaxes_h, self.pseudoaxes_k, self.pseudoaxes_l)
        print(pseudoaxes)
        return pseudoaxes

    def get_axes(self):
        # placeholder, return values depend on geom
        axes = (self.axes_mu, self.axes_omega, self.axes_chi)
        print(axes)
        return axes

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in [a for a in dir(self) if not a.startswith('__')]:
                self.key = value #?
        #otherwise just 'try' all of these:
        self.hkl_wavelength = wavelength
        self.hkl_geom = geom
        self.hkl_lattice = lattice
        self.axes_mu = mu
        self.axes_omega = omega
        self.axes_chi = chi
        self.axes_phi = phi
        self.axes_gamma = gamma
        self.axes_delta = delta
        self.pseudoaxes_h = h
        self.pseudoaxes_k = k
        self.pseudoaxes_l = l
        self.pseudoaxes_psi = psi
        self.pseudoaxes_q = q
        self.pseudoaxes_alpha = alpha
        self.pseudoaxes_qper = qper
        self.pseudoaxes_qpar = qpar
        self.pseudoaxes_tth = tth
        self.pseudoaxes_incidence = incidence
        self.pseudoaxes_azimuth = azimuth
        self.pseudoaxes_emergence = emergence
        
    def test_object(self):
        wavelength = 1. #Angstrom
        geom = 'K6C'
        lattice = [1.54, 1.54, 1.54,
                math.radians(90.0),
                math.radians(90.0),
                math.radians(90.)]
        values_w = [1.0, 1.0, 1.0, 90.0, 90.0, 90.0] # cubic
        #instance of HklCalculator object
        results = self.forward(wavelength=wavelength, geom=geom, latt=lattice, values_w=values_w)
        print(results) 


