import math
#import numpy as np # use to set defaults to np.nan. 0 may be a valid result
import gi
from gi.repository import GLib
gi.require_version('Hkl', '5.0')
from gi.repository import Hkl

#from epics import PV # how do i use this?

class hklCalculator():
    def __init__(self):
        self.hkl_wavelength = 0
        self.hkl_geom = '' #TODO, map to ints
        self.hkl_lattice = [0, 0, 0, 0, 0, 0]
        self.axes_mu = 0
        self.axes_oemga = 0
        self.axes_chi = 0
        self.axes_phi = 0
        self.axes_gamma = 0
        self.axes_delta = 0
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
        #self.pseudoaxes_ 2 azimuth values?
        #self.pseudoaxes_alpha shows up in the gui twice too

    def forward(self, wavelength, geom=None, latt=None, values_w=None):
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
        # wavelength
        detector = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        factory = Hkl.factories()[geom]
        geometry = factory.create_new_geometry()
        try:
            geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
        except:
            print("invalid motor rotation values")
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
        #TODO how do I update self variables without knowing which variables are being updated? Dependent on geom
        self.pseudoaxes_h, self.pseudoaxes_k, self.pseudoaxes_l = values_hkl
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
        pass

#    def get_axes():

#    def get_pseudoaxes():

#    def update_axes():
        
#    def update_pseudoaxes():

def test_object():
    wavelength = 1. #Angstrom
    geom = 'K6C'
    print(type(geom))
    lattice = [1.54, 1.54, 1.54,
            math.radians(90.0),
            math.radians(90.0),
            math.radians(90.)]
    values_w = [1.0, 1.0, 1.0, 90.0, 90.0, 90.0] # cubic
    #instance of HklCalculator object
    hkl_calc = hklCalculator()
    results = hkl_calc.forward(wavelength=wavelength, geom=geom, latt=lattice, values_w=values_w)
    print(results) 

#if __name__ == "__main__":
#    latt = [1.54, 1.54, 1.54,
#            math.radians(90.0),
#            math.radians(90.0),
#            math.radians(90.)]
#    geom = 'K6C' 
#    values_w = [0., 30., 0., 0., 0., 60.]
#    results = forward(wavelength=None, latt=latt, geom=geom, values_w=values_w)
#    print(results) 

