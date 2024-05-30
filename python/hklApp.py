import math
from gi.repository import GLib
from gi.repository import Hkl

def test_object():
    detector = Hkl.Detector.factory_new(Hkl.DetectorType(0))
    #factory = Hkl.factories()['E4CV']
    factory = Hkl.factories()['K6C']
    geometry = factory.create_new_geometry()
    values_w = [0., 30., 0., 0., 0., 60.]
    geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
    axis_names = geometry.axis_names_get()
    print(geometry.name_get(), "diffractometer has", len(axis_names),\
        "axes : ", axis_names)
    print(values_w)
    return "yep"

def forward(wavelength=None, latt=None, geom=None, values_w=None):
    '''
    forward hkl calculation, real -> reciprocal
    inputs
        wavelength :float:
        lattice :: basis vectors of crystal lattice in radians
        geom :str: instrument specific geometry. Options: E4CV (4-circle) ...
        value_w :list: takes in a list of 6 elements corresponding to ...

    outputs
        values_hkl :list: (h,k,l)
        UB_matrix (?)        
    '''
    detector = Hkl.Detector.factory_new(Hkl.DetectorType(0))
    factory = Hkl.factories()[geom]
    geometry = factory.create_new_geometry()
    try:
        geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
    except:
        print("invalid rotations")
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
    return values_hkl

def backward(wavelength, latt, geom, values_hkl):
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

def test():
    latt = [1.54, 1.54, 1.54,
            math.radians(90.0),
            math.radians(90.0),
            math.radians(90.)]
    geom = 'K6C' 
    values_w = [0., 30., 0., 0., 0., 60.]
    results = forward(wavelength=None, latt=latt, geom=geom, values_w=values_w)
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

