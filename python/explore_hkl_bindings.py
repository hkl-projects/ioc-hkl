# in shell: source /epics/iocs/ioc-hkl/iochkl/bin/activate
# export GI_TYPELIB_PATH=/usr/local/lib/girepository-1.0

import numpy as np
import math
import gi
from gi.repository import GLib
gi.require_version('Hkl', '5.0')
from gi.repository import Hkl

# startup
detector = Hkl.Detector.factory_new(Hkl.DetectorType(0))
factory  = Hkl.factories()['E4CV']
geometry = factory.create_new_geometry()
geometry.wavelength_set(0.5, Hkl.UnitEnum.USER)
sample = Hkl.Sample.new("toto") # sample. tab to check attributes

lattice_list = [5.41,5.41,5.41,90.,90.,90.]
a,b,c,alpha,beta,gamma=lattice_list

alpha = math.radians(alpha)
beta  = math.radians(beta)
gamma = math.radians(gamma)
lattice = Hkl.Lattice.new(a,b,c,alpha,beta,gamma)
sample.lattice_set(lattice)

engines = factory.create_new_engine_list()
engines.init(geometry, detector, sample)
engines.get()

# depends on which geom
engine_hkl       = engines.engine_get_by_name("hkl")
engine_psi       = engines.engine_get_by_name("psi")
engine_q         = engines.engine_get_by_name("q")
#engine_q2        = engines.engine_get_by_name("q2")
#engine_qperqpar  = engines.engine_get_by_name("qper_qpar")
#engine_tth       = engines.engine_get_by_name("tth2")
#engine_eulerians = engines.engine_get_by_name("eulerians")
engine_incidence = engines.engine_get_by_name("incidence")
engine_emergence = engines.engine_get_by_name("emergence")

UB_matrix = np.zeros((3,3), dtype=float)
UB = sample.UB_get()
UB_matrix[:,:] = [[UB.get(i, j) for j in range(3)] for i in range(3)]
print(UB_matrix)

### Busing-Levy
#add reflection 1
values_w = [-145., 0., 0., 60.] 
geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
refl1 = sample.add_reflection(geometry, \
                              detector, \
                              0., \
                              0., \
                              4.)

#add reflection 2
values_w = [-145., 90., 0., 60.] 
geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
refl2 = sample.add_reflection(geometry, \
                              detector, \
                              0., \
                              4., \
                              0.)

#compute UB
sample.compute_UB_busing_levy(refl1, refl2) #automatically changes sample UB
UB_matrix[:,:] = [[UB.get(i, j) for j in range(3)] for i in range(3)]
print(UB_matrix)

# refine UB with more reflections
# reflection 3, needs to be sensical
values_w = [-145., 95., 0., 60.] 
geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
refl3 = sample.add_reflection(geometry, \
                              detector, \
                              0., \
                              4., \
                              1.)
sample.affine()

UB_matrix[:,:] = [[UB.get(i, j) for j in range(3)] for i in range(3)]
lattice_list = lattice.get(Hkl.UnitEnum.USER)
print(lattice_list)
print(UB_matrix)

sample.reflections_get()

refl1
sample.del_reflection(refl1)
# explore reflections
# sample.del_reflection()
# sample.reflections_get()


# set pseudoaxes
h = 0
k = 1
l = 1
solutions = engine_hkl.pseudo_axis_values_set([h,k,l], Hkl.UnitEnum.USER)

psi = 56.0
#solutions = engine_psi.pseudo_axis_values_set([psi], Hkl.UnitEnum.USER)

q = 5.
#solutions = engine_q.pseudo_axis_values_set([q], Hkl.UnitEnum.USER)

# set engine parameters
#engine_psi.parameters_values_set([values...], Hkl.UnitEnum.USER)
