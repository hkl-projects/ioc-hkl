import math
from gi.repository import GLib
from gi.repository import Hkl

detector = Hkl.Detector.factory_new(Hkl.DetectorType(0))

factory = Hkl.factories()['K6C']
geometry = factory.create_new_geometry()
values_w = [0., 30., 0., 0., 0., 60.]
geometry.axis_values_set(values_w, Hkl.UnitEnum.USER)
axis_names = geometry.axis_names_get()
print(geometry.name_get(), "diffractometer has", len(axis_names),\
      "axes : ", axis_names)
print(values_w)
sample = Hkl.Sample.new("toto")
lattice = Hkl.Lattice.new(1.54, 1.54, 1.54,
                          math.radians(90.0),
                          math.radians(90.0),
                          math.radians(90.))
sample.lattice_set(lattice)

# compute all the pseudo axes managed by all engines
engines = factory.create_new_engine_list()
engines.init(geometry, detector, sample)
engines.get()

# get the hkl engine and do a computation
hkl = engines.engine_get_by_name("hkl")
values = hkl.pseudo_axis_values_get(Hkl.UnitEnum.USER)
print("read : ", values)

# set the hkl engine and get the results
for _ in range(100):
    try:
        print()
        solutions = hkl.pseudo_axis_values_set(values,
                                               Hkl.UnitEnum.USER)
        print(hkl.pseudo_axis_values_get(Hkl.UnitEnum.USER))

        print("idx".center(15)),
        for name in axis_names:
            print("{}".format(name.center(15))),
        print()

        for i, item in enumerate(solutions.items()):
            read = item.geometry_get().axis_values_get(Hkl.UnitEnum.USER)
            print("{}".format(repr(i).center(15))),
            for value in read:
                print("{}".format(repr(value)[:15].center(15))),
            print()
    except GLib.GError as err:
        print(values, err)
    values[1] += .01
