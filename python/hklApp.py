from enum import IntEnum
from E4CV import hklCalculator_E4CV

class DiffGeometry(IntEnum): #TODO get selection of geom working before instance of class
    E4CV = 0
    K6C = 1
    E4CV_2 = 2
    E4CV_3 = 3
    E4CV_4 = 4
    E4CV_5 = 5
    E4CV_6 = 6

geom_num = 0
if geom_num == 0:
    hkl_calc = hklCalculator_E4CV()
elif geom_num == 1:
    hkl_calc = hklCalculator_E4CV() #placeholder for other geoms

hkl_calc.test()
