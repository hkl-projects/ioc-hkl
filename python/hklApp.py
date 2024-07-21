from enum import IntEnum
from E4CV import hklCalculator_E4CV
from TwoC import hklCalculator_TwoC

def hklCalcs(geom_num=0):
    #TODO need to attach this function to dropdown box in phoebus
    #geom_num = 0
    if geom_num == 0: 
        hkl_calc = hklCalculator_E4CV()
    elif geom_num == 1:
        hkl_calc = hklCalculator_E4CV()
    elif geom_num == 2:
        hkl_calc = hklCalculator_E4CV()
    elif geom_num == 3:
        hkl_calc = hklCalculator_E4CV()
    elif geom_num == 4:
        hkl_calc = hklCalculator_E4CV()
    elif geom_num == 5:
        hkl_calc = hklCalculator_E4CV()
    elif geom_num == 6:
        hkl_calc = hklCalculator_E4CV()
    return hkl_calc
