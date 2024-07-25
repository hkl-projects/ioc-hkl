from enum import IntEnum
#from TwoC import hklCalculator_TwoC
#from E4CH import hklCalculator_E4CH
from E4CV import hklCalculator_E4CV
#from E6C import hklCalculator_E6C
'''
def hklCalcs(geom_num=2):
    #TODO need to attach this function to dropdown box in phoebus
    #geom_num = 0
    if geom_num == 0: 
        print("setting geom to TwoC")
        hkl_calc = hklCalculator_TwoC()
    elif geom_num == 1:
        print("setting geom to E4CH")
        hkl_calc = hklCalculator_E4CH()
    elif geom_num == 2:
        print("setting geom to E4CV")
        hkl_calc = hklCalculator_E4CV()
    elif geom_num == 3:
        print("setting geom to K4CV")
        hkl_calc = hklCalculator_E4CV()
    elif geom_num == 4:
        print("setting geom to E6C")
        hkl_calc = hklCalculator_E6C()
    elif geom_num == 5:
        print("setting geom to K6C")
        hkl_calc = hklCalculator_E4CV()
    elif geom_num == 6:
        print("setting geom to ZAXIS")
        hkl_calc = hklCalculator_E4CV()
    return hkl_calc
'''

#getPV(geom)


class DiffGeometry(IntEnum): #TODO get selection of geom working before instance of class
    E4CV = 0
    K6C = 1
    E4CV_2 = 2
    E4CV_3 = 3
    E4CV_4 = 4
    E4CV_5 = 5
    E4CV_6 = 6

def hklCalcs(geom_num=0):
    #TODO need to attach this function to dropdown box in phoebus
    #geom_num = 0
    if geom_num == 0:
        hkl_calc = hklCalculator_E4CV()
    elif geom_num == 1:
        hkl_calc = hklCalculator_E4CV() #placeholder for other geometries
    return hkl_calc


