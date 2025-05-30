#!../../bin/linux-x86_64/hklApp

< envPaths

# PYTHONPATH points to folders where Python modules are.
epicsEnvSet("PYTHONPATH","$(TOP)/python")

# Designate beamline:IOC_name as conventional prefix
epicsEnvSet("PREFIX", "abtest:ioc-hkl:")

cd ${TOP}

## Register all support components
dbLoadDatabase "${TOP}/dbd/hklApp.dbd"
hklApp_registerRecordDeviceDriver pdbbase

## Load record instances
dbLoadRecords("db/hkl_main.db")

cd ${TOP}/iocBoot/${IOC}

pydev("import hklApp")
pydev("hkl_calc = hklApp.hklCalcs()")

iocInit

epicsThreadSleep(1)

dbpf("$(PREFIX)geom","2")
dbpf("$(PREFIX)wlen","5.431")
dbpf("$(PREFIX)latt_a","5.431")
dbpf("$(PREFIX)latt_b","5.431")
dbpf("$(PREFIX)latt_c","5.431")
dbpf("$(PREFIX)latt_alpha","90")
dbpf("$(PREFIX)latt_beta","90")
dbpf("$(PREFIX)latt_gamma","90")
dbpf("$(PREFIX)omega_e4c","30")
dbpf("$(PREFIX)chi_e4c","20")
dbpf("$(PREFIX)phi_e4c","10")
dbpf("$(PREFIX)tth_e4c","10")
dbpf("$(PREFIX)h","1")
dbpf("$(PREFIX)k","1")
dbpf("$(PREFIX)l","1")
dbpf("$(PREFIX)refl1_h_e4c","0")
dbpf("$(PREFIX)refl1_k_e4c","0")
dbpf("$(PREFIX)refl1_l_e4c","4")
dbpf("$(PREFIX)refl1_omega_e4c","-145")
dbpf("$(PREFIX)refl1_chi_e4c","0")
dbpf("$(PREFIX)refl1_phi_e4c","0")
dbpf("$(PREFIX)refl1_tth_e4c","60")
dbpf("$(PREFIX)refl2_h_e4c","0")
dbpf("$(PREFIX)refl2_k_e4c","4")
dbpf("$(PREFIX)refl2_l_e4c","0")
dbpf("$(PREFIX)refl2_omega_e4c","-145")
dbpf("$(PREFIX)refl2_chi_e4c","90")
dbpf("$(PREFIX)refl2_phi_e4c","0")
dbpf("$(PREFIX)refl2_tth_e4c","69")
#dbpf("$(PREFIX)errors","my string test")


dbl > pvlist.dbl
