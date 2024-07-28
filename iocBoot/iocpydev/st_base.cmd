#!../../bin/linux-x86_64/hklApp

< envPaths

# PYTHONPATH points to folders where Python modules are.
epicsEnvSet("PYTHONPATH","$(TOP)/python")

# Prefix set for Triple Axis Spectrometer -> TAS
epicsEnvSet("PREFIX", "HB3:ioc-hkl:")

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
dbpf("$(PREFIX)a","5.431")
dbpf("$(PREFIX)b","5.431")
dbpf("$(PREFIX)c","5.431")
dbpf("$(PREFIX)alpha","90")
dbpf("$(PREFIX)beta","90")
dbpf("$(PREFIX)gamma","90")
dbpf("$(PREFIX)omega","30")
dbpf("$(PREFIX)chi","20")
dbpf("$(PREFIX)phi","10")
dbpf("$(PREFIX)tth","10")
dbpf("$(PREFIX)h","1")
dbpf("$(PREFIX)k","1")
dbpf("$(PREFIX)l","1")
dbpf("$(PREFIX)refl1h","0")
dbpf("$(PREFIX)refl1k","0")
dbpf("$(PREFIX)refl1l","4")
dbpf("$(PREFIX)refl1omega","-145")
dbpf("$(PREFIX)refl1chi","0")
dbpf("$(PREFIX)refl1phi","0")
dbpf("$(PREFIX)refl1tth","60")
dbpf("$(PREFIX)refl2h","0")
dbpf("$(PREFIX)refl2k","4")
dbpf("$(PREFIX)refl2l","0")
dbpf("$(PREFIX)refl2omega","-145")
dbpf("$(PREFIX)refl2chi","90")
dbpf("$(PREFIX)refl2phi","0")
dbpf("$(PREFIX)refl2tth","69")
dbpf("$(PREFIX)errors","my string test")


#dbl > pvlist.dbl
