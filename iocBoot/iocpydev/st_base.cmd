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

#dbpf("$(PREFIX)geom","0")
#dbl > pvlist.dbl
