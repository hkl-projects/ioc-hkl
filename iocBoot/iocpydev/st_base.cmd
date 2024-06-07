#!../../bin/linux-x86_64/hklApp

< envPaths

# PYTHONPATH points to folders where Python modules are.
epicsEnvSet("PYTHONPATH","$(TOP)/python")

# Prefix set for Triple Axis Spectrometer -> TAS
epicsEnvSet("PREFIX", "TAS:")

cd ${TOP}

## Register all support components
dbLoadDatabase "${TOP}/dbd/hklApp.dbd"
hklApp_registerRecordDeviceDriver pdbbase

## Load record instances
#dbLoadRecords("db/hkl_main.template","P=$(PREFIX),R=hb3:")
dbLoadRecords("db/hkl_main.db")

cd ${TOP}/iocBoot/${IOC}

pydev("import hklApp")
#pydev("from hklApp import hklCalculator_E4CV")
#pydev("hkl_calc = hklCalculator_E4CV()")
#pydev("hkl_calc.test()")

iocInit

dbl > pvlist.dbl
