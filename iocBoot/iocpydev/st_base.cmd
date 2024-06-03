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
dbLoadRecords("${TOP}/db/hklApp.db", "P=$(PREFIX),R=hb3:")
dbLoadRecords("${TOP}/db/hkl.template","P=$(PREFIX),R=hb3:")

cd ${TOP}/iocBoot/${IOC}

pydev("import hklApp")
pydev("from hklApp import hklCalculator")
pydev("hkl_calc = hklCalculator()")
pydev("hkl_calc.test()")

iocInit
