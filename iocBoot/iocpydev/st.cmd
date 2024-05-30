#!../../bin/linux-x86_64/hklApp

< envPaths

# PYTHONPATH points to folders where Python modules are.
epicsEnvSet("PYTHONPATH","$(TOP)/python")

cd ${TOP}

## Register all support components
dbLoadDatabase "${TOP}/dbd/hklApp.dbd"
hklApp_registerRecordDeviceDriver pdbbase

## Load record instances
dbLoadRecords("${TOP}/db/hklApp.db")

cd ${TOP}/iocBoot/${IOC}

pydev("import hklApp")
pydev("hklApp.test()")
# pydev("hklApp.forward()")
# pydev("hklApp.backward()")

iocInit
