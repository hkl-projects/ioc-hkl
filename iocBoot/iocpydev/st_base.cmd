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
dbLoadRecords("db/hkl_main.db")

cd ${TOP}/iocBoot/${IOC}

pydev("import hklApp")
pydev("hkl_calc = hklApp.hklCalcs()")
#pydev("hkl_calc.test()")

iocInit

epicsThreadSleep(1)
dbpf("HB3:wlen","1.54")
dbpf("HB3:a","1.54")
dbpf("HB3:b","1.54")
dbpf("HB3:c","1.54")
dbpf("HB3:alpha","90")
dbpf("HB3:beta","90")
dbpf("HB3:gamma","90")

pydev("hkl_calc.set_sample()")

dbpf("HB3:omega","30")
dbpf("HB3:chi","0")
dbpf("HB3:phi","0")
dbpf("HB3:tth","60")
#dbl > pvlist.dbl
