TOP=../..

include $(TOP)/configure/CONFIG
-include $(TOP)/configure/CONFIG.PyDevice

#----------------------------------------
#  ADD MACRO DEFINITIONS AFTER THIS LINE
#=============================

#=============================
# Build the IOC application
#

PROD_IOC = hklApp
# pydevioc.dbd will be created and installed
DBD += hklApp.dbd

# pydev.dbd will be made up from these files:
hklApp_DBD += base.dbd
hklApp_DBD += pydev.dbd
hklApp_DBD += pycalcRecord.dbd

# Include dbd files from all support applications:
#pydevioc_DBD += xxx.dbd

# Add all the support libraries needed by this IOC
hklApp_LIBS += pydev

# pydevioc_registerRecordDeviceDriver.cpp derives from pydev.dbd
hklApp_SRCS += hklApp_registerRecordDeviceDriver.cpp

# Build the main IOC entry point on workstation OSs.
hklApp_SRCS_DEFAULT += main.cpp
hklApp_SRCS_vxWorks += -nil-

# Add support from base/src/vxWorks if needed
#pydevioc_OBJS_vxWorks += $(EPICS_BASE_BIN)/vxComLibrary

# Finally link to the EPICS Base libraries
hklApp_LIBS += $(EPICS_BASE_IOC_LIBS)

#===========================

include $(TOP)/configure/RULES
#----------------------------------------
#  ADD RULES AFTER THIS LINE

