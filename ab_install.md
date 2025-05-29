# Installation steps for EPICS, IOCs (hkl, motorSim) 
## Assumptions
-Ubuntu

## dependencies
sudo apt-get install build-essential

## Directories
```bash
mkdir ~/epics/{modules,IOCs,support,GUI}
```

## EPICS
download EPICS base https://epics-controls.org/
```bash
cd ~/epics
mv ~/Downloads/base-7.0.9.tar.gz .
tar -xzf base-7.0.9.tar.gz
mv base-7.0.9/ base/
cd base/
make -j4
```
in ~/.bashrc, append to last line 
```
export PATH=$PATH:/epics/base/bin/linux-x86_64
export EPICS_CA_AUTO_ADDR_LIST=YES
```

## GUI
download Phoebus https://controlssoftware.sns.ornl.gov/css\_phoebus/
```bash
cd ~/epics/GUI
mv ~/Downloads/pheobus-linux.zip .
unzip phoebus-linux
```
run with ./phoebus-4.7.4-SNAPSHOT/phoebus.sh

## hkl
### download
```bash
cd ~/epics/support
git clone https://repo.or.cz/hkl.git
```
### install dependencies
```bash
sudo apt install gtk-doc-tools autoconf libgtkmm-3.0-dev libyaml-dev gettext autopoint gobject-introspection libtool autoconf-archive debhelper gnuplot-nox gobject-introspection gtk-doc-tools libbullet-dev libg3d-dev libg3d-plugins libgirepository1.0-dev libgl-dev libgsl-dev libgtk-3-dev libgtkglext1-dev libhdf5-dev python3-gi python3-pip elpa-htmlize dvipng libhdf5-dev povray asymptote libhdf5-dev libcglm-dev libinih-dev cabal-install
```
### configure and build
```bash
cd ~/epics/support/hkl
./autogen.sh
./configure --enable-introspection --disable-binoculars
make -j4
sudo make install
```
/usr/local/lib should now have libhkl.so, among others
```bash
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/lib
export GI_TYPELIB_PATH=/usr/local/lib/girepository-1.0 
```

## hkl IOC
### download
```bash
cd /epics/iocs
git clone https://github.com/hkl-projects/ioc-hkl.git
```
### check python executable, install numpy
```bash
which python3
```
should return
/usr/bin/python3
if on <=ubuntu 20.04
```bash
pip3 install numpy
```
if on >=ubuntu 24.04
```bash
sudo apt install python3-numpy
```
### set paths
if epics dir in ~
in RELEASE, set EPICS\_BASE=/home/ab/epics/base (needs to be absolute path)
if epics dir in /
in RELEASE, set EPICS\_BASE=/epics/base
### build
```bash
cd ioc-hkl
make
```

## motorSim IOC
### dependencies
```bash 
sudo apt install libtirpc-dev re2c flex
```

### download
```bash
cd ~/epics/iocs
git clone https://github.com/hkl-projects/motorMotorSim
```
```bash
cd ~/epics/modules
git clone https://github.com/epics-modules/motor.git
git clone https://github.com/epics-modules/asyn.git
git clone https://github.com/epics-modules/sequencer.git seq #renames dir to seq
```
### configuration files
                            motorMotorSim
For motorSim ioc, in ~/epics/iocs/motorMotorSim/configure/
```bash
cp EXAMPLE_CONFIG_SITE.local CONFIG_SITE.local
cp EXAMPLE_RELEASE.local RELEASE.local
```
in ~/epics/iocs/motorMotorSim/configure/RELEASE.local
```
MOTOR=/home/ab/epics/modules/motor
SEQ=/home/ab/epics/modules/seq
MOTOR_MOTORSIM=/home/ab/epics/iocs/motorMotorSim
EPICS_BASE=/home/ab/epics/base
```
MOTOR and SEQ lines should be added at the top of the file, these will not be here by default

in ~/epics/iocs/motorMotorSim/configure/CONFIG\_SITE.local, uncomment BUILD\_IOCS = YES
module: MOTOR MODULE
For motor module, in ~/epics/modules/motor/configure/RELEASE, change:
```
SUPPORT=/home/ab/epics/modules
ASYN=$(SUPPORT)/asyn
EPICS_BASE=/home/ab/epics/base
```
comment out all submodules except motorSim in ~/epics/motor/modules/Makefile
module: ASYN
in ~/epics/modules/asyn/configure/RELEASE
```
SUPPORT=/home/ab/epics/modules
EPICS_BASE=/home/ab/epics/base
```
module: SEQ
in ~/epics/modules/seq/configure/RELEASE
```
EPICS_BASE=/home/ab/epics/base
```
### build
```bash
cd ~/epics/modules/asyn
make -sj
cd ~/epics/modules/motor
make -sj
cd ~/epics/modules/seq
make -sj
cd ~/epics/iocs/motorMotorSim
make -sj
```
### run motorSim
run with ./st.cmd in ~/epics/iocs/motorMotorSim/iocs/motorSimIOC/iocBoot/iocMotorSim
can find .bob files in ~/epics/modules/motor/motorApp/op/bob/autoconvert
for example, motor8x.bob


### couple PVs between motorsim and hkl-ioc
copy motor/.../op/.../bob/.../motor8x.bob into op dir of hkl-ioc
adjust macros in phoebus to P => "IOC:motorsimtest_ab:"
add FLNK to records in order to bridge the two iocs (motorMotorSim and ioc-hkl)

change /epics/iocs/motorMotorSim/iocs/motorSimIOC/iocBoot/iocMotorSim/motor.substitutions to:
```
file "$(MOTOR)/db/basic_asyn_motor.db"
{
pattern
{P,     N,  M,        DTYP,         PORT,       ADDR,  DESC,          EGU,      DIR,  VELO,  VBAS,  ACCL,  BDST,  BVEL,  BACC,  MRES,  PREC,  DHLM,  DLLM,  INIT, FLNK}
{IOC:motorsimtest_ab:,  1,  "m$(N)",  "asynMotor",  motorSim1,  0,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:omega_e4c.PROC"}
{IOC:motorsimtest_ab:,  2,  "m$(N)",  "asynMotor",  motorSim1,  1,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:chi_e4c.PROC"}
{IOC:motorsimtest_ab:,  3,  "m$(N)",  "asynMotor",  motorSim1,  2,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:phi_e4c.PROC"}
{IOC:motorsimtest_ab:,  4,  "m$(N)",  "asynMotor",  motorSim1,  3,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "",  "HB3:ioc-hkl:tth_e4c.PROC"}
{IOC:motorsimtest_ab:,  5,  "m$(N)",  "asynMotor",  motorSim1,  4,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none1"}
{IOC:motorsimtest_ab:,  6,  "m$(N)",  "asynMotor",  motorSim1,  5,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none2"}
{IOC:motorsimtest_ab:,  7,  "m$(N)",  "asynMotor",  motorSim1,  6,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none3"}
{IOC:motorsimtest_ab:,  8,  "m$(N)",  "asynMotor",  motorSim1,  7,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none4"}
{IOC:motorsimtest_ab:,  9,  "m$(N)",  "asynMotor",  motorSim1,  8,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none5"}
{IOC:motorsimtest_ab:,  10,  "m$(N)",  "asynMotor",  motorSim1,  9,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none6"}
{IOC:motorsimtest_ab:,  11,  "m$(N)",  "asynMotor",  motorSim1,  10,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none7"}
{IOC:motorsimtest_ab:,  12,  "m$(N)",  "asynMotor",  motorSim1,  11,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none8"}
{IOC:motorsimtest_ab:,  13,  "m$(N)",  "asynMotor",  motorSim1,  12,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none9"}
{IOC:motorsimtest_ab:,  14,  "m$(N)",  "asynMotor",  motorSim1,  13,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none10"}
{IOC:motorsimtest_ab:,  15,  "m$(N)",  "asynMotor",  motorSim1,  14,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none11"}
{IOC:motorsimtest_ab:,  16,  "m$(N)",  "asynMotor",  motorSim1,  15,     "motor $(N)",  degrees,  Pos,  1,     .1,    .2,    0,     1,     .2,    0.01,  5,     100,   -100,  "", "HB3:ioc-hkl:none12"}
}

```


then in:

/epics/modules/motor/db/basic_asyn_motor.db
```
record(motor,"$(P)$(M)")
{
	field(DESC,"$(DESC)")
	field(DTYP,"$(DTYP)")
	field(DIR,"$(DIR)")
	field(VELO,"$(VELO)")
	field(VBAS,"$(VBAS)")
	field(ACCL,"$(ACCL)")
	field(ACCU,"$(ACCU=0)")
	field(BDST,"$(BDST)")
	field(BVEL,"$(BVEL)")
	field(BACC,"$(BACC)")
	field(OUT,"@asyn($(PORT),$(ADDR))")
	field(MRES,"$(MRES)")
	field(PREC,"$(PREC)")
	field(EGU,"$(EGU)")
	field(DHLM,"$(DHLM)")
	field(DLLM,"$(DLLM)")
	field(INIT,"$(INIT)")
	field(FLNK,"$(FLNK)")
	field(TWV,"1")
}

# These records make the motor resolution, offset and direction available to the driver
# which is needed for profile moves and other applications

# Motor direction for this axis
record(longout,"$(P)$(M)Direction") {
    field(DESC, "$(M) direction")
    field(DOL,  "$(P)$(M).DIR CP MS")
    field(OMSL, "closed_loop")
    field(DTYP, "asynInt32")
    field(OUT,  "@asyn($(PORT),$(ADDR))MOTOR_REC_DIRECTION")
}


# Motor offset for this axis
record(ao,"$(P)$(M)Offset") {
    field(DESC, "$(M) offset")
    field(DOL,  "$(P)$(M).OFF CP MS")
    field(OMSL, "closed_loop")
    field(DTYP, "asynFloat64")
    field(OUT,  "@asyn($(PORT),$(ADDR))MOTOR_REC_OFFSET")
    field(PREC, "$(PREC)")
}


# Motor resolution for this axis
record(ao,"$(P)$(M)Resolution") {
    field(DESC, "$(M) resolution")
    field(DOL,  "$(P)$(M).MRES CP MS")
    field(OMSL, "closed_loop")
    field(DTYP, "asynFloat64")
    field(OUT,  "@asyn($(PORT),$(ADDR))MOTOR_REC_RESOLUTION")
    field(PREC, "$(PREC)")
}
```

adding the one line for FLNK substitutions

get this warning if exclude .PROC in motor_substitution (I guess this is needed when forward linking across iocs)

WARNING: Forward-link uses Channel Access without pointing to PROC field
    IOC:motorsimtest_ab:m1.FLNK => HB3:ioc-hkl:omega_e4c
WARNING: Forward-link uses Channel Access without pointing to PROC field
    IOC:motorsimtest_ab:m2.FLNK => HB3:ioc-hkl:chi_e4c

to see details of individual PV
(in ioc-hkl epics shell)
```
dbpr HB3:ioc-hkl:omega_e4c, 10
```
PINI: YES
SCAN: Passive

test with increment motors in phoebus and look to e4c (for now) results
```
caget HB3:ioc-hkl:omega_e4c
caget IOC:motorsimtest_ab:m1

camonitor also helps to check if a PV changes, like
camonitor HB3:ioc-hkl:omega_e4c
```

FLNK not working might be an issue with channel access/port accessibilitiy with two IOCs running, getting these warnings

```
cas WARNING: Configured TCP port was unavailable.
cas WARNING: Using dynamically assigned TCP port 46859,
cas WARNING: but now two or more servers share the same UDP port.
cas WARNING: Depending on your IP kernel this server may not be
cas WARNING: reachable with UDP unicast (a host's IP in EPICS_CA_ADDR_LIST)
iocRun: All initialization complete
## motorUtil (allstop & alldone)
motorUtilInit("motorSim:")
# Boot complete
epics> CA.Client.Exception...............................................
    Warning: "User specified timeout on IO operation expired"
    Context: "ca_search_and_connect"
    Source File: ../motorUtil.cc line 210
    Current Time: Thu May 29 2025 09:38:17.303131670
..................................................................
motorUtil.cc: getChID(motorSim:moving.VAL) error: 80
CA.Client.Exception...............................................
    Warning: "User specified timeout on IO operation expired"
    Context: "ca_search_and_connect"
    Source File: ../motorUtil.cc line 210
    Current Time: Thu May 29 2025 09:39:17.303778029
..................................................................
motorUtil.cc: getChID(motorSim:alldone.VAL) error: 80

epics> 
```



can try 
```
sudo lsof -i :5064
```
to see which iocs are on that port, need to look up difference between UDP and TCP ports



The FLNK is working but the value isn't updating, when I run
camonitor HB3:ioc-hkl:omega_e4c.PROC

changing an m1 value in phoebus does update the PV as below:
                        
HB3:ioc-hkl:omega_e4c.PROC     2025-05-29 10:01:20.121099 0  
HB3:ioc-hkl:omega_e4c.PROC     2025-05-29 10:01:20.121099 1  
HB3:ioc-hkl:omega_e4c.PROC     2025-05-29 10:02:10.117501 1  
HB3:ioc-hkl:omega_e4c.PROC     2025-05-29 10:07:44.227275 1  
HB3:ioc-hkl:omega_e4c.PROC     2025-05-29 10:07:54.456788 1  
HB3:ioc-hkl:omega_e4c.PROC     2025-05-29 10:11:25.876751 1  

but the PV for omega doesnt change. Also the update from one IOC to the other is very slow/inconsistent

now seems like I need to add
field(DOL, "IOC1:source CP")
field(OMSL, "closed_loop")
to the target PV in ioc-hkl

works perfectly now, but I need a condition that includes these fields when coupling is on, and doesn't include them when coupling is off
