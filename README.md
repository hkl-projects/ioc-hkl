# Installation instructions

## Dependencies
* EPICS - https://epics.anl.gov/
* PyDevice - https://github.com/klemenv/PyDevice
* hkl - https://repo.or.cz/hkl.git
* cif2hkl https://gitlab.com/soleil-data-treatment/soleil-software-projects/cif2hkl

<!--
## Basic PyDevice directory structure for EPICS IOCs
/epics \
| \
├── base \
├── GUI \
├── iocs \
│   └── ioc-hkl \
├── support \
│   ├── hkl \
│   └── PyDevice \
└── util \
 \
/epics/iocs/ioc-hkl
-->

## EPICS installation
Download the EPICS base from https://epics.anl.gov/download/base/index.php and place tarball into /epics, then unpack and build. 

```bash
tar -xvzf base-7.0.8.tar.gz
mv base-7.0.8 base
cd base
make
```

## Pixi installation
```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL
pixi --version
```

## IOC download
Place this repo in /epics/iocs/
```bash
cd /epics/iocs
git clone https://github.com/hkl-projects/ioc-hkl.git
cd ioc-hkl
```

## hkl installation (from Python environment - recommended)
## IOC & pixi environment install
./run_HKL.sh



## hkl installation (from source - if creating new diffractometer geometry)
hkl - https://repo.or.cz/hkl.git

```bash
cd /epics/support
git clone https://repo.or.cz/hkl.git
git checkout tags/v5.0.0.3357 # optional
cd hkl
```

```bash
sudo apt install gtk-doc-tools autoconf libgtkmm-3.0-dev libyaml-dev gettext autopoint gobject-introspection libtool autoconf-archive debhelper gnuplot-nox gobject-introspection gtk-doc-tools libbullet-dev libg3d-dev libg3d-plugins libgirepository1.0-dev libgl-dev libgsl-dev libgtk-3-dev libgtkglext1-dev libhdf5-dev python3-gi python3-pip elpa-htmlize dvipng libhdf5-dev povray asymptote libhdf5-dev libcglm-dev libinih-dev
```

```bash
./autogen
./configure --enable-introspection --disable-binoculars
make
sudo make install
```

If running hkl outside of this IOC, you will need to set the following environmental variables in your shell/bashrc:
```bash
export GI_TYPELIB_PATH=/usr/local/lib/girepository-1.0 
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/lib
```

## cif2hkl installation
```bash
sudo apt install cif2hkl
```

## Install and run IOC
```bash
cd /epics/iocs/ioc-hkl
make -j4
cd /epics/iocs/ioc-hkl/iocBoot/iocpydev
./st.cmd
```

## To test communication and PV update
in epics shell: pydev("hklApp.test()") \
in epics shell: pydev("hklApp.get\_pseudoaxes()") \
in bash: caget TAS:hb3:in:pseudoaxesh 

