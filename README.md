# Installation instructions

## Dependencies
* EPICS - https://epics.anl.gov/
* PyDevice - https://github.com/klemenv/PyDevice
* hkl - https://repo.or.cz/hkl.git

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

## IOC download
Place this repo in /epics/iocs/
```bash
cd /epics/iocs
git clone https://github.com/hkl-projects/ioc-hkl.git
cd ioc-hkl
```

## hkl installation (from Python environment - recommended)
Download pixi
```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL
pixi --version

Set up ioc-hkl environment from pixi
```bash 
cd /epics/iocs/ioc-hkl
pixi init
pixi add python=3.12 numpy=2.3.4 pandas scipy matplotlib tqdm pygobject hkl
```

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

## Python (venv - If building hkl from source)
Install the python venv environment (named iochkl) with access to system's site-packages via
```
cd /epics/iocs/ioc-hkl
python3 -m venv --system-site-packages /epics/iocs/ioc-hkl/iochkl
source iochkl/bin/activate
pip install -r requirements.txt
pip install numpy==1.26.4 --no-cache-dir --force-reinstall
```
#TODO swap pip numpy version from force-install to version sepcification in requirements.txt


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

