#!/bin/bash

cd ../..

/usr/bin/python3 -m venv iochkl && source iochkl/bin/activate && pip install -r requirements.txt
echo "import sys; sys.path.append('/usr/lib/python3/dist-packages')" > /epics/iocs/ioc-hkl/iochkl/lib/python3.12/site-packages/_gi_patch.pth

cd iocBoot/iocpydev

which python
which python3

export GI_TYPELIB_PATH=/usr/local/lib/girepository-1.0
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

../../bin/linux-x86_64/hklApp st_base.cmd

