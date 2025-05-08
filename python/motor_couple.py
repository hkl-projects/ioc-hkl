# pip install pyepics
# export EPICS_CA_AUTO_ADDR_LIST=YES
# export EPICS_CA_ADDR_LIST="127.0.0.1"

# single PV
#from epics import PV

## source and target PVs
#source_pv = PV('IOC:m1.VAL') # the PV to read from
#target_pv = PV('HB3:ioc-hkl:omega_e4c')  # the PV to write to, hardcoded?
#
## callback function: triggered when source PV changes
#def on_change(pvname=None, value=None, **kwargs):
#    print(f"Updating {target_pv.pvname} to {value}")
#    target_pv.put(value)
#
## add callback to source PV
#source_pv.add_callback(on_change)
#
#print("Running... press Ctrl+C to exit.")
#try:
#    while True:
#        pass  # loop to keep callbacks alive
#except KeyboardInterrupt:
#    print("\nExiting.")
#

from epics import PV
import time

# map source PVs to destination PVs
pv_pairs = {
    'IOC:m1.VAL': 'HB3:ioc-hkl:omega_e4c.VAL',
    'IOC:m2.VAL': 'HB3:ioc-hkl:chi_e4c.VAL',
    'IOC:m3.VAL': 'HB3:ioc-hkl:phi_e4c.VAL',
    'IOC:m4.VAL': 'HB3:ioc-hkl:tth_e4c.VAL',
}

# deadband filtering threshold
DEADBAND = 0.01  # Only forward changes > 0.01

# store last known values
last_values = {}

def make_callback(src_pv, dest_pv_name):
    def callback(pvname=None, value=None, **kwargs):
        last_val = last_values.get(pvname, None)
        if last_val is None or abs(value - last_val) > DEADBAND:
            dest_pv = PV(dest_pv_name)
            dest_pv.put(value)
            print(f"{pvname} → {dest_pv_name} : {value}")
            last_values[pvname] = value
    return callback

# set up source PVs and callbacks
source_pvs = []
for src, dest in pv_pairs.items():
    pv = PV(src, callback=make_callback(src, dest))
    source_pvs.append(pv)
    last_values[src] = None

print("Monitoring PVs. Press Ctrl+C to exit.")
try:
    while True:
        #time.sleep(1)
        pass
except KeyboardInterrupt:
    print("\nExiting...")

