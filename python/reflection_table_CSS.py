from org.csstudio.display.builder.runtime.script import PVUtil, ScriptUtil
import math

data = []
N_COLS = 9

wf = list(PVUtil.getDoubleArray(pvs[0]))
def fmt(x):
    # NaN-safe -> empty string; else %.6g
    try:
        if x != x:           # NaN check (NaN != NaN)
            return ""
    except:
        pass
    return "%.6g" % x

rows = [wf[i:i+N_COLS] for i in range(0, len(wf), N_COLS)]
data = [[fmt(v) for v in row] for row in rows]

widget.setValue(data)

