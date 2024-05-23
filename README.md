# tstim
Support for time-correlated errors in stim

## Example .tstim file

```
from tstim import TStimCircuit
tcirc = TStimCircuit()

tcirc.append_time_correlated_error('XY', [0,1], [0,1], 0.01)
tcirc.append_time_pos(0)
tcirc.append('CX', [0,1])
tcirc.append_time_pos(1)
tcirc.append('H', 0)
tcirc.append_time_correlated_error('ZZ', [0,1], [2,3], 0.01)
tcirc.append_time_pos(2)
tcirc.append('CX', [0,1])
tcirc.append_time_pos(3)
```