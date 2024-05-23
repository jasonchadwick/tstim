# tstim
Support for time-correlated errors in stim

## Example usage

```
from tstim import TStimCircuit
tcirc = TStimCircuit()

tcirc.append_time_pos(0)
tcirc.append('H', [0,1])
tcirc.append_time_pos(1)
tcirc.append_time_correlated_error('XY', [0,1], [0,1], 0.01)
tcirc.append('S', [0,1])
tcirc.append_time_depolarize([0,1], [2,3], 0.01)
tcirc.append_time_pos(2)
tcirc.append('H', [0,1])
tcirc.append_time_pos(3)
```

Then, the TStimCircuit can generate a standard Stim circuit by adding ancillae:
```
>>> tcirc.to_stim()

stim.Circuit('''
    R 1002
    X_ERROR(0.01) 1002
    CX 1002 0
    H 0 1
    CY 1002 1
    S 0 1
    R 1003 1004
    DEPOLARIZE2(0.009375) 1003 1004
    CX 0 1003
    CZ 0 1003
    H 0 1
    CX 1 1004
    CZ 1 1004
''')
```