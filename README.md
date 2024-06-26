# tstim
Support for time-correlated errors in stim

## Example usage

```
from tstim import TStimCircuit
tcirc = TStimCircuit()

q0 = 0
q1 = 1
tcirc.append('R', [q0, q1])
tcirc.append_time_pos(0)
tcirc.append('H', [q0, q1])
tcirc.append_time_pos(1)
tcirc.append_time_correlated_error('XY', [q0, q1], [0,1], 0.01)
tcirc.append('S', [q0, q1])
tcirc.append_time_depolarize([q0, q1], [2,3], 0.01)
tcirc.append_time_pos(2)
tcirc.append('H', [q0, q1])
tcirc.append_time_pos(3)
```

Then, the TStimCircuit can generate a standard Stim circuit by adding ancillae:
```
>>> tcirc.to_stim()

stim.Circuit('''
    R 0 1 1002
    X_ERROR(0.01) 1002
    CX 1002 0
    H 0 1
    CY 1002 1
    S 0 1
    R 1002 1003 1004 1005
    E(0.000625) X1003
    ELSE_CORRELATED_ERROR(0.000625391) X1003 X1005
    ELSE_CORRELATED_ERROR(0.000625782) X1005
    ELSE_CORRELATED_ERROR(0.000626174) X1002
    ELSE_CORRELATED_ERROR(0.000626566) X1002 X1003
    ELSE_CORRELATED_ERROR(0.000626959) X1002 X1003 X1005
    ELSE_CORRELATED_ERROR(0.000627353) X1002 X1005
    ELSE_CORRELATED_ERROR(0.000627746) X1002 X1004
    ELSE_CORRELATED_ERROR(0.000628141) X1002 X1003 X1004
    ELSE_CORRELATED_ERROR(0.000628536) X1002 X1003 X1004 X1005
    ELSE_CORRELATED_ERROR(0.000628931) X1002 X1004 X1005
    ELSE_CORRELATED_ERROR(0.000629327) X1004
    ELSE_CORRELATED_ERROR(0.000629723) X1003 X1004
    ELSE_CORRELATED_ERROR(0.00063012) X1003 X1004 X1005
    ELSE_CORRELATED_ERROR(0.000630517) X1004 X1005
    CX 1002 0
    CZ 1004 0
    H 0 1
    CX 1003 1
    CZ 1005 1
''')
```