import stim
import numpy as np
import copy
from dataclasses import dataclass
import qc_utils.stim

@dataclass
class TimeDepolarize:
    """Probability is *total* (includes the identity gate). Note that this is
    different from Stim's convention.
    """
    target_qubits: list[int]
    target_time_positions: list[int]
    probability: float

@dataclass
class TimeCorrelatedError:
    """TODO"""
    pauli_string: str
    target_qubits: list[int]
    target_time_positions: list[int]
    probability: float

@dataclass 
class TimePos:
    """TODO"""
    time_pos: int

@dataclass
class InstructionToAdd:
    """TODO"""
    first_time_pos: int
    ancilla_idx: int | None
    errors: list[tuple[str, int, int]]

class TStimCircuit:
    """TODO
    
    TODO: re-use ancilla qubits that are done with their correlated error.
    """
    def __init__(self, circuit_str: str = ""):
        """Initialize a TStimCircuit object.
        
        Args:
            circuit: The stim.Circuit object to start from. Defaults to an empty
                circuit.
        """
        if circuit_str:
            raise NotImplementedError
        
        self._bare_stim_circuit = stim.Circuit()
        self._added_instructions: list[stim.CircuitInstruction | stim.CircuitRepeatBlock | TimePos] = []
        self._correlated_errors: list[TimeCorrelatedError | TimeDepolarize] = []

    def append(self, *args):
        """Append a Stim instruction to the circuit. Instruction must be a
        valid stim.Circuit.append instruction.
        """
        self._bare_stim_circuit.append(*args)
        self._added_instructions.append(self._bare_stim_circuit[-1])

    def append_time_correlated_error(
            self, 
            pauli_string: str, 
            target_qubits: list[int], 
            target_time_positions: list[int], 
            probability: float
        ):
        """Append a TIME_CORRELATED_ERROR instruction to the circuit.
        
        Args:
            pauli_string: The Pauli string to apply.
            target_qubits: The qubits to apply the error to.
            target_time_positions: The time positions at which the error occurs.
            probability: The probability of the error occurring.
        """
        self._correlated_errors.append(TimeCorrelatedError(pauli_string, target_qubits, target_time_positions, probability))

    def append_time_depolarize(
            self, 
            target_qubits: list[int], 
            target_time_positions: list[int], 
            probability: float
        ):
        """Append a TIME_DEPOLARIZE instruction to the circuit.
        
        Args:
            target_qubits: The qubits to apply the error to.
            target_time_positions: The time positions at which the error occurs.
            probability: The probability of the error occurring.
        """
        self._correlated_errors.append(TimeDepolarize(target_qubits, target_time_positions, probability))

    def append_time_pos(self, time_pos: int):
        """Append a TIME_POS instruction to the circuit.
        
        Args:
            time_pos: The time position to append.
        """
        self._added_instructions.append(TimePos(time_pos))

    def to_stim(self, include_time_correlations: bool = True, reuse_ancillae: bool = False) -> stim.Circuit:
        """Converts to a stim.Circuit object, either with or without
        time-correlated errors.
        
        Args:
            include_time_correlations: Whether to include time-correlated
                errors.
            reuse_ancillae: Whether to reuse ancillae that are done with their
                correlated error. (Not sure yet whether this is ok to do...)
        
        Returns:
            A stim.Circuit object representing the circuit.
        """
        if include_time_correlations:
            full_circuit = stim.Circuit()
            current_ancilla_idx = self._bare_stim_circuit.num_qubits + 1000
            last_time_pos = -1
            instructions_to_add = self._added_instructions.copy()
            unfinished_correlated_errors = [[min(instr.target_time_positions), [], np.ones_like(instr.target_qubits, bool), copy.copy(instr)] for instr in self._correlated_errors]
            unfinished_correlated_errors.sort(key=lambda x: x[0])

            available_ancillae = []
            for instr in instructions_to_add:
                if isinstance(instr, TimePos):
                    assert instr.time_pos == last_time_pos+1, f'Time positions must be in order, but got {last_time_pos} and {instr.time_pos}.'
                    last_time_pos = instr.time_pos

                    # Apply correlated errors that occur at this time
                    # position
                    inst_indices_to_remove = []
                    for inst_idx, error_to_add in enumerate(unfinished_correlated_errors):
                        if error_to_add[0] > instr.time_pos:
                            # We have reached correlated errors that have not
                            # started happening yet.
                            break

                        num_qubits = len(error_to_add[3].target_qubits)

                        if isinstance(error_to_add[3], TimeCorrelatedError):
                            ancillae = error_to_add[1]
                            if len(ancillae) == 0:
                                # first time seeing this instruction
                                if available_ancillae:
                                    ancilla = available_ancillae.pop()
                                else:
                                    ancilla = current_ancilla_idx
                                    current_ancilla_idx += 1
                                error_to_add[1] = [ancilla]

                                full_circuit.append('R', ancilla)

                                full_circuit.append('X_ERROR', ancilla, error_to_add[3].probability)
                            else:
                                assert len(ancillae) == 1
                                ancilla = ancillae[0]

                            for err_idx, (pauli, target_qubit, time_pos) in enumerate(zip(error_to_add[3].pauli_string, error_to_add[3].target_qubits, error_to_add[3].target_time_positions)):
                                if error_to_add[2][err_idx] and time_pos == instr.time_pos:
                                    if pauli != 'I':
                                        full_circuit.append('C'+pauli, [ancilla, target_qubit])
                                    error_to_add[2][err_idx] = False
                        else:
                            ancillae = error_to_add[1]
                            if len(ancillae) == 0:
                                needed_ancillae = num_qubits
                                ancillae = available_ancillae[:num_qubits]
                                if len(ancillae) < needed_ancillae:
                                    ancillae += [current_ancilla_idx + i for i in range(needed_ancillae - len(ancillae))]
                                    current_ancilla_idx += needed_ancillae - len(ancillae)
                                error_to_add[1] = ancillae
                                full_circuit.append('R', ancillae)
                                
                                qc_utils.stim.depolarize(full_circuit, ancillae, error_to_add[3].probability)

                            for err_idx, (target_qubit, time_pos) in enumerate(zip(error_to_add[3].target_qubits, error_to_add[3].target_time_positions)):
                                if error_to_add[2][err_idx] and time_pos == instr.time_pos:
                                    full_circuit.append('CX', [target_qubit, ancillae[err_idx]])
                                    full_circuit.append('CZ', [target_qubit, ancillae[err_idx]])
                                    error_to_add[2][err_idx] = False

                        # remove completed instructions
                        if np.all(~error_to_add[2]):
                            if reuse_ancillae:
                                available_ancillae.append(ancilla)
                            inst_indices_to_remove.append(inst_idx)
                    for inst_idx in reversed(inst_indices_to_remove):
                        unfinished_correlated_errors.pop(inst_idx)
                else:
                    full_circuit.append(instr)

            if unfinished_correlated_errors:
                print(unfinished_correlated_errors)
                print(full_circuit)
                print(last_time_pos)
                raise ValueError("""
                    Not all correlated errors were resolved. This means that
                    there was TIME_CORRELATED_ERROR instruction that referred to
                    a time position that was never specified.
                    """
                )
            return full_circuit
        else:
            return self._bare_stim_circuit