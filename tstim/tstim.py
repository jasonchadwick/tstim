import stim
import numpy as np
import copy
from dataclasses import dataclass
import qc_utils.stim
import itertools

@dataclass
class TimeDepolarize:
    """Probability does not include the identity gate (same convention as Stim).
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
            ancilla_offset: The offset to start numbering ancilla qubits from.
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
        self._bare_stim_circuit.append('TICK')
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
            probability: The probability of the error occurring. This
                probability INCLUDES the identity gate.
        """
        num_err_strings = 4**len(target_qubits)
        no_identity_prob = probability * (num_err_strings - 1) / num_err_strings
        self._correlated_errors.append(TimeDepolarize(target_qubits, target_time_positions, no_identity_prob))

    def append_time_pos(self, time_pos: int):
        """Append a TIME_POS instruction to the circuit.
        
        Args:
            time_pos: The time position to append.
        """
        self._added_instructions.append(TimePos(time_pos))

    def to_stim(
            self, 
            include_time_correlations: bool = True, 
            reuse_ancillae: bool = True, 
            ancilla_offset=0,
            num_error_strings_to_keep: int = 1000,
        ) -> stim.Circuit:
        """Converts to a stim.Circuit object, either with or without
        time-correlated errors.
        
        Args:
            include_time_correlations: Whether to include time-correlated
                errors.
            reuse_ancillae: Whether to reuse ancillae that are done with their
                correlated error.
            ancilla_offset: The offset to start numbering ancilla qubits from.
                Useful when combining circuits, or for easier reading.
        
        Returns:
            A stim.Circuit object representing the circuit.
        """
        if include_time_correlations:
            full_circuit = stim.Circuit()
            current_ancilla_idx = self._bare_stim_circuit.num_qubits + ancilla_offset
            last_time_pos = -1
            instructions_to_add = self._added_instructions.copy()
            unfinished_correlated_errors = [[min(instr.target_time_positions), [], np.ones_like(instr.target_qubits, bool), copy.copy(instr)] for instr in self._correlated_errors]
            unfinished_correlated_errors.sort(key=lambda x: x[0])

            available_ancillae = []
            for instr in instructions_to_add:
                if isinstance(instr, TimePos):
                    assert instr.time_pos > last_time_pos, f'Time positions must be in order, but got {last_time_pos} and {instr.time_pos}.'
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
                        ancillae = error_to_add[1]
                        if isinstance(error_to_add[3], TimeCorrelatedError):
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
                            if len(ancillae) == 0:
                                needed_ancillae = 2*num_qubits
                                ancillae = available_ancillae[:needed_ancillae]
                                if len(ancillae) < needed_ancillae:
                                    ancillae += [current_ancilla_idx + i for i in range(needed_ancillae - len(ancillae))]
                                    current_ancilla_idx += needed_ancillae - len(ancillae)
                                error_to_add[1] = ancillae

                                x_ancillae = ancillae[:num_qubits]
                                z_ancillae = ancillae[num_qubits:]
                                full_circuit.append('R', ancillae)

                                # apply depolarizing channel
                                # TODO: Justifiable way of setting
                                # num_error_strings_to_keep (based on
                                # probability and num_err_strings). Maybe use
                                # some sort of 'collision probability'?
                                x_paulis, z_paulis_x = get_XZ_depolarize_ops(num_qubits, max_error_strings=num_error_strings_to_keep, all_x=True, include_identity=False)

                                num_err_strings = len(x_paulis)                                

                                independent_prob = error_to_add[3].probability / num_err_strings
                                first_error = True
                                previous_prob = 1
                                previous_prob_prod = 1

                                for i,(xp, zp) in enumerate(zip(x_paulis, z_paulis_x)):
                                    targets = qc_utils.stim.get_stim_targets(ancillae, xp+zp)
                                    if first_error:
                                        prob = independent_prob
                                        full_circuit.append('CORRELATED_ERROR', targets, prob)
                                        first_error = False
                                        previous_prob = prob
                                    else:
                                        previous_prob_prod *= (1-previous_prob)
                                        prob = float(independent_prob / previous_prob_prod)
                                        full_circuit.append('ELSE_CORRELATED_ERROR', targets, prob)
                                        previous_prob = prob
                            else:
                                x_ancillae = ancillae[:num_qubits]
                                z_ancillae = ancillae[num_qubits:]

                            for err_idx, (target_qubit, time_pos) in enumerate(zip(error_to_add[3].target_qubits, error_to_add[3].target_time_positions)):
                                if error_to_add[2][err_idx] and time_pos == instr.time_pos:
                                    full_circuit.append('CX', [x_ancillae[err_idx], target_qubit])
                                    full_circuit.append('CZ', [z_ancillae[err_idx], target_qubit])
                                    error_to_add[2][err_idx] = False

                        # remove completed instructions
                        if np.all(~error_to_add[2]):
                            if reuse_ancillae:
                                available_ancillae.extend(ancillae)
                            inst_indices_to_remove.append(inst_idx)
                    for inst_idx in reversed(inst_indices_to_remove):
                        unfinished_correlated_errors.pop(inst_idx)
                else:
                    full_circuit.append(instr)

            assert len(unfinished_correlated_errors) == 0, 'Not all correlated errors were resolved. This means that there are correlated errors that refer to nonexistent TIME_POS indices.'

            return full_circuit
        else:
            full_circuit = stim.Circuit()
            for instr in self._added_instructions:
                if isinstance(instr, TimePos):
                    continue
                full_circuit.append(instr)
            return full_circuit
        
def get_XZ_depolarize_ops(
        num_qubits: int, 
        max_error_strings: int = 4**10, 
        all_x: bool = False,
        include_identity: bool = True,
        rng: np.random.Generator | int | None = None,
    ):
    """Construct all n-qubit depolarizing operations, then decompose each into
    an X and Z component.
    
    Note: actually only returns Pauli strings consisting of I and X (the "Z
    strings" have their Zs replaced by Xs).
    
    Args:
        num_qubits: Number of qubits to apply the depolarizing channel to.
        max_error_strings: Maximum number of error strings to generate.
        all_x: Whether to replace Zs with Xs in strings.
        include_identity: Whether to include the identity string.
        rng: Random number generator (or integer seed) to use. If None, uses
            np.random.default_rng().
    
    Returns:
        x_ops: List of Pauli strings representing the X component of the
            depolarizing channel. Each string is of length num_qubits consisting
            of I and X.
        z_ops: List of Pauli strings representing the Z component of the
            depolarizing channel. Each string is of length num_qubits consisting
            of I and Z.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    if max_error_strings < 4**num_qubits:
        # Generate random Pauli strings (excluding the identity string)
        if include_identity:
            chosen_indices = rng.choice(4**num_qubits, max_error_strings, replace=False)
        else:
            chosen_indices = rng.choice(4**num_qubits-1, max_error_strings, replace=False)+1
        x_ops = []
        z_ops = []
        for i in chosen_indices:
            # Convert i to a Pauli string
            quaternary_str = np.base_repr(i, base=4).rjust(num_qubits, '0')
            x_op = ''
            z_op = ''
            for digit in quaternary_str:
                if digit == '0':
                    x_op += 'I'
                    z_op += 'I'
                elif digit == '1':
                    x_op += 'X'
                    z_op += 'I'
                elif digit == '2':
                    x_op += 'X'
                    if all_x:
                        z_op += 'X'
                    else:
                        z_op += 'Z'
                elif digit == '3':
                    x_op += 'I'
                    if all_x:
                        z_op += 'X'
                    else:
                        z_op += 'Z'
            x_ops.append(x_op)
            z_ops.append(z_op)
        return x_ops, z_ops
    else:
        # Generate all possible Pauli strings
        x_ops = []
        z_ops = []
        for paulis in itertools.product(['I', 'X', 'Y', 'Z'], repeat=num_qubits):
            x_op = ''
            z_op = ''
            for pauli in paulis:
                if pauli == 'I':
                    x_op += 'I'
                    z_op += 'I'
                elif pauli == 'X':
                    x_op += 'X'
                    z_op += 'I'
                elif pauli == 'Y':
                    x_op += 'X'
                    if all_x:
                        z_op += 'X'
                    else:
                        z_op += 'Z'
                elif pauli == 'Z':
                    x_op += 'I'
                    if all_x:
                        z_op += 'X'
                    else:
                        z_op += 'Z'
            x_ops.append(x_op)
            z_ops.append(z_op)

        if not include_identity:
            x_ops.pop(0)
            z_ops.pop(0)
        return x_ops, z_ops