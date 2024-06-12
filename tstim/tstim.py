import stim
import numpy as np
from numpy.typing import NDArray
import copy
from dataclasses import dataclass
import qc_utils.stim
import itertools
import scipy.stats
import math
import random

@dataclass
class TimeDepolarize:
    """Probability does not include the identity gate (same convention as Stim).
    """
    target_qubits: list[int]
    target_time_positions: list[int]
    probability: float
    annotation: str = ''

@dataclass
class TimeCorrelatedError:
    """TODO"""
    pauli_string: str
    target_qubits: list[int]
    target_time_positions: list[int]
    probability: float
    annotation: str = ''

@dataclass 
class TimePos:
    """TODO"""
    time_pos: int

@dataclass
class UnfinishedCorrelatedError:
    """Used to keep track of correlated errors that have not been fully applied
    yet to an in-progress stim circuit during TStimCircuit.to_stim."""
    instruction: TimeCorrelatedError | TimeDepolarize
    first_time_pos: int
    num_error_strings_to_keep: int
    x_ancillae: list[int]
    z_ancillae: list[int]
    x_paulis: list[list[bool]]
    z_paulis: list[list[bool]]
    x_affected_indices: list[int]
    z_affected_indices: list[int]
    completed_target_qubits: list[bool]
    is_simple_error: bool
    has_been_initialized: bool

def collision_probability(n,k,p_tot):
    """Calculate the probability of sampling the same item more than once in
    n samples, given k equal-probability items and total probability of
    sampling any one item.

    Args:
        n: Number of samples.
        k: Number of items.
        p_tot: Total probability of sampling any one item.

    Returns:
        Probability of sampling the same item more than once.
    """
    assert n >= 1
    assert k >= 1
    assert p_tot >= 0
    assert p_tot <= 1
    d = np.arange(n)
    return 1-np.sum((1-1/k)**(d*(d-1)/2)*scipy.special.comb(n,d)*p_tot**d*(1-p_tot)**(n-d))

def binary_search_prob(num_samples, num_errors, p_err_tot, p_collision_max):
    """Binary search to find the minimum number of error strings to keep in a
    depolarizing error to ensure that the probability of any two error strings
    being sampled is less than p_collision_max, assuming num_samples samples.

    Args:
        num_samples: Number of samples to take.
        num_errors: Number of error strings to consider.
        p_err_tot: Total probability of sampling any one error string.
        p_collision_max: Maximum probability of sampling the same error string
            more than once.

    Returns:
        Minimum number of error strings to keep. Will be between 0 and
        num_errors. Only returns 0 if p_err_tot is 0.
    """
    if np.isclose(p_err_tot, 1):
        return num_errors
    elif np.isclose(p_err_tot, 0):
        return 0

    k_low = 1
    k_high = num_errors
    while k_low < k_high:
        k_test = (k_low + k_high) // 2
        p_test = collision_probability(num_samples, k_test, p_err_tot)
        if p_test > p_collision_max:
            k_low = k_test + 1
        else:
            k_high = k_test
    return k_low

def fast_search_prob(num_samples, num_errors, p_err_tot, p_collision_max):
    """Search powers of 2 to find the minimum number of error strings to keep in
    a depolarizing error to ensure that the probability of any two error strings
    being sampled is less than p_collision_max, assuming num_samples samples.
    
    Args:
        num_samples: Number of samples to take.
        num_errors: Number of error strings to consider.
        p_err_tot: Total probability of sampling any one error string.
        p_collision_max: Maximum probability of sampling the same error string
            more than once.
    
    Returns:
        Minimum number of error strings to keep.
    """
    k_test = 1
    p_test = collision_probability(num_samples, k_test, p_err_tot)
    while p_test > p_collision_max:
        k_test *= 2
        if k_test >= num_errors:
            return num_errors
        p_test = collision_probability(num_samples, k_test, p_err_tot)
    return k_test

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

        self._bare_stim_circuit_str: list[str] = []
        self._added_instructions: list[tuple[str | TimePos, str]] = []
        self._correlated_errors: list[TimeCorrelatedError | TimeDepolarize] = []
        self._num_depolarize_errors: int = 0

    def append(self, instruction: str, annotation: str = '') -> None:
        """Append a Stim instruction (string) to the circuit. Instruction must
        be a string representation of a valid stim.Circuit.append instruction
        (although this is not checked until to_stim is called).

        Args:
            instruction: The instruction to append.
            annotation: An optional annotation for the instruction.
        """
        self._bare_stim_circuit_str.append(instruction)
        self._added_instructions.append((self._bare_stim_circuit_str[-1], annotation))

    def append_time_correlated_error(
            self, 
            pauli_string: str, 
            target_qubits: list[int], 
            target_time_positions: list[int], 
            probability: float,
            annotation: str = '',
        ):
        """Append a TIME_CORRELATED_ERROR instruction to the circuit.

        Args:
            pauli_string: The Pauli string to apply.
            target_qubits: The qubits to apply the error to.
            target_time_positions: The time positions at which the error occurs.
            probability: The probability of the error occurring.
            annotation: An optional annotation for the error.
        """
        self._correlated_errors.append(TimeCorrelatedError(pauli_string, target_qubits, target_time_positions, probability, annotation))

    def append_time_depolarize(
            self, 
            target_qubits: list[int], 
            target_time_positions: list[int], 
            probability: float,
            annotation: str = '',
        ):
        """Append a TIME_DEPOLARIZE instruction to the circuit.

        Args:
            target_qubits: The qubits to apply the error to.
            target_time_positions: The time positions at which the error occurs.
            probability: The probability of the error occurring. This
                probability INCLUDES the identity gate.
            annotation: An optional annotation for the error.
        """
        num_err_strings = 4**len(target_qubits)
        no_identity_prob = probability * (num_err_strings - 1) / num_err_strings
        self._correlated_errors.append(TimeDepolarize(target_qubits, target_time_positions, no_identity_prob, annotation))
        self._num_depolarize_errors += 1

    def append_time_pos(self, time_pos: int):
        """Append a TIME_POS instruction to the circuit.

        Args:
            time_pos: The time position to append.
        """
        self._added_instructions.append((TimePos(time_pos), ''))

    def _presample_correlated_errors(
            self,
            correlated_errors: list[UnfinishedCorrelatedError],
            num_times_to_be_sampled: int,
            allowed_collision_prob_per_error: float,
            rng: np.random.Generator | int | None = None,
        ) -> list[UnfinishedCorrelatedError]:
        """TODO"""
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        elif rng is None:
            rng = np.random.default_rng()
        num_error_strings_to_keep_dict: dict[int, int] = {}
        inst_indices_to_remove = []
        inst_to_insert = []
        correlated_errors_new = []
        for inst_idx, error_to_add in enumerate(correlated_errors):
            skip_instruction = False
            if isinstance(error_to_add.instruction, TimeDepolarize):
                num_qubits = len(error_to_add.instruction.target_qubits)
                assert num_qubits > 0

                if 4**num_qubits-1 not in num_error_strings_to_keep_dict:
                    num_error_strings_to_keep = 4**num_qubits-1
                    if allowed_collision_prob_per_error > 0:
                        # binary search to find minimum number
                        num_error_strings_to_keep = binary_search_prob(num_times_to_be_sampled, 4**num_qubits-1, error_to_add.instruction.probability, allowed_collision_prob_per_error)
                    num_error_strings_to_keep_dict[4**num_qubits-1] = num_error_strings_to_keep
                num_error_strings_to_keep = num_error_strings_to_keep_dict[4**num_qubits-1]

                error_to_add.num_error_strings_to_keep = num_error_strings_to_keep

                x_paulis, z_paulis, x_affected_indices, z_affected_indices = get_XZ_depolarize_ops(num_qubits, max_error_strings=num_error_strings_to_keep, include_identity=False, rng=rng)
                error_to_add.x_paulis = x_paulis
                error_to_add.z_paulis = z_paulis

                # x_affected_indices = np.where(np.any(x_paulis, axis=0))[0]
                # x_affected_indices = [i for i,paulis in enumerate(x_paulis) if len([p for p in paulis if p]) > 0]
                # assert x_affected_indices == list(np.where(np.any(x_paulis, axis=1))[0])
                # z_affected_indices = np.where(np.any(z_paulis, axis=0))[0]
                # z_affected_indices = [i for i,paulis in enumerate(z_paulis) if len([p for p in paulis if p]) > 0]
                error_to_add.x_affected_indices = x_affected_indices
                error_to_add.z_affected_indices = z_affected_indices

                total_affected_indices = list(set(x_affected_indices + z_affected_indices))

                if num_error_strings_to_keep == 0:
                    skip_instruction = True
                if num_error_strings_to_keep == 1:
                    # If we are only adding a single
                    # error string, it is easier to add
                    # it as a TimeCorrelatedError rather
                    # than a TimeDepolarize. It uses
                    # fewer stim qubits and is easier to
                    # visually parse in the circuit.
                    pauli_string = ''
                    for idx in total_affected_indices:
                        if idx in x_affected_indices and idx in z_affected_indices:
                            pauli_string += 'Y'
                        elif idx in x_affected_indices:
                            pauli_string += 'X'
                        elif idx in z_affected_indices:
                            pauli_string += 'Z'
                        else:
                            pauli_string += 'I'

                    target_qubits = [error_to_add.instruction.target_qubits[i] for i in total_affected_indices]
                    target_time_positions = [error_to_add.instruction.target_time_positions[i] for i in total_affected_indices]
                    new_error = UnfinishedCorrelatedError(
                        TimeCorrelatedError(
                            pauli_string,
                            target_qubits,
                            target_time_positions,
                            error_to_add.instruction.probability,
                        ),
                        min(target_time_positions),
                        -1,
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [False]*len(target_qubits),
                        False,
                        False,
                    )
                    skip_instruction = True
                    correlated_errors_new.append(new_error)
            if not skip_instruction:
                correlated_errors_new.append(error_to_add)

        correlated_errors_new.sort(key=lambda x: x.first_time_pos)

        return correlated_errors_new

    def to_stim(
            self, 
            reuse_ancillae: bool = True, 
            ancilla_offset=0,
            num_times_to_be_sampled: int = 10**8,
            allowed_collision_prob: float = 0,
            approximate_independent_errors: bool = False,
            approximate_independent_multiplier: float = 0.0,
            rng: np.random.Generator | int | None = None,
        ) -> stim.Circuit | tuple[stim.Circuit, dict[int, str]]:
        """Converts to a stim.Circuit object, either with or without
        time-correlated errors.

        Args:
            reuse_ancillae: Whether to reuse ancillae that are done with their
                correlated error.
            ancilla_offset: The offset to start numbering ancilla qubits from.
                Useful when combining circuits, or for easier reading.
            num_times_to_be_sampled: The number of times the circuit will be
                sampled. Used with allowed_collision_prob to determine the
                number of error strings to keep for each large correlated error.
            allowed_collision_prob: The allowed chance that any single error
                string within a depolarizing error will be sampled more than
                once (which determines how much we can reduce the number of
                error strings in each depolarizing error), assuming we take
                num_times_to_be_sampled samples.
            approximate_independent_errors: Whether to approximate correlated
                errors with independent errors. If True, spacetime-correlated
                errors are instead applied as independent single-qubit,
                single-time errors. This is much faster to generate, but is
                inaccurate. Useful for building the decoder circuit, but not
                good for sampling.

        Returns:
            A stim.Circuit object representing the circuit. Also returns a
            dictionary mapping indices of instructions in the circuit to their
            annotations, if any annotations are present.
        """
        full_circuit_str = []
        annotations = {}
        current_ancilla_idx = stim.Circuit('\n'.join(self._bare_stim_circuit_str)).num_qubits + ancilla_offset
        last_time_pos = -1
        instructions_to_add = self._added_instructions
        unfinished_correlated_errors = [
            UnfinishedCorrelatedError(
                instr,
                min(instr.target_time_positions),
                4**len(instr.target_qubits)-1,
                [],
                [],
                [],
                [],
                [],
                [],
                [False]*len(instr.target_qubits),
                False,
                False,
            )
            for instr in self._correlated_errors
        ]
        unfinished_correlated_errors.sort(key=lambda x: x.first_time_pos)

        allowed_collision_prob_per_error = 0
        if allowed_collision_prob > 0 and self._num_depolarize_errors > 0:
            allowed_collision_prob_per_error = 1 - (1-allowed_collision_prob)**(1/self._num_depolarize_errors)
        unfinished_correlated_errors = self._presample_correlated_errors(unfinished_correlated_errors, num_times_to_be_sampled, allowed_collision_prob_per_error, rng)

        available_ancillae = []
        for (instr, annotation) in instructions_to_add:
            if isinstance(instr, TimePos):
                assert instr.time_pos > last_time_pos, f'Time positions must be in order, but got {last_time_pos} before {instr.time_pos}.'
                last_time_pos = instr.time_pos

                # Apply correlated errors that occur at this time
                # position
                inst_indices_to_remove = []
                for inst_idx, error_to_add in enumerate(unfinished_correlated_errors):
                    if error_to_add.first_time_pos > instr.time_pos:
                        # We have reached correlated errors that have not
                        # started happening yet (unfinished_correlated_errors is
                        # sorted by first time position).
                        break

                    num_qubits = len(error_to_add.instruction.target_qubits)

                    if isinstance(error_to_add.instruction, TimeCorrelatedError):
                        if not error_to_add.has_been_initialized:
                            # first time seeing this instruction
                            num_time_positions = len(set(error_to_add.instruction.target_time_positions))

                            if num_time_positions == 1:
                                # All time positions are the same (so we can
                                # apply it as a single stim instruction, without 
                                # using ancillae).
                                error_to_add.is_simple_error = True
                            else:
                                if available_ancillae:
                                    ancilla = available_ancillae.pop()
                                else:
                                    ancilla = current_ancilla_idx
                                    current_ancilla_idx += 1
                                error_to_add.x_ancillae = [ancilla]

                                full_circuit_str.append(f'R {ancilla}')

                                full_circuit_str.append(f'X_ERROR({error_to_add.instruction.probability}) {ancilla}')       
                            error_to_add.has_been_initialized = True                     

                        # Apply parts of the error that correspond to this time
                        # position.
                        if error_to_add.is_simple_error:
                            time_pos = error_to_add.instruction.target_time_positions[0]
                            if time_pos == instr.time_pos:
                                err_str = f'E({error_to_add.instruction.probability})'
                                for pauli, target_qubit in zip(error_to_add.instruction.pauli_string, error_to_add.instruction.target_qubits):
                                    if pauli != 'I':
                                        err_str += f' {pauli}{target_qubit}'
                                full_circuit_str.append(err_str)
                                error_to_add.completed_target_qubits = [True]*len(error_to_add.instruction.target_qubits)
                        else:
                            ancilla = error_to_add.x_ancillae[0]
                            for err_idx, (pauli, target_qubit, time_pos) in enumerate(zip(error_to_add.instruction.pauli_string, error_to_add.instruction.target_qubits, error_to_add.instruction.target_time_positions)):
                                if not error_to_add.completed_target_qubits[err_idx] and time_pos == instr.time_pos:
                                    if pauli != 'I':
                                        full_circuit_str.append(f'C{pauli} {ancilla} {target_qubit}')
                                    error_to_add.completed_target_qubits[err_idx] = True
                    else:
                        assert isinstance(error_to_add.instruction, TimeDepolarize)
                        if approximate_independent_errors:
                            # re-weight probability because non-identity support
                            # has different number of terms
                            if approximate_independent_multiplier == 0:
                                prob = error_to_add.instruction.probability * 4**num_qubits / (4**num_qubits - 1) * 3/4
                            else:
                                prob = approximate_independent_multiplier * error_to_add.instruction.probability
                            assert prob <= 3/4 and prob >= 0
                            for err_idx, (target_qubit, time_pos) in enumerate(zip(error_to_add.instruction.target_qubits, error_to_add.instruction.target_time_positions)):
                                if not error_to_add.completed_target_qubits[err_idx] and time_pos == instr.time_pos:
                                    full_circuit_str.append(f'DEPOLARIZE1({prob}) {target_qubit}')
                                    error_to_add.completed_target_qubits[err_idx] = True
                        else:
                            skip_op = False
                            if not error_to_add.has_been_initialized:
                                # This is the first time seeing this
                                # instruction, so we need to create ancillae and
                                # apply the depolarizing errors.
                                if error_to_add.num_error_strings_to_keep > 0:
                                    if num_qubits <= 2 and len(set(error_to_add.instruction.target_time_positions)) == 1:
                                        error_to_add.is_simple_error = True
                                    else:
                                        x_paulis = error_to_add.x_paulis
                                        z_paulis = error_to_add.z_paulis
                                        num_err_strings = len(x_paulis)

                                        x_affected_indices = error_to_add.x_affected_indices
                                        z_affected_indices = error_to_add.z_affected_indices

                                        needed_x_ancillae = len(x_affected_indices)
                                        needed_z_ancillae = len(z_affected_indices)

                                        needed_ancillae = needed_x_ancillae + needed_z_ancillae
                                        ancillae = available_ancillae[:needed_ancillae]
                                        available_ancillae = available_ancillae[needed_ancillae:]
                                        if len(ancillae) < needed_ancillae:
                                            num_to_add = needed_ancillae - len(ancillae)
                                            ancillae += [current_ancilla_idx + i for i in range(num_to_add)]
                                            current_ancilla_idx += num_to_add

                                        x_ancillae = ancillae[:needed_x_ancillae]
                                        z_ancillae = ancillae[needed_x_ancillae:]

                                        error_to_add.x_ancillae = x_ancillae
                                        error_to_add.z_ancillae = z_ancillae
                                        error_to_add.x_affected_indices = x_affected_indices
                                        error_to_add.z_affected_indices = z_affected_indices

                                        reset_layer_idx = len(full_circuit_str)
                                        if len(error_to_add.instruction.annotation) > 0:
                                            annotations[reset_layer_idx] = error_to_add.instruction.annotation
                                        full_circuit_str.append(f'R {" ".join(map(str, ancillae))}')

                                        independent_prob = error_to_add.instruction.probability / num_err_strings
                                        first_error = True
                                        previous_prob_prod = 1

                                        ancilla_used = [False]*needed_ancillae

                                        for i,(xp, zp) in enumerate(zip(x_paulis, z_paulis)):
                                            x_targets = []
                                            z_targets = []
                                            x_idx = 0
                                            z_idx = 0
                                            for j in range(num_qubits):
                                                if j in x_affected_indices:
                                                    if xp[j]:
                                                        x_targets.append(f'X{x_ancillae[x_idx]}')
                                                        ancilla_used[x_idx] = True
                                                    x_idx += 1
                                                if j in z_affected_indices:
                                                    if zp[j]:
                                                        z_targets.append(f'X{z_ancillae[z_idx]}')
                                                        ancilla_used[needed_x_ancillae + z_idx] = True
                                                    z_idx += 1

                                            if first_error:
                                                prob = independent_prob
                                                full_circuit_str.append(f'E({prob}) {" ".join(x_targets + z_targets)}')
                                                first_error = False
                                                previous_prob_prod *= (1-prob)
                                            else:
                                                prob = float(independent_prob / previous_prob_prod)
                                                full_circuit_str.append(f'ELSE_CORRELATED_ERROR({prob}) {" ".join(x_targets + z_targets)}')
                                                previous_prob_prod *= (1-prob)

                                        assert len([x for x in ancilla_used if x == False]) == 0, 'Not all ancillae were used.'
                                else:
                                    # do not apply any errors this round
                                    skip_op = True
                                    # mark for deletion
                                    error_to_add.completed_target_qubits = [True]*len(error_to_add.instruction.target_qubits)

                                error_to_add.has_been_initialized = True

                            if not skip_op:
                                # If we are at the correct time location, apply
                                # CX and CZ to propagate the correlated errors.
                                if error_to_add.is_simple_error:
                                    # 1 or 2 qubit depolarize at single time
                                    time_pos = error_to_add.instruction.target_time_positions[0]
                                    if time_pos == instr.time_pos:
                                        qubits = error_to_add.instruction.target_qubits
                                        if len(qubits) == 1:
                                            full_circuit_str.append(f'DEPOLARIZE1({error_to_add.instruction.probability}) {qubits[0]}')
                                        else:
                                            assert len(qubits) == 2
                                            full_circuit_str.append(f'DEPOLARIZE2({error_to_add.instruction.probability}) {" ".join(map(str, qubits))}')
                                        error_to_add.completed_target_qubits = [True]*len(error_to_add.instruction.target_qubits)
                                else:
                                    x_ancillae = error_to_add.x_ancillae
                                    z_ancillae = error_to_add.z_ancillae
                                    x_affected_indices = error_to_add.x_affected_indices
                                    z_affected_indices = error_to_add.z_affected_indices
                                    for err_idx, (target_qubit, time_pos) in enumerate(zip(error_to_add.instruction.target_qubits, error_to_add.instruction.target_time_positions)):
                                        if not error_to_add.completed_target_qubits[err_idx] and time_pos == instr.time_pos:
                                            if err_idx in x_affected_indices:
                                                ancilla_idx = x_affected_indices.index(err_idx)
                                                full_circuit_str.append(f'CX {x_ancillae[ancilla_idx]} {target_qubit}')

                                            if err_idx in z_affected_indices:
                                                ancilla_idx = z_affected_indices.index(err_idx)
                                                full_circuit_str.append(f'CZ {z_ancillae[ancilla_idx]} {target_qubit}')

                                            error_to_add.completed_target_qubits[err_idx] = True

                    # mark for deletion
                    if len([x for x in error_to_add.completed_target_qubits if x == False]) == 0:
                        if reuse_ancillae:
                            # reclaim ancillae
                            available_ancillae.extend(error_to_add.x_ancillae + error_to_add.z_ancillae)
                            assert len(available_ancillae) == len(set(available_ancillae))
                        inst_indices_to_remove.append(inst_idx)

                # remove old instructions
                for inst_idx in reversed(inst_indices_to_remove):
                    unfinished_correlated_errors.pop(inst_idx)
            else:
                # instr is a regular stim instruction, so we just add it
                if len(annotation) > 0:
                    annotations[len(full_circuit_str)] = annotation
                full_circuit_str.append(str(instr))

        assert len(unfinished_correlated_errors) == 0, f'{len(unfinished_correlated_errors)} correlated errors were not resolved. This means that there are correlated errors that refer to nonexistent TIME_POS indices.'

        if len(annotations) > 0:
            return stim.Circuit('\n'.join(full_circuit_str)), annotations
        else:
            return stim.Circuit('\n'.join(full_circuit_str))

def get_XZ_depolarize_ops(
        num_qubits: int, 
        max_error_strings: int = 4**10, 
        include_identity: bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> tuple[list[list[bool]], list[list[bool]], list[int], list[int]]:
    """Construct all n-qubit depolarizing operations, then decompose each into
    an X and Z component.

    Note: actually only returns Pauli strings consisting of I and X (the "Z
    strings" have their Zs replaced by Xs).

    Args:
        num_qubits: Number of qubits to apply the depolarizing channel to.
        max_error_strings: Maximum number of error strings to generate.
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

    if (include_identity and max_error_strings < 4**num_qubits) or (not include_identity and max_error_strings < 4**num_qubits-1):
        x_ops = [[False]*num_qubits for _ in range(max_error_strings)]
        z_ops = [[False]*num_qubits for _ in range(max_error_strings)]
        x_affected_indices = set()
        z_affected_indices = set()

        # Generate random Pauli strings (excluding the identity string)
        if include_identity:
            if max_error_strings <= 10:
                # random.choices is faster for small k
                chosen_indices = random.choices(range(1, 4**num_qubits), k=max_error_strings)
            else:
                chosen_indices = rng.choice(4**num_qubits, max_error_strings, replace=False)
        else:
            if max_error_strings <= 10:
                # random.choices is faster for small k
                chosen_indices = [1+x for x in random.choices(range(1, 4**num_qubits-1), k=max_error_strings)]
            else:
                chosen_indices = rng.choice(4**num_qubits-1, max_error_strings, replace=False)+1

        for i,idx in enumerate(chosen_indices):
            # Convert i to a Pauli string
            quaternary_str = np.base_repr(idx, base=4).rjust(num_qubits, '0')

            for j,digit in enumerate(quaternary_str):
                if digit == '1':
                    x_ops[i][j] = True
                    x_affected_indices.add(j)
                elif digit == '2':
                    x_ops[i][j] = True
                    x_affected_indices.add(j)
                    z_ops[i][j] = True
                    z_affected_indices.add(j)
                elif digit == '3':
                    z_ops[i][j] = True
                    z_affected_indices.add(j)
    else:
        # Generate all possible Pauli strings
        x_ops = [[False]*num_qubits for _ in range(4**num_qubits)]
        z_ops = [[False]*num_qubits for _ in range(4**num_qubits)]

        for i,paulis in enumerate(itertools.product(['I', 'X', 'Y', 'Z'], repeat=num_qubits)):
            for j,pauli in enumerate(paulis):
                if pauli == 'X':
                    x_ops[i][j] = True
                elif pauli == 'Y':
                    x_ops[i][j] = True
                    z_ops[i][j] = True
                elif pauli == 'Z':
                    z_ops[i][j] = True

        if not include_identity:
            x_ops = x_ops[1:]
            z_ops = z_ops[1:]

        x_affected_indices = range(num_qubits)
        z_affected_indices = range(num_qubits)

    assert len(x_ops) == len(z_ops)
    assert len(x_ops) == min(max_error_strings, 4**num_qubits-1 if not include_identity else 4**num_qubits)

    return x_ops, z_ops, list(x_affected_indices), list(z_affected_indices)