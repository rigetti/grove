"""
This module helps construct sums of non-hermitian operators for measuring
specific sets of amplitudes.  The general procedure for measuring individual
amplitudes was outlined in

Nature Communications 7, Article number: 10439 (2016)
doi:10.1038/ncomms10439

Each amplitude c_{j} is associated with an operator |a><j| where |a> is an
arbitrary computation basis vector.  These are referred to as column operators.
(Cj)

Each column operator has an expectation value
<Cj> = <psi|a>c_{j}

which is proportional to ta complex state vector coefficient...except has
an extra factor <psi|a> that does not depend on 'j'.  Therefore, we can select
a single |a> and for C_{a} = |<psi|a>|^{2}.  What this means is that we can
figure out |<psi|a>|^{2} and thus |<psi|a>| which gives us the normalization
constant we want.
"""
from functools import reduce
import numpy as np
from pyquil.paulis import sX, sY, sZ, sI, PauliSum
from grove.measurements.estimation import estimate_locally_commuting_operator


def _single_projector_generator(ket_op, bra_op, index):
    """
    Generate the pauli sum terms corresponding to |ket_op><brak_op|

    :param ket_op: single qubit computational basis state
    :param bra_op: single qubit computational basis state
    :param index: qubit index to assign to the projector
    :return: pauli sum of single qubit projection operator
    :rtype: PauliSum
    """
    if not isinstance(ket_op, int):
        raise TypeError("ket_op needs to be an integer")
    if not isinstance(bra_op, int):
        raise TypeError("ket_op needs to be an integer")
    if ket_op not in [0, 1] or bra_op not in [0, 1]:
        raise ValueError("bra and ket op needs to be either 0 or 1")

    if ket_op == 0 and bra_op == 0:
        return 0.5 * (sZ(index) + sI(index))
    elif ket_op == 0 and bra_op == 1:
        return 0.5 * (sX(index) + 1j * sY(index))
    elif ket_op == 1 and bra_op == 0:
        return 0.5 * (sX(index) - 1j * sY(index))
    else:
        return 0.5 * (sI(index) - sZ(index))


def projector_generator(ket, bra):
    """
    Generate a Pauli Sum that corresponds to the projection operator |ket><bra|

    note: ket and bra are numerically ordered such that ket = [msd, ..., lsd]
          where msd == most significant digit and lsd = least significant digit.

    :param List ket: string of zeros and ones corresponding to a computational
                     basis state.
    :param List bra: string of zeros and ones corresponding to a computational
                     basis state.
    :return: projector as a pauli sum
    :rytpe: PauliSum
    """
    projectors = []
    for index, (ket_one_qubit, bra_one_qubit) in enumerate(zip(ket[::-1], bra[::-1])):
        projectors.append(_single_projector_generator(ket_one_qubit,
                                                      bra_one_qubit, index))
    return reduce(lambda x, y: x * y, projectors)


def measure_wf_coefficients(prep_program, coeff_list, reference_state,
                            quantum_resource, variance_bound=1.0E-6):
    """
    Measure a set of coefficients with a phase relative to the reference_state

    :param prep_program: pyQuil program to prepare the state
    :param coeff_list: list of integers labeling amplitudes to measure
    :param reference_state: Integer of the computational basis state to use as
                            a reference
    :param quantum_resource: An instance of a quantum abstract machine
    :param variance_bound: Default 1.0E-6.  variance of the monte carlo
                           estimator for the non-hermitian operator
    :return: returns a list of reference_state amplitude + coeff_list amplitudes
    """
    num_qubits = len(prep_program.get_qubits())
    normalizer_ops = projector_generator(reference_state, reference_state)
    c0_coeff, _, _ = estimate_locally_commuting_operator(
        prep_program, normalizer_ops, variance_bound=variance_bound,
        quantum_resource=quantum_resource)
    c0_coeff = np.sqrt(c0_coeff)

    amplitudes = []
    for ii in coeff_list:
        if ii == reference_state:
            amplitudes.append(c0_coeff)
        else:
            bra = list(map(int, np.binary_repr(ii, width=num_qubits)))
            c_ii_op = projector_generator(reference_state, bra)

            result = estimate_locally_commuting_operator(
                prep_program, c_ii_op, variance_bound=variance_bound,
                quantum_resource=quantum_resource)
            amplitudes.append(result[0] / c0_coeff)

    return amplitudes


def measure_pure_state(prep_program, reference_state, quantum_resource,
                       variance_bound=1.0E-6):
    """
    Measure the coefficients of the pure state

    :param prep_program: pyQuil program to prepare the state
    :param reference_state: Integer of the computational basis state to use as
                            a reference
    :param quantum_resource: An instance of a quantum abstract machine
    :param variance_bound: Default 1.0E-6.  variance of the monte carlo
                           estimator for the non-hermitian operator
    :return: an estimate of the wavefunction as a numpy.ndarray
    """
    num_qubits = len(prep_program.get_qubits())
    amplitudes_to_measure = list(range(2 ** num_qubits))

    amplitudes = measure_wf_coefficients(prep_program, amplitudes_to_measure,
                                         reference_state,
                                         quantum_resource,
                                         variance_bound=variance_bound)
    wavefunction = np.asarray(amplitudes)
    return wavefunction.reshape((-1, 1))
