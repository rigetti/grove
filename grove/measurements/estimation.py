"""
Utilities for estimating expected values of Pauli terms given pyquil programs
"""
import numpy as np
from pyquil.paulis import PauliSum, PauliTerm, commuting_sets, sI
from pyquil.quil import Program
from pyquil.gates import RY, RX


class CommutationError(Exception):
    pass


def remove_imaginary_terms(pauli_sums):
    """
    Remove the imaginary components of the Pauli sum term

    :param pauli_sums:
    :return:
    """
    if not isinstance(pauli_sums, PauliSum):
        raise TypeError("not a pauli sum. please give me one")
    new_term = sI(0) * 0.0
    for term in pauli_sums:
        if (not np.isclose(term.coefficient.real, 0.0) and
                np.isclose(term.coefficient.imag, 0.0)):
            new_term += term

    return new_term


def get_rotation_program(pauli_term):
    """
    Generate a rotation program so that the pauli term is diagoanl

    :param pauli_term:
    :return: rotaiton program
    :rtype: Program
    """
    meas_basis_change = Program()
    for index, gate in pauli_term:
        if gate == 'X':
            meas_basis_change.inst(RY(-np.pi / 2, index))
        elif gate == 'Y':
            meas_basis_change.inst(RX(np.pi / 2, index))
        elif gate == 'Z':
            pass
        else:
            raise ValueError()

    return meas_basis_change


def get_parity(pauli_terms, bitstring_results):
    """
    Calculate the parity at each of the marked qubits in pauli_terms

    Each list of bits in the bitstring_results is the maximum weight pauli term
    in pauli_terms.  The bits in each element of bitstring_results correspond
    to a numerical ordering of the maximal weight element of pauli_terms.

    An example:

    A maximum weight pauli term is Z1Z5Z6.

    each element of bitstring_results is something in
    :math:`\{0, 1\}^{\otimes 3}` where the first bit corresponds to the measured
    value of qubit 1, the second bit corresponds to the measured value of qubit
    5, and the third bit corresponds to the measured value of qubit 6.

    :param pauli_terms:
    :param bitstring_results:
    :return:
    """
    max_weight_pauli_index = np.argmax(
        [len(term.get_qubits()) for term in pauli_terms])
    active_qubit_indices = sorted(
        pauli_terms[max_weight_pauli_index]._ops.keys())
    index_mapper = dict(zip(active_qubit_indices,
                            range(len(active_qubit_indices))))

    results = np.zeros((len(pauli_terms), len(bitstring_results)))

    # convert to array so we can fancy index into it later.
    # list() is needed to cast because we don't really want a map object
    bitstring_results = list(map(np.array, bitstring_results))
    for row_idx, term in enumerate(pauli_terms):
        memory_index = np.array(list(map(lambda x: index_mapper[x],
                                         sorted(term.get_qubits()))))

        results[row_idx, :] = [-2 * (sum(x[memory_index]) % 2) + \
                               1 for x in bitstring_results]
    return results


def estimate_pauli_set(pauli_term_set, basis_transform_dict, program,
                       variance_bound, quantum_resource):
    """
    Estimate the mean of a sum of pauli terms to set variance

    The sample variance is calculated by

    .. math::
        \begin{align}
        \mathrm{Var}[\hat{\langle H \rangle}] = \sum_{i, j}h_{i}h_{j}
        \mathrm{Cov}(\hat{\langle P_{i} \rangle}, \hat{\langle P_{j} \rangle})
        \end{align}

    :param pauli_term_set: list of pauli terms to measure simultaneously
    :param basis_transform_dict: basis transform dictionary where the key is
                                 the qubit index and the value is the basis to
                                 rotate into. Valid basis is [I, X, Y, Z].
    :param program: program generating a state to sample from
    :param variance_bound:  Bound on the variance of the estimator for the
                            PauliSum. Remember this is the SQUARE of the
                            standard error!

    :param quantum_resource: quantum abstract machine object
    :return: estimated expected value, covariance matrix, variance of the
             estimator, and the number of shots taken
    """
    if not isinstance(pauli_term_set, list):
        raise TypeError("pauli_term_set needs to be a list")

    # check if each term commutes with everything
    if len(commuting_sets(sum(pauli_term_set))) != 1:
        raise CommutationError("Not all terms commute in the expected way")

    pauli_for_rotations = PauliTerm.from_list(
        [(value, key) for key, value in basis_transform_dict.items()])
    post_rotations = get_rotation_program(pauli_for_rotations)

    coeff_vec = np.array(
        list(map(lambda x: x.coefficient, pauli_term_set))).reshape((-1, 1))

    results = None
    sample_variance = np.infty
    while sample_variance > variance_bound:
        # note: bit string values are returned according to their listed order
        # in run_and_measure.  Therefore, sort beforehand to keep alpha numeric
        tresults = quantum_resource.run_and_measure(
            program + post_rotations,
            sorted(list(basis_transform_dict.keys())),
            trials=10000)

        parity_results = get_parity(pauli_term_set, tresults)

        # Note: easy improvement would be to update mean and variance on the fly
        # instead of storing all these results.
        if results is None:
            results = parity_results
        else:
            results = np.hstack((results, parity_results))

        # calculate the expected values....
        covariance_mat = np.cov(results)
        sample_variance = coeff_vec.T.dot(covariance_mat).dot(coeff_vec) / \
                          results.shape[1]

    return coeff_vec.T.dot(np.mean(results, axis=1)), covariance_mat, \
           sample_variance, results.shape[1]
