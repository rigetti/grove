"""
Utilities for estimating expected values of Pauli terms given pyquil programs
"""
import numpy as np
from pyquil.paulis import (PauliSum, PauliTerm, commuting_sets, sI,
                           term_with_coeff)
from pyquil.quil import Program
from pyquil.gates import RY, RX


class CommutationError(Exception):
    pass


def remove_imaginary_terms(pauli_sums):
    """
    Remove the imaginary component of each term in a Pauli sum

    :param PauliSum pauli_sums: The Pauli sum to process.
    :return: a purely hermitian Pauli sum.
    :rtype: PauliSum
    """
    if not isinstance(pauli_sums, PauliSum):
        raise TypeError("not a pauli sum. please give me one")
    new_term = sI(0) * 0.0
    for term in pauli_sums:
        new_term += term_with_coeff(term, term.coefficient.real)

    return new_term


def get_rotation_program(pauli_term):
    """
    Generate a rotation program so that the pauli term is diagonal

    :param PauliTerm pauli_term: The Pauli term used to generate diagonalizing
                                 one-qubit rotations.
    :return: The rotation program.
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
    Calculate the eigenvalues of Pauli operators given results of projective measurements

    The single-qubit projective measurement results (elements of
    `bitstring_results`) are listed in physical-qubit-label numerical order.

    An example:

    Consider a Pauli term Z1 Z5 Z6 I7 and a collection of single-qubit
    measurement results corresponding to measurements in the z-basis on qubits
    {1, 5, 6, 7}. Each element of bitstring_results is an element of
    :math:`\{0, 1\}^{\otimes 4}`.  If [0, 0, 1, 0] and [1, 0, 1, 1]
    are the two projective measurement results in `bitstring_results` then
    this method returns a 1 x 2 numpy array with values [[-1, 1]]

    :param List pauli_terms: A list of Pauli terms operators to use
    :param bitstring_results: A list of projective measurement results.  Each
                              element is a list of single-qubit measurements.
    :return: Array (m x n) of {+1, -1} eigenvalues for the m-operators in
             `pauli_terms` associated with the n measurement results.
    :rtype: np.ndarray
    """
    qubit_set = []
    for term in pauli_terms:
        qubit_set.extend(list(term.get_qubits()))
    active_qubit_indices = sorted(list(set(qubit_set)))
    index_mapper = dict(zip(active_qubit_indices,
                            range(len(active_qubit_indices))))

    results = np.zeros((len(pauli_terms), len(bitstring_results)))

    # convert to array so we can fancy index into it later.
    # list() is needed to cast because we don't really want a map object
    bitstring_results = list(map(np.array, bitstring_results))
    for row_idx, term in enumerate(pauli_terms):
        memory_index = np.array(list(map(lambda x: index_mapper[x],
                                         sorted(term.get_qubits()))))

        results[row_idx, :] = [-2 * (sum(x[memory_index]) % 2) +
                               1 for x in bitstring_results]
    return results


def estimate_pauli_sum(pauli_terms, basis_transform_dict, program,
                       variance_bound, quantum_resource,
                       commutation_check=True):
    """
    Estimate the mean of a sum of pauli terms to set variance

    The sample variance is calculated by

    .. math::
        \begin{align}
        \mathrm{Var}[\hat{\langle H \rangle}] = \sum_{i, j}h_{i}h_{j}
        \mathrm{Cov}(\hat{\langle P_{i} \rangle}, \hat{\langle P_{j} \rangle})
        \end{align}

    :param pauli_terms: list of pauli terms to measure simultaneously or a
                        PauliSum object
    :param basis_transform_dict: basis transform dictionary where the key is
                                 the qubit index and the value is the basis to
                                 rotate into. Valid basis is [I, X, Y, Z].
    :param program: program generating a state to sample from
    :param variance_bound:  Bound on the variance of the estimator for the
                            PauliSum. Remember this is the SQUARE of the
                            standard error!
    :param quantum_resource: quantum abstract machine object
    :param Bool commutation_check: Optional flag toggling a safety check
                                   ensuring all terms in `pauli_terms`
                                   commute with each other
    :return: estimated expected value, covariance matrix, variance of the
             estimator, and the number of shots taken
    """
    if not isinstance(pauli_terms, (list, PauliSum)):
        raise TypeError("pauli_terms needs to be a list or a PauliSum")

    if isinstance(pauli_terms, PauliSum):
        pauli_terms = PauliSum.terms

    # check if each term commutes with everything
    if commutation_check:
        if len(commuting_sets(sum(pauli_terms))) != 1:
            raise CommutationError("Not all terms commute in the expected way")

    pauli_for_rotations = PauliTerm.from_list(
        [(value, key) for key, value in basis_transform_dict.items()])

    post_rotations = get_rotation_program(pauli_for_rotations)

    coeff_vec = np.array(
        list(map(lambda x: x.coefficient, pauli_terms))).reshape((-1, 1))

    # upper bound on samples given by IV of arXiv:1801.03524
    num_sample_ubound = int(np.ceil(np.sum(np.abs(coeff_vec))**2 / variance_bound))
    results = None
    sample_variance = np.infty
    number_of_samples = 0
    while (sample_variance > variance_bound and
           number_of_samples < num_sample_ubound):
        # note: bit string values are returned according to their listed order
        # in run_and_measure.  Therefore, sort beforehand to keep alpha numeric
        tresults = quantum_resource.run_and_measure(
            program + post_rotations,
            sorted(list(basis_transform_dict.keys())),
            trials=min(10000, num_sample_ubound))
        number_of_samples += len(tresults)

        parity_results = get_parity(pauli_terms, tresults)

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
