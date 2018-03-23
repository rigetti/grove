"""
Utilities for estimating expected values of Pauli terms given pyquil programs
"""
import numpy as np
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.gates import RY, RX


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


def estimate_psum(operator, program, variance_bound,
                  quantum_resource, grouping=None):
    """
    Compute the expected value of an operator given an variance bound

    :param operator: PauliSum or PauliTerm to sample
    :param program: program generating a state to sample from
    :param variance_bound:  Bound on the variance of the estimator for the
                            PauliSum. Remember this is the SQUARE of the
                            standard error!
    :param assignment_prob_map: dictionary mapping qubit terms to assignment
                                error matrix
    :param quantum_resource: quantum abstract machine object
    :param grouping: (Default None) function for grouping the pauli terms of the
                     operator.
    :return: estimated expected alue
    """
    if not isinstance(operator, (PauliTerm, PauliSum)):
        raise TypeError("I only accept PauliTerm or PauliSum")

    if isinstance(operator, PauliTerm):
        operator = PauliSum([operator])

    # # group into z-basis sets
    # diagonal_basis_grouping = commuting_sets_by_zbasis(operator)
    # for key, value in diagonal_basis_grouping.items():
    #     print(key, sum(value))
    if grouping is None:
        # use default grouping which is the trival commuting terms

    else:
        num_terms = len(operator)
        variance_bound_per_term = variance_bound / num_terms
        operator_mean_estimator = 0.0
        operator_mean_variance = 0.0
        for term in operator:
            if term.id() != "":
                post_rotations = get_rotation_program(term)

                number_of_shots = int(np.ceil(abs(term.coefficient)**2 /
                                          variance_bound_per_term))
                num_bins = number_of_shots // 10000
                overflow_bin = number_of_shots % 10000
                results = []
                for _ in range(num_bins):
                    tresults = quantum_resource.run_and_measure(
                        program + post_rotations, list(term.get_qubits()),
                        trials=10000)
                    results.extend(tresults)

                if overflow_bin > 0:
                    tresults = quantum_resource.run_and_measure(
                        program + post_rotations, list(term.get_qubits()),
                        trials=overflow_bin)
                    results.extend(tresults)

                ########################################################
                # observed values and variance given the sample variance
                ########################################################
                obs_val = np.array([-2 * (sum(x) % 2) + 1 for x in results])
                mean_val = np.mean(obs_val)
                sample_var = sum((obs_val - mean_val) ** 2) / (len(obs_val) - 1)
                operator_mean_estimator += term.coefficient * mean_val
                operator_mean_variance += abs(term.coefficient)**2 * sample_var / number_of_shots

            else:
                operator_mean_estimator += term.coefficient

        return operator_mean_estimator, operator_mean_variance
