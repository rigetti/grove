"""
This module computes the expectation value of a cost_hamiltonian (paulisum)
with respect to a state prepared by a pyquil program
"""

import numpy as np
import networkx as nx
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import *


def expectation(pyquil_program, cost_hamiltonian, qvm_connection):
    """
    Computes the expectation value of the pauli_sum over the distribution
    generated from pyquil_prog.

    :param pyquil_program: (pyquil program)
    :param cost_hamiltonian: (PauliSum) PauliSum representing the operator of which
                      to calculate the expectation value.
    :param qvm_connection: (qvm connection)

    :returns: (float) representing the expectation value of pauli_sum given
              the distribution generated from quil_prog.
    """
    operator_programs = []
    operator_coefficients = []
    for cost_term in cost_hamiltonian.terms:
        operator_program = pq.Program()
        for qubit_index, operator in cost_term:
            operator_program.inst(STANDARD_GATES[operator](qubit_index))
        operator_programs.append(operator_program)
        operator_coefficients.append(cost_term.coefficient)
    print(pyquil_program)
    print(operator_programs[0])
    result_overlaps = qvm_connection.expectation(pyquil_program,
        operator_programs=operator_programs)
    result_overlaps = list(result_overlaps)
    assert len(result_overlaps) == len(operator_programs), """The incorrect
        number of results were returned from the QVM"""
    expectation = sum(map(lambda x: x[0]*x[1],
                          zip(result_overlaps, operator_coefficients)))
    return expectation

def get_analytic_expectation_p1(beta, gamma):
    #See Mathematica Notebook for derivation
    return np.sin(2*beta)*np.sin(gamma)

def get_ag_expectation_p1_driver(beta, gamma):
    #See Mathematica Notebook for derivation
    return 2*np.cos(2*beta)*np.sin(gamma)

def get_ag_expectation_p1_cost(beta, gamma):
    #See Mathematica Notebook for derivation
    return np.sin(2*beta)*np.cos(gamma)

#Need tests which assert that the analytical and numerical expecations are equal

