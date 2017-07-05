"""
This module prepares and evaluates the MAXCUT QAOA expectation value.
"""

import numpy as np
import networkx as nx
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import *

import expectation_value


def edges_to_graph(edges_list):
    """Converts a list of edges into a networkx graph object
    """
    maxcut_graph = nx.Graph()
    for edge in edges_list:
        maxcut_graph.add_edge(*edge)
    graph = maxcut_graph.copy()
    return graph

def construct_reference_state_program(num_qubits):
    """Constructs the standard reference state for QAOA - "s"
    """
    reference_state_program = pq.Program()
    for qubit_index in xrange(num_qubits):
        gate = H(qubit_index)
        reference_state_program.inst(gate)
    return reference_state_program

def get_cost_hamiltonian(graph):
    cost_hamiltonian = PauliSum([])
    for qubit_index_A, qubit_index_B in graph.edges():
        cost_term = PauliSum([PauliTerm("Z", qubit_index_A, 1.0)*
                              PauliTerm("Z", qubit_index_B)])
        cost_hamiltonian += cost_term
    return cost_hamiltonian

def get_driver_hamiltonian(graph):
    driver_hamiltonian = PauliSum([])
    for qubit_index in graph.nodes():
        driver_term = PauliSum([PauliTerm("X", qubit_index, 1.0)])
        driver_hamiltonian += driver_term
    return driver_hamiltonian

def exponentiate_hamiltonian(hamiltonian, parameters):
    unitary = pq.Program()
    for term in hamiltonian:
        exponentiated_term = exponential_map(term)
        unitary_term = exponentiated_term(parameters)
        unitary += unitary_term
    return unitary

def get_program_parameterizer_maxcut(steps, reference_state_program,
        cost_hamiltonian, driver_hamiltonian):
    """
    Return a function that accepts parameters and returns a list of
    pyquil programs constituting a full pyquil program
    """

    def program_parameterizer(params):
        """Constructs the pyquil program which rotates the reference state
            using the cost hamiltonian and driver hamiltonian according to
            the given parameters

        :param params: an array of 2*p angles, betas first, then gammas
        :return: a list of pyquil programs
        """
        if len(params) != 2*steps:
            raise ValueError("""len(params) doesn't match the number of
                parameters required by `steps`""")

        betas = params[:steps]
        gammas = params[steps:]

        parameterized_program = pq.Program()
        parameterized_program += reference_state_program

        for step_index in xrange(steps):
            this_step_betas = betas[step_index]
            this_step_gammas = gammas[step_index]

            #The terms for a given step are all sigma_x operators and hence commute,
            #so you can simply exponentiate in sequence.
            this_step_driver_unitary = exponentiate_hamiltonian(
                driver_hamiltonian, this_step_betas)
            #The terms for a given_step are all sigma_z operators and hence commute,
            #so you can simply exponentiate in sequence.
            this_step_cost_unitary = exponentiate_hamiltonian(
                cost_hamiltonian, this_step_gammas)

            parameterized_program += this_step_cost_unitary
            parameterized_program += this_step_driver_unitary

        return parameterized_program

    return program_parameterizer

def maxcut_qaoa_constructor(graph_edges, steps):
    graph = edges_to_graph(graph_edges)
    num_qubits = len(graph.nodes())
    reference_state_program = construct_reference_state_program(num_qubits)
    cost_hamiltonian = get_cost_hamiltonian(graph)
    driver_hamiltonian = get_driver_hamiltonian(graph)
    program_parameterizer = get_program_parameterizer_maxcut(steps,
        reference_state_program, cost_hamiltonian, driver_hamiltonian)
    return program_parameterizer, cost_hamiltonian

if __name__ == "__main__":
    test_graph_edges = [(0,1)]
    steps = 1 #the number of trotterization steps
    program_parameterizer, cost_hamiltonian = maxcut_qaoa_constructor(
        test_graph_edges, steps)
    #beta = 0#np.pi/3
    #gamma = 0#np.pi/3
    #beta = np.pi/8
    #gamma = np.pi/4
    beta = 1.5
    gamma = 1.2

    test_params = [beta, gamma]
    test_program = program_parameterizer(test_params)
    qvm_connection = api.SyncConnection()
    analytic_expectation = expectation_value.get_analytic_expectation_p1(
        beta, gamma)
    numeric_expectation = expectation_value.expectation(test_program,
        cost_hamiltonian, qvm_connection)
    print(analytic_expectation)
    print(numeric_expectation)



