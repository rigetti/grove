"""
This module prepares and evaluates the MAXCUT QAOA expectation value.
"""

import numpy as np
import networkx as nx
import pyquil.quil as pq
from pyquil.gates import *
from pyquil.paulis import *


def edges_to_graph(graph_edges):
    """
    Converts a list of edges into a networkx graph object
    :param (list[tuple[int]]) graph_edges: list of int pairs e.g. [(0,1), (0,2)]
    :return (networkx.Graph) graph: a graph defined by the given edges_list
    """
    maxcut_graph = nx.Graph()
    for edge in graph_edges:
        maxcut_graph.add_edge(*edge)
    graph = maxcut_graph.copy()
    return graph

def construct_reference_state_program(num_qubits):
    """
    Constructs the program for the standard reference state for QAOA - "s"
    :param (int) num_qubits: the number of qubits in the reference state
    :return (Program) reference_state_program: the pyquil program
    """
    reference_state_program = pq.Program()
    for qubit_index in xrange(num_qubits):
        gate = H(qubit_index)
        reference_state_program.inst(gate)
    return reference_state_program

def get_cost_hamiltonian(graph):
    """
    Constructs the cost hamiltonian for a given graph
    :param (networkx.Graph) graph: a graph where each node is a qubit index
    :return (PauliSum) cost_hamiltonian: the hamiltonian for the cost function
    :return (pq.Program) cost_program: evolves a state with the
    """
    cost_hamiltonian = PauliSum([])
    cost_program = pq.Program()
    for qubit_index_A, qubit_index_B in graph.edges():
        cost_term = PauliSum([PauliTerm("Z", qubit_index_A, 1.0)*
                              PauliTerm("Z", qubit_index_B)])
        cost_hamiltonian += cost_term
        cost_program.inst(Z(qubit_index_A))
        cost_program.inst(Z(qubit_index_B))
    return cost_hamiltonian, cost_program

def get_driver_hamiltonian(graph):
    """
    Constructs the driver hamiltonian for a given graph
    :param (networkx.Graph) graph: a graph where each node is a qubit index
    :return (PauliSum) driver_hamiltonian: hamiltonian for the driver function
    :return (pq.Program) driver_program: evolves a state according to the driver
    """
    driver_hamiltonian = PauliSum([])
    driver_program = pq.Program()
    for qubit_index in graph.nodes():
        driver_term = PauliSum([PauliTerm("X", qubit_index, 1.0)])
        driver_hamiltonian += driver_term
        driver_program.inst(X(qubit_index))
    return driver_hamiltonian, driver_program

def exponentiate_hamiltonian(hamiltonian, parameter):
    """
    Generates a unitary operator from a hamiltonian and a set of parameters
    :param (PauliSum) hamiltonian: a hermitian hamiltonian
    :param (float) parameter: the coefficient in the exponential
    :return (pq.Program) unitary: the program generated from the hamiltonian
    """
    unitary = pq.Program()
    for term in hamiltonian:
        exponentiated_term = exponential_map(term)
        unitary_term = exponentiated_term(parameter)
        unitary += unitary_term
    return unitary

def pauli_term_to_gate(pauli_term):
    """
    Converts a PauliTerm to the corresponding Gate
    :param (PauliTerm) pauli_term: consists of a qubit index and operator
    :param (Gate) gate: looked in the STANDARD_GATES dict
    """
    qubit_index, operator = list(pauli_term)[0]
    gate = STANDARD_GATES[operator](qubit_index)
    return gate

def get_program_parameterizer(steps, hamiltonian):
    """
    Creates a function for parametrically generating maxcut qaoa pyquil programs
    :param (int) steps: also known as the "level" or "p"
    :param (PauliSum) hamiltonian: generates the parameterized unitaries
    :return (function) program_parameterizer: maps angles to a pyquil program
    """

    def program_parameterizer(angles):
        """
        Creates a list of parameterized unitaries
        :param (list[floats]) angles: the exponentiation angle for each step
        :return (pq.Program()) parameterized_program: for maxcut qaoa
        """
        if len(angles) != steps:
            raise ValueError("""len(params) doesn't match the number of
                parameters required by `steps`""")
        unitaries_list = [exponentiate_hamiltonian(hamiltonian, step_angle/2)
            for step_angle in angles]
        return unitaries_list

    return program_parameterizer

def operators_lists_to_program(operators_lists):
    zipped_operators = zip(*operators_lists)
    program = pq.Program()
    for operators_slice in zipped_unitaries:
        for operator in operators_slice:
            program += operator
    return program

#def create_hermitian_structure():


#Reduce the complexity
def get_program_parameterizer_maxcut(steps, cost_hamiltonian,
        driver_hamiltonian, cost_program, driver_program):
    """
    Creates a function for parametrically generating maxcut qaoa pyquil programs
    :param (int) steps: also known as the "level" or "p"
    :param (PauliSum) cost_hamiltonian: the hamiltonian for the cost function
    :param (PauliSum) driver_hamiltonian: hamiltonian for the driver function
    :param (pq.Program) cost_program: encodes the hermitian structure
    :param (pq.Program) driver_program: encodes the hermitian structure
    :return (function) program_parameterizer: maps params to a pyquil program
    """

    def program_parameterizer(params):
        """
        Constructs the pyquil program which alters the reference state using the
        cost hamiltonian and driver hamiltonian according to the given params

        :param (list[floats]) params: betas first, then gammas
        :return (pq.Program()) parameterized_program: for maxcut qaoa
        """
        if len(params) != 2*steps:
            raise ValueError("""len(params) doesn't match the number of
                parameters required by `steps`""")

        betas = params[:steps]
        gammas = params[steps:]

        parameterized_program = pq.Program()

        #hermitian_structure = {}
        #unitary_structure = {}

        for step_index in xrange(steps):

            #unitary_structure[step_index + 1] = {}
            #hermitian_structure[step_index + 1] = {}

            this_step_beta = betas[step_index]
            this_step_gamma = gammas[step_index]

            #The terms for a given step are all sigma_x operators and hence
            #commute, so you can simply exponentiate in sequence.`
            this_step_driver_unitary = exponentiate_hamiltonian(
                driver_hamiltonian, this_step_beta/2)
            #The terms for a given_step are all sigma_z operators and hence
            #commute, so you can simply exponentiate in sequence.
            this_step_cost_unitary = exponentiate_hamiltonian(
                cost_hamiltonian, this_step_gamma/2)

            parameterized_program += this_step_cost_unitary
            parameterized_program += this_step_driver_unitary

            """
            hermitian_structure[step_index + 1]["cost"] = [term for term in
                cost_program]
            hermitian_structure[step_index + 1]["driver"] = [term for term in
                driver_program]

            unitary_structure[step_index + 1]["cost"] = [term for term in
                this_step_cost_unitary]
            unitary_structure[step_index + 1]["driver"] = [term for term in
                this_step_driver_unitary]
            """

        return parameterized_program

    return program_parameterizer


def maxcut_qaoa_constructor(graph_edges, steps):
    """
    Convenience function which wraps the intermediate functions
    :param (list[tuple[int]]) graph_edges: list of int pairs e.g. [(0,1), (0,2)]
    """
    graph = edges_to_graph(graph_edges)
    num_qubits = len(graph.nodes())
    reference_state_program = construct_reference_state_program(num_qubits)
    cost_hamiltonian, cost_program = get_cost_hamiltonian(graph)
    driver_hamiltonian, driver_program = get_driver_hamiltonian(graph)
    program_parameterizer = get_program_parameterizer_maxcut(steps,
        cost_hamiltonian, driver_hamiltonian, cost_program, driver_program)
    return (program_parameterizer, reference_state_program,
            cost_hamiltonian, num_qubits)
