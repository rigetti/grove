"""
This module implements the analytical gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
Compute the gradient of a given product of parametric unitary operators.
"""

import networkx as nx
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import PauliTerm, PauliSum


def edges_to_graph(edges_list):
    """Converts a list of edges into a networkx graph object
    """
    maxcut_graph = nx.Graph()
    for edge in edges_list:
        maxcut_graph.add_edge(*edge)
    graph = maxcut_graph.copy()
    return graph

def get_cost_ham(graph):
    """Constructs the cost Hamiltonian - "C"
    Edge-Based
    """
    cost_ham = []
    for qubit_index_A, qubit_index_B in graph.edges():
        cost_ham.append((PauliTerm("Z", qubit_index_A, 0.5)*
                               PauliTerm("Z", qubit_index_B)) +
                              PauliTerm("I", 0, -0.5))
    return cost_ham

def get_driver_ham(graph):
    """Constructs the driving Hamiltonian - "B"
    Node-Based
    """
    driver_ham = []
    for qubit_index in graph.nodes():
        driver_ham.append(PauliTerm("X", qubit_index, -1.0))
    return driver_ham

def define_parametric_program_qaoa_v2(steps, cost_ham, driver_ham)
    cost_para_programs = []
    driver_para_programs = []

    for step in steps:
        for qubit_index_A, qubit_index_B in graph.edges():
            cost_para_program = []
            interaction_term = (PauliTerm("Z", qubit_index_A, 0.5)*
                                PauliTerm("Z", qubit_index_B))
            constant_term = PauliTerm("I", 0, -0.5)
            cost_para_program.append(exponential_map(interaction_term))
            cost_para_program.append(exponential_map(constant_term))

        for qubit_index in graph.nodes():
            driver_para_program = []
            driver_term = PauliTerm("X", qubit_index, -1.0)
            driver_para_program.append(exponential_map(driver_pauli_term))

        cost_para_programs.append(cost_para_program)
        driver_para_programs.append(driver_para_program)

    return (cost_para_programs, driver_para_programs)


def get_parameterized_program_v2(steps, cost_para_programs,
        driver_para_programs, ref_state_prep)
    """
    Return a function that accepts parameters and returns a new Quil
    program

    :returns: a function
    """

    def parameterized_prog(params):
	"""Construct a Quil program for the vector (beta, gamma).

	:param params: array of 2*p angles, betas first, then gammas
	:return: a pyquil program object
	"""
	if len(params) != 2*steps:
	    raise ValueError("""params doesn't match the number of parameters set
				by `steps`""")
	betas = params[:steps]
	gammas = params[steps:]

	prog = pq.Program()
	prog += ref_state_prep
	for step_index in xrange(steps):
            cost_para_program = cost_para_programs[step_index]
            step_gammas = gammas[step_index]
	    for cost_term in cost_para_program:
                step_cost_term = cost_term(step_gammas)
		prog += step_cost_term

            driver_para_program = driver_para_programs[step_index]
            step_betas = betas[step_index]
	    for driver_term in driver_para_program:
                step_driver_term = driver_term(step_betas)
		prog += step_driver_term

	return prog

    return parameterized_prog


def construct_ref_state_prep(num_qubits):
    """Constructs the standard reference state for QAOA - "s"
    """
    ref_prog = pq.Program()
    for i in xrange(num_qubits):
        ref_prog.inst(H(i))
    ref_state_prep = ref_prog
    return ref_state_prep

def define_parametric_program_qaoa(steps, cost_ham, driver_ham)
    cost_para_programs = []
    driver_para_programs = []

    for step in xrange(steps):
	cost_list = []
	driver_list = []

	for cost_pauli_sum in cost_ham:
	    for cost_pauli_term in cost_pauli_sum.terms:
		cost_list.append(exponential_map(cost_pauli_term))

	for driver_pauli_term in driver_ham:
		driver_list.append(exponential_map(driver_pauli_term))

	cost_para_programs.append(cost_list)
	driver_para_programs.append(driver_list)
    return (cost_para_programs, driver_para_programs)


def get_parameterized_program(steps, cost_ham, ref_ham, ref_state_prep):
    """
    Return a function that accepts parameters and returns a new Quil
    program

    :returns: a function
    """

    cost_para_programs, driver_para_programs = define_parametric_program_qaoa(
        cost_ham, ref_ham)

    def parameterized_prog(params):
	"""Construct a Quil program for the vector (beta, gamma).

	:param params: array of 2*p angles, betas first, then gammas
	:return: a pyquil program object
	"""
	if len(params) != 2*steps:
	    raise ValueError("""params doesn't match the number of parameters set
				by `steps`""")
	betas = params[:steps]
	gammas = params[steps:]

	prog = pq.Program()
	prog += ref_state_prep
	for idx in xrange(steps):
	    for cost_prog in cost_para_programs[idx]:
		prog += cost_prog(gammas[idx]) #exponentiated sum becomes a product

	    for driver_prog in driver_para_programs[idx]:
		prog += driver_prog(betas[idx])

	return prog

    return parameterized_prog

#betas correspond to

def get_analytical_gradient_component_qaoa(parameterized_prog, graph_size,
    component_index):
    """
    Maps: Parameterized_Program -> Parameterized_Program

    :returns: a function
    """
    prog = pq.Program()
    parameterized_prog_lower = parameterized_prog[:component_index-1]
    parameterized_prog_upper = parameterized_prog[component_index:]
    analytical_gradients_list = []
    ancilla_qubit_index = graph_size #Check how to add a new qubit
    prog += parameterized_prog_lower
    prog.inst(H(ancilla_qubit_index))
    if component_index > steps: #Then the component_index refers to the betas
        prog.inst(CNOT(component_index, ancilla_qubit_index)) #Check the order on this!
    else:
        prog.inst(CZ(component_index
    #Need to check whether the component index refers to B or C!
    #In the case that the component index refers to part of the driver prog
    #In the case that the component index refers to part of the cost prog
    #Define a controlled version of a product of
    prog += parameterized_prog_upper
    i_one = np.array([[1.0, 0.0], [0.0, 1.0j]])
    prog.defgate("I-ONE", i_one)
    prog.inst("I-ONE", ancilla_qubit_index)
    prog.inst(H(ancilla_qubit_index))
    return prog


if __name__ == "__main__":
    square_ring_edges = [(0,1), (1,2), (2,3), (3,0)]
    graph = edges_to_graph(square_ring_edges)
    cost_ham = get_cost_ham(graph)
    ref_ham = get_ref_ham(graph)
    trotterization_steps = 2 #Referred to as "p" in the paper


