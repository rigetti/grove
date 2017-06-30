"""
This module implements the analytical gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
Compute the gradient of a given product of parametric unitary operators.
"""

import networkx as nx
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import *


def edges_to_graph(edges_list):
    """Converts a list of edges into a networkx graph object
    """
    maxcut_graph = nx.Graph()
    for edge in edges_list:
        maxcut_graph.add_edge(*edge)
    graph = maxcut_graph.copy()
    return graph

def construct_ref_state_prep(num_qubits):
    """Constructs the standard reference state for QAOA - "s"
    """
    ref_prog = pq.Program()
    for i in xrange(num_qubits):
        ref_prog.inst(H(i))
    ref_state_prep = ref_prog
    return ref_state_prep

def define_parametric_program_qaoa(trotterization_steps, graph):
    cost_para_programs = []
    driver_para_programs = []

    for step in xrange(trotterization_steps):
        for qubit_index_A, qubit_index_B in graph.edges():
            cost_para_program = []
            interaction_term = (PauliTerm("Z", qubit_index_A, 0.5)*
                                PauliTerm("Z", qubit_index_B))
            constant_term = PauliTerm("I", 0, -0.5)
            cost_para_program.append(interaction_term)
            cost_para_program.append(constant_term)
            #print(cost_para_program)

        for qubit_index in graph.nodes():
            driver_para_program = []
            driver_term = PauliTerm("X", qubit_index, -1.0)
            driver_para_program.append(driver_term)

        cost_para_programs.append(cost_para_program)
        driver_para_programs.append(driver_para_program)

    return (cost_para_programs, driver_para_programs)


def get_program_parameterizer_qaoa(steps, cost_para_programs,
        driver_para_programs):
    """
    Return a function that accepts parameters and returns a new Quil
    program

    :returns: a function
    """

    def program_parameterizer(params):
	"""Construct a Quil program for the vector (beta, gamma).

	:param params: array of 2*p angles, betas first, then gammas
	:return: a pyquil program object
	"""
	if len(params) != 2*steps:
	    raise ValueError("""params doesn't match the number of parameters set
				by `steps`""")
	betas = params[:steps]
	gammas = params[steps:]

	parameterized_prog = pq.Program()
        underlying_prog = pq.Program()
        terms_list = []

	for step_index in xrange(steps):
            cost_para_program = cost_para_programs[step_index]
            step_gammas = gammas[step_index]
	    for cost_term in cost_para_program:
                parameterized_cost_term = exponential_map(cost_term)
                step_cost_term = parameterized_cost_term(step_gammas)
                terms_list.append(step_cost_term)
		parameterized_prog += step_cost_term
            cost_length = len(cost_para_program)

            driver_para_program = driver_para_programs[step_index]
            step_betas = betas[step_index]
	    for driver_term in driver_para_program:
                parameterized_driver_term = exponential_map(driver_term)
                step_driver_term = parameterized_driver_term(step_betas)
                terms_list.append(step_driver_term)
		parameterized_prog += step_driver_term

	return (terms_list, parameterized_prog, cost_length)

    return program_parameterizer


def get_analytical_gradient_component_qaoa(
        terms_list,
        num_qubits, cost_length, component_index):
    """
    Maps: Parameterized_Program -> Parameterized_Program

    :returns: a function
    """
    gradient_component_prog = pq.Program()
    i_one = np.array([[1.0, 0.0], [0.0, 1.0j]])
    gradient_component_prog.defgate("I-ONE", i_one)
    cz = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, -1.0]])
    gradient_component_prog.defgate("CZ", cz)
    parameterized_prog_lower = terms_list[:component_index-1]
    parameterized_prog_upper = terms_list[component_index:]
    ancilla_qubit_index = num_qubits
    offset = num_qubits
    for parameterized_inst in parameterized_prog_lower:
        gradient_component_prog += parameterized_inst
    gradient_component_prog.inst(H(ancilla_qubit_index))
    if component_index > cost_length + offset: #Then apply driver gate
        print(parameterized_prog[component_index])
        #node_index = component_index_to_graph_element(component_index)
        #Check the argument order on this
        #gradient_component_prog.inst(CNOT(node_index, ancilla_qubit_index))
    elif component_index > offset:
        #print(parameterized_prog[component_index])
        #edge_index_A, edge_index_B = component_index_to_graph_element(
        #    component_index)
        #Check the argument order
        gradient_component_prog.inst(CZ(edge_index_A, ancilla_qubit_index))
        gradient_component_prog.inst(CZ(edge_index_B, ancilla_qubit_index))
    #print(type(gradient_component_prog))
    #print(parameterized_prog_upper)
    gradient_component_prog.inst(parameterized_prog_upper)
    gradient_component_prog.inst("I-ONE", ancilla_qubit_index)
    gradient_component_prog.inst(H(ancilla_qubit_index))
    return gradient_component_prog

if __name__ == "__main__":
    #square_ring_edges = [(0,1), (1,2), (2,3), (3,0)]
    square_ring_edges = [(0,1)]
    graph = edges_to_graph(square_ring_edges)
    num_qubits = len(graph.nodes())
    ref_state_prep = construct_ref_state_prep(num_qubits)
    trotterization_steps = 1 #Referred to as "p" in the paper
    cost_para_programs, driver_para_programs = define_parametric_program_qaoa(
        trotterization_steps, graph)
    program_parameterizer = get_program_parameterizer_qaoa(trotterization_steps,
        cost_para_programs, driver_para_programs)
    component_index = 3
    #start_params = [2.2, 1.2, 2.9, 5.3] #Random test parameters
    start_params = [2.2, 1.2] #Random test parameters
    terms_list, parameterized_prog, cost_length = program_parameterizer(start_params)
    #print(parameterized_prog[0])
    get_analytical_gradient_component_qaoa(terms_list, num_qubits,
        cost_length, component_index)


