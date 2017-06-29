"""
This module implements the analytical gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
Compute the gradient of a given product of parametric unitary operators.
Each unitary operator can be expressed in terms of its Hermitian Generator.
In turn each Hermitian Generator can be decomposed into a sum of pauli gates.
"""

"""
Types:
Pauli < Pauli_Product < Hermitian
Primitive_Unitary < Unitary < Operator
gradient: Float -> Float
product: List(Unitary) -> Unitary
expectation: State, Unitary -> Float
ground_state_expectation: Unitary -> Float
dagger: Unitary -> Unitary
list_primitive_unitaries: Unitary -> List(Primitive_Unitary)
get_generator: Primitive_Unitary -> Hermitian
get_unitary_deriative: Unitary -> Operator
operator_gradient: Unitary -> Unitary
get_pauli_products: Hermitian -> List(Pauli_Product)
get_paulis: Pauli_Product -> List(Pauli)
"""

import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *


def construct_ref_state_prep(num_qubits):
    """Constructs the standard reference state for QAOA "s"
    """
    ref_prog = pq.Program()
    for i in xrange(num_qubits):
        ref_prog.inst(H(i))
    ref_state_prep = ref_prog
    return ref_state_prep

def get_parameterized_program(steps, cost_ham, ref_ham, ref_state_prep):
    """
    Return a function that accepts parameters and returns a new Quil
    program

    :returns: a function
    """
    cost_para_programs = []
    driver_para_programs = []

    for idx in xrange(steps):
	cost_list = []
	driver_list = []
	for cost_pauli_sum in cost_ham:
	    for term in cost_pauli_sum.terms:
		cost_list.append(exponential_map(term))

	for driver_pauli_sum in ref_ham:
	    for term in driver_pauli_sum.terms:
		driver_list.append(exponential_map(term))

	cost_para_programs.append(cost_list)
	driver_para_programs.append(driver_list)

    def psi_ref(params):
	"""Construct a Quil program for the vector (beta, gamma).

	:param params: array of 2 . p angles, betas first, then gammas
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
	    for fprog in cost_para_programs[idx]:
		prog += fprog(gammas[idx])

	    for fprog in driver_para_programs[idx]:
		prog += fprog(betas[idx])

	return prog

    return psi_ref


def make_controlled(gate, control_index):
    pass #Implement this


def get_analytical_gradient_component(cost_ham, ref_ham, psi_ref,
        component_index):
    """Returns a list of operators corresponding to the decomposition of a
    component of the analytic gradient
    """
    prog = pq.Program()
    #Get the first component_index-1 Unitaries from psi_ref
    psi_ref_lower = psi_ref[:component_index-1]
    psi_ref_upper = psi_ref[component_index:]
    component_pauli_sum = hams[component_index] #define hams
    analytical_gradients_list = []
    ancilla_qubit_index = len(prog.qubit_register)
    for pauli_term in component_pauli_sum.terms:
        prog += psi_ref_lower
        prog.inst(H(ancilla_qubit_index))
        controlled_pauli_term = make_controlled(pauli_term, ancilla_qubit_index)
