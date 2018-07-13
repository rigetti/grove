"""
Finding a solution to the travelling salesman problem.
This code is based on the following project made by BOHR.TECHNOLOGY:
https://github.com/BOHRTECHNOLOGY/quantum_tsp

Which was in turn based on the articles by Stuart Hadfield:
https://arxiv.org/pdf/1709.03489.pdf
https://arxiv.org/pdf/1805.03265.pdf
"""

import pyquil.api as api
from grove.pyqaoa.qaoa import QAOA
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.quil import Program
from pyquil.gates import X
import scipy.optimize
import numpy as np


def solve_tsp(nodes_array, connection=None, steps=3, ftol=1.0e-4, xtol=1.0e-4):
    """
    Method for solving travelling salesman problem.

    :param nodes_array: An array of points, which represent coordinates of the cities.
    :param connection: (Optional) connection to the QVM. Default is None.
    :param steps: (Optional. Default=1) Trotterization order for the QAOA algorithm.
    :param ftol: (Optional. Default=1.0e-4) ftol parameter for the Nelder-Mead optimizer
    :param xtol: (Optional. Default=1.0e-4) xtol parameter for the Nelder-Mead optimizer
    """
    if connection is None:
        connection = api.QVMConnection()
    list_of_qubits = list(range(len(nodes_array)**2))
    number_of_qubits = len(list_of_qubits)
    cost_operators = create_cost_hamiltonian(nodes_array)
    driver_operators = create_mixer_operators(len(nodes_array))
    initial_state_program = create_initial_state_program(len(nodes_array))

    minimizer_kwargs = {'method': 'Nelder-Mead',
                            'options': {'ftol': ftol, 'xtol': xtol,
                                        'disp': False}}

    vqe_option = {'disp': print_fun, 'return_all': True,
                  'samples': None}

    qaoa_inst = QAOA(connection, number_of_qubits, steps=steps, cost_ham=cost_operators,
                     ref_hamiltonian=driver_operators, driver_ref=initial_state_program, store_basis=True,
                     minimizer=scipy.optimize.minimize,
                     minimizer_kwargs=minimizer_kwargs,
                     vqe_options=vqe_option)
    
    betas, gammas = qaoa_inst.get_angles()
    most_frequent_string, _ = qaoa_inst.get_string(betas, gammas, samples=1000)
    solution = binary_state_to_points_order(most_frequent_string)
    return solution


def create_cost_hamiltonian(nodes_array):
    """
    Translating the distances between cities into the cost hamiltonian.
    """
    cost_operators = []
    distance_matrix = get_distance_matrix(nodes_array)
    for t in range(len(nodes_array) - 1):
        for city_1 in range(len(nodes_array)):
            for city_2 in range(len(nodes_array)):
                if city_1 != city_2:
                    distance = distance_matrix[city_1, city_2]
                    qubit_1 = t * len(nodes_array) + city_1
                    qubit_2 = (t + 1) * len(nodes_array) + city_2
                    cost_operators.append(PauliTerm("Z", qubit_1, distance) * PauliTerm("Z", qubit_2))
    cost_hamiltonian = [PauliSum(cost_operators)]
    return cost_hamiltonian


def create_mixer_operators(n):
    """
    Creates mixer operators for the QAOA.
    It's based on equations 54 - 58 from https://arxiv.org/pdf/1709.03489.pdf
    Indexing here comes directly from section 4.1.2 from paper 1709.03489, equations 54 - 58.
    """
    mixer_operators = []
    for t in range(n - 1):
        for city_1 in range(n):
            for city_2 in range(n):
                i = t
                u = city_1
                v = city_2
                first_part = 1
                first_part *= s_plus(n, u, i)
                first_part *= s_plus(n, v, i+1)
                first_part *= s_minus(n, u, i+1)
                first_part *= s_minus(n, v, i)

                second_part = 1
                second_part *= s_minus(n, u, i)
                second_part *= s_minus(n, v, i+1)
                second_part *= s_plus(n, u, i+1)
                second_part *= s_plus(n, v, i)
                mixer_operators.append(first_part + second_part)
    return mixer_operators


def create_initial_state_program(number_of_nodes):
    """
    This creates a state, where at t=i we visit i-th city.
    """
    initial_state = Program()
    for i in range(number_of_nodes):
        initial_state.inst(X(i * number_of_nodes + i))

    return initial_state


def s_plus(number_of_nodes, city, time):
    qubit = time * number_of_nodes + city
    return PauliTerm("X", qubit) + PauliTerm("Y", qubit, 1j)


def s_minus(number_of_nodes, city, time):
    qubit = time * number_of_nodes + city
    return PauliTerm("X", qubit) - PauliTerm("Y", qubit, 1j)


def get_distance_matrix(nodes_array):
    """
    Creates distance matrix based on given coordinates.
    """
    number_of_nodes = len(nodes_array)
    matrix = np.zeros((number_of_nodes, number_of_nodes))
    for i in range(number_of_nodes):
        for j in range(i, number_of_nodes):
            matrix[i][j] = distance_between_points(nodes_array[i], nodes_array[j])
            matrix[j][i] = matrix[i][j]
    return matrix


def distance_between_points(point_A, point_B):
    return np.sqrt((point_A[0] - point_B[0])**2 + (point_A[1] - point_B[1])**2)


def binary_state_to_points_order(binary_state):
    """
    Transforms the the order of points from the binary representation: [1,0,0,0,1,0,0,0,1],
    to the binary one: [0, 1, 2]
    """
    points_order = []
    number_of_points = int(np.sqrt(len(binary_state)))
    for p in range(number_of_points):
        for j in range(number_of_points):
            if binary_state[(number_of_points) * p + j] == 1:
                points_order.append(j)
    return points_order


def print_fun(x):
    print(x)


if __name__ == "__main__":
    # Solving for three cities with the following coordinates: 
    # 0: [0, 0], 1: [5, 0], 2: [0, 10]
    nodes_array = np.array([[0, 0], [5, 0], [0, 10]])
    solution = solve_tsp(nodes_array, steps=2, xtol=10e-2, ftol=10e-2)
    print("The most frequent solution")
    print(solution)

