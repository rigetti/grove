
"""
Finding the minimum energy for an Ising problem by QAOA.
"""
import pyquil.api as api
from grove.pyqaoa.qaoa import QAOA
from pyquil.paulis import PauliSum, PauliTerm
from scipy.optimize import minimize
import numpy as np

CXN = api.QVMConnection()


def energy_value(h, J, sol):
    """
    Obtain energy of an Ising solution for a given Ising problem (h,J).

    :param h: External magnectic term of the Ising problem. List.
    :param J: Interaction term of the Ising problem. Dictionary.
    :param sol: Ising solution. List.
    :return: Energy of the Ising string.
    :rtype: Integer or float.

    """
    ener_ising = 0
    for elm in J.keys():
        if elm[0] == elm[1]:
            raise TypeError("""Interaction term must connect two different variables""")
        else:
            ener_ising += J[elm] * int(sol[elm[0]]) * int(sol[elm[1]])
    for i in range(len(h)):
        ener_ising += h[i] * int(sol[i])
    return ener_ising


def print_fun(x):
    print(x)


def ising_trans(x):
    # Transformation to Ising notation
    if x == 1:
        return -1
    else:
        return 1


def ising(h, J, num_steps=0, verbose=True, rand_seed=None, connection=None, samples=None,
          initial_beta=None, initial_gamma=None, minimizer_kwargs=None,
          vqe_option=None):
    """
    Ising set up method

    :param h: External magnectic term of the Ising problem. List.
    :param J: Interaction term of the Ising problem. Dictionary.
    :param num_steps: (Optional.Default=2 * len(h)) Trotterization order for the
                  QAOA algorithm.
    :param verbose: (Optional.Default=True) Verbosity of the code.
    :param rand_seed: (Optional. Default=None) random seed when beta and
                      gamma angles are not provided.
    :param connection: (Optional) connection to the QVM. Default is None.
    :param samples: (Optional. Default=None) VQE option. Number of samples
                    (circuit preparation and measurement) to use in operator
                    averaging.
    :param initial_beta: (Optional. Default=None) Initial guess for beta
                         parameters.
    :param initial_gamma: (Optional. Default=None) Initial guess for gamma
                          parameters.
    :param minimizer_kwargs: (Optional. Default=None). Minimizer optional
                             arguments.  If None set to
                             {'method': 'Nelder-Mead',
                             'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,
                                        'disp': False}
    :param vqe_option: (Optional. Default=None). VQE optional
                             arguments.  If None set to
                       vqe_option = {'disp': print_fun, 'return_all': True,
                       'samples': samples}
    :return: Most frequent Ising string, Energy of the Ising string, Circuit used to obtain result.
    :rtype: List, Integer or float, 'pyquil.quil.Program'.

    """
    if num_steps == 0:
        num_steps = 2 * len(h)

    n_nodes = len(h)

    cost_operators = []
    driver_operators = []
    for i, j in J.keys():
        cost_operators.append(PauliSum([PauliTerm("Z", i, J[(i, j)]) * PauliTerm("Z", j)]))

    for i in range(n_nodes):
        cost_operators.append(PauliSum([PauliTerm("Z", i, h[i])]))

    for i in range(n_nodes):
        driver_operators.append(PauliSum([PauliTerm("X", i, -1.0)]))

    if connection is None:
        connection = CXN

    if minimizer_kwargs is None:
        minimizer_kwargs = {'method': 'Nelder-Mead',
                            'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,
                                        'disp': False}}
    if vqe_option is None:
        vqe_option = {'disp': print_fun, 'return_all': True,
                      'samples': samples}

    if not verbose:
        vqe_option['disp'] = None

    qaoa_inst = QAOA(connection, n_nodes, steps=num_steps, cost_ham=cost_operators,
                     ref_hamiltonian=driver_operators, store_basis=True,
                     rand_seed=rand_seed,
                     init_betas=initial_beta,
                     init_gammas=initial_gamma,
                     minimizer=minimize,
                     minimizer_kwargs=minimizer_kwargs,
                     vqe_options=vqe_option)

    betas, gammas = qaoa_inst.get_angles()
    most_freq_string, sampling_results = qaoa_inst.get_string(
        betas, gammas)
    most_freq_string_ising = [ising_trans(it) for it in most_freq_string]
    energy_ising = energy_value(h, J, most_freq_string_ising)
    param_prog = qaoa_inst.get_parameterized_program()
    circuit = param_prog(np.hstack((betas, gammas)))

    return most_freq_string_ising, energy_ising, circuit
