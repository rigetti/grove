"""Module for Grover's algorithm.
"""

import numpy as np

import pyquil.quil as pq
from pyquil.gates import H
from pyquil.quilbase import Qubit

import grove.alpha.amplification.amplification as amp
from grove.alpha.utility_programs import compute_grover_oracle_matrix


def grover(bitstring_map):
    """Constructs an instance of Grover's Algorithm given a bitstring_map.
    """
    oracle_unitary = compute_grover_oracle_matrix(bitstring_map)
    oracle = pq.Program()
    oracle_name = "ORACLE"
    print oracle_unitary
    oracle.defgate(oracle_name, oracle_unitary)
    number_of_qubits = oracle_unitary.shape[0]
    qubits = [oracle.alloc() for _ in range(number_of_qubits / 2)]
    oracle.inst(tuple([oracle_name] + qubits))
    return oracle_grover(oracle, qubits)


def oracle_grover(oracle, qubits, num_iter=None):
    """Implementation of Grover's Algorithm for a given oracle.

    The query qubit will be left in the zero state afterwards.

    :param Program oracle: An oracle defined as a Program. It should send |x> to (-1)^f(x)|x>,
                           where the range of f is {0, 1}.
    :param qubits: List of qubits for Grover's Algorithm.
    :type qubits: list[int or Qubit]
    :param int num_iter: The number of iterations to repeat the algorithm for.
                         The default is the integer closest to :math:`\\frac{\\pi}{4}\sqrt{N}`,
                         where :math:`N` is the size of the domain.
    :return: A program corresponding to the desired instance of Grover's Algorithm.
    :rtype: Program
    """
    if len(qubits) < 1:
        raise ValueError("Grover's Algorithm requires at least one qubit.")
    if not all((isinstance(qubit, (int, Qubit)) for qubit in qubits)):
        raise ValueError("qubits should be a list of qubits or non-negative integers.")
    if num_iter is None:
        num_iter = int(round(np.pi * 2 ** (len(qubits) / 2.0 - 2.0)))

    uniform_superimposer = pq.Program().inst(list(map(H, qubits)))
    amp_prog = amp.amplify(uniform_superimposer, oracle, qubits, num_iter)
    return amp_prog
