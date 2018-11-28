##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

"""
Module for amplitude amplification, for use in algorithms such as Grover's algorithm.

 See G. Brassard, P. Hoyer, M. Mosca (2000) `Quantum Amplitude Amplification and Estimation
 <https://arxiv.org/abs/quant-ph/0005055>`_ for more information.
"""
from typing import List

import numpy as np
from pyquil import Program
from pyquil.gates import H, X, Z, RZ, STANDARD_GATES

from grove.utils.utility_programs import ControlledProgramBuilder

STANDARD_GATE_NAMES = list(STANDARD_GATES.keys())
X_GATE = np.array([[0, 1], [1, 0]])
X_GATE_LABEL = "NOT"
HADAMARD_DIFFUSION_LABEL = "HADAMARD_DIFFUSION"


def diffusion_program(qubits: List[int]) -> Program:
    diffusion_program = Program()
    dim = 2 ** len(qubits)
    hadamard_diffusion_matrix = np.diag([1.0] + [-1.0] * (dim - 1))
    diffusion_program.defgate(HADAMARD_DIFFUSION_LABEL, hadamard_diffusion_matrix)
    instruction_tuple = (HADAMARD_DIFFUSION_LABEL,) + tuple(qubits)
    diffusion_program.inst(instruction_tuple)
    return diffusion_program


def amplification_circuit(algorithm: Program, oracle: Program,
                          qubits: List[int],
                          num_iter: int,
                          decompose_diffusion: bool = False) -> Program:
    """
    Returns a program that does ``num_iter`` rounds of amplification, given a measurement-less
    algorithm, an oracle, and a list of qubits to operate on.

    :param algorithm: A program representing a measurement-less algorithm run on qubits.
    :param oracle: An oracle maps any basis vector ``|psi>`` to either ``+|psi>`` or
        ``-|psi>`` depending on whether ``|psi>`` is in the desirable subspace or the undesirable
        subspace.
    :param qubits: the qubits to operate on
    :param num_iter: number of iterations of amplifications to run
    :param decompose_diffusion: If True, decompose the Grover diffusion gate into two qubit
     gates. If False, use a defgate to define the gate.
    :return: The amplified algorithm.
    """
    program = Program()

    uniform_superimposer = Program().inst([H(qubit) for qubit in qubits])
    program += uniform_superimposer
    if decompose_diffusion:
        diffusion = decomposed_diffusion_program(qubits)
    else:
        diffusion = diffusion_program(qubits)
    # To avoid redefining gates, we collect them before building our program.
    defined_gates = oracle.defined_gates + algorithm.defined_gates + diffusion.defined_gates
    for _ in range(num_iter):
        program += (oracle.instructions
                 + algorithm.dagger().instructions
                 + diffusion.instructions
                 + algorithm.instructions)
    # We redefine the gates in the new program.
    for gate in defined_gates:
        program.defgate(gate.name, gate.matrix)
    return program


def decomposed_diffusion_program(qubits: List[int]) -> Program:
    """
    Constructs the diffusion operator used in Grover's Algorithm, acted on both sides by an
    a Hadamard gate on each qubit. Note that this means that the matrix representation of this
    operator is diag(1, -1, ..., -1). In particular, this decomposes the diffusion operator, which
    is a :math:`2**{len(qubits)}\times2**{len(qubits)}` sparse matrix, into
     :math:`\mathcal{O}(len(qubits)**2) single and two qubit gates.

    See C. Lavor, L.R.U. Manssur, and R. Portugal (2003) `Grover's Algorithm: Quantum Database
    Search`_ for more information.

    .. _`Grover's Algorithm: Quantum Database Search`: https://arxiv.org/abs/quant-ph/0301079

    :param qubits: A list of ints corresponding to the qubits to operate on.
                   The operator operates on bistrings of the form
                   ``|qubits[0], ..., qubits[-1]>``.
    """
    program = Program()
    if len(qubits) == 1:
        program.inst(Z(qubits[0]))
    else:
        program.inst([X(q) for q in qubits])
        program.inst(H(qubits[-1]))
        program.inst(RZ(-np.pi, qubits[0]))
        program += (ControlledProgramBuilder()
                              .with_controls(qubits[:-1])
                              .with_target(qubits[-1])
                              .with_operation(X_GATE)
                              .with_gate_name(X_GATE_LABEL).build())
        program.inst(RZ(-np.pi, qubits[0]))
        program.inst(H(qubits[-1]))
        program.inst([X(q) for q in qubits])
    return program
