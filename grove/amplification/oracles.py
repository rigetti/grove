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
"""Module containing implementations of oracle Programs.
"""

import numpy as np
import pyquil.quil as pq
from pyquil.gates import X, Z, CNOT

from grove.utils.utility_programs import ControlledProgramBuilder


def basis_selector_oracle(qubits, bitstring):
    """Defines an oracle that selects the ith element of the computational basis.

    Flips the sign of the state :math:`\\vert x\\rangle>`
    if and only if x==bitstring and does nothing otherwise.

    :param qubits: The qubits the oracle is called on. The qubits are assumed to be ordered from
     most significant qubit to least significant qubit.
    :param bitstring: The desired bitstring, given as a string of ones and zeros. e.g. "101"
    :return: A program representing this oracle.
    :rtype: Program
    """
    if len(qubits) != len(bitstring):
        raise ValueError("The bitstring should be the same length as the number of qubits.")
    oracle_prog = pq.Program()

    # In the case of one qubit, we just want to flip the phase of state relative to the other.
    if len(bitstring) == 1:
        oracle_prog.inst(Z(qubits[0]))
        return oracle_prog
    else:
        bitflip_prog = pq.Program()
        for i, qubit in enumerate(qubits):
            if bitstring[i] == '0':
                bitflip_prog.inst(X(qubit))
        oracle_prog += bitflip_prog
        controls = qubits[:-1]
        target = qubits[-1]
        operation = np.array([[1, 0], [0, -1]])
        gate_name = 'Z'
        n_qubit_controlled_z = (ControlledProgramBuilder()
                                .with_controls(controls)
                                .with_target(target)
                                .with_operation(operation)
                                .with_gate_name(gate_name)
                                .build())
        oracle_prog += n_qubit_controlled_z
        oracle_prog += bitflip_prog
    return oracle_prog


def oracle_from_unitary(unitary_funct, qubits, ancilla):
    """
    Defines an oracle that performs the following unitary transformation:
    :math:`|x\rangle |y\rangle \rightarrow |x\rangle|f(x) \otimes y\rangle`

    Allocates one scratch bit.

    :param np.array unitary_funct: Matrix representation of the function f, i.e. the
     unitary transformation that must be applied to a state |x> to put f(x) in qubit 0, with
     little endian notation, where f(x) returns either 0 or 1 for any n-bit string x.
    :param list qubits: List of qubits that enter as input |x>.
    :type qubits: list or Qubit
    :param Qubit ancilla: Qubit to serve as input |y>.
    :return: A program that performs the above unitary transformation.
    :rtype: Program
    """
    unitary_prog = pq.Program()
    p = pq.Program()
    scratch_bit = p.alloc()
    bits_for_funct = [scratch_bit] + qubits
    unitary_prog.defgate("ORACLE", unitary_funct)
    p += unitary_prog
    p.inst(tuple(['FUNCT'] + bits_for_funct))
    p.inst(CNOT(qubits[0], ancilla))
    p += unitary_prog.dagger()
    return p
