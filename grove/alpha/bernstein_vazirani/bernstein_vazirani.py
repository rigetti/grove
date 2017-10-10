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
Module for the Bernstein-Vazirani Algorithm.
For more information, see [Loceff2015]_

.. [Loceff2015] Loceff, M. (2015), `"A Course in Quantum Computing for the Community College"`_, Volume 1, Chapter 18, p 484-541.

.. _`"A Course in Quantum Computing for the Community College"`: http://lapastillaroja.net/wp-content/uploads/2016/09/Intro_to_QC_Vol_1_Loceff.pdf
"""

import pyquil.quil as pq
from pyquil.gates import H, X, CNOT


class BernsteinVazirani(object):
    """
    This class contains an implementation of the Bernstein-Vazirani algorithm using pyQuil. For more
    references see the documentation_

    .. _documentation: http://grove-docs.readthedocs.io/en/latest/bernstein_vazirani.html
    """

    def __init__(self):
        self.n_qubits = None
        self.n_ancillas = 1
        self.computational_qubits = None
        self.ancilla = None
        self.bv_oracle_circuit = None
        self.full_bv_circuit = None

    def _create_bv_oracle_program(self, a, b):
        """
        Creates a black box oracle for a function to be used in the Bernstein-Vazirani algorithm.

        For a function :math:`f` such that

        .. math::

           f:\\{0,1\\}^n\\rightarrow \\{0,1\\}

           \\mathbf{x}\\rightarrow \\mathbf{a}\\cdot\\mathbf{x}+b\\pmod{2}

           (\\mathbf{a}\\in\\{0,1\\}^n, b\\in\\{0,1\\})

        where :math:`(\\cdot)` is the bitwise dot product,
        this function defines a program that performs
        the following unitary transformation:

        .. math::

            \\vert \\mathbf{x}\\rangle\\vert y\\rangle \\rightarrow
            \\vert \\mathbf{x}\\rangle
            \\vert f(\\mathbf{x}) \\text{ xor } y\\rangle

        where :math:`\\text{xor}` is taken bitwise.

        Allocates one scratch bit.

        :param List[Int] a: a vector of length :math:`n`
                                    containing only ones and zeros.
                                    The order is taken to be
                                    most to least significant bit.
        :param Int b: a 0 or 1 as additive bias
        :return: A program that performs the above unitary transformation.
        :rtype: Program
        """
        bv_oracle_circuit = pq.Program()
        if b == 1:
            bv_oracle_circuit.inst(X(self.ancilla))
        for i in range(self.n_qubits):
            if a[i] == 1:
                bv_oracle_circuit.inst(CNOT(self.computational_qubits[self.n_qubits - 1 - i],
                                            self.ancilla))
        return bv_oracle_circuit

    def with_oracle_for_vector(self, a, b=0):
        """
        builder method that constructs an oracle for the given mask vector and bias term

        :param List[Int] a: list of integers (0 and 1) for the bitwise dot-product
        :param Int b: additive bias (0 or 1) for the function. Default: 0
        :return: self
        :rtype: BernsteinVazirani
        """
        self.n_qubits = len(a)
        self.computational_qubits = list(range(self.n_qubits))
        self.ancilla = self.n_qubits  # is the highest index now.
        self.bv_oracle_circuit = self._create_bv_oracle_program(a, b)
        return self

    def _hadarmard_walsh_wrapper(self, oracle_circuit):
        """
        Implementation of the Bernstein-Vazirani Algorithm.

        Given a list of input qubits and an ancilla bit, all initially in the
        :math:`\\vert 0\\rangle` state, create a program that can find :math:`\\vec{a}` with one
        query to the given oracle.

        :param Program oracle_circuit: Program representing unitary application of function
        :rtype: Program
        """
        full_bv_circuit = pq.Program()

        # Put ancilla bit into minus state
        full_bv_circuit.inst(X(self.ancilla), H(self.ancilla))

        full_bv_circuit.inst([H(i) for i in self.computational_qubits])
        full_bv_circuit += oracle_circuit
        full_bv_circuit.inst([H(i) for i in self.computational_qubits])
        return full_bv_circuit

    def run(self, cxn):
        """
        Runs the Bernstein-Vazirani algorithm.

        Given a QVM connection, find the :math:`\\mathbf{a}` and :math:`b` corresponding to the
        function represented by the oracle.

        :param Connection cxn: the QVM connection to use to run the programs
        :rtype: tuple
        """

        self.full_bv_circuit = self._hadarmard_walsh_wrapper(self.bv_oracle_circuit)

        # find vector by running the full bv circuit
        results = cxn.run_and_measure(self.full_bv_circuit, self.computational_qubits)
        bv_vector = results[0][::-1]

        # To get the bias term we skip the Walsh-Hadamard transform
        results = cxn.run_and_measure(self.bv_oracle_circuit, [self.ancilla])
        bv_bias = results[0][0]
        return bv_vector, bv_bias
