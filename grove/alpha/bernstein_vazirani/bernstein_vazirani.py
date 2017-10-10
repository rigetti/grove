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
Additional information for this algorithm can be found at:
http://pages.cs.wisc.edu/~dieter/Courses/2010f-CS880/Scribes/04/lecture04.pdf
"""

import pyquil.quil as pq
from pyquil.gates import *


class BernsteinVazirani(object):

    def __init__(self):
        self.n_qubits = None
        self.n_ancillas = 1
        self.computational_qubits = None
        self.ancilla = None
        self.bv_oracle_circuit = None
        self.full_bv_circuit = None

    def _create_bv_oracle_program(self, vector, bias):
        """
        Creates a black box oracle for a function
        to be used in the Bernstein-Vazirani algorithm.

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

        :param list(int) vec_a: a vector of length :math:`n`
                                    containing only ones and zeros.
                                    The order is taken to be
                                    most to least significant bit.
        :param int b: a 0 or 1
        :param list(int) qubits: List of qubits that enter as input
                                 :math:`\\vert x\\rangle`.
                                 Must be the same length (:math:`n`)
                                 as :math:`\\mathbf{a}`
        :param int ancilla: Ancillary qubit to serve as input
                            :math:`\\vert y\\rangle`,
                            where the answer will be written to.
        :return: A program that performs the above unitary transformation.
        :rtype: Program
        """
        bv_oracle_circuit = pq.Program()
        if bias == 1:
            bv_oracle_circuit.inst(X(self.ancilla))
        for i in range(self.n_qubits):
            if vector[i] == 1:
                bv_oracle_circuit.inst(CNOT(self.computational_qubits[self.n_qubits - 1 - i],
                                            self.ancilla))
        return bv_oracle_circuit

    def with_oracle_for_vector(self, vector, bias):

        self.n_qubits = len(vector)
        self.computational_qubits = list(range(self.n_qubits))
        self.ancilla = self.n_qubits  # is the highest index now.
        self.bv_oracle_circuit = self._create_bv_oracle_program(vector, bias)
        return self

    def with_oracle_circuit(self, oracle_circuit, num_qubits):
        self.n_qubits = num_qubits
        self.computational_qubits = list(range(self.n_qubits))
        self.ancilla = self.n_qubits  # is the highest index now.
        self.bv_oracle_circuit = oracle_circuit
        return self

    def _hadarmard_walsh_wrapper(self, oracle_circuit):
        """
        Implementation of the Bernstein-Vazirani Algorithm.

        Given a list of input qubits and an ancilla bit,
        all initially in the :math:`\\vert 0\\rangle` state,
        create a program that can find :math:`\\vec{a}`
        with one query to the given oracle.

        :param Program oracle: Program representing unitary application of function
        :param list(int) qubits: List of qubits that enter as state
                                 :math:`\\vert x\\rangle`.
        :param int ancilla: Ancillary qubit to serve as input
                            :math:`\\vert y\\rangle`,
                            where the answer will be written to.
        :return: A program corresponding to the desired instance of the
                 Bernstein-Vazirani Algorithm.
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

        Given a QVM connection, an oracle, the input bits, and ancilla,
        find the :math:`\\mathbf{a}` and :math:`b` corresponding to the function
        represented by the oracle.

        :param Connection cxn: the QVM connection to use to run the programs
        :param Program oracle: the oracle to query that represents
                               a function of the form
                               :math:`f(x)=\\mathbf{a}\\cdot\\mathbf{x}+b\\pmod{2}`
        :param list qubits: the input qubits
        :param Qubit ancilla: the ancilla qubit
        :return: a tuple that includes, in order,

                    * the program's determination of :math:`\\mathbf{a}`
                    * the program's determination of :math:`b`
                    * the main program used to determine :math:`\\mathbf{a}`
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

#
#
# def oracle_function(vec_a, b, qubits, ancilla):
#     """
#     Creates a black box oracle for a function
#     to be used in the Bernstein-Vazirani algorithm.
#
#     For a function :math:`f` such that
#
#     .. math::
#
#        f:\\{0,1\\}^n\\rightarrow \\{0,1\\}
#
#        \\mathbf{x}\\rightarrow \\mathbf{a}\\cdot\\mathbf{x}+b\\pmod{2}
#
#        (\\mathbf{a}\\in\\{0,1\\}^n, b\\in\\{0,1\\})
#
#     where :math:`(\\cdot)` is the bitwise dot product,
#     this function defines a program that performs
#     the following unitary transformation:
#
#     .. math::
#
#         \\vert \\mathbf{x}\\rangle\\vert y\\rangle \\rightarrow
#         \\vert \\mathbf{x}\\rangle
#         \\vert f(\\mathbf{x}) \\text{ xor } y\\rangle
#
#     where :math:`\\text{xor}` is taken bitwise.
#
#     Allocates one scratch bit.
#
#     :param list(int) vec_a: a vector of length :math:`n`
#                                 containing only ones and zeros.
#                                 The order is taken to be
#                                 most to least significant bit.
#     :param int b: a 0 or 1
#     :param list(int) qubits: List of qubits that enter as input
#                              :math:`\\vert x\\rangle`.
#                              Must be the same length (:math:`n`)
#                              as :math:`\\mathbf{a}`
#     :param int ancilla: Ancillary qubit to serve as input
#                         :math:`\\vert y\\rangle`,
#                         where the answer will be written to.
#     :return: A program that performs the above unitary transformation.
#     :rtype: Program
#     """
#     assert len(vec_a) == len(qubits), \
#         "vec_a must be the same length as the number of input qubits"
#     assert all(list(map(lambda x: x in {0, 1}, vec_a))), \
#         "vec_a must be a list of 0s and 1s"
#     assert b in {0, 1}, "b must be a 0 or 1"
#
#     n = len(qubits)
#     p = pq.Program()
#     if b == 1:
#         p.inst(X(ancilla))
#     for i in range(n):
#         if vec_a[i] == 1:
#             p.inst(CNOT(qubits[n - 1 - i], ancilla))
#     return p
#
#
# def bernstein_vazirani(oracle, qubits, ancilla):
#     """
#     Implementation of the Bernstein-Vazirani Algorithm.
#
#     Given a list of input qubits and an ancilla bit,
#     all initially in the :math:`\\vert 0\\rangle` state,
#     create a program that can find :math:`\\vec{a}`
#     with one query to the given oracle.
#
#     :param Program oracle: Program representing unitary application of function
#     :param list(int) qubits: List of qubits that enter as state
#                              :math:`\\vert x\\rangle`.
#     :param int ancilla: Ancillary qubit to serve as input
#                         :math:`\\vert y\\rangle`,
#                         where the answer will be written to.
#     :return: A program corresponding to the desired instance of the
#              Bernstein-Vazirani Algorithm.
#     :rtype: Program
#     """
#     p = pq.Program()
#
#     # Put ancilla bit into minus state
#     p.inst(X(ancilla), H(ancilla))
#
#     # Apply Hadamard, Unitary function, and Hadamard again
#     p.inst(list(map(H, qubits)))
#     p += oracle
#     p.inst(list(map(H, qubits)))
#     return p
#
#
# def run_bernstein_vazirani(cxn, oracle, qubits, ancilla):
#     """
#     Runs the Bernstein-Vazirani algorithm.
#
#     Given a QVM connection, an oracle, the input bits, and ancilla,
#     find the :math:`\\mathbf{a}` and :math:`b` corresponding to the function
#     represented by the oracle.
#
#     :param Connection cxn: the QVM connection to use to run the programs
#     :param Program oracle: the oracle to query that represents
#                            a function of the form
#                            :math:`f(x)=\\mathbf{a}\\cdot\\mathbf{x}+b\\pmod{2}`
#     :param list qubits: the input qubits
#     :param Qubit ancilla: the ancilla qubit
#     :return: a tuple that includes, in order,
#
#                 * the program's determination of :math:`\\mathbf{a}`
#                 * the program's determination of :math:`b`
#                 * the main program used to determine :math:`\\mathbf{a}`
#     :rtype: tuple
#     """
#     # First, create the program to find a
#     bv_program = bernstein_vazirani(oracle, qubits, ancilla)
#
#     results = cxn.run_and_measure(bv_program, qubits)
#     bv_a = results[0][::-1]
#
#     # Feed through all zeros to get b
#     results = cxn.run_and_measure(oracle, [ancilla])
#     bv_b = results[0][0]
#
#     return bv_a, bv_b, bv_program
