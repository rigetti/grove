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
"""A module containing useful utility quantum Programs.
"""

from scipy.linalg import sqrtm
import numpy as np

from pyquil.gates import STANDARD_GATES
import pyquil.quil as pq


STANDARD_GATE_NAMES = list(STANDARD_GATES.keys())
SQRT_PREFIX = "SQRT"
CONTROL_PREFIX = "C"
NOT_GATE_LABEL = "NOT"
NOT_GATE = np.array([[0, 1], [1, 0]])
ZERO_PROJECTION = np.array([[1, 0], [0, 0]])
ONE_PROJECTION = np.array([[0, 0], [0, 1]])


class ControlledProgramBuilder(object):
    """
    Returns a controlled u gate with n-1 control_qubits. Useful for constructing oracles.

    Uses a number of gates quadratic in the number of qubits,
    and defines a linear number of new gates. (Roots and adjoints of unitary.)
     See A. Barenco, C. Bennett, R. Cleve (1995) `Elementary Gates for Quantum Computation
     <https://arxiv.org/abs/quant-ph/9503016>`_ for more information.


    :param str gate_name: The name of the gate target.
    :return: The controlled gate.
    """

    def __init__(self):
        self.defined_gates = set()
        self.control_qubits = None
        self.target_qubit = None
        self.operation = None
        self.gate_name = None

    def with_controls(self, control_qubits):
        """Sets the qubits to control on for this controlled operation.

        :param list control_qubits: The indices of the qubits to condition the gate on.
        :return: self, with control_qubits set.
        :rtype: ControlledProgramBuilder
        """
        self.control_qubits = control_qubits
        return self

    def with_target(self, target_qubit):
        """Sets the target qubit for this controlled operation.

        :param int or Qubit target: The Qubit or index of the target of the gate.
        :return: self, with target_qubit set.
        :rtype: ControlledProgramBuilder
        """
        self.target_qubit = target_qubit
        return self

    def with_operation(self, operation):
        """Sets the operation that defines the controlled gate.

        :param numpy.ndarray operation: The unitary gate to be controlled, given as a numpy array.
        :return: self, with operation set.
        :rtype: ControlledProgramBuilder
        """
        self.operation = operation
        return self

    def with_gate_name(self, gate_name):
        """Sets the name for the controlled gate, used in constructing and defining sqrts of the
         gate.

        :param String gate_name:
        :return: self, with gate_name set.
        :rtype: ControlledProgramBuilder
        """
        self.gate_name = gate_name
        return self

    @staticmethod
    def format_gate_name(prefix, gate_name):
        """Formats gate_name by surrounding with brackets and prepending prefix. Used in this case
         to recursively build a program while defining new gates.

        :param str prefix: The prefix to prepend.
        :param str gate_name: The gate name to format and extend.
        :return: The formatted gate name.
        """
        formatted_gate_name = prefix + '-' + gate_name
        return formatted_gate_name

    def build(self):
        """Builds this controlled gate.

        :return: The controlled gate, defined by this object.
        :rtype: Program
        """
        self.defined_gates = set(STANDARD_GATE_NAMES)
        prog = self._recursive_builder(self.operation,
                                       self.gate_name,
                                       self.control_qubits,
                                       self.target_qubit)
        return prog

    def _defgate(self, program, gate_name, gate_matrix):
        """Defines a gate named gate_name with matrix gate_matrix in program. In addition, updates
         self.defined_gates to track what has been defined.

        :param Program program: Pyquil Program to add the defgate and gate to.
        :param str gate_name: The name of the gate to add to program.
        :param numpy.ndarray gate_matrix: The array corresponding to the gate to define.
        :return: the modified Program.
        :retype: Program
        """
        new_program = pq.Program()
        new_program += program
        if gate_name not in self.defined_gates:
            new_program.defgate(gate_name, gate_matrix)
            self.defined_gates.add(gate_name)
        return new_program

    def _recursive_builder(self, operation, gate_name, control_qubits, target_qubit):
        """Helper function used to define the controlled gate recursively. It uses the algorithm in
         the reference above. Namely it recursively constructs a controlled gate by applying a
         controlled square root of the gate, followed by a toffoli gate with len(control_qubits) - 1
         controls, applying the controlled adjoint of the square root, another toffoli with
         len(control_qubits) - 1 controls, and finally another controlled copy of the gate.

        :param numpy.ndarray operation: The matrix for the unitary to be controlled.
        :param String gate_name: The name for the gate being controlled.
        :param Sequence control_qubits: The qubits that are the controls.
        :param Qubit or Int target_qubit: The qubit that the gate should be applied to.
        :return: The intermediate Program being built.
        :rtype: Program
        """
        control_true = np.kron(ONE_PROJECTION, operation)
        control_false = np.kron(ZERO_PROJECTION, np.eye(2, 2))
        control_root_true = np.kron(ONE_PROJECTION, sqrtm(operation, disp=True))
        controlled_gate = control_true + control_false
        controlled_root_gate = control_root_true + control_false
        sqrt_name = self.format_gate_name(SQRT_PREFIX, gate_name)
        controlled_subprogram = pq.Program()
        control_gate = pq.Program()
        # For the base case, we check to see if there is just one control qubit.
        # If it is a CNOT we explicitly break the naming convention so as not to redefine the gate.
        if len(control_qubits) == 1:
            if gate_name == NOT_GATE_LABEL:
                control_name = CONTROL_PREFIX + gate_name
            else:
                control_name = self.format_gate_name(CONTROL_PREFIX, gate_name)
            control_gate = self._defgate(control_gate, control_name, controlled_gate)
            control_gate.inst((control_name, control_qubits[0], target_qubit))
            return control_gate

        else:
            control_sqrt_name = self.format_gate_name(CONTROL_PREFIX, sqrt_name)
            control_gate = self._defgate(control_gate, control_sqrt_name, controlled_root_gate)
            control_gate.inst((control_sqrt_name, control_qubits[-1], target_qubit))
            # Here we recurse to build a toffoli gate on n - 1 of the qubits.
            n_minus_one_toffoli = self._recursive_builder(NOT_GATE,
                                                          NOT_GATE_LABEL,
                                                          control_qubits[:-1],
                                                          control_qubits[-1])

            # We recurse to build a controlled sqrt of the target_gate, excluding the last control.
            n_minus_one_controlled_sqrt = self._recursive_builder(sqrtm(operation, disp=True),
                                                                  sqrt_name,
                                                                  control_qubits[:-1],
                                                                  target_qubit)
            controlled_subprogram += control_gate
            controlled_subprogram += n_minus_one_toffoli
            controlled_subprogram += control_gate.dagger()
            # We only add the instructions so that we don't redefine gates
            controlled_subprogram += n_minus_one_toffoli.instructions
            controlled_subprogram += n_minus_one_controlled_sqrt
            return controlled_subprogram
