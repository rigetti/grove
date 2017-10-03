###############################################################################
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
###############################################################################
"""
The Bravyi-Kitaev transform, generalized using a Fenwick tree for arbitrary
numbers of qubits.

References:
(Regular Bravyi-Kitaev):
https://arxiv.org/abs/1208.5986

(Fenwick generalization for arbitrary n_qubits):
https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.032332
"""
from pyquil.paulis import PauliTerm
from .fenwick_tree import FenwickTree


class BKTransform(object):
    """
    Transform object to create arbitrary creation/annihilation fermionic 
    operators, over a pre-specified number of qubits.

    Uses a Fenwick tree / binary index tree to generalize the Bravyi-Kitaev
    transform for arbitrary number of qubits/fermionic lattice sites.

    :param int n_qubits: number of fermionic lattice sites / qubits to
                         initialize
    """
    def __init__(self, n_qubits):
        assert n_qubits >= 0

        # Build the Fenwick tree
        self.tree = FenwickTree(n_qubits)
        self.n_qubits = n_qubits

    def create(self, index):
        """
        Fermion creation operator at orbital 'index'

        :param int index: orbital index to create fermion at

        :return: qubit operators corresponding to fermion creation
        :rtype: PauliSum
        """
        return self._operator_generator(index, -1.0)

    def kill(self, index):
        """
        Fermion annihilation operator at orbital 'index'

        :param int index: orbital index to annihilate fermion at

        :return: qubit operators corresponding to fermion annihilation
        :rtype: PauliSum
        """
        return self._operator_generator(index, +1.0)

    def product_ops(self, indices, conjugate):
        """
        Convert a list of site indices and coefficients to a Pauli Operators
        list with the generalized Bravyi-Kitaev transformation.

        :param list indices: list of ints specifying the site the fermionic
                             operator acts on, e.g. [0,2,4,6]
        :param list conjugate: List of -1, 1 specifying which of the indices
                               are creation operators (-1) and which are
                               annihilation operators (1).  e.g. [-1,-1,1,1]
        """
        pterm = PauliTerm('I', 0, 1.0)
        for conj, index in zip(conjugate, indices):
            pterm = pterm * self._operator_generator(index, conj)

        pterm = pterm.simplify()
        return pterm

    def _operator_generator(self, index, conj):
        """
        Internal method to generate the appropriate ladder operator at fermion
        orbital at 'index'
        If conj == -1 --> creation
           conj == +1 --> annihilation

        :param int index: fermion orbital to generate ladder operator at
        :param int conj: -1 for creation, +1 for annihilation
        """
        if conj != -1 and conj != +1:
            raise ValueError("Improper conjugate coefficient")
        if index >= self.n_qubits or index < 0:
            raise IndexError("Operator index outside number of qubits for "
                             "current Bravyi-Kitaev transform.")

        # parity set P(j). apply Z to, for parity sign.
        parity_set = [node.index for node in self.tree.get_parity_set(index)]

        # update set U(j). apply X to, for updating purposes.
        ancestors = [node.index for node in self.tree.get_update_set(index)]

        # remainder set C(j) = P(j) \ F(j)
        ancestor_children = [node.index for node in self.tree.get_remainder_set(index)]

        # Under Majorana basis, creation/annihilation operators given by
        # a^{\pm} = (c \mp id) / 2

        # c_j = a_j + a_j^{\dagger} = X_{U(j)} X_j Z_{P(j)}
        c_maj = PauliTerm('X', index)
        for node_idx in parity_set:
            c_maj *= PauliTerm('Z', node_idx)
        for node_idx in ancestors:
            c_maj *= PauliTerm('X', node_idx)

        # d_j = i(a_j^{\dagger} - a_j) = X_{U(j)} Y_j Z_{C(j)}
        d_maj = PauliTerm('Y', index)
        for node_idx in ancestors:
            d_maj *= PauliTerm('X', node_idx)
        for node_idx in ancestor_children:
            d_maj *= PauliTerm('Z', node_idx)

        result = 0.5 * (c_maj + 1j * conj * d_maj)
        return result.simplify()
