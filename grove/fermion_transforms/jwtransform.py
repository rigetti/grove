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
The Jordan-Wigner Transform
"""
from pyquil.paulis import PauliTerm


class JWTransform(object):
    """
    Jordan-Wigner object the appropriate Pauli operators
    """
    def create(self, index):
        """
        Fermion creation operator at orbital 'n'

        :param Int index: creation index
        """
        return self._operator_generator(index, -1.0)

    def kill(self, index):
        """
        Fermion annihilation operator at orbital 'n'

        :param Int index: annihilation index
        """
        return self._operator_generator(index, 1.0)

    def product_ops(self, indices, conjugate):
        """
        Convert a list of site indices and coefficients to a Pauli Operators
        list with the Jordan-Wigner (JW) transformation

        :param List indices: list of ints specifying the site the fermionic operator acts on,
                             e.g. [0,2,4,6]
        :param List conjugate: List of -1, 1 specifying which of the indices are
                               creation operators (-1) and which are annihilation
                               operators (1).  e.g. [-1,-1,1,1]
        """
        pterm = PauliTerm('I', 0, 1.0)
        for conj, index in zip(conjugate, indices):
            pterm = pterm * self._operator_generator(index, conj)

        pterm = pterm.simplify()
        return pterm

    @staticmethod
    def _operator_generator(index, conj):
        """
        Internal method to generate the appropriate operator
        """
        pterm = PauliTerm('I', 0, 1.0)
        Zstring = PauliTerm('I', 0, 1.0)
        for j in range(index):
            Zstring = Zstring*PauliTerm('Z', j, 1.0)

        pterm1 = Zstring*PauliTerm('X', index, 0.5)
        scalar = 0.5 * conj * 1.0j
        pterm2 = Zstring*PauliTerm('Y', index, scalar)
        pterm = pterm * (pterm1 + pterm2)

        pterm = pterm.simplify()
        return pterm
