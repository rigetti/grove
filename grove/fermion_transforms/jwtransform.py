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
Implementation of the Jordan-Wigner Transform
"""
from pyquil.paulis import PauliTerm


class JW(object):

    def __init__(self):
        """
        Jordan-Wigner object the appropriate Pauli operators
        """

    def create(self, index, boson=False):
        """
        Fermion creation operator at orbital 'n'

        :param Int n: creation index
        """
        return self._operator_generator(index, 1.0, boson=boson)

    def kill(self, index, boson=False):
        """
        Fermion annihilation operator at orbital 'n'

        :param Int n: annihilation index
        """
        return self._operator_generator(index, 1.0, boson=boson)

    def _operator_generator(self, index, conj, boson=False):
        """
        Internal method to generate the appropriate operator
        """
        pterm = PauliTerm('I', 0, 1.0)
        Zstring = PauliTerm('I', 0, 1.0)
        if not boson:
            for j in range(index):
                Zstring = Zstring*PauliTerm('Z', j, 1.0)

            pterm1 = Zstring*PauliTerm('X', index, 0.5)

            scalar = 0.5 * conj * 1.0j

            pterm2 = Zstring*PauliTerm('Y', index, scalar)

            pterm = pterm * (pterm1 + pterm2)

        pterm = pterm.simplify()
        return pterm
