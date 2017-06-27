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

import pytest
from grove.order_finding.order_finding import calculate_order, calculate_order_classical, multiplication_operator
import numpy as np


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_order_finding():
    a_values = [2, 3, 7]
    N_values = [3, 5, 15]
    for a, N in zip(a_values, N_values):
        assert calculate_order(a, N) == calculate_order_classical(a, N)

@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_order_finding_edge_case():
    a_values = [5, 11]
    N_values = [9, 21]
    for a, N in zip(a_values, N_values):
        assert calculate_order(a, N) == calculate_order_classical(a, N)

def test_multiplication_operator():
    a = 3
    N = 7
    U = multiplication_operator(a, N)
    one = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    two = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    three = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    five = np.array([0, 0, 0, 0, 0, 1, 0, 0])

    assert np.allclose(U.dot(one), three) # 3 * 1 (mod 7) = 3
    assert np.allclose(U.dot(three), two) # 3 * 3 (mod 7) = 2
    assert np.allclose(U.dot(five), one) # 5 * 3 (mod 7) = 1

