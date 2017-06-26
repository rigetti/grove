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
import numpy as np
from pyquil.gates import *
from grove.amplification.amplification import amplify, n_qubit_control, diffusion_operator
import pyquil.quil as pq
from grove.pyqaoa.utils import compare_progs # This would be nice to have in
                                             # general purpose Util module

# Normal operation

# Setup some variables to reuse
A = pq.Program().inst(H(0)).inst(H(1)).inst(H(2))
A_inv = pq.Program().inst(H(0)).inst(H(1)).inst(H(2))
cz_gate = n_qubit_control([1], 2, np.array([[1,0],[0,-1]]), "CZ")
oracle = pq.Program().inst()
qubits = [0, 1, 2]
iters = 3

def test_qubit_control():
    '''
    Tests the n_qubit_countrol on a generic number of qubits
    '''
    controlled = n_qubit_control([0,1], 2, np.array([[0, 1], [1, 0]]), "X")
    print controlled
    
    
def test_amplify():
    '''
    Test the generic usage of amplify
    '''
    pass
    
    
    # Essentially Grover's to select 011 or 111
       
def test_amplify_init():
    '''
    Test the usage of amplify without init
    '''
    pass

def test_diffusion_operator():
    '''
    Checks that the diffusion operator outputs the correct operation
    '''
    pass

# Edge Cases    
    
def test_edge_case_amplify_0_iters():
    '''
    Checks that the number of iterations needed to be greater than 0
    '''
    with pytest.raises(AssertionError):
        amplify(A, A_inv, oracle, qubits, 0)

def test_edge_case_A_none():
    '''
    Checks that A cannot be None
    '''
    with pytest.raises(AssertionError):
        amplify(None, A_inv, oracle, qubits, iters)
    
def test_edge_case_A_inv_none():
    '''
    Checks that A_inv cannot be None
    '''
    with pytest.raises(AssertionError):
        amplify(A, None, oracle, qubits, iters)
    
def test_edge_case_oracle_none():
    '''
    Checks that U_w cannot be None
    '''
    with pytest.raises(AssertionError):
        amplify(A, A_inv, None, qubits, iters)
        
def test_edge_case_qubits_empty():
    '''
    Checks that the list of qubits to apply the grover
    diffusion operator to must be non-empty
    '''
    with pytest.raises(AssertionError):
        amplify(A, A_inv, oracle, [], iters)

test_amplify()