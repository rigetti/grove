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
from pyquil.gates import *
import pyquil.quil as pq
from grove.superdense.superdense import SDServer, SDClient, MockSDConnection, clear_registers
from grove.pyqaoa.utils import compare_progs # This would be nice to have in
                                             # general purpose Util module
    
# Tests for SDServer ##########################################################

def test_sdserver_normal():
    '''
    Tests the normal creation of the superdense coding server
    '''
    
    bell_pairs = 5
    server = SDServer(bell_pairs, "sdc://rigetti.com/channel")
    
    assert server.qubit_count == bell_pairs * 2
    assert server.namespace == "sdc://rigetti.com/channel"
    assert server.classical_reg == 0
    assert server.registers == range(0, bell_pairs * 2)

def test_sdserver_bad_bell_count():
    '''
    Tests that an exception is raised when a negative bell count is
    requested
    '''
    
    with pytest.raises(AssertionError):
        SDServer(0, "sdc://rigetti.com/channel")

def test_sdserver_bad_classical_reg():
    '''
    Tests than an exception is raised when a negative classical register
    is requested
    '''
    
    with pytest.raises(AssertionError):
        SDServer(1, "sdc://rigetti.com/channel", classical_reg=-1)

def test_sdserver_enough_qubits():
    '''
    Throws an exception when a number of qubit registers are chosen that does
    not satisfy the number of bell pairs requested
    '''
    
    with pytest.raises(AssertionError):
        SDServer(2, "sdc://rigetti.com/channel", qubits=[0, 1, 2])


# Tests for SDClient ##########################################################

def test_sdclient_normal():
    '''
    Tests the normal creation of the superdense coding client
    '''
    
    bell_pairs = 5
    client = SDClient(bell_pairs, "sdc://rigetti.com/channel", range(0, 10))
    
    assert client.qubit_count == bell_pairs * 2
    assert client.namespace == "sdc://rigetti.com/channel"
    assert client.classical_reg == 0
    assert client.registers == range(0, bell_pairs * 2)
    assert len(client.message_regs) == 10

def test_sdclient_bad_bell_count():
    '''
    Tests that an exception is raised when a negative bell count is
    requested
    '''
    
    with pytest.raises(AssertionError):
        SDClient(0, "sdc://rigetti.com/channel", range(0, 10))

def test_sdclient_bad_classical_reg():
    '''
    Tests than an exception is raised when a negative classical register
    is requested
    '''
    
    with pytest.raises(AssertionError):
        SDClient(1, "sdc://rigetti.com/channel", [0,1], classical_reg=-1)

def test_sdclient_enough_qubits():
    '''
    Throws an exception when a number of qubit registers are chosen that does
    not satisfy the number of bell pairs requested
    '''
    
    with pytest.raises(AssertionError):
        SDClient(2, "sdc://rigetti.com/channel", range(4), qubits=[0, 1, 2])
        
def test_sdclient_enough_message_regs():
    '''
    Throws an exception when there are not enough message registers
    '''
    
    with pytest.raises(AssertionError):
        SDClient(2, "sdc://rigetti.com/channel", [1])

# Utilities ###################################################################

def test_clear_qubits():
    '''
    Tests the operation of clearing qubits to the ground state
    '''
    created = pq.Program()
    clear_registers(0, [0], created)
    created_string = created.out()
    desired = "FALSE [0]\nMEASURE 0 [0]\nJUMP-WHEN @THEN1 [0]\nJUMP @END2\nLABEL @THEN1\nX 0\nLABEL @END2\n"
    assert desired == created_string