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

from grove.qft.fourier import inverse_qft
from grove.teleport.teleportation import make_bell_pair
from pyquil.quil import Program, if_then
import pyquil.forest as forest
from pyquil.gates import RESET, FALSE, MEASURE, X, SWAP

class SDServer:
    '''
    Represents a server to be used during the superdense coding protocol
    '''
    
    def __init__(self, bell_count, namespace, classical_reg=0, qubits=None):
        '''
        Instantiates a server which will send information during the superdense
        coding protocol.
        :param bell_count: The number of bell pairs to create. The amount of 
                           information that can be sent between the server
                           and client will be 2 times this value.
        :param namespace: The address to use, i.e. the url for the server and
                          client to connect to.
        :param classical_reg: Optional. Register to use in reset qubits
        :param qubits: Optional. Allows the user to define the indices of the
                       qubits to use for the protocol on the server size.
        '''
        
        self.qubit_count = 2 * bell_count
        if qubits:
            assert self.qubit_count == len(qubits), "Number of requested qubits does not match number of given registers."
            
        self.registers = qubits if qubits else [i for i in range(self.qubit_count)]
        self.namespace = namespace
        self.classical_reg = classical_reg
        
    def connect(self, mc):
        '''
        Makes a connection to the client, using the MockConnection object
        :param mc: A mock connection while we wait for hardware
        '''
        self.mc = mc
        
        
    def dispatch_initial(self):
        '''
        Prepares the bell states and dispatches one qubit to any connected clients
        '''
        p = Program()
        
        # First, clear all registers for purity
        clear_registers(self.classical_reg, self.registers, p)
            
        # Now create bell pairs
        to_send = []
        for i in range(0, len(self.registers), 2):
            p += make_bell_pair(self.registers[i], self.registers[i+1])
            to_send.append(i+1)
            
        # Ideally, in the future, a program would not be sent here. Instead, the bell
        # state preparation will be executed here, and then mc.send will maybe entangle
        # the given registers with photons to send to the client
        self.mc.send(to_send, p)
            
        
class SDClient:
    '''
    Represents a client to be used during the superdense coding protocol
    '''
    
    def __init__(self, bell_count, namespace, classical_reg=0, qubits=None):
        '''
        Instantiates a client, which will receive information during
        the superdense coding protocol.
        '''
        
        self.qubit_count = 2 * bell_count
        if qubits:
            assert self.qubit_count == len(qubits), "Number of requested qubits does not match number of given registers."
        
        self.namespace = namespace
        self.qubit_count = 2 * bell_count
        self.registers = qubits if qubits else [i for i in range(qubit_count)]
        self.classical_reg = 0
        self.used_registers = []
        
    def connect(self, mc):
        '''
        Makes a connection to the client, using the MockConnection object
        :param mc: A mock connection while we wait for hardware
        '''
        self.mc = mc
        
    def prepare():
        '''
        Prepares for information receival by clearing all used registers
        '''
        p = Program()
        clear_registers(self.classical_reg, self.registers, p)
        
    def receive_initial(self, qubit):
        pass
        
class MockConnection:
    '''
    Mocks a connection between the server and client. In the future when
    hardware and infrastructure are available to support it, a real
    connection protocol will need to be created.
    '''
    
    def __init__(self, server, client):
        
        this.server = server
        this.client = client
        assert this.server.qubit_count > this.client.qubit_count, "The client must have enough qubit registers left to receive from the client"
        assert this.server.namespace == this.client.namespace, "The server and client cannot be connected, as they do not share the same namespace address"
        
        this.forest = forest.Connection();
        
        # Server and client will essentially use the same physical QVM or QPU for now
        # If they share any registers, replace them
        # TODO: Make this smarter
        if not set(this.server.registers).isdisjoint(this.client.registers):
            max_reg = max(max(this.server.registers), max(this.client.registers))
            for i, r in enumerate(this.client.registers):
                if r in this.server.registers:
                    this.client.registers[i] = max_reg + 1
                    max_reg += 1
                    
    def increment_index():
        pass
                    
    def send(self, registers, p):
        '''
        The send operation simply swaps qubits into the registers of the client, which
        mocks the operation of sending information over a channel
        '''
        
        for i in len(registers):
            p.inst(SWAP)
            
def clear_registers(classical_reg, registers, p):
    p.inst(FALSE(classical_reg))
    for reg in registers:
        p.inst(MEASURE(reg, classical_reg))
        flip = Program(X(reg))
        p.inst(if_then(classical_reg, flip))
        
        
        
# Create a server and a client (using 2 bell pairs = 4 classical bits)
server = SDServer(2, "sdc://rigetti.com/example")
client = SDClient(server.qubit_count / 2, "sdc://rigetti.com/example")

# Create a mock connection (to be replaced with real connection in future)
connection = MockConnection(server, client)
server.connect(connection)
client.connect(connection)

print server.registers
print client.registers

# Prepare the client for receiving initial entangled qubit payload
client.prepare()

# Server dispatches entangled qubit payload
server.dispatch_initial()

# Server and client sepate some distance, or wait for some time, etc...
# ... until ready to send information from the server to the client