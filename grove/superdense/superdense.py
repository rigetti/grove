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

from grove.teleport.teleportation import make_bell_pair
from pyquil.quil import Program
import pyquil.forest as forest
from pyquil.gates import RESET, FALSE, MEASURE, X, Z, SWAP, CNOT, H
import numpy as np

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
        self.available = []
        
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
            self.available.append(i)
            
        # Ideally, in the future, a program would not be sent here. Instead, the bell
        # state preparation will be executed here, and then mc.send will maybe entangle
        # the given registers with photons to send to the client
        self.mc.execute(p)
        self.mc.send(to_send, 0)
        
    def dispatch_info(self, bit1, bit2):
        '''
        Sends 2 bits of information across the channel
        :param bit1: The first bit to send
        :param bit2: The second bit to send
        '''
        
        # Apply gates to my state
        reg = self.available[0]
        del self.available[0]
        p = Program()
        
        # Apply the encoding
        if bit2 == 1:
            p.inst(X(reg))
        if bit1 == 1:
            p.inst(Z(reg))
            
        self.mc.execute(p)
        self.mc.send([reg], 1)
            
        
class SDClient:
    '''
    Represents a client to be used during the superdense coding protocol
    '''
    
    def __init__(self, bell_count, namespace, message_regs, classical_reg=0, qubits=None):
        '''
        Instantiates a client, which will receive information during
        the superdense coding protocol.
        :param bell_count: The number of bell pairs to create. The amount of 
                           information that can be sent between the server
                           and client will be 2 times this value.
        :param namespace: The address to use, i.e. the url for the server and
                          client to connect to.
        :param message_regs: The classical registers to save the message in
        :param classical_reg: Optional. Register to use in resetting qubits
        :param qubits: Optional. Allows the user to define the indices of the
                       qubits to use for the protocol on the server size.
        '''
        
        self.qubit_count = 2 * bell_count
        if qubits:
            assert self.qubit_count == len(qubits), "Number of requested qubits does not match number of given registers."
        
        self.namespace = namespace
        self.qubit_count = 2 * bell_count
        self.registers = qubits if qubits else [i for i in range(self.qubit_count)]
        self.classical_reg = 0
        self.used_registers = []
        self.message_regs = message_regs
        self.message_index = 0
        
    def connect(self, mc):
        '''
        Makes a connection to the client, using the MockConnection object
        :param mc: A mock connection while we wait for hardware
        '''
        self.mc = mc
        
    def prepare(self):
        '''
        Prepares for information receival by clearing all used registers
        '''
        p = Program()
        clear_registers(self.classical_reg, self.registers, p)
        # [qubit 1, qubit 2]
        self.mappings = [[self.registers[i], self.registers[i+1]] for i in range(0, len(self.registers), 2)]
        self.mc.execute(p)
        
    def lockdown_init(self):
        '''
        Perform operations on the client after receiving parts of the bell
        pairs from the server
        '''
        pass
        
    def get_empty(self):
        '''
        Returns the qubits that have no been used in the protocol yet
        '''
        return [self.mappings[i][0] for i in range(len(self.mappings))]
    
    def get_storage(self):
        '''
        Returns the qubits that are available for information storage
        '''
        return [self.mappings[i][1] for i in range(len(self.mappings))]
    
    def receive_qubit(self):
        '''
        Handles incoming qubits by looking at the next available register
        on the client, and applying the bell measurement
        '''
        
        # Grab the qubit registers
        q2, q1 = (self.mappings[0][0], self.mappings[0][1])
        del self.mappings[0]
        
        # Measure the bell state
        p = Program()
        p.inst(CNOT(q1, q2))
        p.inst(H(q1))
        p.inst(MEASURE(q1, self.message_regs[self.message_index]))
        p.inst(MEASURE(q2, self.message_regs[self.message_index + 1]))
        self.message_index += 2;
        self.mc.execute(p)
        
        # TODO Ideally you would want some sort of sequence detection for
        # when the bitstream stops
        if(self.message_index >= len(self.message_regs)):
            return True
        else:
            return False
        
    def get_message(self):
        '''
        Returns the message received on the client from the server
        '''
        
        return self.mc.finish(self.message_regs)
        
        
        
class MockSDConnection:
    '''
    Mocks a connection between the server and client. In the future when
    hardware and infrastructure are available to support it, a real
    connection protocol will need to be created.
    '''
    
    def __init__(self, server, client):
        '''
        Creates a mock connection between the given server and client
        :param server: The server to provide in this connection
        :param client: The client to provide in this connection
        '''
        
        self.server = server
        self.client = client
        assert self.server.qubit_count >= self.client.qubit_count, "The client must have enough qubit registers left to receive from the client"
        assert self.server.namespace == self.client.namespace, "The server and client cannot be connected, as they do not share the same namespace address"
        
        self.forest = forest.Connection();
        self.final_program = Program()
        
        # Server and client will essentially use the same physical QVM or QPU for now
        # If they share any registers, replace them
        # TODO: Make this smarter
        if not set(self.server.registers).isdisjoint(self.client.registers):
            max_reg = max(max(self.server.registers), max(self.client.registers))
            for i, r in enumerate(self.client.registers):
                if r in self.server.registers:
                    self.client.registers[i] = max_reg + 1
                    max_reg += 1
    
    def execute(self, program):
        '''
        Executes a program. Since this is a mock connection, these are saved
        until the end, and executed then.
        :param program: The program to execute (deferred)
        '''
        self.final_program += program
                    
    def send(self, registers, status):
        '''
        The send operation simply swaps qubits into the registers of the client, which
        mocks the operation of sending information over a channel
        :param registers: A list of registers that hold the qubits to send
        :param status: 0 if this is the initial sending, and 1 if you are sending the
                       final information
        :return: False if more bits can be sent in the final process, and false otherwise
        '''
        
        # IT IS IMPERATIVE THAT QUBITS ARE SENT IN THE CORRECT ORDER. USUALLY,
        # IT WOULD BE NICE TO SEND ALONG INDEX PARAMETERS, BUT THAT WOULD
        # DEFEAT THE PURPOSE!
        
        # Note that these operations will be done on the client side of the connection
        # The server will send a qubit, but that is the only operation
        p = Program()
        open_indices = []
        if status == 0:
            open_indices = self.client.get_empty()
        else:
            open_indices = self.client.get_storage()
        for i, r in enumerate(registers):
            p.inst(SWAP(r, open_indices[i]))
        
        self.execute(p)
        
        if status == 1:
            return self.client.receive_qubit()
        else:
            return False
        
    def finish(self, registers):
        '''
        Closes this connection, returns the received bits
        :param registers: The classical registers to measure
        :return: The received message
        '''
        
        return self.forest.run(self.final_program, registers)[0]
            
def clear_registers(classical_reg, registers, p):
    '''
    Clears the given register, using an empty classical register as an ancillary bit
    :param classical_reg: An extra bit to assist in the clearing process
    :param registers: The registers to clear
    :param p: The program to append this operation to
    '''
    p.inst(FALSE(classical_reg))
    for reg in registers:
        p.inst(MEASURE(reg, classical_reg))
        flip = Program(X(reg))
        p.if_then(classical_reg, flip)
       
if __name__ == "__main__":
    
    # Create a server and a client (using 2 bell pairs = 4 classical bits)
    bell_pairs = 5
    server = SDServer(bell_pairs, "sdc://rigetti.com/channel")
    client = SDClient(bell_pairs, "sdc://rigetti.com/channel", range(1,11))

    # Create a mock connection (to be replaced with real connection in future)
    connection = MockSDConnection(server, client)
    server.connect(connection)
    client.connect(connection)

    # Prepare the client for receiving initial entangled qubit payload
    client.prepare()

    # Server dispatches entangled qubit payload, and client recognizes
    # when this operation is finished
    server.dispatch_initial()
    client.lockdown_init()

    # Server and client separate some distance, or wait for some time, etc...
    # ... until ready to send information from the server to the client
    server_bitstream = list(np.random.randint(2, size=10))
    #print server_bitstream
    for i in range(0, len(server_bitstream), 2):
        bit1 = server_bitstream[i]
        bit2 = server_bitstream[i+1]
        server.dispatch_info(bit1, bit2)

    #print client.get_message()