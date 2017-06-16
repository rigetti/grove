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

import numpy as np

#def test_superdense_protocol_single():
    
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