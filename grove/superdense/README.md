# Superdense Coding Application

This module will allow superdense coding to be used in a client - server type
connection, in the future when qubits can be sent across space and are resistant
to decoherence (for example, a QPU connected to other QPUs through the use of
photonic qubits).

## Setting up the server and client
The `SDServer` will handle the creation of bell pairs, and will send parts of the bell
pair to the `SDClient` using an `SDConnection` object. 

In the context of current quantum communication technology, the example utilizes a 
`MockConnection` object, where instead of using certain processes to send 
photonic qubits between two quantum computers, allows for the client and server
to exist on the same quantum processor, where the transfer of qubits occurs using
the SWAP operation.

To create the server, you pass in the total number of bell pairs that you would
like to allocate space for on the server (which is equal to half the number of 
classical bits that you expect to send you wish to send), as well as a `namespace`
which is used to make the connection between server and client.

```python
bell_pairs = 5
server = SDServer(bell_pairs, "sdc://rigetti.com/experiment")
```

Next, on a separate machine (or in our case the same machine since this is a
local experiment), we create the client, which expects the same number of bell
pairs to work with, the same `namespace` to connect to, as well as `classical_regs`,
which is a list of classical registers to save the bit message sent from server to
client. In this case, we use registers 1 through 10 to save the 10 classical bits
that we want to send across our connection

```python
client = SDClient(bell_pairs, "sdc://rigetti.com/channel", range(1,11))
```

Finally, we want to make a connection between the `SDServer` and `SDClient`, which
we do using the `SDConnection` abstract object. An connection derived from the
`SDConnection` class will handle the specific sending and intake of qubits
between a server and client, which is hardware specific. In the case of the Forest
platform, this will involve "sending" qubits between the server and client by doing
a SWAP from qubits on the server to qubits on the client. This feature is supported
by the `MockSDConnection` class. After making the connnection class with the connected
client, we also tell the client and server what connection they share:

```python
connection = MockSDConnection(server, client)
server.connect(connection)
client.connect(connection)
```

Errors will be thrown if the server and client do not share the same namespace, or if
the number of qubits and classical registers do not work together between the client
and server.

## Preparing for the general superdense coding scheme

Next, we want to prepare our client and server for completing the actual superdense
coding scheme. This is done by first calling `prepare()` on our client, which clears
all qubit and classical registers that are to be used in the protocol.

```python
client.prepare()
```

We can then instruct our server to construct the bell pairs that will be used for later
communication. Using the `dispatch_initial()` method, it will generate these bell pairs
and send one of the qubits for each pair to the client. This sets up the general scheme
of each party holding onto one of the entangled qubits for each pair

```python
server.dispatch_initial()
```

In the case that any client-side operations need to be done after receiving parts of the
bell pairs (for example, activating some long-term storage process), one can then use the
`lockdown_init()` method:

```python
client.lockdown_init()
```

## Completing the superdense coding scheme

We now have a server and client that share pairs of entangled qubits. It is hopeful that
physical advancements in the quantum computing field will allow for long coherence times,
meaning that these machines can now wait for some period of time or move apart some physical
distance before sending information again. This is where the benefit of superdense coding comes
in; although the net transfer of information is 2 qubits for every 2 classical bits,
it is at this later time or far distance that we can use 1 qubit to send 2 classical bits
using **half of the bandwidth** that would need to be used classically. For example, you might
imagine a measuring instrument that measures at 30 GBps, but the channel to your storage destination
only has a bandwidth of 15 GBps.

Let's say that your bitstream is received on the server, and you now want to send it to your
client. You can do this by calling `dispatch_info(bit1, bit2)` from your server, which will
send qubits to your client after encoding the qubit using the general superdense coding
scheme:

```
server_bitstream = [0,1,0,0,....]
for i in range(0, len(server_bitstream), 2):
    bit1 = server_bitstream[i]
    bit2 = server_bitstream[i+1]
    server.dispatch_info(bit1, bit2)
```

The client will receive this messages, finishing once all bell pairs are exhausted, or
until a certain sequence is received (to be implemented). You can then get the final
bitstring received on the client using the `get_message()` method:

```python
result = client.get_message()
print result
```

## Further work

* It seems that the superdense coding protocol actually support the client using up bell pairs to send information to the server, so it may be useful adding support for sending in that direction as well
* Add better streaming support (i.e. the client doesn't need to wait to see the message, measure and output upon eeach receival)
* Better protocol for robustness in transfer; this protocol relies on in-order receival of qubits