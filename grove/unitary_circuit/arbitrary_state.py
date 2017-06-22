"""Class for generating a program that can generate an arbitrary quantum state

The general strategy is: beginning with the |0> state, apply a series of controlled RZ and RY rotations
to move exactly one basis state into place. To control on all qubits except one, the sequence of
Gray codes is followed.
"""

from grove.unitary_circuit.utils import *
from pyquil.api import SyncConnection


def create_arbitrary_state(vector):
    """
    This function makes a program that can generate an arbitrary state.

    Given a complex vector a with components a_i (i from 0 to N-1), produce a function
    that takes in |0> and gives sum_i=1^N a_i/|a| |i>
    :param vector: the vector to put into qubit form.
    :return: a program that takes in |0> and produces a state that represents this vector.
    :rtype: Program
    """
    # start with normalizing
    vector = vector/np.linalg.norm(vector)

    n = int(np.ceil(np.log2(len(vector))))
    N = 2 ** n

    p = pq.Program().inst(map(I, range(n)))

    last = 0  # the last state created

    ones = set()  # the bits that are ones
    zeros = {0}  # the bits that are zeros

    unset_coef = 1

    for i in range(1, N):
        # if there are padded 0s, just add 0s until you reach the end
        if unset_coef == 0:
            return p
        # Generate the Gray Code corresponding to i
        current = i ^ (i >> 1)
        flipped = 0

        # Find the position of the bit that was flipped
        difference = last ^ current
        while difference & 1 == 0:
            difference = difference >> 1
            flipped += 1

        # See if the flip was 0 to 1 or 1 to 0
        flipped_to_one = True
        if flipped in ones:
            ones.remove(flipped)
            flipped_to_one = False

        elif flipped in zeros:
            zeros.remove(flipped)


        a_i = 0. if last >= len(vector) else vector[last]
        z_i = a_i/unset_coef

        # Write z_i = e^(+/- i*alpha/2) * cos(beta/2), depending on if it is 0 to 1 or 1 to 0
        beta = 2 * np.arccos(np.abs(z_i))  # to get the right angle for the magnitude
        z_i /= np.cos(beta/2)
        alpha = -2*np.angle(z_i)  # to get the right angle for the phase

        # set all zero controls to 1
        p.inst(map(X, zeros))

        if not flipped_to_one:
            alpha *= -1

        if a_i == 0:
            alpha = 0

        # make a y rotation to get correct magnitude
        if beta != 0:
            p += n_qubit_controlled_RY(list(zeros | ones), flipped, beta)

        # make a z rotation to get the correct phase
        if alpha != 0:
            p += n_qubit_controlled_RZ(list(zeros | ones), flipped, alpha)

        # flip all zeros back to 1
        p.inst(map(X, zeros))

        if flipped_to_one:
            ones.add(flipped)
            unset_coef *= np.exp(1j*alpha/2)*np.sin(beta/2)

        else:
            zeros.add(flipped)
            unset_coef *= -np.exp(-1j*alpha/2)*np.sin(beta/2)

        last = current

    # fix the phase of the final qubit
    if len(vector) > 1 and unset_coef != 0:
        a_i = vector[last]
        z_i = a_i / unset_coef
        theta = np.angle(z_i)
        p.inst(map(X, zeros))
        p += n_qubit_controlled_PHASE(list(zeros), n-1, theta)
        p.inst(map(X, zeros))

    return p

if __name__ == "__main__":
    print "Example list: -3.2+1j, -7, -0.293j, 1, 0, 0"
    v = input("Input a comma separated list of complex numbers:\n")
    if isinstance(v, int):
        v = [v]
    else:
        v = list(v)
    p = create_arbitrary_state(v)
    qvm = SyncConnection()
    wf, _ = qvm.wavefunction(p)
    print "Normalized Vector: ", list(v / np.linalg.norm(v))
    print "Generated Wavefunction: ",  wf
    print "----------Quil Code Used----------"
    print p.out()