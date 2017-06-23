"""Class for generating a program that can generate an arbitrary quantum state
"""
import pyquil.quil as pq
import numpy as np
from pyquil.gates import *
from pyquil.api import SyncConnection


def create_arbitrary_state(vector, offset=0):
    """
    This function makes a program that can generate an arbitrary state.

    Applies the methods described in:
        - http://140.78.161.123/digital/2016_ismvl_logic_synthesis_quantum_state_generation.pdf
        - https://arxiv.org/pdf/quant-ph/0407010.pdf

    Given a complex vector :math:`a` with components :math:`a_i` (:math:`i` ranging from :math:`0` to :math:`N-1`),
    produce a program that takes in the state :math:`|0\ldots 0>`
    and outputs the state :math:`\sum_{i=0}^{N-1}\frac{a_i}{|a|} |i>`
    where :math:`i` is given in its binary expansion.

    :param vector: the vector to put into qubit form.
    :param offset: Which qubit to begin encoding the state into. The gates of the program will be applied on
                   qubits offset, offset+1, ..., offset+n-1, where n is the number of qubits needed
                   to represent the vector.
    :return: a program that takes in :math:`|0>` and produces a state that represents this vector.
    :rtype: Program
    """
    vec_norm = vector/np.linalg.norm(vector)
    n = int(np.ceil(np.log2(len(vec_norm))))  # number of qubits needed
    if n == 0:  # if vec_norm is of length 1
        n += 1

    qubits = range(offset, offset+n)

    N = 2 ** n  # number of coefficients
    while len(vec_norm) < N:
        vec_norm = np.append(vec_norm, 0)  # pad with zeros

    magnitudes = map(np.abs, vec_norm)
    phases = map(np.angle, vec_norm)

    # because this algorithm starts with a state and makes it into the |0> state,
    # the gates will be constructed in reverse order
    reversed_gates = []

    # generate the matrix that will take the angles of k-fold uniformly controlled rotations
    # and convert them to angles for the equivalent circuit of alternating uncontrolled rotations and CNOTs
    M = np.full((N/2, N/2), 2**-(n-1))
    for i in range(N/2):
        g_i = i ^ (i >> 1)  # Gray code for i
        for j in range(N/2):
            M[i, j] *= (-1)**(bin(j & g_i).count("1"))

    # generate the positions of the controls for the CNOTs
    rotation_cnots = [1, 1]
    for i in range(2, n):
        rotation_cnots[-1] = i
        rotation_cnots = rotation_cnots + rotation_cnots

    for step in range(n):
        z_thetas = []  # will hold the angles for controlled rotations in the phase unification step
        y_thetas = []  # will hold the angles for controlled rotations in the probabilities unification step
        for i in range(0, N, 2):
            phi = phases[i]
            psi = phases[i+1]
            if i % 2**(step+1) == 0:  # due to repetition, only select angles are needed
                z_thetas.append(phi - psi)
            kappa = (phi + psi)/2.
            phases[i], phases[i+1] = kappa, kappa

            a = magnitudes[i]
            b = magnitudes[i+1]
            if i % 2**(step+1) == 0:  # due to repetition, only select angles are needed
                if a == 0 and b == 0:
                    y_thetas.append(0)
                else:
                    y_thetas.append(2*np.arcsin((a-b)/(np.sqrt(2*(a**2 + b**2)))))
            c = np.sqrt((a**2+b**2)/2.)
            magnitudes[i], magnitudes[i+1] = c, c

        # convert these rotation angles to those for uncontrolled rotations + CNOTs
        converted_z_thetas = np.dot(M, z_thetas)
        converted_y_thetas = np.dot(M, y_thetas)

        # phase unification
        for j in range(len(converted_z_thetas)):
            if converted_z_thetas[j] != 0:
                reversed_gates.append(RZ(-converted_z_thetas[j], qubits[0]))  # negative angle in conjugated/reversed circuit
            if step < n-1:
                reversed_gates.append(CNOT(qubits[step + rotation_cnots[j]], qubits[0]))

        # probability unification
        for j in range(len(converted_y_thetas)):
            if converted_y_thetas[j] != 0:
                reversed_gates.append(RY(-converted_y_thetas[j], qubits[0]))  # negative angle in conjugated/reversed circuit
            if step < n-1:
                reversed_gates.append(CNOT(qubits[step + rotation_cnots[j]], qubits[0]))

        # just retain upper left square
        M = M[0:len(M)/2, 0:len(M)/2]*2

        if step < n-1:
            # swap
            reversed_gates.append(SWAP(qubits[0], qubits[step+1]))

            mask = 1 + (1 << (step + 1))
            for i in range(N):
                # only swap the numbers for which the 0th bit and step+1-th bit are different
                # only check for i having 0th bit 1 and step+1-th bit 0 to prevent duplication
                if (i & mask) ^ 1 == 0:
                    phases[i], phases[i ^ mask] = phases[i ^ mask], phases[i]
                    magnitudes[i], magnitudes[i ^ mask] = magnitudes[i ^ mask], magnitudes[i]

            # update next rotation_cnots
            rotation_cnots = rotation_cnots[:len(rotation_cnots) / 2]
            rotation_cnots[-1] -= 1

    # Add Hadamard gates to unentangle to the 0 state
    reversed_gates += map(H, qubits)

    print vector, n
    # Correct the overall phase
    reversed_gates.append(PHASE(2*phases[0], qubits[0]))
    reversed_gates.append(RZ(-2*phases[0], qubits[0]))

    # Apply all gates in reverse
    p = pq.Program().inst([reversed_gates[::-1]])
    return p


if __name__ == "__main__":
    print "Example list: -3.2+1j, -7, -0.293j, 1, 0, 0"
    v = input("Input a comma separated list of complex numbers:\n")
    if isinstance(v, int):
        v = [v]
    else:
        v = list(v)
    offset = input("Input a positive integer offset:\n")
    p = create_arbitrary_state(v, offset)
    qvm = SyncConnection()
    wf, _ = qvm.wavefunction(p)
    print "Normalized Vector: ", list(v / np.linalg.norm(v))
    print "Generated Wavefunction: ",  wf
    print "----------Quil Code Used----------"
    print p.out()