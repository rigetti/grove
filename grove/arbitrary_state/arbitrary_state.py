"""Class for generating a program that can generate an arbitrary quantum state
"""
import pyquil.quil as pq
import numpy as np
from pyquil.gates import *
from pyquil.api import SyncConnection

def create_arbitrary_state(vector):
    """
    This function makes a program that can generate an arbitrary state.

    Uses:
        - http://140.78.161.123/digital/2016_ismvl_logic_synthesis_quantum_state_generation.pdf
        - https://arxiv.org/pdf/quant-ph/0407010.pdf

    Given a complex vector a with components a_i (i from 0 to N-1), produce a function
    that takes in |0> and gives sum_i=1^N a_i/|a| |i>
    :param vector: the vector to put into qubit form.
    :return: a program that takes in |0> and produces a state that represents this vector.
    :rtype: Program
    """
    vector = vector/np.linalg.norm(vector)
    n = int(np.ceil(np.log2(len(vector))))
    N = 2 ** n
    while len(vector) < N:
        vector = np.append(vector, 0)
    magnitudes = map(np.abs, vector)
    phases = map(np.angle, vector)

    reversed_gates = []

    M = np.full((N/2, N/2), 2**-(n-1))
    rotation_cnots = [1, 1]
    for i in range(2, n):
        rotation_cnots[-1] = i
        rotation_cnots = rotation_cnots + rotation_cnots

    for i in range(N/2):
        g_i = i ^ (i >> 1) # Gray code
        for j in range(N/2):
            M[i, j] *= (-1)**(bin(j & g_i).count("1"))

    for step in range(n):
        z_thetas = []
        y_thetas = []
        for i in range(0, N, 2):
            phi = phases[i]
            psi = phases[i+1]
            if i % 2**(step+1) == 0:
                z_thetas.append(phi - psi)
            kappa = (phi+psi)/2.
            phases[i], phases[i+1] = kappa, kappa

            a = magnitudes[i]
            b = magnitudes[i+1]
            if i % 2**(step+1) == 0:
                if a == 0 and b == 0:
                    y_thetas.append(0)
                else:
                    y_thetas.append(2*np.arcsin((a-b)/(np.sqrt(2*(a**2+b**2)))))
            c = np.sqrt((a**2+b**2)/2.)
            magnitudes[i], magnitudes[i+1] = c, c

        converted_z_thetas = np.dot(M, z_thetas)
        converted_y_thetas = np.dot(M, y_thetas)

        # phase
        for j in range(len(converted_z_thetas)):
            if converted_z_thetas[j] != 0:
                reversed_gates.append(RZ(-converted_z_thetas[j], 0))
            if step < n-1:
                reversed_gates.append(CNOT(step + rotation_cnots[j], 0))

        # magnitude
        for j in range(len(converted_y_thetas)):
            if converted_y_thetas[j] != 0:
                reversed_gates.append(RY(-converted_y_thetas[j], 0))
            if step < n-1:
                reversed_gates.append(CNOT(step + rotation_cnots[j], 0))

        # just retain upper left square
        M = M[0:len(M)/2, 0:len(M)/2]*2

        if step < n-1:
            # swap
            reversed_gates.append(SWAP(0, step+1))

            mask = 1 + (1 << (step + 1))
            for i in range(N):
                if (i & mask) ^ 1 == 0:
                    phases[i], phases[i ^ mask] = phases[i ^ mask], phases[i]
                    magnitudes[i], magnitudes[i ^ mask] = magnitudes[i ^ mask], magnitudes[i]

            # update next rotation_cnots
            rotation_cnots = rotation_cnots[:len(rotation_cnots) / 2]
            rotation_cnots[-1] -= 1

    # Hadamards
    reversed_gates += map(H, range(n))

    # overall phase
    reversed_gates.append(PHASE(2*phases[0], 0))
    reversed_gates.append(RZ(-2*phases[0], 0))
    p = pq.Program().inst([reversed_gates[::-1]])
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