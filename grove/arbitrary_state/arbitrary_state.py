"""Class for generating a program that can generate an arbitrary quantum state

- http://140.78.161.123/digital/2016_ismvl_logic_synthesis_quantum_state_generation.pdf
- https://arxiv.org/pdf/quant-ph/0407010.pdf

"""
import numpy as np
import pyquil.quil as pq
from pyquil.api import SyncConnection
from pyquil.gates import *


def get_uniformly_controlled_rotation_matrix(k):
    """
    Returns the matrix represented by :math:`M_{ij}` in arXiv:quant-ph/0407010.

    This matrix converts the angles of :math:`k`-fold uniformly
    controlled rotations to the angles of the efficient gate decomposition.

    :param int k: number of control qubits
    :return: the matrix :math:`M_{ij}`
    :rtype: 2darray
    """
    M = np.full((2**k, 2**k), 2**-k)
    for i in xrange(2**k):
        g_i = i ^ (i >> 1)  # Gray code for i
        for j in xrange(2**k):
            M[i, j] *= (-1)**(bin(j & g_i).count("1"))
    return M


def get_cnot_control_positions(k):
    """
    Returns a list of positions for the controls of the CNOTs used when
    decomposing uniformly controlled rotations, as outlined in
    arXiv:quant-ph/0407010.

    It is assumed that the target is the zeroth qubit
    and higher order bits are in consecutive integer order.
    The list gives the positions going from left to right.

    :param int k: the number of control qubits
    :return: the list of positions of the controls
    :rtype: list
    """
    rotation_cnots = [1, 1]
    for i in xrange(2, k+1):
        # algorithm described is to replace the last control
        # with a control to the new qubit
        # and then repeat the sequence twice
        rotation_cnots[-1] = i
        rotation_cnots = rotation_cnots + rotation_cnots
    return rotation_cnots


def create_arbitrary_state(vector, qubits=None):
    """
    This function makes a program that can generate an arbitrary state.

    Applies the methods described in references above.

    Given a complex vector :math:`a` with components :math:`a_i`
    (:math:`i` ranging from :math:`0` to :math:`N-1`),
    produce a program that takes in the state :math:`\\vert 0\\ldots 0\\rangle`
    and outputs the state

    .. math::

        \\sum_{i=0}^{N-1}\\frac{a_i}{\\vert a\\vert} \\vert i\\rangle

    where :math:`i` is given in its binary expansion.

    :param 1darray vector: the vector to put into qubit form.
    :param list(int) qubits: Which qubits to encode the vector into.
                             Must contain at least the minimum
                             number of qubits :math:`n` needed for all elements
                             of vector to be present as a coefficient in the
                             final state. If more than :math:`n` are provided,
                             only the first :math:`n` will be used.
                             If no list is provided, the default will be
                             qubits :math:`0, 1, \ldots, n-1`.
    :return: a program that takes in :math:`\\vert 0\\rangle^{\otimes n}`
             and produces a state that represents this vector, as
             described above.
    :rtype: Program
    """
    vec_norm = vector/np.linalg.norm(vector)
    n = int(np.ceil(np.log2(len(vec_norm))))  # number of qubits needed
    if n == 0:  # if vec_norm is of length 1
        n += 1

    if qubits is None:
        qubits = range(n)

    N = 2 ** n  # number of coefficients
    while len(vec_norm) < N:
        vec_norm = np.append(vec_norm, 0)  # pad with zeros

    magnitudes = map(np.abs, vec_norm)
    phases = map(np.angle, vec_norm)

    # because this algorithm starts with a state and makes it into the |0> state,
    # the gates will be constructed in reverse order
    reversed_gates = []

    # matrix that converts angles of uniformly controlled rotation to angles of uncontrolled rotations
    # at first, all qubits except for the 0 qubit act as controls
    M = get_uniformly_controlled_rotation_matrix(n - 1)

    # generate the positions of the controls for the CNOTs
    # at first, all qubits except for the 0 qubit act as controls
    rotation_cnots = get_cnot_control_positions(n - 1)

    for step in xrange(n):
        z_thetas = []  # will hold the angles for controlled rotations in the phase unification step
        y_thetas = []  # will hold the angles for controlled rotations in the probabilities unification step
        for i in xrange(0, N, 2):
            # find z rotation angles
            phi = phases[i]
            psi = phases[i+1]
            if i % 2**(step+1) == 0:  # due to repetition, only select angles are needed
                z_thetas.append(phi - psi)
            # update phases after applying such rotations
            kappa = (phi + psi)/2.
            phases[i], phases[i+1] = kappa, kappa

            # find y rotation angles
            a = magnitudes[i]
            b = magnitudes[i+1]
            if i % 2**(step+1) == 0:  # due to repetition, only select angles are needed
                if a == 0 and b == 0:
                    y_thetas.append(0)
                else:
                    y_thetas.append(2*np.arcsin((a-b)/(np.sqrt(2*(a**2 + b**2)))))
            # update magnitudes after applying such rotations
            c = np.sqrt((a**2+b**2)/2.)
            magnitudes[i], magnitudes[i+1] = c, c

        # convert these rotation angles to those for uncontrolled rotations + CNOTs
        converted_z_thetas = np.dot(M, z_thetas)
        converted_y_thetas = np.dot(M, y_thetas)

        # just retain upper left square for the next iteration (one less control)
        M = M[0:len(M)/2, 0:len(M)/2]*2

        # phase unification
        for j in xrange(len(converted_z_thetas)):
            if converted_z_thetas[j] != 0:
                # angle is negated in conjugated/reversed circuit
                reversed_gates.append(RZ(-converted_z_thetas[j], qubits[0]))
            if step < n-1:
                reversed_gates.append(CNOT(qubits[step + rotation_cnots[j]], qubits[0]))

        # probability unification
        for j in xrange(len(converted_y_thetas)):
            if converted_y_thetas[j] != 0:
                # angle is negated in conjugated/reversed circuit
                reversed_gates.append(RY(-converted_y_thetas[j], qubits[0]))
            if step < n-1:
                reversed_gates.append(CNOT(qubits[step + rotation_cnots[j]], qubits[0]))

        # swaps are applied after all rotation steps except the last
        if step < n-1:
            reversed_gates.append(SWAP(qubits[0], qubits[step+1]))

            mask = 1 + (1 << (step + 1))
            for i in xrange(N):
                # only swap the numbers for which the 0th bit and step+1-th bit are different
                # only check for i having 0th bit 1 and step+1-th bit 0 to prevent duplication
                if (i & mask) ^ 1 == 0:
                    phases[i], phases[i ^ mask] = phases[i ^ mask], phases[i]
                    magnitudes[i], magnitudes[i ^ mask] = magnitudes[i ^ mask], magnitudes[i]

            # update next rotation_cnots
            rotation_cnots = rotation_cnots[:len(rotation_cnots) / 2]
            rotation_cnots[-1] -= 1

    # Add Hadamard gates to remove superposition
    reversed_gates += map(H, qubits)

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