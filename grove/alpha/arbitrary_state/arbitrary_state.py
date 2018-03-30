"""Class for generating a program that can generate an arbitrary quantum state.
References are available at:

- http://140.78.161.123/digital/\
2016_ismvl_logic_synthesis_quantum_state_generation.pdf
- https://arxiv.org/pdf/quant-ph/0407010.pdf

Note that the algorithm used creates a circuit that begins with a target state
and brings it to the all zero state. Thus, many of this module's functions
involve finding gates to be applied in the reversed circuit.
"""
import numpy as np
import pyquil.quil as pq
from pyquil.api import QVMConnection
from pyquil.gates import *
from six.moves import input


def get_uniformly_controlled_rotation_matrix(k):
    """
    Returns the matrix represented by :math:`M_{ij}` in arXiv:quant-ph/0407010.

    This matrix converts the angles of :math:`k`-fold uniformly
    controlled rotations to the angles of the efficient gate decomposition.

    :param int k: number of control qubits
    :return: the matrix :math:`M_{ij}`
    :rtype: 2darray
    """
    M = np.full((2 ** k, 2 ** k), 2 ** -k)
    for i in range(2 ** k):
        g_i = i ^ (i >> 1)  # Gray code for i
        for j in range(2 ** k):
            M[i, j] *= (-1) ** (bin(j & g_i).count("1"))
    return M


def get_cnot_control_positions(k):
    """
    Returns a list of positions for the controls of the CNOTs used when
    decomposing uniformly controlled rotations, as outlined in
    arXiv:quant-ph/0407010.

    Referencing Fig. 2 in the aforementioned paper, this method
    uses the convention that, going up from the target qubit,
    the control qubits are labelled :math:`1, 2, \ldots, k`,
    where :math:`k` is the number of control qubits.
    The returned list provides the qubit that controls
    each successive CNOT, in order from left to right.

    :param int k: the number of control qubits
    :return: the list of positions of the controls
    :rtype: list
    """
    rotation_cnots = [1, 1]
    for i in range(2, k + 1):
        # algorithm described is to replace the last control
        # with a control to the new qubit
        # and then repeat the sequence twice
        rotation_cnots[-1] = i
        rotation_cnots = rotation_cnots + rotation_cnots
    return rotation_cnots


def get_rotation_parameters(phases, magnitudes):
    """
    Simulates one step of rotations.

    Given lists of phases and magnitudes of the same length :math:`N`,
    such that :math:`N=2^n` for some positive integer :math:`n`,
    finds the rotation angles required for one step of phase and magnitude
    unification.

    :param list phases: real valued phases from :math:`-\\pi` to :math:`\\pi`.
    :param list magnitudes: positive, real value magnitudes such that
                       the sum of the square of each magnitude is
                       :math:`2^{-m}` for some nonnegative integer :math:`m`.
    :return: A tuple t of four lists such that

        - t[0] are the z-rotations needed to unify adjacent pairs of phases
        - t[1] are the y-rotations needed to unify adjacent pairs of magnitudes
        - t[2] are the updated phases after these rotations are applied
        - t[3] are the updated magnitudes after these rotations are applied

    :rtype: tuple
    """
    # will hold the angles for controlled rotations
    # in the phase unification and probability unification steps,
    # respectively
    z_thetas = []
    y_thetas = []

    # will hold updated phases and magnitudes after rotations
    new_phases = []
    new_magnitudes = []

    for i in range(0, len(phases), 2):
        # find z rotation angles
        phi = phases[i]
        psi = phases[i + 1]
        z_thetas.append(phi - psi)

        # update phases after applying such rotations
        kappa = (phi + psi) / 2.
        new_phases.append(kappa)

        # find y rotation angles
        a = magnitudes[i]
        b = magnitudes[i + 1]
        if a == 0 and b == 0:
            y_thetas.append(0)
        else:
            y_thetas.append(
                2 * np.arcsin((a - b) / (np.sqrt(2 * (a ** 2 + b ** 2)))))

        # update magnitudes after applying such rotations
        c = np.sqrt((a ** 2 + b ** 2) / 2.)
        new_magnitudes.append(c)

    return z_thetas, y_thetas, new_phases, new_magnitudes


def get_reversed_unification_program(angles, control_indices,
                                     target, controls, mode):
    """
    Gets the Program representing the reversed circuit
    for the decomposition of the uniformly controlled
    rotations in a unification step.

    If :math:`n` is the number of controls, the indices within control indices
    must range from 1 to :math:`n`, inclusive. The length of control_indices
    and the length of angles must both be :math:`2^n`.

    :param list angles: The angles of rotation in the the decomposition,
                        in order from left to right
    :param list control_indices: a list of positions for the controls of the
                                 CNOTs used when decomposing uniformly
                                 controlled rotations; see
                                 get_cnot_control_positions for labelling
                                 conventions.
    :param int target: Index of the target of all rotations
    :param list controls: Index of the controls, in order from bottom to top.
    :param str mode: The unification mode. Is either 'phase', corresponding
                     to controlled RZ rotations, or 'magnitude', corresponding
                     to controlled RY rotations.
    :return: The reversed circuit of this unification step.
    :rtype: Program
    """
    if mode == 'phase':
        gate = RZ
    elif mode == 'magnitude':
        gate = RY
    else:
        raise ValueError("mode must be \'phase\' or \'magnitude\'")

    reversed_gates = []

    for j in range(len(angles)):
        if angles[j] != 0:
            # angle is negated in conjugated/reversed circuit
            reversed_gates.append(gate(-angles[j], target))
        if len(controls) > 0:
            reversed_gates.append(CNOT(controls[control_indices[j] - 1],
                                       target))

    return pq.Program().inst(reversed_gates[::-1])


def create_arbitrary_state(vector, qubits=None):
    """
    This function makes a program that can generate an arbitrary state.

    Applies the methods described in references above.

    Given a complex vector :math:`\\mathbf{a}` with components :math:`a_i`
    (:math:`i` ranging from :math:`0` to :math:`N-1`),
    produce a program that takes in the state :math:`\\vert 0 \\rangle`
    and outputs the state

    .. math::

        \\sum_{i=0}^{N-1}\\frac{a_i}{\\vert \\mathbf{a}\\vert} \\vert i\\rangle

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
    vec_norm = vector / np.linalg.norm(vector)
    n = max(1, int(np.ceil(np.log2(len(vec_norm)))))  # number of qubits needed

    if qubits is None:
        qubits = range(n)

    N = 2 ** n  # number of coefficients
    while len(vec_norm) < N:
        vec_norm = np.append(vec_norm, 0)  # pad with zeros

    magnitudes = list(map(np.abs, vec_norm))
    phases = list(map(np.angle, vec_norm))

    # matrix that converts angles of uniformly controlled rotation
    # to angles of uncontrolled rotations
    # at first, all qubits except for the 0 qubit act as controls
    M = get_uniformly_controlled_rotation_matrix(n - 1)

    # generate the positions of the controls for the CNOTs
    # at first, all qubits except for the 0 qubit act as controls
    rotation_cnots = get_cnot_control_positions(n - 1)

    # because conceptually this algorithm starts with the end state
    # and makes it into the |0> state,
    # the gates will be applied in reverse order
    # from the circuit given in the paper
    reversed_prog = pq.Program()

    for step in range(n):
        # Will hold reversed program corresponding to this particular step.
        reversed_step_prog = pq.Program()

        # get the y and z rotation angles needed to unify pairs
        # of phases and magnitudes, respectively,
        # as well as the new phases and magnitudes that these
        # rotations will update the states to
        z_thetas, y_thetas, phases, magnitudes = \
            get_rotation_parameters(phases, magnitudes)

        # convert these rotation angles
        # to those for uncontrolled rotations + CNOTs
        converted_z_thetas = np.dot(M, z_thetas)
        converted_y_thetas = np.dot(M, y_thetas)

        # phase unification
        phase_prog = get_reversed_unification_program(converted_z_thetas,
                                                      rotation_cnots,
                                                      qubits[0],
                                                      qubits[step + 1:],
                                                      'phase')

        # probability unification
        prob_prog = get_reversed_unification_program(converted_y_thetas,
                                                     rotation_cnots,
                                                     qubits[0],
                                                     qubits[step + 1:],
                                                     'magnitude')

        # swaps are applied first, then reversed probability unification,
        # then reversed phase unification

        if step < n - 1:
            # swaps are applied after all rotation steps except the last
            reversed_step_prog += SWAP(qubits[0], qubits[step + 1])

            # just retain upper left square
            # for the next iteration (one less control)
            M = M[0:int(len(M) / 2), 0:int(len(M) / 2)] * 2

            # update next set of controls (one less control)
            rotation_cnots = rotation_cnots[:int(len(rotation_cnots) / 2)]
            rotation_cnots[-1] -= 1

        reversed_step_prog += prob_prog
        reversed_step_prog += phase_prog

        # This step must be applied to the front
        # of the currently built up reversed_prog because
        # of the reversal of the steps in the final reversed program.
        reversed_prog = reversed_step_prog + reversed_prog

    # Add Hadamard gates to remove superposition
    reversed_prog = pq.Program().inst(list(map(H, qubits))) + reversed_prog

    # Correct the overall phase
    reversed_prog = pq.Program().inst(RZ(-2 * phases[0], qubits[0])) \
                      .inst(PHASE(2 * phases[0], qubits[0])) + reversed_prog

    # Apply all gates in reverse
    return reversed_prog


if __name__ == "__main__":
    print("Example list: -3.2+1j, -7, -0.293j, 1, 0, 0")
    v = input("Input a comma separated list of complex numbers:\n")
    if isinstance(v, int):
        v = [v]
    else:
        v = list(v)
    p = create_arbitrary_state(v)
    qvm = QVMConnection()
    wf = qvm.wavefunction(p)
    print("Normalized Vector: ", list(v / np.linalg.norm(v)))
    print("Generated Wavefunction: ", wf)
    if input("Show Program? (y/n): ") == 'y':
        print("----------Quil Code Used----------")
        print(p.out())
