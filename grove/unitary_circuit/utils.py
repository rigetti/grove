"""Utils for generating circuits from unitaries."""
from pyquil.gates import *
import numpy as np
from scipy.linalg import sqrtm
import pyquil.quil as pq
from pyquil.parametric import ParametricProgram

############### Circuits from single bit unitaries and controlled unitaries ######################
# see                                                                                            #
#       -http://d.umn.edu/~vvanchur/2015PHYS4071/Chapter4.pdf                                    #
#       - http://fab.cba.mit.edu/classes/862.16/notes/computation/Barenco-1995.pdf               #
# for more information                                                                           #
# NOTE: these papers may use different conventions.                                              #
##################################################################################################


def get_one_qubit_gate_params(U):
    """
    Decompose U into e^i*d_phase * RZ(alpha)*RY(theta)*RZ(beta).
    :param U: a 2x2 unitary matrix
    :return: d_phase, alpha, theta, beta
    :rtype: tuple
    """
    d = np.sqrt(np.linalg.det(U)+0j)  # +0j to ensure the complex square root is taken
    U = U/d
    d_phase = np.angle(d)
    if U[0, 0] == 0:
        alpha = np.angle(U[1, 0])
        beta = -alpha
    elif U[1, 0] == 0:
        alpha = -np.angle(U[0, 0])
        beta = alpha
    else:
        alpha = -np.angle(U[0,0]) + np.angle(U[1,0])
        beta = -np.angle(U[0,0]) - np.angle(U[1,0])

    theta = 2*np.arctan2(np.abs(U[1, 0]), np.abs(U[0, 0]))
    return d_phase, alpha, beta, theta

def get_one_qubit_gate_from_unitary_params(params, qubit):
    """
    Creates a circuit that simulates acting the unitary on the qubit, given angle parameters as above.
    Uses a combination of PHASE, RY and RZ gates to achieve this.
    :param params: tuple in the form (d_phase, alpha, theta, beta)
    :param qubit: the qubit on which to act on
    :return: the program that represents acting the unitary on the qubit
    :rtype: Program
    """
    p = pq.Program()
    d_phase, alpha, beta, theta = params

    # can make simplifications if theta is 0
    if theta == 0:
        z_angle = beta + alpha - 2*d_phase
        if z_angle != 0:
            p.inst(RZ(z_angle, qubit))
        if d_phase != 0:
            p.inst(PHASE(2*d_phase, qubit))
    else:
        if beta != 0:
            p.inst(RZ(beta, qubit))

        p.inst(RY(theta, qubit))

        if (alpha - 2*d_phase) != 0:
            p.inst(RZ(alpha-2*d_phase, qubit))
        if d_phase != 0:
            p.inst(PHASE(2*d_phase, qubit))
    return p


def get_one_qubit_controlled_from_unitary_params(params, control, target):
    """
    Get the controlled version of the unitary, given angular parameters.
    Uses PHASE, RZ, RY and CNOT gates.

    :param params: tuple in the form (d_phase, alpha, theta, beta)
    :param control: the control qubit
    :param target: the target qubit
    :return: the program that simulates acting a controlled unitary on the target, given the control.
    :rtype: Program
    """
    p = pq.Program()
    d_phase, alpha, beta, theta = params
    # p.inst(RZ((beta-alpha)/2, target))\
    #     .inst(CNOT(control, target)) \
    #     .inst(RZ(-(beta + alpha) / 2, target)).inst(RY(-theta/2, target))\
    #     .inst(CNOT(control, target))\
    #     .inst(RY(theta/2, target)).inst(RZ(alpha, target))\
    #     .inst(PHASE(d_phase, control))
    #
    if beta-alpha != 0:
        p.inst(RZ((beta-alpha)/2, target))
    p.inst(CNOT(control, target))
    if beta+alpha != 0:
        p.inst(RZ(-(beta+alpha)/2, target))
    if theta != 0:
        p.inst(RY(-theta/2, target))
    p.inst(CNOT(control, target))
    if theta != 0:
        p.inst(RY(theta/2, target))
    if alpha != 0:
        p.inst(RZ(alpha, target))
    if d_phase != 0:
        p.inst(PHASE(d_phase, control))
    return p


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


def n_qubit_controlled_RZ(controls, target, theta):
    """
    :param controls: The list of control qubits
    :param target: The target qubit
    :param theta: The angle of rotation
    :return: the program that applies a RZ(theta) gate to target, given controls
    :rtype: Program
    """
    if len(controls) == 0:
        return pq.Program().inst(RZ(theta, target))
    u = np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])
    return n_qubit_control(controls, target, u)


def n_qubit_controlled_PHASE(controls, target, theta):
    """
    :param controls: The list of control qubits
    :param target: The target qubit
    :param theta: The angle of rotation
    :return: the program that applies a PHASE(theta) gate to target, given controls
    :rtype: Program
    """
    if len(controls) == 0:
        return pq.Program().inst(PHASE(theta, target))
    u = np.array([[1, 0], [0, np.exp(1j*theta)]])
    p = n_qubit_control(controls, target, u)
    return p


def n_qubit_controlled_RY(controls, target, theta):
    """
    :param controls: The list of control qubits
    :param target: The target qubit
    :param theta: The angle of rotation
    :return: the program that applies a RY(theta) gate to target, given controls
    :rtype: Program
    """
    if len(controls) == 0:
        return pq.Program().inst(RY(theta, target))
    u = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
    return n_qubit_control(controls, target, u)


def n_qubit_control(controls, target, u):
    """
    Returns a controlled u gate with n-1 controls.

    Does not define new gates. Follows arXiv:quant-ph/9503016, but flips the order of the gates due to convention.

    :param controls: The indices of the qubits to condition the gate on.
    :param target: The index of the target of the gate.
    :param u: The unitary gate to be controlled, given as a numpy array.
    :return: The controlled gate.
    :rtype: Program
    """
    def controlled_program_builder(controls, target, target_gate):

        p = pq.Program()

        params = get_one_qubit_gate_params(target_gate)

        sqrt_gate = sqrtm(target_gate)
        sqrt_params = get_one_qubit_gate_params(sqrt_gate)

        adj_sqrt_params = get_one_qubit_gate_params(np.conj(sqrt_gate).T)

        if len(controls) == 0:
            p += get_one_qubit_gate_from_unitary_params(params, target)

        elif len(controls) == 1:
            # controlled U
            p += get_one_qubit_controlled_from_unitary_params(params, controls[0], target)

        else:
            many_toff = controlled_program_builder(controls[:-1], controls[-1], np.array([[0, 1], [1, 0]]))

            # n-2 controlled V
            p += controlled_program_builder(controls[:-1], target, sqrt_gate)

            p += many_toff

            # controlled V_adj
            p += get_one_qubit_controlled_from_unitary_params(adj_sqrt_params, controls[-1], target)

            p += many_toff

            # controlled V
            p += get_one_qubit_controlled_from_unitary_params(sqrt_params, controls[-1], target)

        return p

    p = controlled_program_builder(controls, target, u)
    return p