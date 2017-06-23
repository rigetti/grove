"""Utils for generating circuits from unitaries."""
from pyquil.gates import *
import numpy as np
import pyquil.quil as pq

############### Circuits from single bit unitaries and controlled unitaries ######################
# see                                                                                            #
#       -http://d.umn.edu/~vvanchur/2015PHYS4071/Chapter4.pdf                                    #
#       - http://fab.cba.mit.edu/classes/862.16/notes/computation/Barenco-1995.pdf               #
# for more information                                                                           #
# NOTE: these papers may use different conventions sign and phase conventions                    #
#       for rotation gates.                                                                      #
##################################################################################################


def get_one_qubit_gate_params(U):
    """
    Decompose U into e^i*d_phase * RZ(alpha)*RY(theta)*RZ(beta).
    :param U: a 2x2 unitary matrix
    :return: d_phase, alpha, beta, theta
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
    :param params: tuple in the form (d_phase, alpha, beta, theta)
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

    :param params: tuple in the form (d_phase, alpha, beta, theta)
    :param control: the control qubit
    :param target: the target qubit
    :return: the program that simulates acting a controlled unitary on the target, given the control.
    :rtype: Program
    """
    p = pq.Program()
    d_phase, alpha, beta, theta = params

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


# X = e^i*pi/2*RZ(-pi/2)*RY(pi)*RZ(pi/2)
#   = PHASE(pi/2)*RZ(-pi/2)*RZ(-pi/2)*RY(pi)*RZ(pi/2)
# PHASE(alpha) = e^(-i*pi/2) * RZ(alpha)
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
    p = pq.Program()
    p += get_one_qubit_controlled_from_unitary_params((0, theta/4, theta/4, 0), controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += get_one_qubit_controlled_from_unitary_params((0, -theta/4, -theta/4, 0), controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += n_qubit_controlled_RZ(controls[:-1], target, theta/2)
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
    p = pq.Program()
    p += get_one_qubit_controlled_from_unitary_params((0, 0, 0, theta/2), controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += get_one_qubit_controlled_from_unitary_params((0, 0, 0, -theta/2), controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += n_qubit_controlled_RY(controls[:-1], target, theta/2)
    return p


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
    p = pq.Program()
    p += get_one_qubit_controlled_from_unitary_params((theta/4, theta/4, theta/4, 0), controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += get_one_qubit_controlled_from_unitary_params((-theta/4, -theta/4, -theta/4, 0), controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += n_qubit_controlled_PHASE(controls[:-1], target, theta/2)
    return p

def n_qubit_controlled_X(controls, target):
    """
    :param controls: The list of control qubits
    :param target: The target qubit
    :return: the program that applies a X gate to target, given controls (i.e. n-1 Toffoli)
    :rtype: Program
    """
    if len(controls) == 0:
        return pq.Program().inst(X(target))
    if len(controls) == 1:
        return pq.Program().inst(CNOT(controls[0], target))
    if len(controls) == 2:
        return pq.Program().inst(CCNOT(controls[0], controls[1], target))

    p = pq.Program()
    p += n_qubit_controlled_PHASE(controls, target, np.pi)
    p += n_qubit_controlled_RZ(controls, target, -np.pi)
    p += n_qubit_controlled_RY(controls, target, np.pi)
    p += n_qubit_controlled_RZ(controls, target, np.pi/2)

    return p

# def n_qubit_control(controls, target, u):
#     """
#     Returns a controlled u gate with n-1 controls.
#
#     Does not define new gates. Follows arXiv:quant-ph/9503016. Uses the same format as in grove.grover.grover.
#
#     :param controls: The indices of the qubits to condition the gate on.
#     :param target: The index of the target of the gate.
#     :param u: The unitary gate to be controlled, given as a numpy array.
#     :return: The controlled gate.
#     :rtype: Program
#     """
#     def controlled_program_builder(controls, target, target_gate):
#
#         p = pq.Program()
#
#         params = get_one_qubit_gate_params(target_gate)
#
#         sqrt_gate = sqrtm(target_gate)
#         sqrt_params = get_one_qubit_gate_params(sqrt_gate)
#
#         adj_sqrt_params = get_one_qubit_gate_params(np.conj(sqrt_gate).T)
#
#         if len(controls) == 0:
#             p += get_one_qubit_gate_from_unitary_params(params, target)
#
#         elif len(controls) == 1:
#             # controlled U
#             p += get_one_qubit_controlled_from_unitary_params(params, controls[0], target)
#
#         else:
#             # controlled V
#             many_toff = controlled_program_builder(controls[:-1], controls[-1], np.array([[0, 1], [1, 0]]))
#
#             p += get_one_qubit_controlled_from_unitary_params(sqrt_params, controls[-1], target)
#
#             p += many_toff
#
#             # controlled V_adj
#             p += get_one_qubit_controlled_from_unitary_params(adj_sqrt_params, controls[-1], target)
#
#             p += many_toff
#
#             # n-2 controlled V
#             p += controlled_program_builder(controls[:-1], target, sqrt_gate)
#
#         return p
#
#     p = controlled_program_builder(controls, target, u)
#     return p