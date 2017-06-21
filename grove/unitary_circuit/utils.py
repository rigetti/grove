"""Utils for generating circuits from unitaries."""
from pyquil.gates import *
import numpy as np
from scipy.linalg import sqrtm
import pyquil.quil as pq
from pyquil.parametric import ParametricProgram
from pyquil.api import SyncConnection

############### Circuits from single bit unitaries and controlled unitaries ######################
# see http://d.umn.edu/~vvanchur/2015PHYS4071/Chapter4.pdf for more information                  #
##################################################################################################


def get_one_qubit_gate_params(U):
    """
    :param U: a 2x2 unitary matrix
    :return: d_phase, alpha, theta, beta
    :rtype: list
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
        alpha = -np.angle(U[0,0]/U[1,0])
        beta = -np.angle(U[0,0]*U[1,0])

    theta = 2*np.arctan2(np.abs(U[1, 0]), np.abs(U[0, 0]))
    return d_phase, alpha, beta, theta

def get_one_qubit_gate_from_unitary_params(params, qubit):
    p = pq.Program()
    d_phase, alpha, beta, theta = params
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

        if alpha != 0:
            p.inst(RZ(alpha-2*d_phase, qubit))
        if d_phase != 0:
            p.inst(PHASE(2*d_phase, qubit))
    return p


def get_one_qubit_controlled_from_unitary_params(params, control, target):
    p = pq.Program()
    d_phase, alpha, beta, theta = params
    p.inst(RZ((beta-alpha)/2, target))\
        .inst(CNOT(control, target)) \
        .inst(RZ(-(beta + alpha) / 2, target)).inst(RY(-theta/2, target))\
        .inst(CNOT(control, target))\
        .inst(RY(theta/2, target)).inst(RZ(alpha, target))\
        .inst(PHASE(d_phase, control))
    return p


def create_arbitrary_state(vector, verbose=False):
    vector = vector/np.linalg.norm(vector)
    if verbose: print list(vector)
    n = int(np.ceil(np.log2(len(vector))))
    N = 2 ** n
    if verbose: print n, N
    p = pq.Program()
    last = 0
    ones = set()
    zeros = {0}
    unset_coef = 1
    if verbose: cxn = SyncConnection()
    for i in range(1, N):
        current = i ^ (i >> 1)
        flipped = 0
        difference = last ^ current
        while difference & 1 == 0:
            difference = difference >> 1
            flipped += 1

        flipped_to_one = True
        if flipped in ones:
            ones.remove(flipped)
            flipped_to_one = False

        else:
            if flipped in zeros:
                zeros.remove(flipped)

        a_i = 0. if last >= len(vector) else vector[last]
        z_i = a_i/unset_coef
        alpha = -np.angle(z_i**2)
        beta = 2 * np.arccos(np.abs(z_i))

        if not flipped_to_one:
            alpha *= -1

        if a_i == 0:
            alpha = 0
            beta = np.pi

        if verbose: print bin(last), bin(current), flipped_to_one, unset_coef, a_i, z_i, alpha, beta
        # set all zero controls to 1
        if verbose: print zeros, ones

        p.inst(map(X, zeros))

        # make a z rotation to get the correct phase
        p += n_qubit_controlled_RZ(list(zeros | ones), flipped, alpha)

        # make a y rotation to get correct magnitude
        p += n_qubit_controlled_RY(list(zeros | ones), flipped, beta)

        # flip all zeros back to 1
        p.inst(map(X, zeros))

        if flipped_to_one:
            ones.add(flipped)
            unset_coef *= np.exp(-1j*alpha/2)*np.sin(beta/2)

        else:
            zeros.add(flipped)
            unset_coef *= -np.exp(1j*alpha/2)*np.sin(beta/2)

        last = current
        if verbose: wf, _ = cxn.wavefunction(p)
        if verbose: print wf, "\n---------------------"

    # fix the phase of the final qubit
    a_i = vector[last]
    z_i = a_i / unset_coef
    theta = np.angle(z_i)
    p.inst(map(X, zeros))
    p += n_qubit_controlled_PHASE(list(zeros), n-1, theta)
    p.inst(map(X, zeros))
    return p


def n_qubit_controlled_RZ(controls, target, theta):
    print "Controlled RZ ", controls, target, theta
    if len(controls) == 0:
        return pq.Program().inst(RZ(theta, target))
    u = np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])
    return n_qubit_control(controls, target, u)


def n_qubit_controlled_PHASE(controls, target, theta):
    print "Controlled PHASE ", controls, target, theta

    if len(controls) == 0:
        return pq.Program().inst(PHASE(theta, target))
    u = np.array([[1, 0], [0, np.exp(1j*theta)]])
    p = n_qubit_control(controls, target, u)
    return p


def n_qubit_controlled_RY(controls, target, theta):
    print "Controlled RY ", controls, target, theta

    if len(controls) == 0:
        return pq.Program().inst(RY(theta, target))
    u = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
    return n_qubit_control(controls, target, u)


def n_qubit_control(controls, target, u):
    """
    Returns a controlled u gate with n-1 controls.

    Does not define new gates.

    :param controls: The indices of the qubits to condition the gate on.
    :param target: The index of the target of the gate.
    :param u: The unitary gate to be controlled, given as a numpy array.
    :return: The controlled gate.
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
            # controlled V
            p += get_one_qubit_controlled_from_unitary_params(sqrt_params, controls[-1], target)

            many_toff = controlled_program_builder(controls[:-1], controls[-1], np.array([[0, 1], [1, 0]]))
            p += many_toff

            # controlled V_adj
            p += get_one_qubit_controlled_from_unitary_params(adj_sqrt_params, controls[-1], target)

            p += many_toff

            # n-2 controlled V
            p += controlled_program_builder(controls[:-1], target, sqrt_gate)

        return p

    p = controlled_program_builder(controls, target, u)
    return p