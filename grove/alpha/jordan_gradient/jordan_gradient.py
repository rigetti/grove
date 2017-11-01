from __future__ import division
import numpy as np
import pyquil.quil as pq
from pyquil.gates import X, H
from grove.alpha.phaseestimation.phase_estimation import controlled
from grove.qft.fourier import qft, inverse_qft

def initialize_system(input_qubits, ancilla_qubits):
    """ Prepare initial state

    :param list input_qubit: Qubits of input registers.
    :param list ancilla_qubits: Qubits of output register.
    :return Program p_ic: Quil program to initialize this system.
    """

    # ancilla qubits to plane wave state
    ic_out = list(map(X, ancilla_qubits))
    ft_out = qft(ancilla_qubits)
    p_ic_out = pq.Program(ic_out) + ft_out
    # input qubits to equal superposition
    ic_in = list(map(H, input_qubits))
    p_ic_in = pq.Program(ic_in)
    # combine programs
    p_ic = p_ic_out + p_ic_in
    
    return p_ic

def phase_kickback(f_h, input_qubits, ancilla_qubits, precision):
    """ Encode f_h into ancilla eigenvalue and kickback to input registers

    :param np.array f_h: Oracle outputs for function f at domain value h.
    :param list input_qubit: Qubits of input registers.
    :param list ancilla_qubits: Qubits of output register.
    :param int precision: Bit precision of gradient.
    :return Program p_kickback: Quil program to perform phase kickback.
    """
    
    # encode f_h into CPHASE gate
    U = np.array([[1, 0],
                  [0, np.exp(1.0j * np.pi * f_h)]])
    p_kickback = pq.Program()
    # apply c-U^{2^j} to ancilla register
    for i in input_qubits:
        if i > 0:
            U = np.dot(U, U)
        cU = controlled(U)
        name = "c-U{0}".format(2 ** i)
        p_kickback.defgate(name, cU)
        p_kickback.inst((name, i, ancilla_qubits[0]))

    # iqft to pull out fractional component of eigenphase
    p_kickback += inverse_qft(input_qubits)

    for q_out in input_qubits:
        p_kickback.measure(q_out, q_out)
        
    return p_kickback

def gradient_estimator(f_h, input_qubits, ancilla_qubits, precision=16):
    """ Gradient estimation via Jordan's algorithm
    10.1103/PhysRevLett.95.050501

    :param np.array f: Oracle outputs.
    :param list input_qubit: Qubits of input registers.
    :param list ancilla_qubits: Qubits of output register.
    :param int precision: Bit precision of gradient.
    :return Program p_gradient: Quil program to estimate gradient of f.
    """    
    
    # intialize input and output registers
    p_ic = initialize_system(input_qubits, ancilla_qubits)
    # encode oracle values into phase
    p_kickback = phase_kickback(f_h, input_qubits, ancilla_qubits, precision)
    # combine steps of algorithm into one program
    p_gradient = p_ic + p_kickback
    
    return p_gradient
