from __future__ import division
import numpy as np
import pyquil.quil as pq
from pyquil.gates import X, H
from grove.alpha.phaseestimation.phase_estimation import controlled
from grove.qft.fourier import inverse_qft

def initialize_system(input_qubits, ancilla_qubit):
    """ Prepare initial state

    :param list input_qubit: Qubits of input registers.
    :param list ancilla_qubit: Qubit of ancilla register.
    :return Program p_ic: Quil program to initialize this system.
    """

    # ancilla qubit to plane wave state
    ic_ancilla = pq.Program([X(ancilla_qubit), H(ancilla_qubit)])
    p_ic_out = pq.Program(ic_ancilla)

    # input qubits to equal superposition
    ic_in = list(map(H, input_qubits))
    p_ic_in = pq.Program(ic_in)

    # combine programs
    p_ic = p_ic_out + p_ic_in
    
    return p_ic

def phase_kickback(f_h, input_qubits, ancilla_qubit, precision):
    """ Encode f_h into ancilla eigenvalue and kickback to input registers

    :param np.array f_h: Oracle outputs for function f at domain value h.
    :param list input_qubit: Qubits of input registers.
    :param list ancilla_qubit: Qubit of ancilla register.
    :param int precision: Bit precision of gradient.
    :return Program p_kickback: Quil program to perform phase kickback.
    """
    
    # encode f_h / 2 into CPHASE gate
    U = np.array([[np.exp(1.0j * np.pi * f_h), 0],
                  [0, np.exp(1.0j * np.pi * f_h)]])

    # apply cU^{2^j} to ancilla register
    p_kickback = pq.Program()
    for i in input_qubits:
        if i > 0:
            U = np.dot(U, U)
        name = "cU{0}".format(2 ** i)
        p_kickback.defgate(name, controlled(U))
        p_kickback.inst((name, i, ancilla_qubit))

    # iqft to pull out fractional component of eigenphase
    p_kickback += inverse_qft(input_qubits)

    return p_kickback

def gradient_estimator(f_h, input_qubits, ancilla_qubit, precision=16):
    """ Gradient estimation via Jordan's algorithm
    10.1103/PhysRevLett.95.050501

    :param np.array f: Oracle outputs.
    :param list input_qubit: Qubits of input registers.
    :param list ancilla_qubits: Qubits of output register.
    :param int precision: Bit precision of gradient.
    :return Program p_gradient: Quil program to estimate gradient of f.
    """    
    
    # intialize input and output registers
    p_ic = initialize_system(input_qubits, ancilla_qubit)

    # encode oracle values into phase
    p_kickback = phase_kickback(f_h, input_qubits, ancilla_qubit, precision)

    # combine steps of algorithm into one program
    p_gradient = p_ic + p_kickback

    # measure input qubits
    for q_out in input_qubits:
        p_gradient.measure(q_out, q_out)
        
    return p_gradient
