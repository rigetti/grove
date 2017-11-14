from __future__ import division
import numpy as np
import pyquil.quil as pq
from pyquil.gates import X, H
from grove.alpha.phaseestimation.phase_estimation import controlled
from grove.qft.fourier import inverse_qft
from gradient_helper import real_to_binary, binary_to_real, stats_to_bf

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

def phase_kickback(f_h, input_qubits, ancilla_qubit):
    """ Encode f_h into ancilla eigenvalue and kickback to input registers

    :param np.array f_h: Oracle outputs for function f at domain value h.
    :param list input_qubit: Qubits of input registers.
    :param list ancilla_qubit: Qubit of ancilla register.
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

def gradient_estimator(f_h, input_qubits, ancilla_qubit):
    """ Gradient estimation via Jordan's algorithm
    10.1103/PhysRevLett.95.050501

    :param np.array f: Oracle output at perturbation h.
    :param list input_qubit: Qubits of input registers.
    :param list ancilla_qubits: Qubits of output register.
    :return Program p_gradient: Quil program to estimate gradient of f.
    """    
    
    perturbation_sign = np.sign(f_h)

    # intialize input and output registers
    p_ic = initialize_system(input_qubits, ancilla_qubit)

    # encode oracle values into phase
    p_kickback = phase_kickback(abs(f_h), input_qubits, ancilla_qubit)

    # combine steps of algorithm into one program
    p_gradient = p_ic + p_kickback

    # measure input qubits
    for q_measure in input_qubits:
        p_gradient.measure(q_measure, q_measure)
        
    return perturbation_sign, p_gradient

def estimate_gradient(f_h, precision, qvm=False, n_measurements=500):
    """ Estimate the gradient from point of evaluation
        to point of perturbation, h

    :param np.array f: Oracle output at perturbation h.
    :param int precision: Bit precision of gradient.
    :param int n_measurements: Number of times to measure system.
    """

    # enumerate input and ancilla qubits
    input_qubits = list(range(precision))
    ancilla_qubit = precision
    
    # generate gradient program
    perturbation_sign, p_gradient = gradient_estimator(f_h, input_qubits, ancilla_qubit)
    
    # run gradient program
    if not qvm:
        from pyquil.api import SyncConnection
        qvm = SyncConnection()
    measurement = qvm.run(p_gradient, input_qubits, n_measurements)
    measurements = np.array(measurement)

    # summarize measurements
    stats = measurements.sum(axis=0) / len(measurements)
    bf_estimate = perturbation_sign * stats_to_bf(stats)
    bf_explicit = '{0:.16f}'.format(bf_estimate)
    deci_estimate = binary_to_real(bf_explicit)
        
    return deci_estimate    
