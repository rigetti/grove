from __future__ import division

import numpy as np
import pyquil.quil as pq
from pyquil.gates import X, H
from grove.alpha.phaseestimation.phase_estimation import phase_estimation

from grove.alpha.jordan_gradient.gradient_utils import binary_to_real, \
    measurements_to_bf

def initialize_system(ancilla_qubit):
    """Prepare initial state

    :param list ancilla_qubit: Qubit of ancilla register.
    :return: Quil program to initialize this system.
    :rtype: Program
    """

    # ancilla qubit to plane wave state
    ic_ancilla = pq.Program([X(ancilla_qubit), H(ancilla_qubit)])
    p_ic = pq.Program(ic_ancilla)

    return p_ic

def phase_kickback(f_h, precision):
    """Encode f_h into ancilla eigenvalue and kickback to input registers

    :param float f_h: Oracle output at perturbation h.
    :param int precision: Bit precision of gradient.
    :return: Quil program to perform phase kickback.
    :rtype: Program
    """

    # encode f_h / 2 into CPHASE gate
    U = np.array([[np.exp(-1.0j * np.pi * f_h), 0],
                  [0, np.exp(-1.0j * np.pi * f_h)]])
    p_kickback = phase_estimation(U, precision)

    return p_kickback

def gradient_estimator(f_h, ancilla_qubit):
    """Gradient estimation via Jordan's algorithm
    10.1103/PhysRevLett.95.050501

    :param float f_h: Oracle output at perturbation h.
    :param list ancilla_qubits: Qubits of output register.
    :return: Quil program to estimate gradient of f.
    :rtype: Program
    """

    # intialize input and output registers
    p_ic = initialize_system(ancilla_qubit)

    # encode oracle values into phase
    p_kickback = phase_kickback(abs(f_h), ancilla_qubit)

    # combine steps of algorithm into one program
    p_gradient = p_ic + p_kickback

    return p_gradient

def estimate_gradient(f_h, precision, n_measurements=50, cxn=False):
    """Estimate the gradient using function evaluation at perturbation, h

    :param float f_h: Oracle output at perturbation h.
    :param int precision: Bit precision of gradient.
    :param int n_measurements: Number of times to measure system.
    :param Connection cxn: connection to the QPU or QVM
    :return: Decimal estimate of gradient.
    :rtype: float
    """

    # enumerate input and ancilla qubits
    input_qubits = list(range(precision))
    ancilla_qubit = precision

    # generate gradient program
    perturbation_sign = np.sign(f_h)
    p_gradient = gradient_estimator(f_h, ancilla_qubit)

    # run gradient program
    if not cxn:
        from pyquil.api import QVMConnection
        cxn = QVMConnection()
    measurements = cxn.run(p_gradient, input_qubits, n_measurements)
    if isinstance(measurements, str):
        raise ValueError(measurements)

    # summarize measurements
    bf_estimate = perturbation_sign * measurements_to_bf(measurements)
    bf_explicit = '{0:.16f}'.format(bf_estimate)
    deci_estimate = binary_to_real(bf_explicit)

    return deci_estimate
