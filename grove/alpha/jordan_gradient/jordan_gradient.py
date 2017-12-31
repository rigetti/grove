from __future__ import division

import numpy as np
from grove.alpha.phaseestimation.phase_estimation import phase_estimation

from grove.alpha.jordan_gradient.gradient_utils import binary_to_real, \
    measurements_to_bf


def gradient_program(f_h, precision):
    """Gradient estimation via Jordan's algorithm
    10.1103/PhysRevLett.95.050501

    :param float f_h: Oracle output at perturbation h.
    :param list ancilla_qubits: Qubits of output register.
    :return: Quil program to estimate gradient of f.
    :rtype: Program
    """

    # encode oracle values into phase
    phase_factor = np.exp(-1.0j * 2 * np.pi * abs(f_h))
    U = np.array([[phase_factor, 0],
                  [0, phase_factor]])
    p_gradient = phase_estimation(U, precision)

    return p_gradient


def estimate_gradient(f_h, precision, gradient_max=1, n_measurements=50, 
                      cxn=None):
    """Estimate the gradient using function evaluation at perturbation, h

    :param float f_h: Oracle output at perturbation h.
    :param int precision: Bit precision of gradient.
    :param int gradient_max: OOM estimate of largest gradient value
    :param int n_measurements: Number of times to measure system.
    :param Connection cxn: Connection to the QPU or QVM.
    :return: Decimal estimate of gradient.
    :rtype: float
    """

    # scale f_h by range of values gradient can take on
    f_h *= 1. / gradient_max

    # generate gradient program
    perturbation_sign = np.sign(f_h)
    p_gradient = gradient_program(f_h, precision)

    # run gradient program
    if cxn is None:
        from pyquil.api import QVMConnection
        cxn = QVMConnection()
    measured_qubits = list(range(precision + 1))
    measurements = cxn.run(p_gradient, measured_qubits, n_measurements)

    # summarize measurements
    bf_estimate = perturbation_sign * measurements_to_bf(measurements)
    bf_explicit = '{0:.16f}'.format(bf_estimate)
    deci_estimate = binary_to_real(bf_explicit)
    
    # rescale gradient
    deci_estimate *= gradient_max

    return deci_estimate
