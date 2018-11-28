import numpy as np
from pyquil import Program
from pyquil.api import QuantumComputer, get_qc

from grove.alpha.jordan_gradient.gradient_utils import (binary_float_to_decimal_float,
                                                        measurements_to_bf)
from grove.alpha.phaseestimation.phase_estimation import phase_estimation


def gradient_program(f_h: float, precision: int) -> Program:
    """
    Gradient estimation via Jordan's algorithm (10.1103/PhysRevLett.95.050501).

    :param f_h: Oracle output at perturbation h.
    :param precision: Bit precision of gradient.
    :return: Quil program to estimate gradient of f.
    """

    # encode oracle values into phase
    phase_factor = np.exp(1.0j * 2 * np.pi * abs(f_h))
    U = np.array([[phase_factor, 0],
                  [0, phase_factor]])
    p_gradient = phase_estimation(U, precision)

    return p_gradient


def estimate_gradient(f_h: float, precision: int,
                      gradient_max: int = 1,
                      n_measurements: int = 50,
                      qc: QuantumComputer = None) -> float:
    """
    Estimate the gradient using function evaluation at perturbation, h.

    :param f_h: Oracle output at perturbation h.
    :param precision: Bit precision of gradient.
    :param gradient_max: OOM estimate of largest gradient value.
    :param n_measurements: Number of times to measure system.
    :param qc: The QuantumComputer object.
    :return: Decimal estimate of gradient.
    """

    # scale f_h by range of values gradient can take on
    f_h *= 1. / gradient_max

    # generate gradient program
    perturbation_sign = np.sign(f_h)
    p_gradient = gradient_program(f_h, precision)

    # run gradient program
    if qc is None:
        qc = get_qc(f"{len(p_gradient.get_qubits())}q-qvm")

    p_gradient.wrap_in_numshots_loop(n_measurements)
    executable = qc.compiler.native_quil_to_executable(p_gradient)
    measurements = qc.run(executable)

    # summarize measurements
    bf_estimate = perturbation_sign * measurements_to_bf(measurements)
    bf_explicit = '{0:.16f}'.format(bf_estimate)
    deci_estimate = binary_float_to_decimal_float(bf_explicit)
    
    # rescale gradient
    deci_estimate *= gradient_max

    return deci_estimate
