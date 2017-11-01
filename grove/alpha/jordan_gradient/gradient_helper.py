import numpy as np
from jordan_gradient import gradient_estimator

def real_to_binary(number, precision=16):
    """ Convert real decimal to precision-bit binary fraction
    
    :param float number: Real decimal over (0, 1].
    :param int precision: Number of bits of binary precision.
    :return float bf: Binary fraction representation of real decimal.
    """
    
    bf = ''
    for val in range(precision):
        number = 2 * (number % 1)
        bf += str(int(number))
    bf = float('.' + bf)
    
    return bf

def binary_to_real(number):
    """ Convert binary fraction to real decimal
    
    :param float number: Floating point representation of binary fraction.
    :return float deci: Real decimal representation of binary fraction.
    """    
    
    deci = 0
    for ndx, val in enumerate(str(number).split('.')[-1]):
        deci += float(val) / 2**(ndx+1)
        
    return deci

def stats_to_bf(stats):
    """ Convert measurement into gradient binary fraction
    
    :param np.array stats: Output measurement statistics of gradient program.
    :return float bf: Binary fraction representation of gradient estimate.
    """
    
    stats_str = [str(int(i)) for i in np.ceil(stats[::-1][1:])]
    bf_str = '0.' + ''.join(stats_str)
    bf = float(bf_str)
    
    return bf

def gradient_error(f_h, precision=5, n_measurements=100):
    """ Computes error of gradient estimates for an input perturbation value
    
    :param np.array/float f_h: Value of f at perturbation h.
    :param int n_measurements: Number of times to run the gradient program.
    :return float error: Error of gradient estimate.
    """

    if isinstance(f_h, float):
        d = 1 # f_h = np.array(f_h)
    else:
        d = f_h.ndim

    # enumerate qubit register
    N_qi = d * precision
    input_qubits = list(range(N_qi))
    ancilla_qubits = [N_qi]

    # build program and run n_measurements times
    p_g = gradient_estimator(f_h, input_qubits, ancilla_qubits, precision)

    from pyquil.api import SyncConnection
    qvm = SyncConnection()
    measurements = np.array(qvm.run(p_g, input_qubits, n_measurements))

    # summarize measurements and compute error
    stats = measurements.sum(axis=0) / len(measurements)
    bf_estimate = stats_to_bf(stats)
    deci_estimate = binary_to_real(bf_estimate)
    error = f_h - deci_estimate
    
    return error
