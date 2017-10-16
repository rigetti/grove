from __future__ import division
import numpy as np
from pyquil.gates import X, H, CPHASE
import pyquil.quil as pq
from grove.qft.fourier import qft, inverse_qft

def real_to_binary(number, precision=16):
    """ Convert real number into a binary fraction

    :param float number: Number to convert.
    :param int precision: Precision of binary fraction.
    :return str out: Precision-bit binary fraction. 
    """

    out = ''
    for val in range(precision):
        number = 2 * (number % 1)
        out += str(int(number))
    return out

def initialize_system(d_i, precision_i, precision_o):
    """ Prepare program initial conditions

    Input qubits to equal superposition
    Output qubits to plane wave state

    :param int d_i: Number of dimensions of function domain.
    :param int precision_i: Bit precision of input qubits.
    :param int precision_o: Bit precision of output qubits.
    :return list ics: List of gates needed to prepare IC state.
    """

    N_qi = d_i * precision_i
    input_qubits = list(range(N_qi))
    ic_in = list(map(H, input_qubits))
    ancilla_qubits = list(range(N_qi, N_qi + precision_o))
    ic_out = list(map(X, ancilla_qubits))
    ft_out = qft(ancilla_qubits)
    ics = pq.Program(ic_in + ic_out) + ft_out
    return ics, input_qubits, ancilla_qubits

def oracle(f, x, eval_ndx, qubits, ancilla, precision, eval_shift):    
    """ Phase kickback of gradient values

    :param np.array f: Oracle outputs.
    :param np.array x: Domain of f.
    :param int eval_ndx: Index of domain value to shift over linear regime.
    :param list qubits: Indices of input qubits.
    :param list ancilla: Indices of ancilla qubits.
    :param int precision: Bit precision of gradient.
    :param int eval_shift: Number indicies over which function is linear.
    :return Program p_cR: Quil program that encodes gradient values via cRz.
    """

    N_q = len(qubits)
    dx = x[eval_ndx+eval_shift] - x[eval_ndx]
    y_1 = f[eval_ndx+eval_shift] - y[eval_ndx]
    scale = real_to_binary(y_1 / dx, precision=precision)
    cR = []
    for bit_ndx, a_bit in enumerate(ancilla):
        angle = np.pi * int(scale[bit_ndx]) # 2**(1+bit_ndx-precision)
        gate = CPHASE(angle)(qubits[-1*(1+bit_ndx)], a_bit)
        cR.append(gate)
    p_cR = pq.Program(cR)
    return p_cR

def gradient_estimator(f, x, eval_ndx, precision=16, eval_shift=1):
    """ Gradient estimation via Jordan's algorithm
    10.1103/PhysRevLett.95.050501

    :param np.array f: Oracle outputs.
    :param np.array x: Domain of f.
    :param int eval_ndx: Index of domain value to shift over linear regime.
    :param int precision: Bit precision of gradient.
    :param int eval_shift: Number indicies over which function is linear.
    :return Program p_gradient: Quil program to estimate gradient of f.
    """

    d = f.ndim
    # initialize registers
    p_ic, q_i, q_a = initialize_system(d, precision, precision)
    # feed function and circuit into oracle
    p_oracle = oracle(f, x, eval_ndx, q_i, q_a, precision, eval_shift)
    # qft result
    p_iqft = inverse_qft(q_i)
    # combine steps of algorithm into one program
    p_gradient = p_ic + p_oracle + p_iqft
    return p_gradient

if __name__ == '__main__':
    from pyquil.api import SyncConnection
    qvm = SyncConnection()
    
    x = np.linspace(0, .1, 100)
    # test function with analytic gradient 0.011
    y = .375*x 
    
    p_eval = 0
    eval_shift = 99
    precision = 3
    ca = list(range(precision))
    p_gradient = gradient_estimator(y, x, p_eval, precision=precision,
            eval_shift=eval_shift)
    measurements = []
    for m in range(1000):
        measurements.append(qvm.run_and_measure(p_gradient, ca)[0])
    m = np.vstack(measurements)
    
    probability_m = m.sum(axis=0)[::-1] / m.shape[0]
    print (probability_m)
    probability_m[probability_m < .5] = 0
    probability_m[probability_m > 0] = 1
    estimate = ''.join(str(int(np.round(i))) for i in probability_m)
    print (estimate)
