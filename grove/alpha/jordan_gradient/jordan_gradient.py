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
    :param int precision_i: Precision of input qubits.
    :param int precision_o: Precision of output qubits.
    :return list ics: list of gates needed to prepare IC state 
    """

    N_qi = d_i * precision_i
    input_qubits = list(range(N_qi))
    ic_in = list(map(H, input_qubits))
    ancilla_qubits = list(range(N_qi, N_qi + precision_o))
    ic_out = list(map(X, ancilla_qubits))
    ft_out = qft(ancilla_qubits)
    ics = pq.Program(ic_in + ic_out) + ft_out
    return ics, input_qubits, ancilla_qubits

def oracle(f, x, eval_ndx, qubits, ancilla):    
    N_q = len(qubits)
    dx = 2*(x[eval_ndx+1] - x[eval_ndx])
    y_1 = f[eval_ndx+1]
    scale = real_to_binary(y_1 / dx)
    cR = []
    for register, a_bit in enumerate(ancilla):
        angle = 2 * np.pi * int(scale[register])
        cR.append(CPHASE(angle)(qubits[N_q - 1 - register], a_bit))
    return cR

def gradient_estimator(function, x, p_ev, precision=3):
    d = function.ndim

    # initialize registers
    ic, q_i, q_a = initialize_system(d, precision, precision)

    # feed function and circuit into oracle
    p_oracle = oracle(function, x, p_ev, q_i, q_a)

    # qft result
    p_iqft = inverse_qft(q_i)

    # combine steps of algorithm into one program
    p_gradient = pq.Program(ic + p_oracle) + p_iqft

    # measure output registers
    for q in q_i:
        p_gradient.measure(q, q)

    return p_gradient

import pyquil.api as api
qvm = api.SyncConnection()

x = np.linspace(0, .1, 100)
y = 1.2*x

p_eval = 0
precision = 12
p_g = gradient_estimator(y, x, p_eval, precision=precision)

import IPython
IPython.embed()

# wf = qvm.wavefunction(p_g)[0]
# print (wf)
