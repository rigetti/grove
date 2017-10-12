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

def oracle(f, x, eval_ndx, qubits, ancilla, precision, eval_shift):    
    N_q = len(qubits)
    dx = x[eval_ndx+eval_shift] - x[eval_ndx]
    y_1 = f[eval_ndx+eval_shift]
    scale = real_to_binary(y_1 / dx, precision=precision)
    cR = []
    for register, a_bit in enumerate(ancilla):
        angle = int(scale[register]) # 2 * np.pi * 
        cR.append(CPHASE(angle)(qubits[N_q - 1 - register], a_bit))
    return cR

def gradient_estimator(function, x, p_ev, precision=16, eval_shift=1):
    d = function.ndim

    # initialize registers
    ic, q_i, q_a = initialize_system(d, precision, precision)

    # feed function and circuit into oracle
    p_oracle = oracle(function, x, p_ev, q_i, q_a, precision, eval_shift)

    # qft result
    p_iqft = inverse_qft(q_i)

    # combine steps of algorithm into one program
    p_gradient = pq.Program(ic + p_oracle) + p_iqft

    # measure input registers
    for q in q_i:
       p_gradient.measure(q, q)

    return p_gradient


import IPython
from pyquil.api import JobConnection
job_qvm = JobConnection(endpoint="https://job.rigetti.com/beta")

from pyquil.api import SyncConnection
qvm = SyncConnection()

x = np.linspace(0, .1, 100)
y = 1.2*x

p_eval = 0
eval_shift = 99
precision = 8
p_g = gradient_estimator(y, x, p_eval, precision=precision,
        eval_shift=eval_shift)

ca = list(range(precision))

IPython.embed()

