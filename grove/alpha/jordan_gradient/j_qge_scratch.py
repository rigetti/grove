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

def initialize_system(input_qubits, ancilla_qubits):
    """ Prepare program initial conditions

    Input qubits to equal superposition
    Output qubits to plane wave state

    :param list input_qubit: Qubits of input registers.
    :param list ancilla_qubits: Qubits of output register.
    :return Program p_ic: Quil program to initialize this state.
    """

    ic_out = list(map(X, ancilla_qubits))
    ft_out = qft(ancilla_qubits)
    p_ic_out = pq.Program(ic_out) + ft_out
    
    ic_in = list(map(H, input_qubits))
    p_ic_in = pq.Program(ic_in)
    
    p_ic = p_ic_out + p_ic_in
    return p_ic

def phase_kickback(f, x_eval, dx, precision, input_qubits, ancilla_qubits):
    """ Phase kickback of gradient values

    :param np.array f: Oracle outputs.
    :param int x_eval: Point at which the gradient is to be estimated.
    :param int dx: Domain shift over which f is approximately linear.
    :param int precision: Bit precision of gradient.
    :param list input_qubit: Qubits of input registers.
    :param list ancilla_qubits: Qubits of output register.
    :return Program p_cR: Quil program to encode gradient values via cRz.
    """

    y_1 = f[x_eval+dx] - f[x_eval]
    scale = real_to_binary(y_1 / dx, precision=precision)
    
    cR = []
    for bit_ndx, a_bit in enumerate(ancilla_qubits):
        angle = np.pi * int(scale[bit_ndx])
        gate = CPHASE(angle)(input_qubits[-1*(1+bit_ndx)], a_bit)
        cR.append(gate)
    
    p_kickback = pq.Program(cR) + inverse_qft(input_qubits)
    return p_kickback

def gradient_estimator(f, x_eval, dx, precision=16):
    """ Gradient estimation via Jordan's algorithm
    10.1103/PhysRevLett.95.050501

    :param np.array f: Oracle outputs.
    :param int x_eval: Point at which the gradient is to be estimated.
    :param int dx: Domain shift over which f is approximately linear.
    :param int precision: Bit precision of gradient.
    :return Program p_gradient: Quil program to estimate gradient of f.
    """
    
    # enumerate registers
    d = f.ndim
    N_qi = d * precision
    input_qubits = list(range(N_qi))
    ancilla_qubits = list(range(N_qi, N_qi + precision))
    
    # intialize input and output registers
    p_ic = initialize_system(input_qubits, ancilla_qubits)
    
    # encode oracle values into phase
    p_kickback = phase_kickback(f, x_eval, dx, precision, input_qubits,
            ancilla_qubits)
    
    # combine steps of algorithm into one program
    p_gradient = p_ic + p_kickback
    return p_gradient

if __name__ == '__main__':
    


    # test function with analytic gradient 0.01
    import numpy as np
    x = np.linspace(0, .1, 100)
    y = .25*x
    
    x_eval = 0
    dx = 4
    precision = 2

    from jordan_gradient import gradient_estimator
    p_gradient = gradient_estimator(y, x, x_eval, precision=precision,
            eval_shift=eval_shift)
            
    from pyquil.api import SyncConnection
    qvm = SyncConnection()
    ca = list(range(precision))
    measurement = qvm.run(p_gradient, ca)
    
    
    
    
    
    m = np.vstack(measurements)
    probability_m = np.round(m.sum(axis=0)[::-1] / m.shape[0])
    estimate = ''.join(str(int(i)) for i in probability_m)
    print (estimate)
