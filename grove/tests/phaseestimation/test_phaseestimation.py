import numpy as np

from pyquil.gates import H, CPHASE, SWAP, MEASURE
import pyquil.quil as pq

from grove.qft.fourier import inverse_qft
from grove.alpha.phaseestimation.phase_estimation import controlled
from grove.alpha.phaseestimation.phase_estimation import phase_estimation


def test_phase_estimation():
    phase = 0.75
    precision = 4
    
    phase_factor = np.exp(1.0j * 2 * np.pi * phase)
    U = np.array([[phase_factor, 0],
                  [0, -1*phase_factor]])
    
    trial_prog = phase_estimation(U, precision)
    
    result_prog = pq.Program([H(i) for i in range(precision)])
    
    q_out = range(precision, precision+1)
    for i in range(precision):
        if i > 0:
            U = np.dot(U, U)
        cU = controlled(U)
        name = "CONTROLLED-U{0}".format(2 ** i)
        result_prog.defgate(name, cU)
        result_prog.inst((name, i) + tuple(q_out))
    
    result_prog += inverse_qft(range(precision))
    
    result_prog += [MEASURE(i, [i]) for i in range(precision)]
    
    assert(trial_prog == result_prog)
