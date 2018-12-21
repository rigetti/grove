import numpy as np
from unittest.mock import patch
from pyquil import Program
from pyquil.gates import H, CPHASE, SWAP, MEASURE

from grove.alpha.phaseestimation.phase_estimation import controlled
from grove.alpha.jordan_gradient.jordan_gradient import gradient_program, estimate_gradient


def test_gradient_program():
    f_h = 0.25
    precision = 2
    
    trial_prog = gradient_program(f_h, precision)
    
    result_prog = Program([H(0), H(1)])

    phase_factor = np.exp(1.0j * 2 * np.pi * abs(f_h))
    U = np.array([[phase_factor, 0],
                  [0, phase_factor]])
    q_out = range(precision, precision+1)
    for i in range(precision):
        if i > 0:
            U = np.dot(U, U)
        cU = controlled(U)
        name = "CONTROLLED-U{0}".format(2 ** i)
        result_prog.defgate(name, cU)
        result_prog.inst((name, i) + tuple(q_out))

    result_prog.inst([SWAP(0, 1), H(0), CPHASE(-1.5707963267948966, 0, 1),
                      H(1), MEASURE(0, 0), MEASURE(1, 1)])

    assert(trial_prog == result_prog)


def test_estimate_gradient():
    test_perturbation = .25
    test_precision = 3
    test_measurements = 10

    with patch("pyquil.api.QuantumComputer") as qc:
        qc.run.return_value = np.asarray([[0, 1, 0, 0] for i in range(test_measurements)])

    gradient_estimate = estimate_gradient(test_perturbation, test_precision,
                                          n_measurements=test_measurements,
                                          qc=qc)

    assert(np.isclose(gradient_estimate, test_perturbation))
