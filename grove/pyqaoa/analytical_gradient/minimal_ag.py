"""
This module assumes that p=1

Unitary Structure:
{1:
    {'cost':
        [(0, <Gate: CNOT 0 1>),
         (0, <Gate: RZ(1.5707963267948966) 1>),
         (0, <Gate: CNOT 0 1>)],
     'driver': [(0, <Gate: H 0>),
                (0, <Gate: RZ(1.5707963267948966) 0>),
                (0, <Gate: H 0>), (0, <Gate: H 1>),
                (0, <Gate: RZ(1.5707963267948966) 1>),
                (0, <Gate: H 1>)]
    }
}

Hermitian  Structure:
{1:
    {'cost':
        [(0, <Gate: Z 0>),
         (0, <Gate: Z 1>)],
     'driver':
        [(0, <Gate: X 0>),
         (0, <Gate: X 1>)]
    }
}

Cost Gradient Program:

H 0
H 1
H 2
CPHASE(3.141592653589793) 2 0
CPHASE(3.141592653589793) 2 1
CNOT 0 1
RZ(2.6) 1
CNOT 0 1
H 0
RZ(2.4) 0
H 0
H 1
RZ(2.4) 1
H 1
S 2
H 2

"""
import numpy as np
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *

if __name__ == "__main__":
    """Constructs a minimal ag implementation
    """

    gamma, beta = 1.2, 1.3

    #Constructs the cost gradient
    cost_gradient_program = pq.Program()

    cost_gradient_program.inst(H(0))
    cost_gradient_program.inst(H(1))
    cost_gradient_program.inst(H(2))

    cost_gradient_program.inst(CPHASE(np.pi)(2,0))
    cost_gradient_program.inst(CPHASE(np.pi)(2,1))

    cost_gradient_program.inst(CNOT(0,1))
    cost_gradient_program.inst(RZ(gamma)(1))
    cost_gradient_program.inst(CNOT(0,1))

    cost_gradient_program.inst(H(0))
    cost_gradient_program.inst(RZ(beta)(0))
    cost_gradient_program.inst(H(0))

    cost_gradient_program.inst(H(1))
    cost_gradient_program.inst(RZ(beta)(1))
    cost_gradient_program.inst(H(1))

    cost_gradient_program.inst(S(2))
    cost_gradient_program.inst(H(2))
