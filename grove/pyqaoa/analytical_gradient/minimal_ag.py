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

    qvm = api.SyncConnection()

    #########################################
    #Constructs the cost hamiltonian program#
    #########################################
    cost_hamiltonian_program = pq.Program()

    #Cost Hamiltonian
    cost_hamiltonian_program.inst(Z(0))
    cost_hamiltonian_program.inst(Z(1))
    #Ancilla Hamiltonian
    cost_hamiltonian_program.inst(Z(2))

    gamma, beta = 1.2, 1.3

    ######################################
    #Constructs the cost gradient program#
    ######################################
    cost_gradient_program = pq.Program()

    #Reference State Prep
    cost_gradient_program.inst(H(0))
    cost_gradient_program.inst(H(1))
    #Ancilla Prep
    cost_gradient_program.inst(H(2))

    #Z(0) and Z(1) gates become CZ(2,0) and CZ(2,1) gates respectively
    cost_gradient_program.inst(CPHASE(np.pi)(2,0))
    cost_gradient_program.inst(CPHASE(np.pi)(2,1))

    #The exponentiated Z(0)Z(1) term
    cost_gradient_program.inst(CNOT(0,1))
    cost_gradient_program.inst(RZ(gamma)(0))
    cost_gradient_program.inst(CNOT(0,1))

    #The exponentiated X(0) Term
    cost_gradient_program.inst(H(0))
    cost_gradient_program.inst(RZ(beta)(0))
    cost_gradient_program.inst(H(0))

    #The exponentiated X(1) Term
    cost_gradient_program.inst(H(1))
    cost_gradient_program.inst(RZ(beta)(1))
    cost_gradient_program.inst(H(1))

    #The final ancilla Operations
    cost_gradient_program.inst(S(2))
    cost_gradient_program.inst(H(2))

    print(cost_gradient_program)

    expectation = qvm.expectation(cost_gradient_program,
            operator_programs=[cost_hamiltonian_program])[0]
    print(expectation)

    #########################################
    #Constructs the driver gradient programs#
    #########################################
    driver_gradient_program_0 = pq.Program()

    #Reference State Prep
    driver_gradient_program_0.inst(H(0))
    driver_gradient_program_0.inst(H(1))
    #Ancilla Prep
    driver_gradient_program_0.inst(H(2))

    #The exponentiated Z(0)Z(1) term
    driver_gradient_program_0.inst(CNOT(0,1))
    driver_gradient_program_0.inst(RZ(gamma)(1))
    driver_gradient_program_0.inst(CNOT(0,1))

    #X(0) and X(1) gates become CNOT(2,0) and CNOT(2,1) gates respectively
    driver_gradient_program_0.inst(CNOT(2,0))
    #driver_gradient_program_0.inst(CNOT(2,1))

    #The exponentiated X(0) Term
    driver_gradient_program_0.inst(H(0))
    driver_gradient_program_0.inst(RZ(beta)(0))
    driver_gradient_program_0.inst(H(0))

    #The exponentiated X(1) Term
    driver_gradient_program_0.inst(H(1))
    driver_gradient_program_0.inst(RZ(beta)(1))
    driver_gradient_program_0.inst(H(1))

    #The final entanling ancilla operations
    driver_gradient_program_0.inst(S(2))
    driver_gradient_program_0.inst(H(2))

    #print(driver_gradient_program_0)
    #print(cost_hamiltonian_program_0)


    #print(np.cos(2*beta)*np.sin(gamma))

    #expectation = qvm.expectation(driver_gradient_program_0,
    #        operator_program_0s=[cost_hamiltonian_program])[0]
    #print(expectation)
