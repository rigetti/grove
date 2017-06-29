'''
An implementation of the HHL algorithm from Harrow, Hassidim, and Lloyd
Original paper: https://arxiv.org/abs/0811.3171 (1)
Useful applications: https://cstheory.stackexchange.com/questions/25306/applications-of-hhls-algorithm-for-solving-linear-equations

"A quantum algorithm to estimate features of the solution of a set of linear equations"

Development notes:
    - ill-conditioned matrices can cause issues during this algorithm
    - The strength of the algorithm is that it works only with O(log N)-qubit registers, and never has to write down all
    of A, ~b or ~x.

'''

import pyquil.quil as pq
import numpy as np

def HHL(A, b_regs):
    '''
    Returns a program that accomplishes the HHL algorithm using the provided
    matrix A and vector b, using the given qubit working space
    
    :param A: A numpy matrix to be used in solved A|x> = |b>
    :param b_regs: A list of registers holding the value of |b>
    :return: A Program that performs the HHL algorithm
    '''
    
    p = pq.Program()
    
    # If A is not Hermitian, use definition 1 from (1) to create a Hermitian
    # matrix, keeping track to make a reduction later in the algorithm
    hermitian = np.equal(A, A.getH())
    if !hermitian:
        raise ValueError("Not yet excepting Hermitian matrices")
    
    # Create a unitary operator e^(iAt) using arXiv:quant-ph/0508139
    unitary_program = eiat(A)
    
    # Prepare state |b> using either a pre-computed state on a given register,
    # or by constructing it via arXiv:quant-ph/0208112
    
    # Decompose |b> in the eigenvector basis using phase estimation
    # A parameter 'T' can be set by the user to change the error bounds
    
    # Apply the conditional Hamiltonian evolution
    
    # Fourier transform the first register
    
    # Add ancilla qubit and perform conditional rotations
    
    # Undo the phase estimation
    
    # Measure the last qubit. Conditioned on getting |1> back, we have
    # |x> with some normalization required