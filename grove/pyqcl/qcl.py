##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

import itertools
import numpy as np
from scipy import linalg

import pyquil.api as api
import pyquil.quil as pq
from pyquil.gates import RX, RY, RZ

from grove.pyqcl.optimizer import GradientOptimizer

class QCL(object):
    """
    The Quantum Circuit Learning algorithm

    QCL is an object that encapsulates the Quantum Circuit Learning algorithm -
    a classical-quantum hybrid algorithm for machine learning on near-term 
    quantum processors. The main components of the QCL algorithm are options for 
    gradient descend options performing the loss function minimization,
    a unitary encoding input data, a unitary generating output state and 
    operator programs on which expectation values are measured in the output state.

    Using this object:

        1) initilze with `inst = QCL(state_generators, operator_programs, **params)

        2) call `inst.fit(X, y)` to fit QCL to input data 
    
        3) call `inst.predict(X)` to get predictions of fitted QCL

    :param state_generators: (dict) Dictionary with pyQuil programs generating
        input state, output state and gradient states.
    :param operator_programs: (list) A list of Programs, each specifying an 
        operator whose expectation to compute. Default is a list containing 
        only the empty Program.
    :param qvm: (optional, QVM) forest connection object.
    :param initial_theta: (ndarray) initial parameters for optimization.
    :param loss: (string) loss function definition.
    :params learning_rate: (float) learning rate for gradient descend method.
                           Default=1.0
    :params epochs: (int) number of gradient descend epochs. Default=1 
    :params batch_size: (int) batch size. Default=None
    :params verbose: (bool) Print intermediate results. Default=False
                                   
    """

    def __init__(self, state_generators, initial_theta, loss, operator_programs=None, 
                 learning_rate=1.0, epochs=1, batch_size=None, qvm=None, verbose=False): 
                 
        self.loss_mse = 'mean_squared_error'
        self.loss_entropy = 'binary_crossentropy'
        
        self.initial_theta = initial_theta
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.state_generators = state_generators
        if 'input' not in self.state_generators.keys():
            raise ValueError("Provide pyQuil program responsible for " + 
                             "encoding X into quantum state!")
        if 'output' not in self.state_generators.keys():
            raise ValueError("Provide pyQuil program responsible for " + 
                             "genering X output state!")
        if 'grad' not in self.state_generators.keys():
            raise ValueError("Provide pyQuil program responsible for " + 
                             "calculating a gradient!")
           
        self.operator_programs = operator_programs

        if qvm is None:
            self.qvm = api.QVMConnection()
        else:
            self.qvm = qvm
        
        self.verbose = verbose
        
        self.fit_OK = False
    
    def fit(self, X, y):
        """
        Fit QCL.
        
        :param X: (ndarray) Training data of shape (n_samples,n_features).
        :param y: (ndarray) Training labels of shape (n_samples,n_classes) for 
            classification task and (n_samples,) for regression task.  

        :return self: returns an instance of self.
        """
    
        X, y = self._make_data(X, y)  
        
        optimizer = GradientOptimizer(self.initial_theta, self.loss, 
                                      self.learning_rate, self.epochs, 
                                      self.batch_size, self.verbose)            
        self.results = optimizer.gradient_descend(X, y, self.state_generators, 
                                                  self.operator_programs, self.qvm)
        
        self.fit_OK = True
        
        return self

    def predict(self, X):
        """
        Predict QCL.
        
        :param X: (ndarray) Testing data of shape (n_samples,n_features). 

        :return y_pred: (ndarray) Predictions of shape (n_samples,n_classes) for 
            classification task and (n_samples,) for regression task. 
        """        
        if not self.fit_OK:
            raise ValueError("The QCL instance must be fitted before.")   
                
        X = self._make_data(X) 
        n_samples = len(X)
        
        # Check operators
        if not isinstance(self.operator_programs, list):
            operator_programs = [self.operator_programs]
        else:
            operator_programs = self.operator_programs
        n_operators = len(self.operator_programs)
        
        prog_input_gen = self.state_generators['input']
        prog_output_gen = self.state_generators['output']
        
        y_pred = np.zeros((n_samples, n_operators))
        for k in range(n_samples):
            prog = prog_input_gen(X[k,:])
            prog += prog_output_gen(self.results.theta)
            y_pred[k,:] = self.results.coeff * np.array(self.qvm.expectation(prog, operator_programs))
            if self.loss == self.loss_entropy:
                y_pred[k,:] = np.exp(y_pred[k,:]) / np.sum(np.exp(y_pred[k,:])) 

        return y_pred
                    
    def get_results(self):
        """
        Extract fitting results.

        :return (qcl.OptResult()) object :func:`OptResult <vqe.OptResult>`.
                 The following fields are initialized in OptResult:
                 -theta: set of optimized parameters
                 -coeff: scalar value of optimized mulitiplicative coefficient. 
                 -history_theta: a list of all intermediate parameter vectors.
                 -history_loss: a list of all intermediate losses.
                 -history_grad: a list of all intermediate gradient arrays.
        """ 
        if not self.fit_OK:
            raise ValueError("The QCL instance must be fitted before.")
        
        return self.results
        
    def _make_data(self, X, y=None):
        if np.ndim(X) == 1:
            X = X.reshape(-1,1)
            
        if y is not None:
            if np.ndim(y) == 1:
                y = y.reshape(-1,1)
            if self.loss == self.loss_entropy and y.shape[1] == 1:
                y = np.c_[y, 1-y]
            return X, y
        else:
            return X
        
def default_input_state_gen(n_qubits):
    """
    Generates Quil program encoding input data in a way described in 
    https://arxiv.org/abs/1803.00745
    
    :param n_qubits: (int) number of qubits in a circuit. 

    :return (Program) Quil program for input data encoding 
    """ 
    def prog(sample):
        p = pq.Program()
        n_features = len(sample)
        for j in range(n_qubits):
            p.inst(RY(np.arcsin(sample[j % n_features]), j))
            p.inst(RZ(np.arccos(sample[j % n_features]**2), j))
        return p 
    return prog

def default_output_state_gen(ising_prog, n_qubits, depth):
    """
    Generates Quil program preparing output state in a way described in 
    https://arxiv.org/abs/1803.00745
    
    :param ising_prog: (Program) Program evolving system according to fully 
        connected transverse Ising hamiltionian. 
    :param n_qubits: (int) number of qubits in a circuit. 
    :param depth: (int) depth of a circuit. 
    
    :return (Program) Quil program for preparing output state 
    """
    def prog(theta):
        p = pq.Program()
        theta = theta.reshape(3,n_qubits,depth)
        for i in range(depth):
            p += ising_prog
            for j in range(n_qubits):
                p.inst(RX(theta[0,j,i], j))
                p.inst(RZ(theta[1,j,i], j))
                p.inst(RX(theta[2,j,i], j))
        return p
    return prog

def default_grad_state_gen(ising_prog, n_qubits, depth):
    """
    Generates Quil program preparing a state allowing for calculation of 
    gradients in a way described in https://arxiv.org/abs/1803.00745. Gradients
    are created by inserting R(+/- pi/2) rotations in a chain of unitary 
    transformations.
    
    :param ising_prog: (Program) Program evolving system according to fully 
        connected transverse Ising hamiltionian. 
    :param n_qubits: (int) number of qubits in a circuit. 
    :param depth: (int) depth of a circuit. 
    
    :return (Program) Quil program for preparing a state allowing for gradient 
        calculation.
    """
    def prog(theta, idx, sign):
        theta = theta.reshape(3,n_qubits,depth)
        idx = np.unravel_index(idx, theta.shape)
        p = pq.Program()
        for i in range(depth):
            p += ising_prog
            for j in range(n_qubits):
                p.inst(RX(theta[0,j,i], j))
                if idx == (0,j,i):
                    p.inst(RX(sign*np.pi/2.0, j))
                p.inst(RZ(theta[1,j,i], j))
                if idx == (1,j,i):
                    p.inst(RZ(sign*np.pi/2.0, j))
                p.inst(RX(theta[2,j,i], j))
                if idx == (2,j,i):
                    p.inst(RX(sign*np.pi/2.0, j))
        return p
    return prog  
   
def ising_prog_gen(trotter_steps, T, n_qubits):
    """
    Generates Quil program evolving system according to fully connected
    transverse Ising hamiltionian. The exponential of a sum of non-commuting 
    Pauli operators is generated by Trotter Suzuki approximation of first order 
    (Lie product formula). 
    Important: The generation of matrix exponential could be done only on QVM 
    connection. 
    
    :param trotter_steps: (int) trotter steps. 
    :param T: (int) evolution time. 
    :param n_qubits: (int) number of qubits in a circuit.  
    
    :return (Program) Quil program evolving system according to fully connected
        transverse Ising hamiltionian.
    """
    gate_I = np.eye(2)
    gate_X = np.array([[0.0,1.0],
                       [1.0,0.0]])
    gate_Z = np.array([[1.0,0.0],
                       [0.0,-1.0]])
       
    def multi_kron(*args):
        ret = np.array([[1.0]])
        for q in args:
            ret = np.kron(ret, q)
        return ret
    
    def multi_dot(*args):
        for i, q in enumerate(args):
            if i == 0:
                ret = q
            else:
                ret = np.dot(ret, q)
        return ret
        
    # Initilize coefficients
    h_coeff = np.random.uniform(-1.0, 1.0, size=n_qubits)
    J_coeff = dict()
    for val in itertools.combinations(range(n_qubits),2):
        J_coeff[val] = np.random.uniform(-1.0, 1.0)
       
    # Unitary
    for steps in range(trotter_steps):
        
        non_inter = [linalg.expm(-(1j)*T/trotter_steps*multi_kron(*[h * gate_X if i == j else gate_I for i in range(n_qubits)])) for j, h in enumerate(h_coeff)]
        inter = [linalg.expm(-(1j)*T/trotter_steps*multi_kron(*[J * gate_Z if i == k[0] else gate_Z if i == k[1] else gate_I for i in range(n_qubits)])) for k, J in J_coeff.items()]
        ising_step = multi_dot(*non_inter+inter)
    
        if steps == 0:
            ising_gate = ising_step
        else:
            ising_gate = multi_dot(ising_step, ising_gate) 
            
    ising_name = 'ISING_GATE'
    ising_prog = pq.Program().defgate(ising_name, ising_gate)
    ising_prog.inst(tuple([ising_name] + list(reversed(range(n_qubits)))))
    
    return ising_prog