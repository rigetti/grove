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
import math
import numpy as np

import pyquil.api as api

class OptResults(dict):
    """
    Object for holding theta optimization results from QCL.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
   
class GradientOptimizer():
    """
    Implementation of gradient descent method. 

    :param initial_theta: (ndarray) initial parameters for optimization.
    :param loss: (string) loss function definition.
    :params learning_rate: (float) learning rate for gradient descend method.
                           Default=1.0
    :params epochs: (int) number of gradient descend epochs. Default=1 
    :params batch_size: (int) batch size. Default=None
    :params verbose: (bool) Print intermediate results. Default=False                      
    """
    def __init__(self, initial_theta, loss, learning_rate=1.0, epochs=1, 
                 batch_size=None, verbose=False):
        
        self.loss_mse = 'mean_squared_error'
        self.loss_entropy = 'binary_crossentropy'
        self.available_losses = [self.loss_mse, self.loss_entropy]
        
        self.initial_theta = initial_theta
        
        # Loss function
        self.loss = loss
        if self.loss not in self.available_losses:
            raise ValueError("Available losses are " + self.available_losses)
            
        # Learning rate 
        self.learning_rate = learning_rate
        if not isinstance(self.learning_rate, float) and not isinstance(self.learning_rate, int):
            raise TypeError("learning_rate variable must be a number")
        if isinstance(self.learning_rate, int):
            self.learning_rate = float(self.learning_rate)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate variable must be a postive number")
        
        # Epochs and batch size
        self.epochs = epochs
        if not isinstance(self.epochs, int):
            raise TypeError("epochs variable must be an integer")
        if self.epochs <= 0:
            raise ValueError("epochs variable must be a postive integer")
        self.batch_size = batch_size
        if self.batch_size is not None:
            if not isinstance(self.batch_size, int):
                raise TypeError("batch size variable must be an integer")
            if self.batch_size <= 0:
                raise ValueError("batch size variable must be a postive integer")   
               
        self.verbose = verbose
        
    def gradient_descend(self, X, y, state_generators, operator_programs=None,
                         qvm=None): 
        """
        Perform theta optimization using gradient descend method.
 
        :param X: (ndarray) Training data of shape (n_samples,n_features).
        :param y: (ndarray) Training labels of shape (n_samples,n_classes) for 
            classification task and (n_samples,) for regression task.
        :param state_generators: (dict) Dictionary with pyQuil programs generating
            input state, output state and gradient states.
        :param list operator_programs: A list of Programs, each specifying an 
            operator whose expectation to compute. Default is a list containing 
            only the empty Program.
        :param qvm: (optional, QVM) forest connection object.
                               
        :return (qcl.OptResult()) object :func:`OptResult <vqe.OptResult>`.
                 The following fields are initialized in OptResult:
                 -theta: set of optimized parameters
                 -coeff: scalar value of optimized mulitiplicative coefficient. 
                 -history_theta: a list of all intermediate parameter vectors.
                 -history_loss: a list of all intermediate losses.
                 -history_grad: a list of all intermediate gradient arrays.
        """
        history_theta, history_loss, history_grad = [], [], []
        coeff, theta = 1.0, self.initial_theta
        
        prog_input_gen = state_generators['input']
        prog_output_gen = state_generators['output']
        prog_output_grad = state_generators['grad']
        
        n_samples = len(X)
        n_theta = len(theta)
        
        if qvm is None:
            self.qvm = api.QVMConnection()
        else:
            self.qvm = qvm
            
        # Check operators
        if not isinstance(operator_programs, list):
            operator_programs = [operator_programs]
        n_operators = len(operator_programs)
        
        # Check batch size
        if self.batch_size is None:
            self.batch_size = n_samples
        self.batch_size = min(self.batch_size, n_samples)
        
        # Loop over epochs
        for e in range(self.epochs):  
            
            # Loop over batches
            batches = self.generate_batches(X, y, self.batch_size)
            n_batches = len(batches)
            for i, batch in enumerate(batches):
                
                batch_X, batch_y = batch
                n_samples_in_batch = len(batch_X)
                
                # Predictions
                batch_y_pred = np.zeros((n_samples_in_batch, n_operators))
                for k in range(n_samples_in_batch):
                    prog = prog_input_gen(batch_X[k,:])
                    prog += prog_output_gen(theta)
                    batch_y_pred[k,:] = coeff * np.array(qvm.expectation(prog, operator_programs))
                    if self.loss == self.loss_entropy:
                       batch_y_pred[k,:] = np.exp(batch_y_pred[k,:]) / np.sum(np.exp(batch_y_pred[k,:]))
                    
                # Comput loss
                loss_value = self._compute_loss(batch_y, batch_y_pred)
                
                # Display status
                if self.verbose:
                    print('Epoch: {}/{} ::: Batch: {}/{} ::: Loss: {:.5f}'.format(e+1, self.epochs, i+1, n_batches, loss_value)) 
                
                # Gradient
                if not (e == self.epochs - 1 and i == n_batches - 1):
                    grad = np.zeros((n_samples_in_batch, n_operators, n_theta))
                    for k in range(n_samples_in_batch):
                        
                        # Define input state 
                        prog_input = prog_input_gen(batch_X[k,:])
                        
                        # Caclulate gradient for each theta_j
                        for j in range(n_theta):
                            
                            # Gradient +/- 
                            for sign in [1,-1]:
                                grad_sign = np.zeros(n_operators)
                                grad_progs = prog_output_grad(theta, j, sign)
                                # Generally, the gradient programs could return
                                # a program or list of programs (in case the 
                                # gradient +/- is the sum of expectations)
                                if not isinstance(grad_progs, list):
                                    grad_progs = [grad_progs]
                                for grad_prog in grad_progs:
                                    prog = prog_input
                                    prog += grad_prog
                                    # B_j +/- expectation
                                    grad_sign += np.array(qvm.expectation(prog, operator_programs))
                                # Gradient = (B_j+ - B_j-) / 2
                                grad[k, :, j] += sign / 2.0 * grad_sign
                                
                    # Gradient update
                    grad_full = self._compute_grad_full(batch_y, batch_y_pred, grad)
                    if self.loss == self.loss_mse:
                        grad_full_coeff = -2.0 * np.mean((batch_y - batch_y_pred) * batch_y_pred)
                    
                    # Update theta
                    theta -= self.learning_rate * grad_full
                    if self.loss == self.loss_mse:
                        coeff -= self.learning_rate * grad_full_coeff
                    
                # Append to history
                history_loss.append(loss_value)
                history_theta.append(theta)
                history_grad.append(grad)
            
            # Prepare results
            results = OptResults()
            results.theta, results.coeff = theta, coeff
            results.loss = loss_value
            results.history_loss = history_loss
            results.history_theta = history_theta
            results.history_grad = history_grad
            
        return results
    
    @staticmethod
    def generate_batches(X, y, batch_size):
        """
        Creates a list of shuffled batches from (X, y)
     
        :param X: (ndarray) Training data of shape (n_samples,n_features).
        :param y: (ndarray) Training labels of shape (n_samples,n_classes) for 
            classification task and (n_samples,) for regression task.   
        :param batch_size: (int) batch size
        
        :return batches: returns a list of tuples (batch_X, batch_y)
        """
        m = len(X)
        batches = []
            
        # Shuffle 
        permutation = list(np.random.permutation(m))
        shuff_X = X[permutation,:]
        shuff_y = y[permutation]
    
        # Partition
        num_complete_batches = math.floor(m/batch_size)
        for k in range(0, num_complete_batches):
            batch_X = shuff_X[k*batch_size:(k+1)*batch_size, :]
            batch_y = shuff_y[k*batch_size:(k+1)*batch_size]
            batches.append((batch_X, batch_y))
        
        # End case (last mini-batch < mini_batch_size)
        if m % batch_size != 0:
            batch_X = shuff_X[num_complete_batches*batch_size:m]
            batch_y = shuff_y[num_complete_batches*batch_size:m]
            batches.append((batch_X, batch_y))
        
        return batches 
    
    def _compute_loss(self, y, y_pred):
        if self.loss == self.loss_mse:
            return np.mean((y - y_pred) ** 2)
        elif self.loss == self.loss_entropy:
            return -1.0 * np.mean(np.sum(y * np.log(y_pred), axis=1)) 
        else:
            raise ValueError("Available losses are " + self.available_losses)
            
    def _compute_grad_full(self, y, y_pred, grad):
        if self.loss == self.loss_mse:
            return -2.0 * np.mean((y - y_pred) * grad[:,0,:], axis=0)
        elif self.loss == self.loss_entropy:
            return -1.0 * np.mean((grad[:,0,:] - grad[:,1,:]) * (y[:,0,np.newaxis] - y_pred[:,0,np.newaxis]), axis=0)
        else:
            raise ValueError("Available losses are " + self.available_losses)