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

import pyquil.forest as qvm_module
import pyquil.quil as pq
import numpy as np
import inspect
from pyquil.gates import STANDARD_GATES
from pyquil.paulis import PauliTerm, PauliSum


class OptResults(dict):
    """
    Object for holding optimization results from VQE.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class VQE(object):
    """
    The Variational-Quantum-Eigensolver algorithm

    VQE is an object that encapsulates the VQE algorithm (functional
    minimization). The main components of the VQE algorithm are a minimizer
    function for performing the functional minimization, a function that takes a
    vector of parameters and returns a parameterized Quil program, and a
    Hamiltonian of which to calculate the expectation value.

    Using this object:

        1) initialize with `inst = VQE(minimizer)` where `minimizer` is a
        function that performs a gradient free minization--i.e
        scipy.optimize.minimize(. , ., method='Nelder-Mead')

        2) call `inst.vqe_run(parametric_state_evolve, hamiltonian,
        initial_parameters)`. Returns the optimal parameters and minimum
        expecation

    :param minimizer: function that minimizes objective f(obj, param). For
                      example the function scipy.optimize.minimize() needs
                      at least two parameters, the objective and an initial
                      point for the optimization.  The args for minimizer
                      are the cost function (provided by this class),
                      initial parameters (passed to vqe_run() method, and
                      jacobian (defaulted to None).  kwargs can be passed
                      in below.
    :param minimizer_args: (list) arguments for minimizer function. Default=None
    :param minimizer_kwargs: (dict) arguments for keyword args.
                              Default=None

    """

    def __init__(self, minimizer, minimizer_args=[], minimizer_kwargs={}):
        self.minimizer = minimizer
        self.minimizer_args = minimizer_args
        self.minimizer_kwargs = minimizer_kwargs
        self.n_qubits = None

    def vqe_run(self, parametric_state_evolve, hamiltonian, initial_params,
                gate_noise=None, measurement_noise=None,
                jacobian=None, qvm=None, disp=None, return_all=False):
        """
        functional minimization loop.

        :param parametric_state_evolve: function that takes a set of parameters
                                        and returns a quil program.
        :param hamiltonian: (PauliSum) object representing the hamiltonian of
                            which to take the expectation value.
        :param initial_params: (ndarray) vector of initial parameters for the
                               optimization
        :param gate_noise: list of Px, Py, Pz probabilities of gate being
                           applied to every gate after each get application
        :param measurement_noise: list of Px', Py', Pz' probabilities of a X, Y
                                  or Z being applied before a measurement.
        :param jacobian: (optional) method of generating jacobian for parameters
                         (Default=None).
        :param qvm: (optional, QVM) forest connection object.
        :param disp: (optional, bool) display level. If True then each iteration
                     expectation and parameters are printed at each
                     optimization iteration.
        :param return_all: (optional, bool) request to return all intermediate
                           parameters determined during the optimization.
        :return: (vqe.OptResult()) object :func:`OptResult <vqe.OptResult>`.
                 The following fields are initialized in OptResult:
                 -x: set of w.f. ansatz parameters
                 -fun: scalar value of the objective function
                 -iteration_params: a list of all intermediate parameter vectors. Only
                                    returned if 'return_all=True' is set as a vqe_run()
                                    option.
                 -expectation_vals: a list of all intermediate expectation values. Only
                                    returned if 'return_all=True' is set as a
                                    vqe_run() option.
        """
        disp_fun = disp if disp is not None else lambda x: None
        iteration_params = []
        expectation_vals = []
        if qvm is None:
            qvm = qvm_module.Connection(
                    gate_noise=gate_noise,
                    measurement_noise=measurement_noise)
        else:
            self.qvm = qvm

        def objective_function(params):
            """
            closure representing the functional

            :param params: (ndarray) vector of parameters for generating the
                           the function of the functional.
            :return: (float) expectation value
            """
            quil_prog = parametric_state_evolve(params)
            E = self.expectation(quil_prog, hamiltonian, qvm)
            return E

        def print_current_iter(iter_vars):
            quil_prog = parametric_state_evolve(iter_vars)
            E = self.expectation(quil_prog, hamiltonian, self.qvm)
            disp_fun("\tParameters: {} ".format(iter_vars))
            if jacobian is not None:
                grad = jacobian(iter_vars)
                disp_fun("\tGrad-L1-Norm: {}".format(np.max(np.abs(grad))))
                disp_fun("\tGrad-L2-Norm: {} ".format(np.linalg.norm(grad)))

            disp_fun("\tE => {}".format(E))
            if return_all:
                iteration_params.append(iter_vars)
                expectation_vals.append(E)

        # using self.minimizer
        arguments, _, _, _ = inspect.getargspec(self.minimizer)

        if disp is not None and 'callback' in arguments:
            self.minimizer_kwargs['callback'] = print_current_iter

        args = [objective_function, initial_params]
        args.extend(self.minimizer_args)
        if 'jac' in arguments:
            self.minimizer_kwargs['jac'] = jacobian

        result = self.minimizer(*args, **self.minimizer_kwargs)

        if hasattr(result, 'status'):
            if result.status != 0:
                disp_fun("Classical optimization exited with an error index: %i" % result.status)

        results = OptResults()
        if hasattr(result, 'x'):
            results.x = result.x
            results.fun = result.fun
        else:
            results.x = result

        if return_all:
            results.iteration_params = iteration_params
            results.expectation_vals = expectation_vals
        return results

    def expectation(self, quil_prog, pauli_sum, qvm):
        """
        Computes the expectation value of pauli_sum over the distribution
        generated from quil_prog.

        :param quil_prog: (quil program)
        :param pauli_sum: (PauliSum, ndarray) PauliSum representing the
                          operator of which to calculate the expectation value
                          or a numpy matrix representing the Hamiltonian
                          tensored up to the appropriate size.
        :param qvm: (qvm connection)

        :returns: (float) representing the expectation value of pauli_sum given
                  given the distribution generated from quil_prog.
        """
        if isinstance(pauli_sum, np.ndarray):
            # debug mode by passing an array
            wf = qvm.wavefunction(quil_prog)
            average_exp = np.conj(wf).T.dot(pauli_sum.dot(wf)).real
            return average_exp
        else:
            if not isinstance(pauli_sum, (PauliTerm, PauliSum)):
                raise TypeError("pauli_sum variable must be a PauliTerm or"
                                "PauliSum object")
            if isinstance(pauli_sum, PauliTerm):
                pauli_sum = PauliSum([pauli_sum])

            operator_progs = []
            operator_coeffs = []
            for p_term in pauli_sum.terms:
                op_prog = pq.Program()
                for qindex, op in p_term:
                    op_prog.inst(STANDARD_GATES[op](qindex))
                operator_progs.append(op_prog)
                operator_coeffs.append(p_term.coefficient)

            result_overlaps = qvm.expectation(quil_prog,
                                              operator_programs=operator_progs)
            result_overlaps = list(result_overlaps)
            assert len(result_overlaps) == len(operator_progs), """Somehow we
            didn't get the correct number of results back from the QVM"""
            expectation = sum(map(lambda x: x[0]*x[1], zip(result_overlaps, operator_coeffs)))
            return expectation.real
