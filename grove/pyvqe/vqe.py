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

import pyquil.api as api
import pyquil.quil as pq
import numpy as np
import inspect
from collections import Counter
from pyquil.gates import STANDARD_GATES, RX, RY
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
    vector of parameters and returns a pyQuil program, and a
    Hamiltonian of which to calculate the expectation value.

    Using this object:

        1) initialize with `inst = VQE(minimizer)` where `minimizer` is a
        function that performs a gradient free minization--i.e
        scipy.optimize.minimize(. , ., method='Nelder-Mead')

        2) call `inst.vqe_run(variational_state_evolve, hamiltonian,
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

    def vqe_run(self, variational_state_evolve, hamiltonian, initial_params,
                gate_noise=None, measurement_noise=None,
                jacobian=None, qvm=None, disp=None, samples=None, return_all=False):
        """
        functional minimization loop.

        :param variational_state_evolve: function that takes a set of parameters
                                        and returns a pyQuil program.
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
        :param samples: (int) Number of samples for calculating the expectation
                        value of the operators.  If `None` then faster method
                        ,dotting the wave function with the operator, is used.
                        Default=None.
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
        self._disp_fun = disp if disp is not None else lambda x: None
        iteration_params = []
        expectation_vals = []
        self._current_expectation = None
        if samples is None:
            print """WARNING: Fast method for expectation will be used. Noise
                     models will be ineffective"""

        if qvm is None:
            qvm = api.SyncConnection(
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
            pyquil_prog = variational_state_evolve(params)
            mean_value = self.expectation(pyquil_prog, hamiltonian, samples, qvm)
            self._current_expectation = mean_value  # store for printing
            return mean_value

        def print_current_iter(iter_vars):
            self._disp_fun("\tParameters: {} ".format(iter_vars))
            if jacobian is not None:
                grad = jacobian(iter_vars)
                self._disp_fun("\tGrad-L1-Norm: {}".format(np.max(np.abs(grad))))
                self._disp_fun("\tGrad-L2-Norm: {} ".format(np.linalg.norm(grad)))

            self._disp_fun("\tE => {}".format(self._current_expectation))
            if return_all:
                iteration_params.append(iter_vars)
                expectation_vals.append(self._current_expectation)

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
                self._disp_fun("Classical optimization exited with an error index: %i" % result.status)

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

    def expectation(self, pyquil_prog, pauli_sum, samples, qvm):
        """
        Computes the expectation value of pauli_sum over the distribution
        generated from pyquil_prog.

        :param pyquil_prog: (pyQuil program)
        :param pauli_sum: (PauliSum, ndarray) PauliSum representing the
                          operator of which to calculate the expectation value
                          or a numpy matrix representing the Hamiltonian
                          tensored up to the appropriate size.
        :param samples: (int) number of samples used to calculate the
                        expectation value.  If samples is None then the expectation
                        value is calculated by calculating <psi|O|psi> on the
                        QVM.  Error models will not work if samples is None.

        :param qvm: (qvm connection)

        :returns: (float) representing the expectation value of pauli_sum given
                  given the distribution generated from quil_prog.
        """
        if isinstance(pauli_sum, np.ndarray):
            # debug mode by passing an array
            wf, _ = qvm.wavefunction(pyquil_prog)
            wf = np.reshape(wf.amplitudes, (-1, 1))
            average_exp = np.conj(wf).T.dot(pauli_sum.dot(wf)).real
            return average_exp
        else:
            if not isinstance(pauli_sum, (PauliTerm, PauliSum)):
                raise TypeError("pauli_sum variable must be a PauliTerm or"
                                "PauliSum object")

            if isinstance(pauli_sum, PauliTerm):
                pauli_sum = PauliSum([pauli_sum])

            if samples is None:
                operator_progs = []
                operator_coeffs = []
                for p_term in pauli_sum.terms:
                    op_prog = pq.Program()
                    for qindex, op in p_term:
                        op_prog.inst(STANDARD_GATES[op](qindex))
                    operator_progs.append(op_prog)
                    operator_coeffs.append(p_term.coefficient)

                result_overlaps = qvm.expectation(pyquil_prog,
                                                  operator_programs=operator_progs)
                result_overlaps = list(result_overlaps)
                assert len(result_overlaps) == len(operator_progs), """Somehow we
                didn't get the correct number of results back from the QVM"""
                expectation = sum(map(lambda x: x[0]*x[1], zip(result_overlaps, operator_coeffs)))
                return expectation.real
            else:
                if not isinstance(samples, int):
                    raise TypeError("samples variable must be an integer")
                if samples <= 0:
                    raise ValueError("samples variable must be a postive integer")

                # normal execution via fake sampling
                expectation = 0.0  # stores the sum of contributions to the energy from each operator term
                for j, term in enumerate(pauli_sum.terms):
                    meas_basis_change = pq.Program()
                    qubits_to_measure = []
                    if term.id() == "":
                        meas_outcome = 1.0
                    else:
                        for index, gate in term:
                            qubits_to_measure.append(index)
                            if gate == 'X':
                                meas_basis_change.inst(RY(-np.pi / 2, index))
                            elif gate == 'Y':
                                meas_basis_change.inst(RX(np.pi / 2, index))

                            meas_outcome = expectation_from_sampling(pyquil_prog + meas_basis_change,
                                                                     qubits_to_measure,
                                                                     qvm, samples)

                    expectation += term.coefficient * meas_outcome

                return expectation.real


def parity_even_p(state, marked_qubits):
    """
    Calculates the parity of elements at indexes in marked_qubits

    Parity is relative to the binary representation of the integer state.

    :param state: The wavefunction index that corresponds to this state.
    :param marked_qubits: The indexes to be considered in the parity sum.
    :returns: A boolean corresponding to the parity.
    """
    assert isinstance(state, int), "{} is not an integer. Must call " \
                                   "parity_even_p with an integer " \
                                   "state.".format(state)
    mask = 0
    for q in marked_qubits:
        mask |= 1 << q
    return bin(mask & state).count("1") % 2 == 0


def expectation_from_sampling(pyquil_program, marked_qubits, qvm, samples):
    """
    Calculation of Z_{i} at marked_qubits

    Given a wavefunctions, this calculates the expectation value of the Zi
    operator where i ranges over all the qubits given in marked_qubits.

    :param pyquil_program: pyQuil program generating some state
    :param marked_qubits: The qubits within the support of the Z pauli
                          operator whose expectation value is being calculated
    :param qvm: A QVM connection.
    :param samples: Number of bitstrings collected to calculate expectation
                    from sampling.
    :returns: The expectation value as a float.
    """
    # construct program to measure
    for qindex in marked_qubits:
        pyquil_program.measure(qindex, qindex)

    bitstring_samples = qvm.run(pyquil_program, range(max(marked_qubits) + 1), trials=samples)
    bitstring_tuples = map(tuple, bitstring_samples)

    freq = Counter(bitstring_tuples)

    # perform weighted average
    expectation = 0
    for bitstring, count in freq.items():
        bitstring_int = int("".join([str(x) for x in bitstring[::-1]]), 2)
        if parity_even_p(bitstring_int, marked_qubits):
            expectation += float(count)/samples
        else:
            expectation -= float(count)/samples
    return expectation
