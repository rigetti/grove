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

from collections import Counter
from scipy import optimize
import numpy as np
from grove.pyvqe.vqe import VQE
import pyquil.quil as pq
from pyquil.gates import H
from pyquil.paulis import exponential_map, PauliSum


class QAOA(object):
    def __init__(self, qvm, n_qubits, steps=1, init_betas=None,
                 init_gammas=None, cost_ham=[],
                 ref_hamiltonian=[], driver_ref=None,
                 minimizer=None, minimizer_args=[],
                 minimizer_kwargs={}, rand_seed=None,
                 vqe_options={}, store_basis=False):
        """
        QAOA object.

        Contains all information for running the QAOA algorthm to find the
        ground state of the list of cost clauses.

        :param qvm: (Connection) The qvm connection to use for the algorithm.
        :param n_qubits: (int) The number of qubits to use for the algorithm.
        :param steps: (int) The number of mixing and cost function steps to use.
                      Default=1.
        :param init_betas: (list) Initial values for the beta parameters on the
                           mixing terms. Default=None.
        :param init_gammas: (list) Initial values for the gamma parameters on the
                            cost function. Default=None.
        :param cost_ham: list of clauses in the cost function. Must be
                    PauliSum objects
        :param ref_hamiltonian: list of clauses in the cost function. Must be
                    PauliSum objects
        :param driver_ref: (pyQuil.quil.Program()) object to define state prep
                           for the starting state of the QAOA algorithm.
                           Defaults to tensor product of \|+> states.
        :param rand_seed: integer random seed for initial betas and gammas
                          guess.
        :param minimizer: (Optional) Minimization function to pass to the
                          Variational-Quantum-Eigensolver method
        :param minimizer_kwargs: (Optional) (dict) of optional arguments to pass to
                                 the minimizer.  Default={}.
        :param minimizer_args: (Optional) (list) of additional arguments to pass to the
                               minimizer. Default=[].
        :param minimizer_args: (Optional) (list) of additional arguments to pass to the
                               minimizer. Default=[].
        :param vqe_options: (optinal) arguents for VQE run.
        :param store_basis: (optional) boolean flag for storing basis states.
                            Default=False.
        """
        self.qvm = qvm
        self.steps = steps
        self.n_qubits = n_qubits
        self.nstates = 2 ** n_qubits
        if store_basis:
            self.states = [np.binary_repr(i, width=self.n_qubits) for i in range(
                           self.nstates)]
        self.betas = init_betas
        self.gammas = init_gammas
        self.vqe_options = vqe_options

        if driver_ref is not None:
            if not isinstance(driver_ref, pq.Program):
                raise TypeError("""Please provide a pyQuil Program object as a
                                   to generate initial state""")
            else:
                self.ref_state_prep = driver_ref
        else:
            ref_prog = pq.Program()
            for i in xrange(self.n_qubits):
                ref_prog.inst(H(i))
            self.ref_state_prep = ref_prog

        if not isinstance(cost_ham, (list, tuple)):
            raise TypeError("""cost_hamiltonian must be a list of PauliSum
                               objects""")
        if not all([isinstance(x, PauliSum) for x in cost_ham]):
            raise TypeError("""cost_hamiltonian must be a list of PauliSum
                                   objects""")
        else:
            self.cost_ham = cost_ham

        if not isinstance(ref_hamiltonian, (list, tuple)):
            raise TypeError("""cost_hamiltonian must be a list of PauliSum
                               objects""")
        if not all([isinstance(x, PauliSum) for x in ref_hamiltonian]):
            raise TypeError("""cost_hamiltonian must be a list of PauliSum
                                   objects""")
        else:
            self.ref_ham = ref_hamiltonian

        if minimizer is None:
            self.minimizer = optimize.minimize
        else:
            self.minimizer = minimizer
        # minimizer_kwargs initialized to empty dictionary
        if len(minimizer_kwargs) == 0:
            self.minimizer_kwargs = {'method': 'Nelder-Mead',
                                     'options': {'disp': True,
                                                 'ftol': 1.0e-2,
                                                 'xtol': 1.0e-2}}
        else:
            self.minimizer_kwargs = minimizer_kwargs

        self.minimizer_args = minimizer_args

        if rand_seed is not None:
            np.random.seed(rand_seed)
        if self.betas is None:
            self.betas = np.random.uniform(0, np.pi, self.steps)[::-1]
        if self.gammas is None:
            self.gammas = np.random.uniform(0, 2*np.pi, self.steps)

    def get_parameterized_program(self):
        """
        Return a function that accepts parameters and returns a new Quil
        program

        :returns: a function
        """
        cost_para_programs = []
        driver_para_programs = []

        for idx in xrange(self.steps):
            cost_list = []
            driver_list = []
            for cost_pauli_sum in self.cost_ham:
                for term in cost_pauli_sum.terms:
                    cost_list.append(exponential_map(term))

            for driver_pauli_sum in self.ref_ham:
                for term in driver_pauli_sum.terms:
                    driver_list.append(exponential_map(term))

            cost_para_programs.append(cost_list)
            driver_para_programs.append(driver_list)

        def psi_ref(params):
            """Construct a Quil program for the vector (beta, gamma).

            :param params: array of 2 . p angles, betas first, then gammas
            :return: a pyquil program object
            """
            if len(params) != 2*self.steps:
                raise ValueError("""params doesn't match the number of parameters set
                                    by `steps`""")
            betas = params[:self.steps]
            gammas = params[self.steps:]

            prog = pq.Program()
            prog += self.ref_state_prep
            for idx in xrange(self.steps):
                for fprog in cost_para_programs[idx]:
                    prog += fprog(gammas[idx])

                for fprog in driver_para_programs[idx]:
                    prog += fprog(betas[idx])

            return prog

        return psi_ref

    def get_angles(self):
        """
        Finds optimal angles with the quantum variational eigensolver method.

        Stored VQE result

        :returns: ([list], [list]) A tuple of the beta angles and the gamma
                  angles for the optimal solution.
        """
        stacked_params = np.hstack((self.betas, self.gammas))
        vqe = VQE(self.minimizer, minimizer_args=self.minimizer_args,
                  minimizer_kwargs=self.minimizer_kwargs)
        cost_ham = reduce(lambda x, y: x + y, self.cost_ham)
        # maximizing the cost function!
        param_prog = self.get_parameterized_program()
        result = vqe.vqe_run(param_prog, cost_ham, stacked_params, qvm=self.qvm,
                             **self.vqe_options)
        self.result = result
        betas = result.x[:self.steps]
        gammas = result.x[self.steps:]
        return betas, gammas

    def probabilities(self, angles):
        """
        Computes the probability of each state given a particular set of angles.

        :param angles: [list] A concatenated list of angles [betas]+[gammas]
        :return: [list] The probabilities of each outcome given those angles.
        """
        if isinstance(angles, list):
            angles = np.array(angles)

        assert angles.shape[0] == 2 * self.steps, "angles must be 2 * steps"

        param_prog = self.get_parameterized_program()
        prog = param_prog(angles)
        wf, _ = self.qvm.wavefunction(prog)
        wf = wf.amplitudes.reshape((-1, 1))
        probs = np.zeros_like(wf)
        for xx in xrange(2 ** self.n_qubits):
            probs[xx] = np.conj(wf[xx]) * wf[xx]
        return probs

    def get_string(self, betas, gammas, samples=100):
        """
        Compute the most probable string.

        The method assumes you have passed init_betas and init_gammas with your
        pre-computed angles or you have run the VQE loop to determine the
        angles.  If you have not done this you will be returning the output for
        a random set of angles.

        :param samples: (int, Optional) number of samples to get back from the
                        QVM.
        :returns: tuple representing the bitstring, Counter object from
                  collections holding all output bitstrings and their frequency.
        """
        if samples <= 0 and not isinstance(samples, int):
            raise ValueError("samples variable must be positive integer")
        param_prog = self.get_parameterized_program()
        stacked_params = np.hstack((betas, gammas))
        sampling_prog = param_prog(stacked_params)
        for i in xrange(self.n_qubits):
            sampling_prog.measure(i, [i])

        bitstring_samples = self.qvm.run_and_measure(sampling_prog, range(self.n_qubits), trials=samples)
        bitstring_tuples = map(tuple, bitstring_samples)
        freq = Counter(bitstring_tuples)
        most_frequent_bit_string = max(freq, key=lambda x: freq[x])
        return most_frequent_bit_string, freq
