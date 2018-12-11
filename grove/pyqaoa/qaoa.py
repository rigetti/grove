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
from functools import reduce
from typing import Any, Dict, List, Tuple

import numpy as np
from pyquil import Program
from pyquil.api import QuantumComputer, WavefunctionSimulator
from pyquil.gates import H, MEASURE
from pyquil.paulis import exponential_map, PauliSum
from scipy import optimize, ufunc

from grove.pyvqe.vqe import VQE


class QAOA(object):
    def __init__(self, qc: QuantumComputer, qubits: List[int],
                 steps: int = 1,
                 init_betas: List[float] = None,
                 init_gammas: List[float] = None,
                 cost_ham: List[PauliSum] = None,
                 ref_ham: List[PauliSum]= None,
                 driver_ref: Program = None,
                 minimizer: ufunc = None,
                 minimizer_args: List[Any] = None,
                 minimizer_kwargs: Dict[str, Any] = None,
                 rand_seed: int = None,
                 vqe_options = None,
                 store_basis: bool = False):
        """
        QAOA object constructor.

        Contains all information for running the QAOA algorithm to find the
        ground state of the list of cost clauses.

        N.B. This only works if all the terms in the cost Hamiltonian commute with each other.

        :param qc: The QuantumComputer connection to use for the algorithm.
        :param qubits: The list of qubits to use for the algorithm.
        :param steps: The number of mixing and cost function steps to use.
                      Default=1.
        :param init_betas: Initial values for the beta parameters on the
                           mixing terms. Default=None.
        :param init_gammas: Initial values for the gamma parameters on the
                            cost function. Default=None.
        :param cost_ham: list of clauses in the cost function. Must be PauliSum objects.
        :param ref_ham: list of clauses in the mixer function. Must be PauliSum objects.
        :param driver_ref: The object to define state prep
                           for the starting state of the QAOA algorithm.
                           Defaults to tensor product of \|+> states.
        :param minimizer: (Optional) Minimization function to pass to the
                          Variational-Quantum-Eigensolver method.
        :param minimizer_args: (Optional) (list) of additional arguments to pass to the
                               minimizer. Default=[].
        :param minimizer_kwargs: (Optional) (dict) of optional arguments to pass to
                                 the minimizer.  Default={}.
        :param rand_seed: integer random seed for initial betas and gammas
                          guess.
        :param vqe_options: (optional) arguments for VQE run.
        :param store_basis: (optional) boolean flag for storing basis states.
                            Default=False.
        """

        # Seed the random number generator, if a seed is provided.
        if rand_seed is not None:
            np.random.seed(rand_seed)

        # Set attributes values, considering their defaults
        self.qc = qc
        self.steps = steps
        self.qubits = qubits
        self.nstates = 2 ** len(qubits)

        self.cost_ham = cost_ham or []
        self.ref_ham = ref_ham or []

        self.minimizer = minimizer or optimize.minimize
        self.minimizer_args = minimizer_args or []
        self.minimizer_kwargs = minimizer_kwargs or {
            'method': 'Nelder-Mead',
            'options': {
                'disp': True,
                'ftol': 1.0e-2,
                'xtol': 1.0e-2
            }
        }

        self.betas = init_betas or np.random.uniform(0, np.pi, self.steps)[::-1]
        self.gammas = init_gammas or np.random.uniform(0, 2*np.pi, self.steps)
        self.vqe_options = vqe_options or {}

        self.ref_state_prep = (
            driver_ref or
            Program([H(i) for i in self.qubits])
        )

        if store_basis:
            self.states = [
                np.binary_repr(i, width=len(self.qubits))
                for i in range(self.nstates)
            ]

        # Check argument types
        if not isinstance(self.cost_ham, (list, tuple)):
            raise TypeError("cost_ham must be a list of PauliSum objects.")
        if not all([isinstance(x, PauliSum) for x in self.cost_ham]):
            raise TypeError("cost_ham must be a list of PauliSum objects")

        if not isinstance(self.ref_ham, (list, tuple)):
            raise TypeError("ref_ham must be a list of PauliSum objects")
        if not all([isinstance(x, PauliSum) for x in self.ref_ham]):
            raise TypeError("ref_ham must be a list of PauliSum objects")

        if not isinstance(self.ref_state_prep, Program):
            raise TypeError("Please provide a pyQuil Program object "
                            "to generate initial state.")

    def get_parameterized_program(self):
        """
        Return a function that accepts parameters and returns a new Quil program.

        :returns: a function
        """
        cost_para_programs = []
        driver_para_programs = []

        for idx in range(self.steps):
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
            """
            Construct a Quil program for the vector (beta, gamma).

            :param params: array of 2 . p angles, betas first, then gammas
            :return: a pyquil program object
            """
            if len(params) != 2*self.steps:
                raise ValueError("params doesn't match the number of parameters set by `steps`")
            betas = params[:self.steps]
            gammas = params[self.steps:]

            prog = Program()
            prog += self.ref_state_prep
            for idx in range(self.steps):
                for fprog in cost_para_programs[idx]:
                    prog += fprog(gammas[idx])

                for fprog in driver_para_programs[idx]:
                    prog += fprog(betas[idx])

            return prog

        return psi_ref

    def get_angles(self) -> Tuple[List[float], List[float]]:
        """
        Finds optimal angles with the quantum variational eigensolver method.

        Stored VQE result

        :returns: A tuple of the beta angles and the gamma angles for the optimal solution.
        """
        stacked_params = np.hstack((self.betas, self.gammas))
        vqe = VQE(self.minimizer, minimizer_args=self.minimizer_args,
                  minimizer_kwargs=self.minimizer_kwargs)
        cost_ham = reduce(lambda x, y: x + y, self.cost_ham)
        # maximizing the cost function!
        param_prog = self.get_parameterized_program()
        result = vqe.vqe_run(param_prog, cost_ham, stacked_params, qc=self.qc,
                             **self.vqe_options)
        self.result = result
        betas = result.x[:self.steps]
        gammas = result.x[self.steps:]
        return betas, gammas

    def probabilities(self, angles: List[float]) -> np.ndarray:
        """
        Computes the probability of each state given a particular set of angles.

        :param angles: A concatenated list of angles [betas]+[gammas]
        :return: The probabilities of each outcome given those angles.
        """
        if isinstance(angles, list):
            angles = np.array(angles)

        assert angles.shape[0] == 2 * self.steps, "angles must be 2 * steps"

        param_prog = self.get_parameterized_program()
        prog = param_prog(angles)
        wf = WavefunctionSimulator().wavefunction(prog)
        wf = wf.amplitudes.reshape((-1, 1))
        probs = np.zeros_like(wf)
        for xx in range(2 ** len(self.qubits)):
            probs[xx] = np.conj(wf[xx]) * wf[xx]
        return probs

    def get_string(self, betas: List[float], gammas: List[float], samples: int = 100):
        """
        Compute the most probable string.

        The method assumes you have passed init_betas and init_gammas with your
        pre-computed angles or you have run the VQE loop to determine the
        angles.  If you have not done this you will be returning the output for
        a random set of angles.

        :param betas: List of beta angles
        :param gammas: List of gamma angles
        :param samples: (Optional) number of samples to get back from the QuantumComputer.
        :returns: tuple representing the bitstring, Counter object from
                  collections holding all output bitstrings and their frequency.
        """
        if samples <= 0 and not isinstance(samples, int):
            raise ValueError("samples variable must be positive integer")
        param_prog = self.get_parameterized_program()
        stacked_params = np.hstack((betas, gammas))

        sampling_prog = Program()
        ro = sampling_prog.declare('ro', 'BIT', len(self.qubits))
        sampling_prog += param_prog(stacked_params)
        sampling_prog += [MEASURE(qubit, r) for qubit, r in zip(self.qubits, ro)]
        sampling_prog.wrap_in_numshots_loop(samples)
        executable = self.qc.compile(sampling_prog)
        bitstring_samples = self.qc.run(executable)
        bitstring_tuples = list(map(tuple, bitstring_samples))
        freq = Counter(bitstring_tuples)
        most_frequent_bit_string = max(freq, key=lambda x: freq[x])
        return most_frequent_bit_string, freq
