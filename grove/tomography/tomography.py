##############################################################################
# Copyright 2017-2018 Rigetti Computing
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

"""
Module for quantum state and process tomography.

Quantum state and process tomography are algorithms that take as input many copies of a quantum
state or process, and output an estimate of what that state or process is. For more information, see
the documentation.
"""

from __future__ import print_function

import logging
from collections import namedtuple, OrderedDict
from itertools import product as cartesian_product

from pyquil.quilbase import Pragma

import grove.tomography.operator_utils

try:
    # Python 2
    from itertools import izip
except ImportError:  # pragma no coverage
    # Python 3
    izip = zip

import numpy as np

from pyquil.gates import I, RX, RY, MEASURE
from pyquil.quil import Program

import grove.tomography.utils as ut
import grove.tomography.operator_utils as o_ut
from grove.tomography.operator_utils import QI, QX, QY
from grove.tomography.utils import bitlist_to_int, sample_assignment_probs

_log = logging.getLogger(__name__)

qt = ut.import_qutip()
cvxpy = ut.import_cvxpy()

# We constrain the number of qubits to prevent the use of large amounts of memory, and prohibitively
# long programs.
MAX_QUBITS_STATE_TOMO = 4
MAX_QUBITS_PROCESS_TOMO = MAX_QUBITS_STATE_TOMO // 2
SEED = 137
SOLVER = "SCS"
if qt:
    TOMOGRAPHY_GATES = OrderedDict([(I, QI),
                                    (RX(np.pi / 2), (-1j * np.pi / 4 * QX).expm()),
                                    (RY(np.pi / 2), (-1j * np.pi / 4 * QY).expm()),
                                    (RX(np.pi), (-1j * np.pi / 2 * QX).expm())])
else:  # pragma no coverage
    TOMOGRAPHY_GATES = {}


def default_rotations(*qubits):
    """
    Generates the Quil programs for the tomographic pre- and post-rotations of any number of qubits.

    :param list qubits: A list of qubits to perform tomography on.
    """
    for gates in cartesian_product(TOMOGRAPHY_GATES.keys(), repeat=len(qubits)):
        tomography_program = Program()
        for qubit, gate in izip(qubits, gates):
            tomography_program.inst(gate(qubit))
        yield tomography_program


def default_channel_ops(nqubits):
    """
    Generate the tomographic pre- and post-rotations of any number of qubits as qutip operators.

    :param int nqubits: The number of qubits to perform tomography on.
    :return: Qutip object corresponding to the tomographic rotation.
    :rtype: Qobj
    """
    for gates in cartesian_product(TOMOGRAPHY_GATES.values(), repeat=nqubits):
        yield qt.tensor(*gates)


class _SDP_SOLVER(object):
    """
    Helper object that allows to test whether a working convex solver with SDP capabilities is
    installed. Not all solvers supported by cvxpy support positivity constraints. Examples of ones
    that do are CVXOPT and SCS.

    Usage:
        if _SDP_SOLVER.is_functional():
            # code to solve SDP
    """
    _functional = False
    _tested = False

    @classmethod
    def is_functional(cls):
        """
        Checks lazily whether a convex solver is installed that handles positivity constraints.

        :return: True if a solver supporting positivity constraints is installed.
        :rtype: bool
        """
        if not cls._tested:
            cls._tested = True
            np.random.seed(SEED)
            test_problem_dimension = 10
            mat = np.random.randn(test_problem_dimension, test_problem_dimension)
            posmat = mat.dot(mat.T)
            posvar = cvxpy.Variable(test_problem_dimension, test_problem_dimension)
            prob = cvxpy.Problem(cvxpy.Minimize((cvxpy.trace(posmat * posvar)
                                                 + cvxpy.norm(posvar))),
                                 [posvar >> 0, cvxpy.trace(posvar) >= 1.])

            try:
                prob.solve(SOLVER)
                cls._functional = True
            except cvxpy.SolverError:  # pragma no coverage
                _log.warning("No convex SDP solver found. You will not be able to solve"
                             " tomography problems with matrix positivity constraints.")
        return cls._functional


TomographySettings = namedtuple('TomographySettings', ('constraints', 'solver_kwargs'))
"""
Encapsulate the TomographySettings, i.e., the constraints to be applied to the Maximum
Likelihood Estimation and the keyword arguments to be passed to the convex solver.

:param set constraints: The constraints to be applied:
For state tomography the maximal constraints are `{'positive', 'unit_trace'}`.
For process tomography the maximal constraints are `{'cpositive', 'trace_preserving'}`.
:param dict solver_kwargs: Keyword arguments to be passed to the convex solver.
"""


DEFAULT_SOLVER_KWARGS = dict(verbose=False, max_iters=20000)


class TomographyBaseError(Exception):
    """
    Base class for errors raised during Tomography analysis.
    """
    pass


class IncompleteTomographyError(TomographyBaseError):
    """
    Raised when a tomography SignalTensor has circuit results that are all 0. indicating that the
    measurement did not complete successfully.
    """
    pass


class BadReadoutPOVM(TomographyBaseError):
    """
    Raised when the tomography analysis fails due to a bad readout calibration.
    """
    pass


class TomographyBase(object):
    def estimate_from_ssr(self):  # pragma no coverage
        raise NotImplementedError()


def _do_tomography(target_program, nsamples, cxn, qubits, max_num_qubits, tomography_class,
                   program_generator, settings, use_run=False):
    """

    :param Program target_program: The program to run to generate the state or process.
    :param int nsamples: The number of samples to take per measurement.
    :param QPUConnection|QVMConnection cxn: The connection to Forest.
    :param lists qubits: The list of qubits to perform tomography on.
    :param int max_num_qubits: The maximum allowed number of qubits.
    :param type tomography_class: The type of tomography to perform.
    :param function program_generator: The function that yields the tomography experiments.
    :param TomographySettings settings: The settings for running the optimizer.
    :param bool use_run: If ``True``, use append measurements on all qubits and use ``cxn.run``
        instead of ``cxn.run_and_measure``.
    :return: The tomography result, the assignment probabilities, and the histograms of counts
     measured.
    :rtype: tuple
    """
    if qubits is None:
        qubits = sorted(target_program.get_qubits())

    num_qubits = len(qubits)
    dimension = 2 ** num_qubits

    if num_qubits > max_num_qubits:
        raise ValueError("Too many qubits!")

    assignment_probs = sample_assignment_probs(qubits, nsamples, cxn)
    tomo_seq = list(program_generator(target_program, qubits))
    histograms = np.zeros((len(tomo_seq), dimension))

    jobs = []
    _log.info('Submitting jobs...')
    for i, tomo_prog in izip(ut.TRANGE(len(tomo_seq)), tomo_seq):
        if use_run:
            jobs.append(cxn.run_async(tomo_prog + Program([MEASURE(q, q) for q in qubits]),
                                      qubits, nsamples))
        else:
            jobs.append(cxn.run_and_measure_async(tomo_prog, qubits, nsamples))
    _log.info('Waiting for results...')
    for i, job_id in izip(ut.TRANGE(len(jobs)), jobs):
        job = cxn.wait_for_job(job_id)
        results = job.result()
        idxs = list(map(bitlist_to_int, results))
        histograms[i] = ut.make_histogram(idxs, dimension)

    povm = o_ut.make_diagonal_povm(grove.tomography.operator_utils.POVM_PI_BASIS ** num_qubits, assignment_probs)
    channel_ops = list(default_channel_ops(num_qubits))

    # Currently the analysis pathways are slightly different, so we branch on which type of
    # tomography is being done.
    if tomography_class.__tomography_type__ == "PROCESS":
        histograms = histograms.reshape((len(channel_ops), len(channel_ops), dimension))
        tomo_result = tomography_class.estimate_from_ssr(histograms, povm, channel_ops, channel_ops,
                                                         settings)
    else:
        tomo_result = tomography_class.estimate_from_ssr(histograms, povm, channel_ops, settings)
    return tomo_result, assignment_probs, histograms
