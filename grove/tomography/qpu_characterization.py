import logging
from collections import namedtuple
from time import sleep

from typing import Sequence
import numpy as np
from pyquil.api import QVMConnection, QPUConnection
from pyquil.api.errors import DeviceRetuningError
from pyquil.quil import Program
from pyquil.gates import I, RX, CZ

import grove.tomography.operator_utils as o_ut
import grove.tomography.utils as ut
from grove.tomography.process_tomography import (process_tomography_programs, ProcessTomography,
                                                 DEFAULT_PROCESS_TOMO_SETTINGS)
from grove.tomography.tomography import default_channel_ops
from grove.tomography.utils import run_in_parallel
qt = ut.import_qutip()

_log = logging.getLogger(__name__)

# Qubit 3 not tunable on Acorn.
ACORN_QUBITS = [q for q in range(20) if q not in [3]]

ACORN_CLIQUES = [
    [(0, 5), (10, 15), (1, 6), (11, 16), (2, 7), (12, 17), (13, 18), (4, 9), (14, 19)],
    [(0, 6), (10, 16), (1, 7), (11, 17), (2, 8), (12, 18), (13, 19)],
    [(5, 10), (6, 11), (7, 12), (8, 13), (9, 14)]
]

AGAVE_QUBITS = [q for q in range(8) if q not in []]


AGAVE_CLIQUES = [
    [(0, 1), (2, 3), (4, 5), (6, 7)],
    [(0, 7), (1, 2), (3, 4), (5, 6)],
]

if qt:
    # generate ideal targets
    QCZ = qt.cphase(np.pi)
    QRX90 = (-np.pi * .25j * qt.sigmax()).expm()
    QI = qt.qeye(2)
else:  # pragma no coverage
    QCZ = None
    QRX90 = None
    QI = None


# This class as well as NoiseModel should ultimately live in pyquil
class ISA(object):
    """
    Basic Instruction Set Architecture specification.
    """

    def __init__(self, name, qubits, cz_edges):
        """
        Create an Instruction Set Architecture ISA

        :param str name: Name of the QPU.
        :param Sequence[int] qubits:
        :param Sequence[Tuple] cz_edges:
        """
        self.name = name
        self.qubits = sorted(qubits)
        self.cz_edges = sorted([tuple(sorted(e)) for e in cz_edges])

    def to_dict(self):
        """
        Create a JSON serializable representation of the ISA.

        :return: A dictionary representation of self.
        :rtype: Dict[str,Any]
        """
        return {
            "name": self.name,
            "qubits": self.qubits,
            "cz_edges": self.cz_edges
        }

    @classmethod
    def from_dict(cls, d):
        """
        Re-create the ISA from a dictionary representation.

        :param Dict[str,Any] d: The dictionary representation.
        :return: The restored ISA.
        :rtype: ISA
        """
        return cls(**d)

    def __eq__(self, other):
        return isinstance(other, ISA) and self.__dict__ == other.__dict__


ACORN_ISA = ISA("19Q-Acorn", ACORN_QUBITS, sum(ACORN_CLIQUES, []))
AGAVE_ISA = ISA("8Q-Agave", AGAVE_QUBITS, sum(AGAVE_CLIQUES, []))

QPU_TOMO_SETTINGS = DEFAULT_PROCESS_TOMO_SETTINGS._replace(
 constraints={'cpositive', 'trace_preserving'}
)


_KrausModel = namedtuple("_KrausModel", ["gate", "params", "targets", "kraus_ops", "fidelity"])


class KrausModel(_KrausModel):

    @staticmethod
    def unpack_kraus_matrix(m):
        """
        Helper to optionally unpack a JSON compatible representation of a complex Kraus matrix.

        :param list|np.array m: The representation of a Kraus operator. Either a complex square
        matrix (as numpy array or nested lists) or a pair of real square matrices (as numpy arrays
        or nested lists) representing the element-wise real and imaginary part of m.
        :return: A complex square numpy array representing the Kraus operator.
        :rtype: np.array
        """
        m = np.asarray(m, dtype=complex)
        if m.ndim == 3:
            m = m[0] + 1j*m[1]
        if not m.ndim == 2:  # pragma no coverage
            raise ValueError("Need 2d array.")
        if not m.shape[0] == m.shape[1]:  # pragma no coverage
            raise ValueError("Need square matrix.")
        return m

    def _asdict(self):
        res = super(KrausModel, self)._asdict()
        res['kraus_ops'] = [[k.real.tolist(), k.imag.tolist()] for k in self.kraus_ops]
        return res

    @classmethod
    def from_dict(cls, d):
        kraus_ops = [KrausModel.unpack_kraus_matrix(k) for k in d['kraus_ops']]
        return cls(d['gate'], d['params'], d['targets'], kraus_ops, d['fidelity'])

    def __eq__(self, other):
        return self._asdict() == other._asdict()


class NoiseModel(object):
    """
    Encapsulate the QPU noise model
    """

    def __init__(self, isa, identities, rx90s, czs, assignment_probs, cz_cliques):
        """
        Create a Noise model for a QPU containing information about the noisy identity gates,
        RX(pi/2) gates and CZ gates on the defined graph edges.
        The tomographies and assignment probabilities are ordered in the same way as they appear
        in the ISA.

        :param ISA isa: The instruction set architecture for the QPU.
        :param Sequence[KrausModel] identities: The tomographic estimates of single qubit
        identity gates.
        :param Sequence[KrausModel] rx90s: The tomographic estimates of single qubit RX(pi/2)
        gates.
        :param Sequence[KrausModel] czs: The tomographic estimates of two qubit CZ gates.
        :param Sequence[np.array] assignment_probs: The single qubit readout assignment
        probability matrices.
        :param Sequence[List[tuple]] cz_cliques: Gates within a CZ clique
        are executed simultaneously during noise model characterization.
        """
        self.isa = isa
        self.identities = identities
        self.rx90s = rx90s
        self.czs = czs
        self.assignment_probs = assignment_probs
        self.cz_cliques = cz_cliques

    def to_dict(self):
        """
        Create a JSON serializable representation of the noise model.

        :return: A dictionary representation of self.
        :rtype: Dict[str,Any]
        """
        return {
            "isa": self.isa.to_dict(),
            "identities": [t._asdict() for t in self.identities],
            "rx90s":  [t._asdict() for t in self.rx90s],
            "czs": [t._asdict() for t in self.czs],
            "assignment_probs": [a.tolist() for a in self.assignment_probs],
            "cz_cliques": self.cz_cliques,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Re-create the noise model from a dictionary representation.

        :param Dict[str,Any] d: The dictionary representation.
        :return: The restored noise model.
        :rtype: NoiseModel
        """
        return cls(
            isa=ISA.from_dict(d["isa"]),
            identities=[KrausModel.from_dict(t) for t in d["identities"]],
            rx90s=[KrausModel.from_dict(t) for t in d["rx90s"]],
            czs=[KrausModel.from_dict(t) for t in d["czs"]],
            assignment_probs=[np.array(a) for a in d["assignment_probs"]],
            cz_cliques=[map(tuple, c) for c in d["cz_cliques"]]
        )

    def __eq__(self, other):
        return isinstance(other, NoiseModel) and self.to_dict() == other.to_dict()


def estimate(isa, nsamples, cxn, cz_cliques, shuffle=True, max_retries=100,
             settings=QPU_TOMO_SETTINGS, retune_sleep=60.):
    """
    Generate a ``NoiseModel`` for a QPU specified by its ``isa`` by sampling from a connection
    ``cxn``.

    :param ISA isa: The instruction set architecture of the QPU.
    :param int nsamples: The number of samples to use in each measurement.
    :param QPUConnection|QVMConnection cxn: The connection to the virtual or actual quantum
    processor.
    :param bool shuffle: If True (default), the tomographic sequences between different qubit
    cliques are randomized in order to avoid introducing bias, e.g., by having identity gate
    pre/post rotations occur on all qubits simultaneously.
    :param Sequence[List[Tuple]] cz_cliques: Gates within a CZ clique
    are executed simultaneously during noise model characterization.
    :param int max_retries: How often to restart tomographies that are interrupted by the QPU
    retuning.
    :param float retune_sleep: How long to wait before the next retry after a DeviceRetuningError.
    :param TomographySettings settings: The settings for the MLE tomography.
    :return: The estimated NoiseModel
    """
    single_qubit_cliques = [[q] for q in isa.qubits]
    missing_edges = set(isa.cz_edges) - set(map(tuple, sum(cz_cliques, [])))
    if missing_edges:
        raise ValueError("Incomplete specification of cz_cliques. Missing edges: {}".format(
            missing_edges
        ))

    def _execute_tomos(gate, cliques):
        trials = 0
        while True:
            try:
                return parallel_process_tomographies(gate, cliques, nsamples, cxn, shuffle=shuffle,
                                                     settings=settings)
            except DeviceRetuningError as e:
                _log.info("Device retuning, sleeping for 60 seconds.")
                sleep(retune_sleep)
                trials += 1
                if trials >= max_retries:  # pragma no coverage
                    _log.error("Did not complete tomography of:\n\t {}".format(
                        "\n\t".join(map(str, map(gate, cliques)))))
                    raise e

    # Identities
    i_tomos, _ , i_assignment_probs  = _execute_tomos(I, single_qubit_cliques)
    i_kraus = [KrausModel("I", (), q, t.to_kraus(), t.avg_gate_fidelity(QI))
               for q, t in zip(isa.qubits, i_tomos)]

    # RX(pi/2)
    rx90_tomos, _, rx90_assignment_probs = _execute_tomos(RX(np.pi/2), single_qubit_cliques)
    rx90_kraus = [KrausModel("RX", (np.pi/2,), q, t.to_kraus(), t.avg_gate_fidelity(QRX90))
                  for q, t in zip(isa.qubits, rx90_tomos)]

    # combine assignment probs through averaging
    assignment_probs = [.5*(p1 + p2)
                        for p1, p2 in zip(i_assignment_probs, rx90_assignment_probs)]

    cz_tomo_dict = {}
    for cliques in cz_cliques:
        tomos = _execute_tomos(CZ, cliques)[0]
        for edge, tomo in zip(cliques, tomos):
            cz_tomo_dict[edge] = tomo

    cz_tomos = [cz_tomo_dict[e] for e in isa.cz_edges]
    cz_kraus = [KrausModel("CZ", (), e, t.to_kraus(), t.avg_gate_fidelity(QCZ))
                for e, t in zip(isa.cz_edges, cz_tomos)]

    return NoiseModel(isa, i_kraus, rx90_kraus, cz_kraus, assignment_probs, cz_cliques)


def parallel_sample_assignment_probs(qubit_cliques, nsamples, cxn, shuffle=True):
    """
    Sample the assignment probabilities of qubits using nsamples per measurement, and then compute
    the estimated assignment probability matrix. See the docstring for estimate_assignment_probs for
    more information.

    :param Sequence[tuple|list] qubit_cliques: The qubit targets for each parallel tomography. All
    cliques must contain the same number of qubits and they must be disjoint.
    :param int nsamples: The number of samples to use in each measurement.
    :param QPUConnection|QVMConnection cxn: The connection to the virtual or actual quantum
    processor.
    :param bool shuffle: If True (default), the tomographic sequences between different qubit
    cliques are randomized in order to avoid introducing bias, e.g., by having identity gate
    pre/post rotations occur on all qubits simultaneously.
    :return: A list of assignment probability matrices for each qubit clique.
    :rtype: List[numpy.ndarray]
    """
    n_qubits_per_clique = [len(c) for c in qubit_cliques]
    if not all(n_qubits == n_qubits_per_clique[0] for n_qubits in n_qubits_per_clique):
        raise ValueError("Can only handle cliques of equal size.")

    programs = [list(ut.basis_state_preps(*sorted(c))) for c in qubit_cliques]
    all_hists = run_in_parallel(programs, nsamples, cxn, shuffle=shuffle)
    return [ut.estimate_assignment_probs(hists) for hists in all_hists]


def parallel_process_tomographies(gate, qubit_cliques, nsamples, cxn,
                                  shuffle=True, settings=DEFAULT_PROCESS_TOMO_SETTINGS):
    """
    Execute process tomographies of the same pyquil ``gate`` on several disjoint ``qubit_cliques``
    simultaneously.

    :param gate: The gate under investigation (can be any function that takes qubit indices as
    argument and outputs a Quil program.
    :param Sequence[tuple|list] qubit_cliques: The qubit targets for each parallel tomography. All
    cliques must contain the same number of qubits and they must be disjoint.
    :param int nsamples: The number of samples
    :param QPUConnection|QVMConnection cxn: The connection to the virtual or actual quantum
    processor.
    :param bool shuffle: If True (default), the tomographic sequences between different qubit
    cliques are randomized in order to avoid introducing bias, e.g., by having identity gate
    pre/post rotations occur on all qubits simultaneously.
    :param TomographySettings settings: The settings for the MLE tomography.
    :return: A tuple ``(qpts, qpt_histograms, assignment_probs)`` where ``qpts`` is a list of
    ``ProcessTomography`` objects, one for each qubit clique, ``qpt_histograms`` is a sequence of
    2d-arrays containing the individual QPT measurement histograms for each clique, and
    ``assignment_probs`` is a list of assignment probability matrices.
    """

    n_qubits_per_clique = [len(c) for c in qubit_cliques]
    n_qubits = n_qubits_per_clique[0]
    if not all(_n_qubits == n_qubits for _n_qubits in n_qubits_per_clique):
        raise ValueError("Can only handle cliques of equal size.")

    # generate assignment probs
    assignment_probs_per_clique = parallel_sample_assignment_probs(qubit_cliques, nsamples, cxn,
                                                                   shuffle=shuffle)

    programs = []
    for c in qubit_cliques:
        programs.append(list(process_tomography_programs(Program(gate(*sorted(c))), sorted(c))))

    tomo_hists_per_clique = run_in_parallel(programs, nsamples, cxn, shuffle=shuffle)
    results_per_clique = []

    # same for all cliques
    channel_ops = list(default_channel_ops(n_qubits))

    for assignment_probs, hists in zip(assignment_probs_per_clique, tomo_hists_per_clique):
        povm = o_ut.make_diagonal_povm(o_ut.POVM_PI_BASIS ** n_qubits,
                                       assignment_probs)
        hists = hists.reshape((len(channel_ops), len(channel_ops), 2 ** n_qubits))
        results_per_clique.append(ProcessTomography.estimate_from_ssr(hists, povm, channel_ops,
                                                                      channel_ops, settings))

    return results_per_clique, tomo_hists_per_clique, assignment_probs_per_clique
