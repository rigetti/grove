import pytest
from grove.alpha.jordan_gradient.jordan_gradient import estimate_gradient
from mock import patch
import pyquil.quil as pq
import numpy as np

def test_gradient_estimator():
    test_perturbation = .25
    test_precision = 3
    test_measurements = 10

    with patch("pyquil.api.SyncConnection") as cxn:
        cxn.run.return_value = [[1,0,0] for i in range(test_measurements)]
        
    gradient_estimate = estimate_gradient(test_perturbation, test_precision,
            n_measurements=test_measurements, cxn=cxn) 

    assert(np.isclose(gradient_estimate, test_perturbation))
