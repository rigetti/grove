import numpy as np
bnds = ((-0.1, np.pi), (-0.1, np.pi))

minimizer_kwargs_dict = {
 'Nelder-Mead': {'method': 'Nelder-Mead',
                                   'options': {'disp': True,
                                               'ftol': 1.0e-2,
                                               'xtol': 1.0e-2}},
 'Powell': {'method': 'Powell',
                                   'options': {'disp': True,
                                               'ftol': 1.0e-2,
                                               'xtol': 1.0e-2}},
 'CG': {'method': 'CG',
                                   'options': {'disp': True,
                                               'gtol': 1.0e-2}},
 'BFGS': {'method': 'BFGS',
                                   'options': {'disp': True,
                                               'gtol': 1.0e-2}},
 'Newton-CG': {'method': 'Newton-CG',
                                   'options': {'disp': True,
                                               'gtol': 1.0e-2}},
 'L-BFGS-B': {'method': 'L-BFGS-B',
                                   'options': {'disp': True}},
 'TNC': {'method': 'TNC',
                                   'options': {'disp': True}},
 'COBYLA': {'method': 'COBYLA',
                                   'options': {'disp': True}},
 'SLSQP': {'method': 'SLSQP',
                                   'options': {'disp': True}},
 'dogleg': {'method': 'dogleg',
                                   'options': {'disp': True}},
 'trust-ncg': {'method': 'trust-ncg',
                                   'options': {'disp': True}}

}
