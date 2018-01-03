import numpy as np


def binary_to_real(number):
    """Convert binary fraction to real decimal

    :param string number: String representation of binary fraction.
    :return: Real decimal representation of binary fraction.
    :rtype: float
    """

    if isinstance(number, str):
        if number[0] == '-':
            n_sign = -1
        else:
            n_sign = 1
    elif isinstance(number, float):
        n_sign = np.sign(number)
        number = str(number)

    deci = 0
    for ndx, val in enumerate(number.split('.')[-1]):
        deci += float(val) / 2**(ndx+1)
    deci *= n_sign

    return deci


def measurements_to_bf(measurements):
    """Convert measurements into gradient binary fraction

    :param list measurements: Output measurements of gradient program.
    :return: Binary fraction representation of gradient estimate.
    :rtype: float
    """

    measurements = np.array(measurements)
    stats = measurements.sum(axis=0) / len(measurements)
    stats_str = [str(int(i)) for i in np.round(stats[::-1][1:])]
    bf_str = '0.' + ''.join(stats_str)
    bf = float(bf_str)

    return bf
