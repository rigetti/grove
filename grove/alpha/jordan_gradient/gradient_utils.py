from typing import Union

import numpy as np


def binary_float_to_decimal_float(number: Union[float, str]) -> float:
    """
    Convert binary floating point to decimal floating point.

    :param number: Binary floating point.
    :return: Decimal floating point representation of binary floating point.
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


def measurements_to_bf(measurements: np.ndarray) -> float:
    """
    Convert measurements into gradient binary fraction.

    :param measurements: Output measurements of gradient program.
    :return: Binary fraction representation of gradient estimate.
    """
    try:
        measurements.sum(axis=0)
    except AttributeError:
        measurements = np.asarray(measurements)
    finally:
        stats = measurements.sum(axis=0) / len(measurements)

    stats_str = [str(int(i)) for i in np.round(stats[::-1][1:])]
    bf_str = '0.' + ''.join(stats_str)
    bf = float(bf_str)

    return bf
