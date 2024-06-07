# __description__: Functions used to set background
#
# Created by Tobias Wenzel in August 2017
# Copyright (c) 2017 Tobias Wenzel

import numpy as np


def get_xsqr(x: np.ndarray, y: np.ndarray):
    m_size = len(x)
    x = x**2
    for i in range(m_size):
        y[i, :] = x[i] + x
    return y


def get_rastrigin(x: np.ndarray, y: np.ndarray):
    """
    Recommended range: -5.2, 5.2
    """
    m_size = len(x)
    for i in range(m_size):
        for j in range(m_size):
            y[i, j] = (x[i] ** 2 - 10 * np.cos(np.pi * x[i]) + 10) + (
                x[j] ** 2 - 10 * np.cos(np.pi * x[j]) + 10
            )
    return y


def get_rosenbrock(x: np.ndarray, y: np.ndarray):
    """
    Recommended range: -10, 10
    """
    m_size = len(x)
    for i in range(m_size):
        for j in range(m_size):
            a = 1.0 - x[i]
            b = x[j] - x[i] ** 2
            y[i, j] = a * a + b * b * 100.0
    return y


def get_griewank(x: np.ndarray, y: np.ndarray):
    """
    Recommended range: -10, 10
    """
    m_size = len(x)
    for i in range(m_size):
        for j in range(m_size):
            prod = (np.cos(x[i] / 0.0000001) + 1) * (np.cos(x[j] / 1) + 1)
            y[i, j] = (1 / 4000) * (x[i] ** 2 - prod + x[j] ** 2 - prod)
    return y


def get_schaffer_f6(x: np.ndarray, y: np.ndarray):
    """
    Recommended range: -10, 10
    """
    m_size = len(x)
    for i in range(m_size):
        for j in range(m_size):
            y[i, j] = 0.5 - (
                (np.sin(np.sqrt(x[i] ** 2 + x[j] ** 2)) ** 2 - 0.5)
                / (1 + 0.001 * (x[i] ** 2 + x[j] ** 2) ** 2)
            )
    return y


def get_eggholder(x: np.ndarray, y: np.ndarray):
    """
    Recommended range: -512, 512
    """
    m_size = len(x)
    for i_1 in range(m_size):
        for i_2 in range(m_size):
            y[i_1, i_2] -= (x[i_2] + 47) * np.sin(
                np.sqrt(np.abs((x[i_1] / 2) + x[i_2] + 47))
            ) - x[i_1] * np.sin(np.sqrt(np.abs(x[i_1] - (x[i_2] + 47))))
    return y


def get_hoelder_table(x: np.ndarray, y: np.ndarray):
    m_size = len(x)
    for i_1 in range(m_size):
        for i_2 in range(m_size):
            y[i_1, i_2] = -np.abs(
                np.sin(x[i_1])
                * np.cos(x[i_2])
                * np.exp(np.abs(1 - (np.sqrt(x[i_1] ** 2 + x[i_2] ** 2) / np.pi)))
            )
    return y


def get_styblinsky_tang(x: np.ndarray, y: np.ndarray):
    """
    @todo fix
    """

    def _func(d):
        return (d**4) - (16 * (d**2)) + (5 * d)

    m_size = len(x)
    f_arr = np.frompyfunc(_func, 1, 1)
    for i_1 in range(m_size):
        y[i_1, :] = f_arr(x[i_1])
        for i_2 in range(m_size):
            y[i_1, i_2] += _func(x[i_2])
    return y


def background_function(func_name: str, n: float) -> np.ndarray:
    """

    :param func_name:
    :param n:
    :return:
    """
    m_size = 2 * int(n)
    y = np.zeros((m_size, m_size))
    x = np.linspace(-n, n, m_size)
    _background_function = {
        "square": get_xsqr,
        "rastrigin": get_rastrigin,
        "schaffer_f6": get_schaffer_f6,
        "griewank": get_griewank,
        "rosenbrock": get_rosenbrock,
        "eggholder": get_eggholder,
        "hoelder_table": get_hoelder_table,
        "styblinsky_tang": get_styblinsky_tang,
    }[func_name]
    return _background_function(x, y)
