# __filename__: herzblatt.py
#
# __description__: Functions used to set background (PSO, but could be
#  also other optimizer.
#
# __remark__:
#
# __todos__:
#
# Created by Tobias Wenzel in August 2017
# Copyright (c) 2017 Tobias Wenzel

import numpy as np
from scipy.optimize import rosen
import sys

def get_xsqr(vals: np.ndarray, bg_fn: np.ndarray):
    """
    ...used for background image
    :param n:
    :return:
    """
    print("background-function: x^2")
    m_size = len(vals)
    vals = vals**2
    for i in range(m_size):
        bg_fn[i, :] = vals[i]+ vals
    return vals, bg_fn


def get_rastrigin(vals: np.ndarray, bg_fn: np.ndarray):
    """
    -5.2, 5.2
    :param n:
    :return:
    """

    print(f"background-function: {sys._getframe().f_code.co_name}")
    #if n != 10:
    #    print(f"use n={n} as bound. n=6 is recommended.")
    m_size = len(vals)
    for i in range(m_size):
        for j in range(m_size):
            bg_fn[i,j]= (vals[i]**2-10*np.cos(np.pi*vals[i])+10)\
        +(vals[j]**2-10*np.cos(np.pi*vals[j])+10)
    return vals, bg_fn


def get_rosenbrock(vals: np.ndarray, bg_fn: np.ndarray):
    """

    :param n:
    :return:
    """
    print(f"background-function: {sys._getframe().f_code.co_name}")
    # if n != 10:
    #     print(f"use n={n} as bound. n=10 is recommended.")
    m_size = len(vals)

    for i in range(m_size):
        for j in range(m_size):
            a = 1. - vals[i]
            b = vals[j] - vals[i]**2
            bg_fn[i, j]= a * a + b * b * 100.
    return vals, bg_fn


def get_griewank(vals: np.ndarray, bg_fn: np.ndarray):
    """
    :param n:
    :return:
    """
    print(f"background-function: {sys._getframe().f_code.co_name}")
    # if n != 10:
    #     print(f"use n={n} as bound. n=10 is recommended.")
    m_size = len(vals)
    for i in range(m_size):
        for j in range(m_size):
            prod = (np.cos(vals[i] / 0.0000001) + 1) * (np.cos(vals[j] / 1) + 1)
            bg_fn[i,j] = (1 / 4000) * (vals[i] ** 2 - prod + vals[j] ** 2 - prod)
    return vals, bg_fn


def get_schaffer_f6(vals: np.ndarray, bg_fn: np.ndarray):
    """
    :param n:
    :return:
    """
    print(f"background-function: {sys._getframe().f_code.co_name}")
    # if n != 10:
    #     print(f"use n={n} as bound. n=10 is recommended.")
    m_size = len(vals)

    for i in range(m_size):
        for j in range(m_size):
            bg_fn[i,j] =(0.5 - ((np.sin(np.sqrt(vals[i]**2 + vals[j]**2))**2 - 0.5) \
                    / (1 + 0.001*(vals[i]**2 + vals[j]**2)**2)))
    return vals, bg_fn


def get_eggholder(vals: np.ndarray, bg_fn: np.ndarray):
    """
    -512, 512
    :param n:
    :return vals
    :return bg_fn:
    """
    print(f"background-function: {sys._getframe().f_code.co_name}")
    # if n != 512:
    #     print(f"use n={n} as bound. n=512 is recommended.")
    m_size = len(vals)

    for i_1 in range(m_size):
        for i_2 in range(m_size):
            bg_fn[i_1,i_2] -= (vals[i_2] + 47) \
                              *np.sin(np.sqrt(np.abs((vals[i_1] / 2) + vals[i_2] + 47)))\
                              - vals[i_1] \
                              * np.sin(np.sqrt(np.abs(vals[i_1] - (vals[i_2] + 47))))
    return vals, bg_fn


def get_hoelder_table(vals: np.ndarray,
                      bg_fn: np.ndarray):
    """
    :return vals
    :return bg_fn:
    """
    m_size = len(vals)
    for i_1 in range(m_size):
        for i_2 in range(m_size):
            bg_fn[i_1,i_2] = -np.abs(np.sin(vals[i_1])*np.cos(vals[i_2])\
                                * np.exp(np.abs(1-(np.sqrt(vals[i_1]**2+vals[i_2]**2)/np.pi))))
    return vals, bg_fn


def get_styblinsky_tang(vals: np.ndarray, bg_fn: np.ndarray):
    """
    @not yet working
    :param n:
    :return:
    """
    m_size = len(vals)
    f = lambda d: (d ** 4) - (16 * (d**2)) + (5 * d)
    f_arr = np.frompyfunc(f, 1, 1)

    for i_1 in range(m_size):
        bg_fn[i_1, :] = f_arr(vals[i_1])
        for i_2 in range(m_size):
            bg_fn[i_1, i_2] += f(vals[i_2])
    return vals, bg_fn


background_function = {}

background_function['square'] = get_xsqr
background_function['rastrigin'] = get_rastrigin
background_function['schaffer_f6'] = get_schaffer_f6
background_function['griewank'] = get_griewank
background_function['rosenbrock'] = get_rosenbrock
background_function['eggholder'] = get_eggholder
background_function['hoelder_table'] = get_hoelder_table
background_function['styblinsky_tang'] = get_styblinsky_tang


def generate_2d_background(func_name: str, n: int):
    """

    :param func_name:
    :param n:
    :return:
    """
    m_size = 2*n
    bg_fn = np.zeros((m_size,m_size))
    vals = np.linspace(-n, n, m_size)
    return background_function[func_name](vals, bg_fn)