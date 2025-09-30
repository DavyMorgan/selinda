from typing import Callable, Union

import numpy as np


def gene_dynamics_np(
        t,
        x,
        a: np.ndarray,
        b: Union[float, np.ndarray] = 1.0,
        f: float = 1.0,
        h: float = 2.0) -> np.ndarray:
    """
    Gene dynamics by the Michaelis-Menten equation
    :param t: time
    :param x: initial value:  is 1d row vector feature, n
    :param a: adjacency matrix
    :param b: strength of the degradation or dimmerization
    :param f: degradation (1) or dimmerization (2)
    :param h: Hill coefficient

    :return:
    dx/dt = -b x_i^f + sum_{j=1}^n a_{ij} x_j^h / (x_j^h + 1)
    """
    self_dynamics = -b * (x ** f)
    interaction = np.dot(a, x ** h / (x ** h + 1))
    f = self_dynamics + interaction
    return f


def neuronal_dynamics_np(
        t,
        x,
        a: np.ndarray,
        b: Union[float, np.ndarray] = 1.0,
        u: float = 3.5,
        d: float = 2.0) -> np.ndarray:
    """
    Wilson–Cowan neuronal dynamics

    :param t: time
    :param x: initial value:  is 1d row vector feature, n
    :param a: adjacency matrix
    :param b: strength of the self-inhibition
    :param u: threshold
    :param d: dispersal rate
    :return:
    dx/dt = -x_i + sum_{j=1}^n a_{ij} / (1 + exp(u - d x_j))
    """
    self_dynamics = -b * x
    interaction = np.dot(a, 1 / (1 + np.exp(u - d * x)))
    f = self_dynamics + interaction
    return f


def epidemic_dynamics_np(
        t,
        x,
        a: np.ndarray,
        b: Union[float, np.ndarray] = 6.0) -> np.ndarray:
    """
    SIS epidemic dynamics

    :param t: time
    :param x: initial value:  is 1d row vector feature, n
    :param a: adjacency matrix
    :param b: strength of the self-recovery
    :return:
    dx/dt = -b x_i + (1 - x_i) sum_{j=1}^n a_{ij} x_j
    """
    self_dynamics = -b * x
    interaction = (1 - x) * np.dot(a, x)
    f = self_dynamics + interaction
    return f


def load_dynamics(dynamics_type: str) -> Callable:
    dynamic_map = {
        'gene': gene_dynamics_np,
        'epidemic': epidemic_dynamics_np,
        'neuron': neuronal_dynamics_np,
    }
    dynamics = dynamic_map[dynamics_type]
    return dynamics
