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


def mutual_dynamics_np(
        t,
        x,
        a: np.ndarray,
        b: Union[float, np.ndarray] = 0.1,
        k: Union[float, np.ndarray] = 5.0,
        c: Union[float, np.ndarray] = 1.0,
        d: Union[float, np.ndarray] = 5.0,
        e: Union[float, np.ndarray] = 0.9,
        h: Union[float, np.ndarray] = 0.1) -> np.ndarray:
    """
    Mutualistic dynamics

    :param t: time
    :param x: initial value:  is 1d row vector feature, n
    :param a: adjacency matrix
    :param b: incoming migration rate
    :param k: system carrying capacity for logistic growth
    :param c: low abundance for Allee effect
    :param d: mutual interaction term
    :param e: mutual interaction term
    :param h: mutual interaction term

    :return:
    dx/dt = b + x * (1 - x / k) * (x / c - 1) + sum_{j=1}^n a_{ij} x_i x_j / (d + e x_i + h x_j)
    """
    n = len(x)
    x = x.reshape(-1, 1)
    self_dynamics = b + x * (1 - x / k) * (x / c - 1)
    h = np.broadcast_to(np.asarray(h), (n, 1)).reshape(1, n)
    outer = np.dot(a, np.dot(x, x.T) / (np.broadcast_to(d, (n, n)) + np.broadcast_to(e * x, (n, n)) + np.broadcast_to(h * x.T, (n, n))))
    interaction = np.diag(outer).reshape(-1, 1)
    f = (self_dynamics + interaction).reshape(-1)
    return f


def load_dynamics(dynamics_type: str) -> Callable:
    dynamic_map = {
        'gene': gene_dynamics_np,
        'neuron': neuronal_dynamics_np,
        'epidemic': epidemic_dynamics_np,
        'mutualistic': mutual_dynamics_np,
    }
    dynamics = dynamic_map[dynamics_type]
    return dynamics
