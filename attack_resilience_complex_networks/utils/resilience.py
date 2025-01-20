from typing import Tuple

import numpy as np
import torch


def compute_beta(topology: torch.tensor) -> float:
    """
    :param topology: adjacency matrix
    :return: current beta
    """
    with torch.no_grad():
        beta = torch.sum(torch.matmul(topology, topology)) / torch.sum(topology)
        beta = beta.item()
    return beta


def compute_beta_np(topology: np.ndarray) -> Tuple[float, Tuple[float, float, float]]:
    """
    :param topology: adjacency matrix
    :return: current beta
    """
    s_in = topology.sum(axis=0)
    s_out = topology.sum(axis=1)
    beta_direct = (s_out * s_in).sum() / topology.sum()
    density = s_in.mean()
    heterogeneity = s_in.std()*s_out.std()/density
    if abs(heterogeneity) < 1e-6:
        symmetry = 0
    else:
        symmetry = ((s_in*s_out).mean() - s_in.mean()*s_out.mean())/(s_in.std()*s_out.std())
    beta_factorized = density + symmetry*heterogeneity
    if not np.isclose(beta_direct, beta_factorized):
        print('beta_direct is not consistent with beta_factorized')
        print(f'beta_direct: {beta_direct}, beta_factorized: {beta_factorized}, density: {density}, symmetry: {symmetry}, heterogeneity: {heterogeneity}')
    return beta_direct, (density, symmetry, heterogeneity)
