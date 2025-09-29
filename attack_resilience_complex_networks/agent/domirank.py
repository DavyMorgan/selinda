from typing import List, Optional

import os
import sys

domirank_root_path = os.path.join(os.getcwd(), 'thirdparty', 'DomiRank')
if domirank_root_path not in sys.path:
    sys.path.append(domirank_root_path)

import numpy as np
import networkx as nx
import src.DomiRank as dr


class PY_DOMIRANK:
    def __init__(
        self,
        analytical: bool = False,
        sigma: Optional[float] = None,
        auto_sigma: bool = True,
        dt: float = 0.1,
        epsilon: float = 1e-5,
        max_iter_1: int = 100,
        max_iter_2: int = 1000,
        check_step: int = 10,
        sampling: int = 0,
    ) -> None:
        self._analytical = analytical
        self._sigma = sigma
        self._auto_sigma = auto_sigma
        self._dt = dt
        self._epsilon = epsilon
        self._max_iter_1 = max_iter_1
        self._max_iter_2 = max_iter_2
        self._check_step = check_step
        self._sampling = sampling

    def _compute_sigma(self, adjacency):
        sigma = self._sigma
        if sigma is not None:
            return sigma
        if not self._auto_sigma:
            return -1
        sigma_candidate = dr.optimal_sigma(
            adjacency,
            analytical=self._analytical,
            dt=self._dt,
            epsilon=self._epsilon,
            maxIter=self._max_iter_1,
            checkStep=self._check_step,
            sampling=self._sampling,
        )
        if isinstance(sigma_candidate, tuple):
            sigma = sigma_candidate[0]
        else:
            sigma = sigma_candidate
        return sigma

    def get_solution(self, network: nx.Graph) -> List[int]:
        node_order = list(network.nodes())
        if not node_order:
            return []

        adjacency = nx.to_scipy_sparse_array(network, nodelist=node_order)
        sigma = self._compute_sigma(adjacency)

        success, scores = dr.domirank(
            adjacency,
            analytical=self._analytical,
            sigma=sigma,
            dt=self._dt,
            epsilon=self._epsilon,
            maxIter=self._max_iter_2,
            checkStep=self._check_step,
        )

        if not success:
            # Retry with iterative mode if analytical solver fails to converge.
            success, scores = dr.domirank(
                adjacency,
                analytical=False,
                sigma=sigma,
                dt=self._dt,
                epsilon=self._epsilon,
                maxIter=self._max_iter_2,
                checkStep=self._check_step,
            )

        if not success:
            degree_order = sorted(network.degree(node_order), key=lambda item: item[1], reverse=True)
            return [node for node, _ in degree_order]

        centrality = np.asarray(scores).reshape(-1)
        sorted_indices = np.argsort(-centrality)
        return [node_order[idx] for idx in sorted_indices]
