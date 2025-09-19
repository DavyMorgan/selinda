from typing import List

import os
import sys

import networkx as nx

from attack_resilience_complex_networks.agent.reinsertion import reinsert

finder_root_path = os.path.join(os.getcwd(), 'thirdparty', 'FINDER')
finder_code_path = os.path.join(finder_root_path, 'code', 'FINDER_ND')
sys.path.append(finder_code_path)
finder_model_path = os.path.join(finder_code_path, 'models', 'nrange_30_50_iter_78000.ckpt')
from FINDER import FINDER


class PY_FINDER:
    def __init__(self, reinsertion: bool=False):
        self.dqn = FINDER()
        self.dqn.LoadModel(finder_model_path)
        self._cursor = 0
        self.reinsertion = reinsertion

    def get_solution(self, topology: nx.Graph, oneshot: bool = False) -> List[int]:
        if not oneshot:
            solution, _ = self.dqn.EvaluateRealData(topology, 0)
        else:
            solution, _ = self.dqn.EvaluateRealData(topology, 1)
        if self.reinsertion:
            solution = reinsert(topology, solution)
        return solution
