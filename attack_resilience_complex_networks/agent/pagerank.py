from typing import List

import networkx as nx


class PY_PAGERANK:
    def __init__(self):
        self._name = "PageRank"

    def get_solution(self, network: nx.Graph) -> List[int]:

        nodes = []
        pr = nx.pagerank(network)
        pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        for node, _ in pr_sorted:
            nodes.append(node)
        return nodes
