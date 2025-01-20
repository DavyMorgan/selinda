from typing import List

import os
import sys
ci_root_path = os.path.join(os.getcwd(), 'thirdparty', 'CI')
sys.path.append(ci_root_path)
import tempfile

import networkx as nx
import subprocess


class PY_CI:
    def __init__(self, l: int = 2, stop_condition: int = 1, oneshot: bool = True, use_reinsert: bool = False):
        self.executable = "CI"
        self._l = l
        self._stop_condition = stop_condition
        self._oneshot = oneshot
        self._use_reinsert = use_reinsert

    def get_solution(self, network: nx.Graph) -> List[int]:

        nodes = []
        temp_input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt')
        temp_output_file = tempfile.NamedTemporaryFile(mode='r', suffix='.txt')
        try:
            with open(temp_input_file.name, "w") as tmp:
                for node in network.nodes():
                    tmp.write(f"{node+1}")
                    for neighbor in network.neighbors(node):
                        tmp.write(f" {neighbor+1}")
                    tmp.write("\n")

            cmds = [
                f"{ci_root_path}/{self.executable}",
                f"{temp_input_file.name}",
                f"{self._l}",
                f"{self._stop_condition}",
                f"{temp_output_file.name}",
                f"{int(self._oneshot)}",
                f"{int(self._use_reinsert)}",
            ]

            subprocess.run(
                cmds,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)

            with open(temp_output_file.name, "r") as tmp:
                for line in tmp.readlines():
                    node = line.strip().split(" ")[1]

                    nodes.append(int(node)-1)
        except Exception as e:
            print(e)
        finally:
            temp_input_file.close()
            temp_output_file.close()

        return nodes

