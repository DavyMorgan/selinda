from typing import List

import os
import sys
ei_root_path = os.path.join(os.getcwd(), 'thirdparty', 'EI')
sys.path.append(ei_root_path)
import tempfile

import numpy as np
import networkx as nx
import subprocess


class PY_EI:
    def __init__(self):
        self.executable = "exploimmun"

    def get_solution(self, network: nx.Graph) -> List[int]:
        num_nodes = network.number_of_nodes()

        nodes = []
        try:
            temp_input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt')
            temp_output_file = tempfile.NamedTemporaryFile(mode='r', suffix='.txt')
            temp_thres_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt')

            with open(temp_input_file.name, "w") as tmp:
                tmp.write("{}\n".format(num_nodes))
                for edge in network.edges():
                    tmp.write(
                        "{} {}\n".format(
                            edge[0], edge[1]
                        )
                    )

            cmds = [
                f"{ei_root_path}/{self.executable}",
                f"{num_nodes}",
                f"{temp_input_file.name}",
                f"{temp_output_file.name}",
                "1",
                "1",
                f"{temp_thres_file.name}",
            ]

            subprocess.run(
                cmds,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)

            with open(temp_output_file.name, "r") as tmp:
                for line in tmp.readlines():
                    node = line.strip()

                    nodes.append(int(node))

            all_nodes = np.arange(num_nodes)
            mask = np.ones_like(all_nodes, dtype=bool)
            mask[nodes] = False
            selected_nodes = all_nodes[mask]
        except Exception as e:
            print(e)
        finally:
            temp_input_file.close()
            temp_output_file.close()
            temp_thres_file.close()

        return selected_nodes

