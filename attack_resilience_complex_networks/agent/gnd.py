from typing import List

import os
import sys
gnd_root_path = os.path.join(os.getcwd(), 'thirdparty', 'GND')
sys.path.append(gnd_root_path)
import tempfile

import networkx as nx
import subprocess


class PY_GND:
    def __init__(self):
        self.executable = "GND"

    def get_solution(self, network: nx.Graph) -> List[int]:
        num_nodes = network.number_of_nodes()

        nodes = []
        temp_input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt')
        temp_output_file = tempfile.NamedTemporaryFile(mode='r', suffix='.txt')
        try:
            with open(temp_input_file.name, "w") as tmp:
                for edge in network.edges():
                    tmp.write(
                        "{} {}\n".format(
                            edge[0] + 1, edge[1] + 1
                        )
                    )

            cmds = [
                f"{gnd_root_path}/{self.executable}",
                f"{num_nodes}",
                f"{temp_input_file.name}",
                f"{temp_output_file.name}",
            ]


            subprocess.run(
                cmds,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)

            with open(temp_output_file.name, "r") as tmp:
                for line in tmp.readlines():
                    node = line.strip()

                    nodes.append(int(node) - 1)
        except Exception as e:
            print(e)
        finally:
            temp_input_file.close()
            temp_output_file.close()

        return nodes

