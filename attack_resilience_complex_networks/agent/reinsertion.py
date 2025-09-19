from typing import List

import os
import sys
reinsertion_root_path = os.path.join(os.getcwd(), 'thirdparty', 'reinsertion')
executable = "reinsertion"
sys.path.append(reinsertion_root_path)
import tempfile

import networkx as nx
import subprocess


def reinsert(network: nx.Graph, solution: List[int]):
    nodes = []
    temp_input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt')
    temp_output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt')
    temp_output_file2 = tempfile.NamedTemporaryFile(mode='r', suffix='.txt')
    try:
        with open(temp_input_file.name, "w") as tmp:
            for edge in network.edges():
                tmp.write(
                    "{} {}\n".format(
                        edge[0] + 1, edge[1] + 1
                    )
                )

        with open(temp_output_file.name, "w") as tmp:
            for node in solution:
                tmp.write(
                    "{}\n".format(
                        node + 1
                    )
                )

        cmds = [
            f"{reinsertion_root_path}/{executable}",
            f"{temp_input_file.name}",
            f"{temp_output_file.name}",
            f"{temp_output_file2.name}",
        ]

        subprocess.run(
            cmds,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)
        output_filename = temp_output_file2.name

        with open(output_filename, "r") as tmp:
            for line in tmp.readlines():
                node = line.strip()

                nodes.append(int(node) - 1)
    except Exception as e:
        print(e)
    finally:
        temp_input_file.close()
        temp_output_file.close()
        temp_output_file2.close()

    return nodes