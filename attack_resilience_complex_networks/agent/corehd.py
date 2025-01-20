from typing import List

import os
import sys
corehd_root_path = os.path.join(os.getcwd(), 'thirdparty', 'CoreHD')
sys.path.append(corehd_root_path)
import tempfile

import networkx as nx
import subprocess
from subprocess import check_output


class PY_COREHD:
    def __init__(self):
        self.executable = "coreHD"

    def get_solution(self, network: nx.Graph) -> List[int]:
        num_nodes = network.number_of_nodes()
        num_edges = network.number_of_edges()

        nodes = []
        temp_input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt')
        temp_feedback_file = tempfile.NamedTemporaryFile(mode='r', suffix='.txt')
        temp_time_file = tempfile.NamedTemporaryFile(mode='r', suffix='.txt')
        temp_output_file = tempfile.NamedTemporaryFile(mode='r', suffix='.txt')
        try:
            with open(temp_input_file.name, "w") as tmp:
                tmp.write(f"{num_nodes} {num_edges}\n")
                for edge in network.edges():
                    tmp.write(
                        "{} {}\n".format(
                            edge[0] + 1, edge[1] + 1
                        )
                    )

            cmds = [
                f"{corehd_root_path}/{self.executable} "
                f'--NetworkFile "{temp_input_file.name}" '
                f"--VertexNumber {num_nodes} "
                f"--EdgeNumber {num_edges} "
                f'--Afile "{temp_output_file.name}" '
                f'--FVSfile "{temp_feedback_file.name}" '
                f'--Timefile "{temp_time_file.name}" '
                f'--Csize 1'
                # f'--seed {kwargs["seed"]} '
                #     int rdseed = 93276792; //you can set this seed to another value
                #     int prerun = 14000000; //you can set it to another value
            ]

            for cmd in cmds:
                check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)

            with open(temp_output_file.name, "r") as tmp:
                lines = tmp.readlines()
                for line in lines[2:]:
                    node = line.strip()

                    nodes.append(int(node) - 1)

        except Exception as e:
            print(e)
        finally:
            temp_input_file.close()
            temp_feedback_file.close()
            temp_time_file.close()
            temp_output_file.close()

        return nodes

