import gzip
import pickle
from typing import Union, Tuple, Optional, Dict

import networkx as nx
import numpy as np
import scipy.sparse as sp


def normalized_laplacian(adjacency_matrix: np.ndarray):
    """
    Input adjacency_matrix: np.ndarray
    :return:  np.ndarray  D^-1/2 * ( D - A ) * D^-1/2 = I - D^-1/2 * ( A ) * D^-1/2
    """
    out_degree = adjacency_matrix.sum(1)
    int_degree = adjacency_matrix.sum(0)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.eye(adjacency_matrix.shape[0]) \
                  - np.diag(out_degree_sqrt_inv) @ adjacency_matrix @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def sparse_to_dense(obs: Dict[str, np.ndarray]) -> np.ndarray:
    edge_index = obs['edge_index']
    edge_attr = obs['edge_attr']
    edge_mask = obs['edge_mask']
    edge_attr = edge_attr[edge_mask]
    edge_index = edge_index[:, edge_mask]
    num_nodes = obs['action_mask'].sum()
    adjacency_matrix = sp.coo_matrix((edge_attr, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    adjacency_matrix = adjacency_matrix.toarray()
    return adjacency_matrix


def load_topology(filename: str, graph_type: str, dynamics_type: str, scale: float = 1.0) \
        -> Union[nx.Graph, Tuple[nx.Graph, Optional[np.ndarray]]]:
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rb') as f:
            topology_dict = pickle.load(f)
        topology = topology_dict[graph_type]
        if dynamics_type in ['gene', 'neuron', 'epidemic']:
            g = nx.from_numpy_array(topology)
            gcc = np.array(list(max(nx.connected_components(g), key=len)))
            g.remove_nodes_from(np.setdiff1d(np.arange(topology.shape[0]), gcc))
            g = nx.convert_node_labels_to_integers(g)
            topology = nx.to_numpy_array(g)
        elif dynamics_type is not None:
            raise ValueError(f'Unknown dynamics type: {dynamics_type}')
        topology *= scale
        topology = nx.from_numpy_array(topology)
        return topology
    else:
        assert filename.endswith('.txt')
        topology = nx.read_edgelist(filename)
        topology = nx.convert_node_labels_to_integers(topology)
        nx.set_edge_attributes(topology, 1.0, 'weight')
        return topology
