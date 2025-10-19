import gzip
import pickle
from typing import Union, Tuple, Optional, Dict

import networkx as nx
import numpy as np
import pandas as pd
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
    elif filename.endswith('.csv'):
        topology = pd.read_csv(filename).to_numpy()
        topology = generate_projection_net(topology)
        g = nx.from_numpy_array(topology)
        gcc = np.array(list(max(nx.connected_components(g), key=len)))
        g.remove_nodes_from(np.setdiff1d(np.arange(topology.shape[0]), gcc))
        g = nx.convert_node_labels_to_integers(g)
        topology = nx.to_numpy_array(g)
        topology *= scale
        topology = nx.from_numpy_array(topology)
        return topology
    else:
        assert filename.endswith('.txt')
        topology = nx.read_edgelist(filename)
        topology = nx.convert_node_labels_to_integers(topology)
        nx.set_edge_attributes(topology, 1.0, 'weight')
        return topology


def generate_projection_net(m: np.ndarray) -> np.ndarray:
    """
    :param m: bipartite graph adjacency matrix
    :param axis: axis to project on
    :return: projection graph adjacency matrix
    """

    axis = 0
    m_sum = m / (m.sum(axis=axis, keepdims=True) + 1e-8)

    memory_cost = m_sum.shape[0] * m_sum.shape[0] * m_sum.shape[1] * 8 / 1024 / 1024
    if memory_cost > 50:
        g = np.zeros((m.shape[axis], m.shape[axis]), dtype=np.float64)
        for i in range(0, g.shape[0] - 1):
            for j in range(i + 1, g.shape[0]):
                mij = np.sum((m_sum[i] * m_sum[j] != 0) * (m_sum[i] + m_sum[j]))
                g[i, j] = mij
                g[j, i] = mij
    else:
        mask = (m_sum[:, None, :] * m_sum[None, :, :] != 0).astype(np.float64)
        interaction = m_sum[:, None, :] + m_sum[None, :, :]
        g = (mask * interaction).sum(-1)
        g = g - np.diag(np.diag(g))

    return g
