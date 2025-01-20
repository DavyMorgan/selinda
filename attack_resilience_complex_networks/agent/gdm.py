from typing import Dict, List

import os
import sys
gdm_root_path = os.path.join(os.getcwd(), 'thirdparty', 'GDM-slim')
sys.path.append(gdm_root_path)
gdm_model_path = os.path.join(
    gdm_root_path,
    'Fchi_degree_clustering_coefficient_degree_kcore_CL20_20_20_20_H1_1_1_1_FL40_30_20_1_CTrue_True_True_True_NS0.2_0.2_0.2_0.2_D0.3_0.3_0.3_0.3_BTrue_True_True_True_S0_L0.003_WD1e-05_E50_SNone.h5')

from GAT import GAT_Model

import numpy as np
import networkx as nx
import torch as th


def chi(o, e):
    if e == 0:
        return 0
    return (o - e) ** 2 / e


def compute_gdm_features(nodes: np.ndarray, edge_index: np.ndarray):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    edge_list = edge_index.T.tolist()
    g.add_edges_from(edge_list)
    degree = np.array([d for (n, d) in g.degree()])
    max_degree = np.max(degree)
    degree = np.divide(degree, max_degree)
    average_degree = np.mean(degree)
    chi_degree = np.array([chi(d, average_degree) for d in degree])
    core_number = nx.core_number(g)
    kcore = np.array([core_number[n] for n in g.nodes])
    max_kcore = np.max(kcore)
    kcore = np.divide(kcore, max_kcore)
    cc = nx.clustering(g)
    clustering_coefficient = np.array([cc[n] for n in g.nodes])
    gdm_features = np.column_stack([chi_degree, clustering_coefficient, degree, kcore])
    return gdm_features


def transform_obs(obs: Dict):
    edge_index = obs['edge_index'].astype(int)  # (2, M)
    edge_mask = obs['edge_mask'].astype(bool)  # (M)
    edge_index = edge_index[:, edge_mask]

    num_nodes = obs['node_mask'].sum()
    indices = np.arange(num_nodes)
    action_mask = obs['action_mask'].astype(bool)  # (N)
    action_mask = action_mask[:num_nodes]

    node_features = compute_gdm_features(indices, edge_index)
    node_features = th.from_numpy(node_features).float()
    edge_index = th.from_numpy(edge_index).long()
    return node_features, edge_index, action_mask, indices


class PY_GDM:
    def __init__(self):
        self.model = GAT_Model()
        self.model.load_state_dict(th.load(gdm_model_path, map_location='cpu'))

    def predict(self, obs: Dict, deterministic=True):
        x, edge_index, action_mask, indices = transform_obs(obs)
        pred = self.model(x, edge_index).numpy()
        action = indices[action_mask][np.argmax(pred[action_mask])]
        return action, pred

