from typing import Dict

import numpy as np

from attack_resilience_complex_networks.env.cnc import ComplexNetworkClient
from attack_resilience_complex_networks.utils.config import Config


class ObsExtractor:
    def __init__(self, cfg: Config, cnc: ComplexNetworkClient, dynamics_type: str,
                 num_nodes: int, num_edges: int, norm_obs: bool = False):
        self.cfg = cfg
        self._cnc = cnc
        self._dynamics_type = dynamics_type
        self._num_nodes = num_nodes
        self._num_edges = num_edges
        self._norm_obs = norm_obs

    @staticmethod
    def _normalize_obs(obs: np.ndarray) -> np.ndarray:
        obs -= obs.min(axis=0)
        obs /= obs.max(axis=0) + 1e-6
        obs = 2 * obs - 1

        return obs

    def get_obs(self) -> Dict[str, np.ndarray]:
        edge_index, edge_attr = self._cnc.get_edge_pyg()
        actual_num_edges = len(edge_attr)
        edge_index = np.pad(edge_index,
                            ((0, 0), (0, self._num_edges - edge_index.shape[1])),
                            'constant',
                            constant_values=self._num_nodes - 1,)
        edge_attr = np.pad(edge_attr,
                           (0, self._num_edges - edge_attr.shape[0]),
                           'constant',
                           constant_values=0,).astype(np.float32)
        edge_mask = np.ones((actual_num_edges,), dtype=bool)
        edge_mask = np.pad(edge_mask,
                           (0, self._num_edges - actual_num_edges),
                           'constant',
                           constant_values=False,)
        node_mask = np.zeros((self._num_nodes,), dtype=bool)
        actual_num_nodes = self._cnc.get_current_num_nodes()
        node_mask[:actual_num_nodes] = True
        action_mask = self._cnc.get_action_mask_pyg()
        action_mask = np.pad(action_mask,
                             (0, self._num_nodes - action_mask.shape[0]),
                             'constant',
                             constant_values=False,)

        if self._dynamics_type is not None:
            obs_nodes = self._cnc.get_nil_obs_pyg()
            obs_nodes_dynamics_param = self._cnc.get_dynamics_param_obs_pyg()
            obs_nodes = np.concatenate([obs_nodes, obs_nodes_dynamics_param], axis=1)
            obs_nodes_env = self._cnc.get_env_obs()
            obs_nodes = np.concatenate([obs_nodes, obs_nodes_env], axis=1)
        else:
            obs_nodes = self._cnc.get_topology_obs_pyg()
        if self._norm_obs:
            obs_nodes = self._normalize_obs(obs_nodes)
        obs_nodes = np.pad(obs_nodes,
                           ((0, self._num_nodes - obs_nodes.shape[0]), (0, 0)),
                           'constant',
                           constant_values=0,).astype(np.float32)
        obs = {
            'node_features': obs_nodes,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_mask': edge_mask,
            'node_mask': node_mask,
            'action_mask': action_mask,
        }
        return obs
