from typing import Dict, Tuple
import numpy as np


def unpack_obs(obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_features = obs['node_features']
    action_mask = obs['action_mask']
    valid_actions, = np.nonzero(action_mask)
    indices = np.arange(node_features.shape[0])
    return node_features, valid_actions, indices
