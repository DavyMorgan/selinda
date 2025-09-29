import numpy as np
import networkx as nx

from attack_resilience_complex_networks.utils.obs import unpack_obs


def eigen_state(network: nx.Graph, obs: dict[str, np.ndarray], oneshot: bool=False) -> int:

    node_features, valid_actions, indices = unpack_obs(obs)
    if len(valid_actions):
        eigen_centrality = nx.eigenvector_centrality_numpy(network)
        eigen_score = np.array([eigen_centrality[i] for i in range(len(eigen_centrality))])
        stable_state = node_features[:, -4][valid_actions]
        sr_score = eigen_score * stable_state
        if not oneshot:
            # get the node with the highest score and with a true action mask
            action = indices[valid_actions][np.argmax(sr_score)]
            return action
        else:
            rank_one_shot = indices[valid_actions][np.argsort(sr_score)]
            return rank_one_shot[::-1].tolist()

    # If there is no valid choice, then `0` is returned which results in an
    # infeasible action ending the episode.
    return 0
