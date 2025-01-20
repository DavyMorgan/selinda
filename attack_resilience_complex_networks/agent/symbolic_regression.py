from typing import Dict

import numpy as np

from attack_resilience_complex_networks.utils.obs import unpack_obs


def sr(obs: Dict[str, np.ndarray], oneshot: bool = False) -> int:
    node_features, valid_actions, indices = unpack_obs(obs)
    if len(valid_actions):
        degree = node_features[:, 0]
        stable_state = node_features[:, -4]
        sr_score = degree*stable_state
        if not oneshot:
            # get the node with the highest score and with a true action mask
            action = indices[valid_actions][np.argmax(sr_score[valid_actions])]
            return action
        else:
            rank_one_shot = indices[valid_actions][np.argsort(sr_score[valid_actions])]
            return rank_one_shot[::-1].tolist()

    # If there is no valid choice, then `0` is returned which results in an
    # infeasible action ending the episode.
    return 0
