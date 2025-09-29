from typing import Dict, List, Union

import numpy as np

from attack_resilience_complex_networks.utils.obs import unpack_obs


def random_policy(obs: Dict[str, np.ndarray]) -> int:
    _, valid_actions, _ = unpack_obs(obs)
    if len(valid_actions):
        action = np.random.choice(valid_actions)
        return action

    # If there is no valid choice, then `0` is returned which results in an
    # infeasable action ending the episode.
    return 0


def degree_centrality(obs: Dict[str, np.ndarray], oneshot: bool = False) -> Union[int, List[int]]:
    node_features, valid_actions, indices = unpack_obs(obs)
    if len(valid_actions):
        degree = node_features[:, 0]
        if not oneshot:
            # get the node with the highest degree and with a true action mask
            action = indices[valid_actions][np.argmax(degree[valid_actions])]
            return action
        else:
            rank_one_shot = indices[valid_actions][np.argsort(degree[valid_actions])]
            return rank_one_shot[::-1].tolist()

    # If there is no valid choice, then `0` is returned which results in an
    # infeasible action ending the episode.
    return 0


def resilience_centrality(obs: Dict[str, np.ndarray], oneshot: bool = False) -> Union[int, List[int]]:
    node_features, valid_actions, indices = unpack_obs(obs)
    if len(valid_actions):
        resilience = node_features[:, 3]
        if not oneshot:
            # get the node with the highest resilience centrality value and with a true action mask
            action = indices[valid_actions][np.argmax(resilience[valid_actions])]
            return action
        else:
            rank_one_shot = indices[valid_actions][np.argsort(resilience[valid_actions])]
            return rank_one_shot[::-1].tolist()

    # If there is no valid choice, then `0` is returned which results in an
    # infeasible action ending the episode.
    return 0


def resilience_centrality_revised(obs: Dict[str, np.ndarray], oneshot: bool = False) -> Union[int, List[int]]:
    node_features, valid_actions, indices = unpack_obs(obs)
    if len(valid_actions):
        resilience = node_features[:, 3]
        degree = node_features[:, 0]
        weighted_degree = node_features[:, 2]
        resilience_revised = resilience + degree * (weighted_degree - degree)
        if not oneshot:
            # get the node with the highest resilience centrality value and with a true action mask
            action = indices[valid_actions][np.argmax(resilience_revised[valid_actions])]
            return action
        else:
            rank_one_shot = indices[valid_actions][np.argsort(resilience_revised[valid_actions])]
            return rank_one_shot[::-1].tolist()

    # If there is no valid choice, then `0` is returned which results in an
    # infeasible action ending the episode.
    return 0


def state(obs: Dict[str, np.ndarray], oneshot: bool = False) -> int:
    node_features, valid_actions, indices = unpack_obs(obs)
    if len(valid_actions):
        stable_state = node_features[:, -4]
        sr_score = stable_state
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


def rc_state(obs: Dict[str, np.ndarray], oneshot: bool = False) -> Union[int, List[int]]:
    node_features, valid_actions, indices = unpack_obs(obs)
    if len(valid_actions):
        resilience = node_features[:, 3]
        stable_state = node_features[:, -4]
        sr_score = resilience * stable_state
        if not oneshot:
            # get the node with the highest resilience centrality value and with a true action mask
            action = indices[valid_actions][np.argmax(sr_score[valid_actions])]
            return action
        else:
            rank_one_shot = indices[valid_actions][np.argsort(sr_score[valid_actions])]
            return rank_one_shot[::-1].tolist()

    # If there is no valid choice, then `0` is returned which results in an
    # infeasible action ending the episode.
    return 0

