import random
from typing import Tuple, Union, Dict, List, Optional

import pandas as pd
from absl import flags

import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from attack_resilience_complex_networks.env import EvalCN
from attack_resilience_complex_networks.utils.config import Config
from attack_resilience_complex_networks.utils.policy import get_policy_kwargs
from attack_resilience_complex_networks.eval_utils import get_model, get_budget_env, get_model_path_rl_eval_env
from attack_resilience_complex_networks.agent.policy import ExplainPolicy, ExplainValue

FLAGS = flags.FLAGS


def get_explain_policy(cfg: Config,
                       env: Union[DummyVecEnv, VecNormalize],
                       model_path: Optional[str]) -> ExplainPolicy:
    test_model = get_model(cfg, env)
    if model_path:
        trained_model = PPO.load(model_path)
        test_model.set_parameters(trained_model.get_parameters())
    agent = test_model
    params = agent.policy.state_dict()
    num_node_features = env.env_method('get_num_node_features')[0]
    policy_kwargs = get_policy_kwargs(cfg, num_node_features, explain=True, observation_space=env.observation_space)
    device = test_model.policy.device
    if not FLAGS.explain_value:
        explain_agent = ExplainPolicy(device, **policy_kwargs)
    else:
        explain_agent = ExplainValue(device, **policy_kwargs)
    explain_agent.load_state_dict(params, strict=False)

    return explain_agent


def prepare_explain_sr() -> Tuple[Config, float, EvalCN, ExplainPolicy]:
    print('start evaluation.')

    th.manual_seed(FLAGS.global_seed)
    np.random.seed(FLAGS.global_seed)
    random.seed(FLAGS.global_seed)

    cfg = Config(FLAGS.cfg, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)
    budget, eval_env = get_budget_env(cfg)

    assert cfg.agent.startswith('rl')
    model_path, rl_eval_env = get_model_path_rl_eval_env(cfg, eval_env, FLAGS.model_path)

    explain_policy = get_explain_policy(cfg, rl_eval_env, model_path)
    return cfg, budget, eval_env, explain_policy


def compute_sr_x(obs: Dict[str, np.ndarray], env: EvalCN) -> pd.DataFrame:
    node_features = obs['node_features']  # (B, N, F)
    node_mask = obs['action_mask']  # (B, N)
    node_features = node_features[node_mask]
    degree = node_features[:, 0]
    if not FLAGS.explain_value:
        if FLAGS.has_dynamics:
            weighted_degree = node_features[:, 2]
            stable_state = node_features[:, -4]
            weighted_stable_state = node_features[:, -2]
            derivative = node_features[:, -3]
            weighted_derivative = node_features[:, -1]
            sr_x = pd.DataFrame(
                {'d': degree,
                 'w_d': weighted_degree,
                 's': stable_state,
                 'w_s': weighted_stable_state,
                 'g': derivative,
                 'w_g': weighted_derivative}
            )
        else:
            weighted_degree = node_features[:, 1]
            kcore = node_features[:, 2]
            weighted_kcore = node_features[:, 3]
            betweenness = node_features[:, 8]
            weighted_betweenness = node_features[:, 9]
            sr_x = pd.DataFrame(
                {'d': degree,
                 'w_d': weighted_degree,
                 'k': kcore,
                 'w_k': weighted_kcore,
                 'b': betweenness,
                 'w_b': weighted_betweenness}
            )
    else:
        mean_degree = degree.mean()
        mean_squared_degree = (degree ** 2).mean()
        mean_cubed_degree = (degree ** 3).mean()
        var_degree = mean_squared_degree - mean_degree ** 2
        max_degree = degree.max()

        weighted_degree = node_features[:, 2] if FLAGS.has_dynamics else node_features[:, 1]
        mean_weighted_degree = weighted_degree.mean()
        mean_squared_weighted_degree = (weighted_degree ** 2).mean()
        mean_cubed_weighted_degree = (weighted_degree ** 3).mean()
        var_weighted_degree = mean_squared_weighted_degree - mean_weighted_degree ** 2
        max_weighted_degree = weighted_degree.max()
        sr_x = pd.DataFrame(
            {'d_mean': [mean_degree],
             'd_squared_mean': [mean_squared_degree],
             'd_cubed_mean': [mean_cubed_degree],
             'd_var': [var_degree],
             'd_max': [max_degree],
             'w_d_mean': [mean_weighted_degree],
             'w_d_squared_mean': [mean_squared_weighted_degree],
             'w_d_cubed_mean': [mean_cubed_weighted_degree],
             'w_d_var': [var_weighted_degree],
             'w_d_max': [max_weighted_degree]}
        )

    return sr_x


def batch_graph(episode_data: List[Tuple[th.Tensor, th.Tensor, th.Tensor]],
                device: th.device) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    if len(episode_data) < 2:
        return episode_data[0]
    x, edge_index, edge_attr = zip(*episode_data)
    num_nodes = th.as_tensor([_x.shape[0] for _x in x]).to(device)
    x = th.cat(x, dim=0)
    num_nodes_cum = th.cumsum(num_nodes, dim=0)
    num_edges = th.as_tensor([_edge_index.shape[1] for _edge_index in edge_index]).to(device)
    edge_offset = th.repeat_interleave(num_nodes_cum[:-1], num_edges[1:])
    edge_index = th.cat(edge_index, dim=1)
    edge_index[:, -edge_offset.shape[0]:] += edge_offset
    edge_attr = th.cat(edge_attr, dim=0)
    return x, edge_index, edge_attr


def batch_sr_data(sr_data: List[Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]]) \
        -> Union[
               Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
               Tuple[pd.DataFrame, np.ndarray]]:
    if not FLAGS.explain_value:
        sr_x, sr_logit_y, sr_prob_y, sr_pair_pos_neg = zip(*sr_data)
        sr_x = pd.concat(sr_x, axis=0)
        sr_logit_y = np.concatenate(sr_logit_y, axis=0)
        sr_prob_y = np.concatenate(sr_prob_y, axis=0)
        sr_pair_pos_neg = np.concatenate(sr_pair_pos_neg, axis=0)
        return sr_x, sr_logit_y, sr_prob_y, sr_pair_pos_neg
    else:
        sr_x, sr_logit_y = zip(*sr_data)
        sr_x = pd.concat(sr_x, axis=0)
        sr_logit_y = np.concatenate(sr_logit_y, axis=0)
        return sr_x, sr_logit_y


def transform_obs(obs: Dict[str, np.ndarray], device: th.device) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    obs = {key: th.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    edge_index = obs['edge_index'].long()  # (B, 2, M)
    edge_attr = obs['edge_attr']  # (B, M)
    edge_mask = obs['edge_mask'].bool()  # (B, M)
    node_mask = obs['action_mask'].bool()  # (B, N)
    node_features = obs['node_features']  # (B, N, F)
    node_features = node_features[node_mask]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask]
    return node_features, edge_index, edge_attr


def evaluate_explain_policy(agent: ExplainPolicy, env: EvalCN,
                            single_step: bool = False, symbolic_regression: bool = False) \
        -> Union[
            Tuple[float, float, List[int], Tuple[th.Tensor, th.Tensor, th.Tensor]],
            Tuple[
                float, float, List[int], Tuple[th.Tensor, th.Tensor, th.Tensor],
                Union[
                    Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
                    Tuple[pd.DataFrame, np.ndarray]]]]:
    episode_data = []
    sr_data = []
    obs, _ = env.reset()
    sr_x = compute_sr_x(obs, env)
    x, edge_index, edge_attr = transform_obs(obs, agent.device)
    reward = -1.0
    done = False
    while not done:
        action, prob = agent.predict(x, edge_index, edge_attr, deterministic=FLAGS.deterministic)
        logit = agent(x, edge_index, edge_attr).cpu().numpy()
        prob = prob.cpu().numpy()
        num_nodes = x.shape[0]
        pair_pos_neg = np.full(num_nodes, -1)
        high_scored_nodes = prob > 0.001
        num_high_scored_nodes = high_scored_nodes.sum()
        pair_pos_neg[high_scored_nodes] = 0
        if num_high_scored_nodes > 0:
            extra = 0
        else:
            extra = 1
        pair_pos_neg[action.item()] = num_nodes - num_high_scored_nodes - extra
        if not FLAGS.explain_value:
            sr_data.append((sr_x, logit, prob, pair_pos_neg))
        else:
            sr_data.append((sr_x, logit))
        episode_data.append((x, edge_index, edge_attr))
        obs, reward, terminated, truncated, info = env.step(action.item())
        sr_x = compute_sr_x(obs, env)
        x, edge_index, edge_attr = transform_obs(obs, agent.device)
        done = terminated or truncated
    action_history = env.get_attacked_nodes()
    if FLAGS.explain_value:
        logit = agent(x, edge_index, edge_attr).cpu().numpy()
        sr_data.append((sr_x, logit))
    if not single_step:
        episode_data = batch_graph(episode_data, agent.device)
        sr_data = batch_sr_data(sr_data)
    else:
        episode_data = episode_data[0]
        sr_data = sr_data[0]
    if symbolic_regression:
        return reward, 0.0, action_history, episode_data, sr_data
    return reward, 0.0, action_history, episode_data

