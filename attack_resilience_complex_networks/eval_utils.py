import os
from typing import Tuple, Union, List, Optional

import numpy as np
import torch as th
from tqdm.rich import tqdm

import networkx as nx
from absl import flags

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from attack_resilience_complex_networks.env import EvalCN
from attack_resilience_complex_networks.utils.config import Config
from attack_resilience_complex_networks.utils.obs import unpack_obs

FLAGS = flags.FLAGS


def get_budget_env(cfg: Config) -> Tuple[float, EvalCN]:
    eval_env = EvalCN(cfg, 'test',
                      FLAGS.has_dynamics, FLAGS.random_episode, FLAGS.num_instances,
                      FLAGS.block_feature,
                      getattr(FLAGS, 'protected_nodes', None), getattr(FLAGS, 'pre_attacked_nodes', None))
    budget = eval_env.get_budget()

    eval_env.reset()
    print(f'env reset!')

    if FLAGS.debug:
        check_env(eval_env)

    return budget, eval_env


def get_model_path_rl_eval_env(cfg: Config, eval_env: EvalCN, model_path: Optional[str] = None) \
        -> Tuple[Optional[str], VecNormalize]:
    rl_eval_env = Monitor(eval_env)
    rl_eval_env = DummyVecEnv([lambda: rl_eval_env])
    rl_eval_env = VecNormalize(rl_eval_env, norm_obs=False, norm_reward=False)

    if model_path:
        model_path = os.path.join(cfg.root_dir, 'output', model_path)
    else:
        print('No model path provided, using random policy.')
    return model_path, rl_eval_env


def get_model(cfg: Config,
              env: DummyVecEnv,
              explain: bool=False) -> PPO:
    from attack_resilience_complex_networks.agent.policy import MaskedActorCriticPolicy
    from attack_resilience_complex_networks.utils.policy import get_policy_kwargs
    num_node_features = env.env_method('get_num_node_features')[0]
    policy_kwargs = get_policy_kwargs(cfg, explain=explain)
    model = PPO(MaskedActorCriticPolicy,
                env,
                verbose=1,
                policy_kwargs=policy_kwargs)
    return model


def get_agent(cfg: Config,
              env: Optional[Union[DummyVecEnv, VecNormalize]],
              model_path: Optional[str],
              reinsertion: bool = False):
    if cfg.agent == 'random':
        from attack_resilience_complex_networks.agent.heuristic import random_policy
        agent = random_policy
    elif cfg.agent == 'degree':
        from attack_resilience_complex_networks.agent.heuristic import degree_centrality
        agent = degree_centrality
    elif cfg.agent == 'resilience':
        from attack_resilience_complex_networks.agent.heuristic import resilience_centrality
        agent = resilience_centrality
    elif cfg.agent == 'state':
        from attack_resilience_complex_networks.agent.heuristic import state
        agent = state
    elif cfg.agent == 'rc-state':
        from attack_resilience_complex_networks.agent.heuristic import rc_state
        agent = rc_state
    elif cfg.agent == 'pagerank':
        from attack_resilience_complex_networks.agent.pagerank import PY_PAGERANK
        agent = PY_PAGERANK()
    elif cfg.agent == 'domirank':
        from attack_resilience_complex_networks.agent.domirank import PY_DOMIRANK
        agent = PY_DOMIRANK()
    elif cfg.agent == 'finder':
        from attack_resilience_complex_networks.agent.finder import PY_FINDER
        agent = PY_FINDER(reinsertion=reinsertion)
    elif cfg.agent == 'gdm':
        from attack_resilience_complex_networks.agent.gdm import PY_GDM
        agent = PY_GDM(reinsertion=reinsertion)
    elif cfg.agent == 'gnd':
        from attack_resilience_complex_networks.agent.gnd import PY_GND
        agent = PY_GND(reinsertion=reinsertion)
    elif cfg.agent == 'ei':
        from attack_resilience_complex_networks.agent.ei import PY_EI
        agent = PY_EI()
    elif cfg.agent == 'ci':
        from attack_resilience_complex_networks.agent.ci import PY_CI
        agent = PY_CI(use_reinsert=reinsertion)
    elif cfg.agent == 'corehd':
        from attack_resilience_complex_networks.agent.corehd import PY_COREHD
        agent = PY_COREHD()
    elif cfg.agent == 'pagerank-state':
        from attack_resilience_complex_networks.agent.pagerank_state import pagerank_state
        agent = pagerank_state
    elif cfg.agent == 'eigen-state':
        from attack_resilience_complex_networks.agent.eigen_state import eigen_state
        agent = eigen_state
    elif cfg.agent == 'selinda-dynamic':
        from attack_resilience_complex_networks.agent.symbolic_regression import sr
        agent = sr
    elif cfg.agent == 'selinda-topology':
        from attack_resilience_complex_networks.agent.sr import PY_SR
        agent = PY_SR(use_reinsert=reinsertion)
    elif cfg.agent == 'selinda-homogeneous':
        from attack_resilience_complex_networks.agent.heuristic import resilience_centrality_revised
        agent = resilience_centrality_revised
    elif cfg.agent == 'rl-gnn':
        assert env is not None
        test_model = get_model(cfg, env)
        if model_path:
            trained_model = PPO.load(model_path)
            test_model.set_parameters(trained_model.get_parameters())
        agent = test_model
    else:
        raise ValueError(f'Agent {cfg.agent} not supported.')
    return agent


def evaluate(agent,
             env: EvalCN) -> Tuple[float, float, List[int], int, Optional[Tuple]]:
    if FLAGS.agent == 'rl-gnn':
        return evaluate_ppo(agent, env)
    elif FLAGS.agent == 'gdm':
        return evaluate_gdm(agent, env)
    else:
        return evaluate_baseline(agent, env)


def evaluate_ppo(agent, env: EvalCN) -> Tuple[float, float, List[int], int, Optional[Tuple]]:
    obs, _ = env.reset()
    reward = -1.0
    done = False
    if not FLAGS.oneshot:
        num_nodes = env.get_num_nodes()
        with tqdm(total=num_nodes) as pbar:
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                pbar.update(1)
        action_history = env.get_attacked_nodes()
        num_nodes_after_attack = env.get_current_num_nodes()
    else:
        _, valid_actions, indices = unpack_obs(obs)
        if FLAGS.agent != 'gdm':
            obs, _ = agent.policy.obs_to_tensor(obs)
            actions = th.tensor(indices).to(agent.device)
            _, log_prob, _ = agent.policy.evaluate_actions(obs, actions)
            log_prob = log_prob.detach().cpu().numpy()
        else:
            _, log_prob = agent.predict(obs, deterministic=True)
        rank_one_shot = indices[valid_actions][np.argsort(log_prob[valid_actions])]
        solution = rank_one_shot[::-1].tolist()
        reward, _, action_history, num_nodes_after_attack, _, _, _, _, _ = env.evaluate_finder_solution(solution)
    return reward, 0.0, action_history, num_nodes_after_attack, None


def evaluate_gdm(agent, env: EvalCN) \
        -> Tuple[float, float, List[int], int, Tuple[List[float], List[float], List[float], List[float], List[float]]]:
    obs, _ = env.reset()
    _, valid_actions, indices = unpack_obs(obs)
    g = env.get_topology()
    nx.set_edge_attributes(g, 1.0, 'weight')
    _, log_prob = agent.predict(obs, deterministic=True)
    rank_one_shot = indices[valid_actions][np.argsort(log_prob[valid_actions])]
    solution = rank_one_shot[::-1].tolist()
    solution = agent.update_solution(g, solution)
    reward, _, action_history, num_nodes_after_attack, _, _, _, _, _ = env.evaluate_finder_solution(solution)
    return reward, 0.0, action_history, num_nodes_after_attack, None


def evaluate_finder(agent, env: EvalCN) \
        -> Tuple[float, float, List[int], int, Tuple[Union[float, List[float]], Union[float, List[float]], Union[float, List[float]], Union[float, List[float]], Union[float, List[float]]]]:
    obs, _ = env.reset()
    if env.cfg.agent in ['pagerank', 'domirank', 'finder', 'gnd', 'ei', 'ci', 'corehd', 'selinda-topology']:
        g = env.get_topology()
        nx.set_edge_attributes(g, 1.0, 'weight')
        if env.cfg.agent == 'finder' and FLAGS.oneshot:
            solution = agent.get_solution(g, oneshot=True)
        else:
            solution = agent.get_solution(g)
    elif FLAGS.oneshot:
        solution = agent(obs, oneshot=True)
    else:
        raise ValueError(f'Only support finder, gnd, ei, ci, corehd, selinda-topology and oneshot, but got {env.cfg.agent}.')
    reward, _, action_history, num_nodes, d, h, b, s_lo, s_hi = env.evaluate_finder_solution(
        solution, return_d_h_b_s_list=True)
    return reward, 0.0, action_history, num_nodes, (d, h, b, s_lo, s_hi)


def evaluate_baseline(
        agent,
        env: EvalCN) -> Tuple[float, float, List[int], int, Optional[Tuple]]:
    if env.cfg.agent in ['pagerank', 'domirank', 'finder', 'gnd', 'ei', 'ci', 'corehd', 'selinda-topology'] or FLAGS.oneshot:
        return evaluate_finder(agent, env)
    else:
        obs, _ = env.reset()
        reward = -1.0
        done = False
        num_nodes = env.get_num_nodes()
        with tqdm(total=num_nodes) as pbar:
            while not done:
                if env.cfg.agent in ['pagerank-state', 'eigen-state']:
                    g = env.get_topology()
                    nx.set_edge_attributes(g, 1.0, 'weight')
                    action = agent(g, obs)
                else:
                    action = agent(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                pbar.update(1)
        action_history = env.get_attacked_nodes()
        num_nodes_after_attack = env.get_current_num_nodes()
        return reward, 0.0, action_history, num_nodes_after_attack, None


def report_eval(eval_time: float,
                cfg: Config,
                eval_env: EvalCN,
                budget: float,
                mean_reward: float,
                attacked_nodes: List[int],
                num_nodes_after_attack: int,
                proxy: Tuple = None) -> None:
    print(f'method: {cfg.agent}')
    print(f'\t number of nodes: {eval_env.get_num_nodes()}')
    print(f'\t attack budget: {budget}')
    if eval_env.has_dynamics:
        print(f'\t attack cost: {mean_reward}')
    else:
        attack_cost = len(attacked_nodes)
        print(f'\t attack cost: {attack_cost}')
        print(f'\t ANC: {mean_reward}')
    print(f'\t attacked nodes (first 100): {attacked_nodes[:100]}')
    print(f'\t attacked nodes slim (first 100): {eval_env.get_slim_action_history()[:100]}')
    print(f'\t number of nodes after attack: {num_nodes_after_attack}')
    print(f'\t time: {eval_time}')
    if getattr(FLAGS, 'case_study', False):
        d, h, b_connectivity, s_lo, s_hi = proxy
        if FLAGS.has_dynamics:
            print(f'\t d: {d}')
            print(f'\t h: {h}')
            print(f'\t b: {b_connectivity}')
            print(f'\t s_lo: {s_lo}')
            print(f'\t s_hi: {s_hi}')
        else:
            print(f'\t connectivity: {b_connectivity}')
