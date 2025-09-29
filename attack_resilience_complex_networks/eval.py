from absl import app, flags
import pickle
import time
from typing import Tuple, List

import numpy as np
import torch as th

from attack_resilience_complex_networks.eval_utils import (evaluate, get_agent, report_eval,
                                                           get_model_path_rl_eval_env, get_budget_env)
from attack_resilience_complex_networks.utils.config import Config
from attack_resilience_complex_networks.utils.exp import set_proc_and_seed
from attack_resilience_complex_networks.utils.obs import unpack_obs

flags.DEFINE_string('cfg', None, 'Configuration file.')
flags.DEFINE_integer('global_seed', None, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_string('root_dir', '/data/selinda', 'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_bool('tmp', False, 'Whether to use temporary storage.')
flags.DEFINE_bool('debug', False, 'Whether to use debug mode.')
flags.DEFINE_enum('agent', None,
                  ['random', 'degree', 'resilience', 'pagerank', 'domirank', 'finder', 'gdm', 'gnd', 'ei', 'ci', 'corehd',
                   'pagerank-state', 'eigen-state', 'state', 'rc-state',
                   'selinda-dynamic', 'selinda-topology', 'selinda-homogeneous', 'rl-gnn'],
                  'Agent type.')
flags.DEFINE_bool('oneshot', False, 'Whether to use oneshot test.')
flags.DEFINE_bool('reinsertion', False, 'Whether to use reinsertion.')
flags.DEFINE_bool('has_dynamics', True, 'Whether the network has a system dynamic.')
flags.DEFINE_string('model_path', None, 'Path to saved mode to evaluate.')
flags.DEFINE_bool('random_episode', False, 'Whether to use a random network in each episode.')
flags.DEFINE_integer('num_instances', 0, 'Number of instances to test.')
flags.DEFINE_multi_string('block_feature', None, 'The blocked feature that will be set as 0.')
flags.DEFINE_multi_integer('protected_nodes', None, 'The protected nodes.')
flags.DEFINE_multi_integer('pre_attacked_nodes', None, 'The nodes that have been attacked before the episode.')
flags.DEFINE_bool('case_study', False, 'Whether to print d and h values for case study.')
flags.DEFINE_bool('early_warning', False, 'Whether to study early warning.')
FLAGS = flags.FLAGS


def main(_):
    print('start evaluation.')
    set_proc_and_seed(FLAGS.global_seed)

    if FLAGS.agent in ['resilience', 'selinda-homogeneous', 'selinda-dynamic']:
        assert FLAGS.has_dynamics

    if FLAGS.early_warning:
        assert FLAGS.num_instances == 0
        custom_agent = FLAGS.agent
        if FLAGS.has_dynamics:
            FLAGS.agent = 'selinda-dynamic'
        else:
            FLAGS.agent = 'selinda-topology'
        cfg = Config(FLAGS.cfg, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)
        budget, eval_env = get_budget_env(cfg)
        agent = get_agent(cfg, None, None)
        _mean_reward, _, attacked_nodes, num_nodes_after_attack, _ = evaluate(agent, eval_env)
        obs, _ = eval_env.reset()
        node_features, _, _ = unpack_obs(obs)
        if FLAGS.has_dynamics:
            degree = node_features[:, 0]
            stable_state = node_features[:, -4]
            sr_score = degree * stable_state
            ew_thres = sr_score[attacked_nodes].sum()
        else:
            degree = node_features[:, 0]
            w_degree = node_features[:, 1]
            sr_score = degree ** 2 / w_degree.clip(min=1)
            ew_thres = sr_score[attacked_nodes].sum()
        print(f'optimal attack cost: {_mean_reward}')
        print(f'early warning threshold: {ew_thres}')
        FLAGS.agent = custom_agent

    all_results = []

    cfg = Config(FLAGS.cfg, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)
    budget, eval_env = get_budget_env(cfg)

    if cfg.agent.startswith('rl'):
        model_path, rl_eval_env = get_model_path_rl_eval_env(cfg, eval_env, FLAGS.model_path)
    else:
        model_path = None
        rl_eval_env = None

    agent = get_agent(cfg, rl_eval_env, model_path, FLAGS.reinsertion)

    def test_core() -> Tuple[float, float, List[int], List[float]]:
        with th.no_grad():
            start_time = time.time()
            if FLAGS.agent in ['pagerank', 'domirank', 'finder', 'gnd', 'ei', 'ci', 'corehd', 'selinda-topology'] or FLAGS.oneshot:
                _mean_reward, _, attacked_nodes, num_nodes_after_attack, proxy = evaluate(agent, eval_env)
            else:
                _mean_reward, _, attacked_nodes, num_nodes_after_attack, _ = evaluate(agent, eval_env)
                if FLAGS.has_dynamics:
                    d, h = eval_env.report_proxy_history()
                    b, s_lo, s_hi = eval_env.report_dynamic_proxy_history()
                    proxy = (d, h, b, s_lo, s_hi)
                else:
                    proxy = (None, None, eval_env.report_connectivity_history(), None, None)
            case_time = time.time() - start_time
            report_eval(case_time, cfg, eval_env, budget, _mean_reward, attacked_nodes, num_nodes_after_attack, proxy)
        if proxy:
            _network_proxy = proxy[3] if FLAGS.has_dynamics else proxy[2]
        else:
            _network_proxy = None
        return _mean_reward, case_time, attacked_nodes, _network_proxy

    if FLAGS.num_instances == 0:
        mean_reward, eval_time, attacked_nodes, network_proxy = test_core()
        if FLAGS.early_warning:
            ew_score = sr_score[attacked_nodes]
            ew_score = np.clip(ew_score.cumsum() / ew_thres, 0, 1)
            print(f'\t early warning score: {ew_score}')
            save_dict = {
                'network_proxy': network_proxy,
                'ew_score': ew_score
            }
            with open(f'{FLAGS.root_dir}/early_warning.pkl', 'wb') as f:
                pickle.dump(save_dict, f)
    else:
        eval_env.reset_instance_id()
        overall_eval_time = 0.0
        for _ in range(FLAGS.num_instances):
            mean_reward, eval_time, _, _ = test_core()
            overall_eval_time = overall_eval_time + eval_time
            all_results.append(mean_reward)
        print(f'all results: {all_results}')
        print(f'sum: {sum(all_results)}')
        print(f'avg: {sum(all_results)/len(all_results)}')
        print(f'eval time: {overall_eval_time}')
        print(f'avg eval time: {overall_eval_time/len(all_results)}')


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'cfg',
        'global_seed',
        'agent'
    ])
    app.run(main)

