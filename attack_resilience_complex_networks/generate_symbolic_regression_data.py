import numpy as np
from absl import app, flags
import time

import torch as th

from attack_resilience_complex_networks.eval_utils import report_eval
from attack_resilience_complex_networks.explain_utils import (evaluate_explain_policy, batch_sr_data, prepare_explain_sr)

flags.DEFINE_string('cfg', None, 'Configuration file.')
flags.DEFINE_integer('global_seed', None, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_string('root_dir', '/data/selinda', 'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_bool('tmp', False, 'Whether to use temporary storage.')
flags.DEFINE_bool('debug', False, 'Whether to use debug mode.')
flags.DEFINE_bool('has_dynamics', True, 'Whether the network has system dynamics.')
flags.DEFINE_bool('single_step', False, 'Whether to only use the data of the first single step.')
flags.DEFINE_bool('explain_value', False, 'Whether to explain the value network.')
flags.DEFINE_string('model_path', None, 'Path to saved mode to evaluate.')
flags.DEFINE_bool('random_episode', False, 'Whether to use a random graph in each episode.')
flags.DEFINE_integer('num_instances', None, 'Number of instances to test.')
flags.DEFINE_multi_string('block_feature', None, 'The blocked feature that will be set as 0.')
flags.DEFINE_bool('deterministic', True, 'Whether to use deterministic mode for attack.')
flags.DEFINE_string('save_filename', './sr_data.csv', 'Path to save the symbolic regression data.')
FLAGS = flags.FLAGS


def main(_):
    cfg, budget, eval_env, explain_policy = prepare_explain_sr()

    def test_core():
        with th.no_grad():
            mean_reward, _, _attacked_nodes, _episode_data, _sr_data = evaluate_explain_policy(
                explain_policy, eval_env,
                single_step=FLAGS.single_step,
                symbolic_regression=True)
        num_nodes_after_attack = eval_env.get_current_num_nodes()
        eval_time = time.time() - start_time
        report_eval(eval_time, cfg, eval_env, budget, mean_reward, _attacked_nodes, num_nodes_after_attack)
        return _episode_data, _attacked_nodes, _sr_data

    start_time = time.time()
    if FLAGS.num_instances is None:
        _, attacked_nodes, sr_data = test_core()
        label_y = np.full(eval_env.get_num_nodes(), -1)
        label_y[attacked_nodes] = 1
        if not FLAGS.explain_value:
            step_y = np.zeros(len(attacked_nodes), dtype=np.int32)
        else:
            step_y = np.zeros(len(attacked_nodes) + 1, dtype=np.int32)
        step_y[-1] = 1
    else:
        eval_env.reset_instance_id()
        sr_data = []
        label_y = []
        step_y = []
        for _ in range(FLAGS.num_instances):
            _, instance_attacked_nodes, instance_sr_data = test_core()
            sr_data.append(instance_sr_data)
            instance_label_y = np.full(eval_env.get_num_nodes(), -1)
            instance_label_y[instance_attacked_nodes] = 1
            label_y.append(instance_label_y)
            if not FLAGS.explain_value:
                instance_step_y = np.zeros(len(instance_attacked_nodes), dtype=np.int32)
            else:
                instance_step_y = np.zeros(len(instance_attacked_nodes) + 1, dtype=np.int32)
            instance_step_y[-1] = 1
            step_y.append(instance_step_y)
        sr_data = batch_sr_data(sr_data)
        label_y = np.concatenate(label_y)
        step_y = np.concatenate(step_y)

    if not FLAGS.explain_value:
        sr_data_x, sr_data_logit_y, sr_data_prob_y, sr_data_pair_pos_neg = sr_data
    else:
        sr_data_x, sr_data_logit_y = sr_data
    sr_data_x = sr_data_x.reset_index(drop=True)

    sr_data = sr_data_x.copy()
    sr_data['logit_y'] = sr_data_logit_y
    sr_data['step_y'] = step_y
    if not FLAGS.explain_value:
        sr_data['prob_y'] = sr_data_prob_y
        sr_data['pair_pos_neg'] = sr_data_pair_pos_neg
        if FLAGS.single_step:
            sr_data['label_y'] = label_y
    sr_data.to_csv(FLAGS.save_filename, index=False)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'cfg',
        'global_seed',
        'agent'
    ])
    app.run(main)
