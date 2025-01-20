from absl import app, flags
import time

import torch as th

from torch_geometric.explain import Explainer, GNNExplainer

from attack_resilience_complex_networks.eval_utils import report_eval
from attack_resilience_complex_networks.explain_utils import evaluate_explain_policy, batch_graph, prepare_explain_sr

flags.DEFINE_string('cfg', None, 'Configuration file.')
flags.DEFINE_integer('global_seed', None, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_string('root_dir', '/data/selinda', 'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_bool('tmp', False, 'Whether to use temporary storage.')
flags.DEFINE_bool('debug', False, 'Whether to use debug mode.')
flags.DEFINE_bool('has_dynamics', True, 'Whether the network has system dynamics.')
flags.DEFINE_enum('explanation_type', 'model', ['model', 'phenomenon'], 'Agent type.')
flags.DEFINE_bool('single_step', False, 'Whether to only use the data of the first single step.')
flags.DEFINE_bool('explain_value', False, 'Whether to explain the value network.')
flags.DEFINE_string('model_path', None, 'Path to saved mode to evaluate.')
flags.DEFINE_bool('random_episode', False, 'Whether to use a random graph in each episode.')
flags.DEFINE_integer('num_instances', None, 'Number of instances to test.')
flags.DEFINE_multi_string('block_feature', None, 'The blocked feature that will be set as 0.')
flags.DEFINE_bool('deterministic', True, 'Whether to use deterministic mode for attack.')
FLAGS = flags.FLAGS


def main(_):
    cfg, budget, eval_env, explain_policy = prepare_explain_sr()

    def test_core():
        with th.no_grad():
            mean_reward, _, _attacked_nodes, _episode_data = evaluate_explain_policy(explain_policy, eval_env,
                                                                                     single_step=FLAGS.single_step)
        num_nodes_after_attack = eval_env.get_current_num_nodes()
        eval_time = time.time() - start_time
        report_eval(eval_time, cfg, eval_env, budget, mean_reward, _attacked_nodes, num_nodes_after_attack)
        return _episode_data, _attacked_nodes

    start_time = time.time()
    if FLAGS.num_instances is None:
        episode_data, attacked_nodes = test_core()
        y = th.zeros(eval_env.get_num_nodes()).to(explain_policy.device)
        y[attacked_nodes] = 1
    else:
        eval_env.reset_instance_id()
        episode_data = []
        y = []
        for _ in range(FLAGS.num_instances):
            instance_episode_data, instance_attacked_nodes = test_core()
            episode_data.append(instance_episode_data)
            instance_y = th.zeros(eval_env.get_num_nodes()).to(explain_policy.device)
            instance_y[instance_attacked_nodes] = 1
            y.append(instance_y)
        episode_data = batch_graph(episode_data, explain_policy.device)
        y = th.stack(y, dim=0)

    if FLAGS.explanation_type == 'phenomenon':
        assert FLAGS.single_step
        explanation_type = 'phenomenon'
        target = y
    else:
        explanation_type = 'model'
        target = None

    if not FLAGS.explain_value:
        task_level = 'node'
        mode = 'binary_classification'
    else:
        task_level = 'graph'
        mode = 'regression'

    explainer = Explainer(
        model=explain_policy,
        algorithm=GNNExplainer(epochs=5000, lr=0.001),
        explanation_type=explanation_type,
        node_mask_type='attributes',
        edge_mask_type=None,
        model_config=dict(
            mode=mode,
            task_level=task_level,
            return_type='raw',
        ),
    )

    x, edge_index, edge_attr = episode_data

    explanation = explainer(x, edge_index, target=target, edge_attr=edge_attr)
    print(f'Generated explanations in {explanation.available_explanations}')

    path = 'feature_importance.png'
    explanation.visualize_feature_importance(path)
    print(f"Feature importance plot has been saved to '{path}'")


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'cfg',
        'global_seed',
    ])
    app.run(main)

