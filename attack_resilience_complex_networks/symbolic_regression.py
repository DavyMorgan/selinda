import numpy as np
from absl import app, flags

import pandas as pd
from pysr import PySRRegressor


flags.DEFINE_string('sr_data_filename', './sr_data.csv', 'Path to symbolic regression data.')
flags.DEFINE_enum('loss', 'L2',
                  ['L2', 'L1', 'LogitDist', 'Sigmoid', 'L1Hinge', 'Pairwise'],
                  'loss type.')
flags.DEFINE_enum('target', 'prob',
                  ['prob', 'logit', 'pair'],
                  'target type.')
flags.DEFINE_integer('num_nodes', 80, 'Number of nodes in the graph.')
flags.DEFINE_bool('single_step', False, 'Whether to only use the data of the first single step.')
flags.DEFINE_bool('explain_value', False, 'Whether to explain the value network.')
flags.DEFINE_multi_string('primitives', None, 'Variable names for symbolic regression.')
flags.DEFINE_multi_string('binary_operators', None, 'Binary operators for symbolic regression.')
flags.DEFINE_multi_string('unary_operators', [], 'Unary operators for symbolic regression.')
FLAGS = flags.FLAGS


def main(_):

    sr_data = pd.read_csv(FLAGS.sr_data_filename)
    variable_names = FLAGS.primitives
    sr_data_x = sr_data[variable_names].to_numpy()

    if FLAGS.explain_value:
        assert FLAGS.loss in ['L2', 'L1', 'LogitDist', 'Pairwise']
        assert FLAGS.target in ['logit', 'pair']

    if FLAGS.loss == 'Pairwise':
        assert FLAGS.target == 'pair'
        batching = False
    elif len(sr_data_x) > 10000:
        batching = True
    else:
        batching = False

    if FLAGS.target == 'prob':
        sr_data_y = sr_data['prob_y']
    elif FLAGS.target == 'logit':
        sr_data_y = sr_data['logit_y']
    elif FLAGS.target == 'pair':
        if not FLAGS.explain_value:
            if FLAGS.single_step:
                sr_data_y = sr_data['label_y']
                num_graphs = len(sr_data_y)//FLAGS.num_nodes
                sr_data_x_pair = []
                sr_data_y_pair = []
                num_samples = 0
                for graph_idx in range(num_graphs):
                    sr_data_x_i = sr_data_x[graph_idx*FLAGS.num_nodes:(graph_idx+1)*FLAGS.num_nodes, :]
                    sr_data_y_i = sr_data_y[graph_idx*FLAGS.num_nodes:(graph_idx+1)*FLAGS.num_nodes]
                    pos_data_x_i = sr_data_x_i[sr_data_y_i > 0]
                    pos_data_y_i = sr_data_y_i[sr_data_y_i > 0]
                    neg_data_x_i = sr_data_x_i[sr_data_y_i < 0]
                    neg_data_y_i = sr_data_y_i[sr_data_y_i < 0]
                    num_pos_i = (sr_data_y_i == 1).sum()
                    num_neg_i = FLAGS.num_nodes - num_pos_i
                    pos_data_x_i_repeat = pos_data_x_i.repeat(num_neg_i, axis=0)
                    pos_data_y_i_repeat = np.ones((len(pos_data_x_i_repeat),), dtype=pos_data_y_i.dtype)
                    neg_data_x_i_repeat = np.tile(neg_data_x_i, (num_pos_i, 1))
                    neg_data_y_i_repeat = np.zeros((len(neg_data_x_i_repeat),), dtype=neg_data_y_i.dtype)

                    sr_data_x_i_pair = np.empty(
                        (len(pos_data_x_i_repeat) + len(neg_data_x_i_repeat), pos_data_x_i_repeat.shape[1]),
                        dtype=pos_data_x_i_repeat.dtype)
                    sr_data_x_i_pair[0::2, :] = pos_data_x_i_repeat
                    sr_data_x_i_pair[1::2, :] = neg_data_x_i_repeat
                    sr_data_y_i_pair = np.empty(
                        (len(pos_data_y_i_repeat) + len(neg_data_y_i_repeat),),
                        dtype=pos_data_y_i_repeat.dtype)
                    sr_data_y_i_pair[0::2] = pos_data_y_i_repeat
                    sr_data_y_i_pair[1::2] = neg_data_y_i_repeat
                    sr_data_x_pair.append(sr_data_x_i_pair)
                    sr_data_y_pair.append(sr_data_y_i_pair)
                    print(len(sr_data_x_i_pair), len(sr_data_y_i_pair), num_pos_i*num_neg_i*2)
                    num_samples = num_samples + num_pos_i*num_neg_i
                    print(num_samples)
                sr_data_x = np.concatenate(sr_data_x_pair, axis=0)
                sr_data_y = np.concatenate(sr_data_y_pair, axis=0)
                print(len(sr_data_x), len(sr_data_y))
            else:
                sr_data_y = sr_data['pair_pos_neg']
                pos_data_x = sr_data_x[sr_data_y > 0]
                pos_data_y = sr_data_y[sr_data_y > 0]
                neg_data_x = sr_data_x[sr_data_y < 0]
                neg_data_y = sr_data_y[sr_data_y < 0]
                pos_data_x_repeat = pos_data_x.repeat(pos_data_y, axis=0)
                pos_data_y_repeat = np.ones((len(pos_data_x_repeat),), dtype=pos_data_y.dtype)
                sr_data_x_pair = np.empty((len(pos_data_x_repeat) + len(neg_data_x), pos_data_x_repeat.shape[1]),
                                          dtype=pos_data_x_repeat.dtype)
                sr_data_x_pair[0::2, :] = pos_data_x_repeat
                sr_data_x_pair[1::2, :] = neg_data_x
                sr_data_y_pair = np.empty((len(pos_data_y_repeat) + len(neg_data_y),), dtype=pos_data_y_repeat.dtype)
                sr_data_y_pair[0::2] = pos_data_y_repeat
                sr_data_y_pair[1::2] = neg_data_y
                sr_data_x = sr_data_x_pair
                sr_data_y = sr_data_y_pair
        else:
            sr_data_y = sr_data['step_y']
            pos_data_x = sr_data_x[sr_data_y == 0]
            neg_data_x = sr_data_x[1:][sr_data_y[:-1] == 0]
            sr_data_x_pair = np.empty((len(pos_data_x) + len(neg_data_x), pos_data_x.shape[1]),
                                      dtype=pos_data_x.dtype)
            sr_data_x_pair[0::2, :] = pos_data_x
            sr_data_x_pair[1::2, :] = neg_data_x
            sr_data_y_pair = np.empty((len(pos_data_x) + len(neg_data_x),), dtype=sr_data_y.dtype)
            sr_data_y_pair[0::2] = 1
            sr_data_y_pair[1::2] = 0
            sr_data_x = sr_data_x_pair
            sr_data_y = sr_data_y_pair

    else:
        raise ValueError(f'Unknown target type: {FLAGS.target}')

    print('start symbolic regression.')

    if FLAGS.loss == 'L2':
        loss = 'L2DistLoss()'
        objective = None
    elif FLAGS.loss == 'L1':
        loss = 'L1DistLoss()'
        objective = None
    elif FLAGS.loss == 'LogitDist':
        loss = 'LogitDistLoss()'
        objective = None
    elif FLAGS.loss == 'Sigmoid':
        loss = 'SigmoidLoss()'
        objective = None
    elif FLAGS.loss == 'L1Hinge':
        loss = 'L1HingeLoss()'
        objective = None
    elif FLAGS.loss == 'Pairwise':
        # a julia snippet to generate the Pairwise loss
        loss = None
        objective = f"""
        function PairwiseLoss(tree, dataset::Dataset{{T, L}}, options) where {{T, L}}
            prediction, flag = eval_tree_array(tree, dataset.X, options)
            if !flag
                return L(Inf)
            end
            prediction_pos = prediction[1:2:end]
            prediction_neg = prediction[2:2:end]
            num_pair = length(prediction_pos)
            margin = prediction_pos - prediction_neg
            margin_sigmoid = 1.0 ./ (1.0 .+ exp.(-margin))
            loss = -log.(margin_sigmoid)
            loss = sum(loss)/num_pair
            return loss
        end
        """
    else:
        raise ValueError(f'Unknown loss type: {FLAGS.loss}')

    model = PySRRegressor(
        niterations=100,  # < Increase me for better results
        populations=60,
        population_size=40,
        binary_operators=FLAGS.binary_operators,
        unary_operators=FLAGS.unary_operators,
        batching=batching,
        batch_size=5000,
        loss=loss,
        full_objective=objective,
    )
    if FLAGS.explain_value:
        sr_data_y = -sr_data_y
        print(sr_data_y)
    model.fit(sr_data_x, sr_data_y, variable_names=variable_names)
    print(model)
    print('Sympy expression:')
    for i in range(len(model.equations_)):
        print(f'{i}\t {model.sympy(i)}')


if __name__ == '__main__':
    app.run(main)
