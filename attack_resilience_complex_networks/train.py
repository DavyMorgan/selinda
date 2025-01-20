import os

from absl import app, flags
import time
from typing import Union, Optional

import numpy as np
import torch as th

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper, DummyVecEnv, VecCheckNan
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from attack_resilience_complex_networks.env import CN, EvalCN
from attack_resilience_complex_networks.utils.config import Config
from attack_resilience_complex_networks.utils.exp import set_proc_and_seed
from attack_resilience_complex_networks.utils.policy import get_policy_kwargs
from attack_resilience_complex_networks.agent.policy import MaskedActorCriticPolicy
from attack_resilience_complex_networks.utils.callback import (HParamCallback,
                                                               UpdateValEnvAndStopTrainingOnNoModelImprovement,
                                                               UpdateValEnv)

flags.DEFINE_string('cfg', None, 'Configuration file.')
flags.DEFINE_integer('global_seed', None, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_bool('debug', False, 'Whether to use debug mode.')
flags.DEFINE_string('root_dir', '/data/selinda', 'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_bool('tmp', False, 'Whether to use temporary storage.')
flags.DEFINE_bool('save_ckpt', True, 'Whether to save checkpoints.')
flags.DEFINE_bool('reset_num_timesteps', True, 'Whether to reset the current timestamp number.')
flags.DEFINE_integer('save_freq', 10000, 'Save ckpt every save_freq steps.')
flags.DEFINE_bool('validate', True, 'Whether to evaluate on validation set during training.')
flags.DEFINE_integer('val_freq', 5000, 'Test on validation set every val_freq steps.')
flags.DEFINE_bool('early_stop', True, 'Whether to stop training if no improvements are made.')
flags.DEFINE_integer('early_stop_patience', 10, 'Patience of early stop.')
flags.DEFINE_integer('early_stop_min_num_vals', 50, 'Patience of early stop.')
flags.DEFINE_bool('has_dynamics', True, 'Whether the network has system dynamics.')
flags.DEFINE_integer('num_envs', 20, 'Number of environments for parallel training.')
flags.DEFINE_float('lr', 3e-4, 'Learning rate.')
flags.DEFINE_integer('steps_per_iteration', 5000, 'Number of timestamps per training iteration.')
flags.DEFINE_integer('batch_size', 100, 'Mini-batch size.')
flags.DEFINE_integer('optim_epochs_per_iteration', 10, 'Number of epochs for optimization per iteration.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_float('gae_lambda', 0.95, 'Factor for trade-off of bias vs variance for Generalized Advantage Estimator.')
flags.DEFINE_float('ent_coef', 0.01, 'Weight for entropy loss.')
flags.DEFINE_float('vf_coef', 0.5, 'Weight for value loss.')
flags.DEFINE_integer('train_steps', 1_000_000, 'Total number of training steps.')
flags.DEFINE_bool('normalize_reward', True, 'Whether to normalize reward during training.')
flags.DEFINE_bool('random_episode', False, 'Whether to use a random graph in each episode.')
flags.DEFINE_integer('num_instances', None, 'Number of instances to train/test.')
flags.DEFINE_multi_string('block_feature', None, 'The blocked feature that will be set as 0.')
FLAGS = flags.FLAGS


def get_model(cfg: Config,
              env: Union[VecEnvWrapper, DummyVecEnv, EvalCN],
              training: bool = True,
              load_from_file: bool = False,
              ckpt_path: str = None) -> PPO:
    policy_kwargs = get_policy_kwargs(cfg)
    tb_log_path = cfg.tb_log_path if training else None
    n_steps = max(FLAGS.steps_per_iteration // FLAGS.num_envs, 10) if training else 10
    if not load_from_file:
        model = PPO(MaskedActorCriticPolicy,
                    env,
                    learning_rate=FLAGS.lr,
                    n_steps=n_steps,
                    batch_size=FLAGS.batch_size,
                    n_epochs=FLAGS.optim_epochs_per_iteration,
                    gamma=FLAGS.gamma,
                    gae_lambda=FLAGS.gae_lambda,
                    ent_coef=FLAGS.ent_coef,
                    vf_coef=FLAGS.vf_coef,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tb_log_path)
    else:
        model = PPO.load(ckpt_path,
                         env=env,
                         learning_rate=FLAGS.lr,
                         n_steps=n_steps,
                         batch_size=FLAGS.batch_size,
                         n_epochs=FLAGS.optim_epochs_per_iteration,
                         gamma=FLAGS.gamma,
                         gae_lambda=FLAGS.gae_lambda,
                         ent_coef=FLAGS.ent_coef,
                         vf_coef=FLAGS.vf_coef,
                         verbose=1,
                         tensorboard_log=tb_log_path)
    return model


def get_best_model(cfg: Config) -> PPO:
    best_model_path = os.path.join(cfg.best_model_path, 'best_model.zip')
    model = PPO.load(best_model_path)
    return model


def get_callbacks(cfg: Config) -> Optional[CallbackList]:
    callback_list = []
    hparam_callback = HParamCallback()
    callback_list.append(hparam_callback)
    if FLAGS.save_ckpt:
        save_freq = max(FLAGS.save_freq // FLAGS.num_envs, 1)
        ckpt_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=cfg.ckpt_save_path,
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        callback_list.append(ckpt_callback)
    if FLAGS.validate:
        val_env = EvalCN(cfg, 'val',
                         has_dynamics=FLAGS.has_dynamics,
                         random_episode=False,
                         num_instances=FLAGS.num_instances,
                         block_feature=FLAGS.block_feature)
        val_num_instances = val_env.get_eval_num_instances()

        val_env = Monitor(val_env)
        val_env = DummyVecEnv([lambda: val_env])
        val_env = VecNormalize(val_env, norm_obs=False, norm_reward=False)
        if FLAGS.debug:
            val_env = VecCheckNan(val_env, raise_exception=True)

        if FLAGS.early_stop:
            callback_after_eval = UpdateValEnvAndStopTrainingOnNoModelImprovement(
                val_env=val_env,
                max_no_improvement_evals=FLAGS.early_stop_patience,
                min_evals=FLAGS.early_stop_min_num_vals,
            )
        else:
            callback_after_eval = UpdateValEnv(val_env=val_env)

        val_freq = max(FLAGS.val_freq // FLAGS.num_envs, 1)
        val_callback = EvalCallback(
            val_env,
            callback_after_eval=callback_after_eval,
            best_model_save_path=cfg.best_model_path,
            n_eval_episodes=val_num_instances,
            log_path=cfg.best_model_path,
            eval_freq=val_freq,
            deterministic=True,
            render=False)
        callback_list.append(val_callback)

    if len(callback_list) == 0:
        callback_list = None
    else:
        callback_list = CallbackList(callback_list)

    return callback_list


def main(_):
    set_proc_and_seed(FLAGS.global_seed)

    cfg = Config(FLAGS.cfg, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, 'rl-gnn', FLAGS.reset_num_timesteps)

    env = make_vec_env(CN, n_envs=FLAGS.num_envs, seed=FLAGS.global_seed,
                       env_kwargs={
                           'cfg': cfg,
                           'has_dynamics': FLAGS.has_dynamics,
                           'random_episode': FLAGS.random_episode,
                           'num_instances': FLAGS.num_instances,
                           'block_feature': FLAGS.block_feature})
    env = VecNormalize(env, norm_obs=False, norm_reward=FLAGS.normalize_reward)

    if FLAGS.debug:
        th.autograd.set_detect_anomaly(True)
        np.seterr(all='warn')
        env = VecCheckNan(env, raise_exception=True)

    if FLAGS.reset_num_timesteps:
        model = get_model(cfg, env)
    else:
        latest_model_path = os.path.join(cfg.latest_model_path, 'latest_model.zip')
        model = get_model(cfg, env, load_from_file=True, ckpt_path=latest_model_path)
    callback_list = get_callbacks(cfg)
    model.learn(
        total_timesteps=FLAGS.train_steps,
        callback=callback_list,
        tb_log_name=cfg.tb_log_name,
        reset_num_timesteps=FLAGS.reset_num_timesteps,
        progress_bar=True)
    latest_model_path = os.path.join(cfg.latest_model_path, 'latest_model.zip')
    model.save(latest_model_path)

    eval_env = EvalCN(cfg, 'test',
                      has_dynamics=FLAGS.has_dynamics,
                      random_episode=False,
                      num_instances=FLAGS.num_instances,
                      block_feature=FLAGS.block_feature)
    num_eval_instances = eval_env.get_eval_num_instances()
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)
    if FLAGS.debug:
        eval_env = VecCheckNan(eval_env, raise_exception=True)
    test_model = get_model(cfg, eval_env, training=False)
    trained_best_model = get_best_model(cfg)
    test_model.set_parameters(trained_best_model.get_parameters())
    start_time = time.time()
    mean_reward, _ = evaluate_policy(test_model, eval_env, n_eval_episodes=num_eval_instances)
    eval_time = time.time() - start_time
    if FLAGS.num_instances <= 1:
        print(f'\t number of nodes: {eval_env.env_method("get_num_nodes")[0]}')
        print(f'\t attack budget: {eval_env.env_method("get_budget")[0]}')
    if FLAGS.has_dynamics:
        print(f'\t attack cost: {mean_reward}')
    else:
        print(f'\t ANC: {mean_reward}')
    print(f'\t time: {eval_time}')


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'cfg',
        'global_seed'
    ])
    app.run(main)
