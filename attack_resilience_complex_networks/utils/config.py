import os
import re
import yaml
import glob

from absl import flags
from stable_baselines3.common.utils import get_latest_run_id

FLAGS = flags.FLAGS


class TupleSafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


TupleSafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    TupleSafeLoader.construct_python_tuple)


def load_yaml(file_path):
    files = glob.glob(file_path, recursive=True)
    assert(len(files) == 1)
    cfg = yaml.load(open(files[0], 'r'), Loader=TupleSafeLoader)
    return cfg


class Config:

    def __init__(self, cfg_id: str, global_seed: int, tmp: bool, root_dir: str,
                 agent: str = 'rl-gnn', reset_num_timesteps: bool = True,
                 cfg_dict: dict = None, code_dir: str = None):
        self.cfg_id = cfg_id
        self.seed = global_seed
        self.code_dir = os.getcwd() if code_dir is None else code_dir
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            file_path = os.path.join(self.code_dir, f'attack_resilience_complex_networks/cfg/**/{self.cfg_id}.yaml')
            cfg = load_yaml(file_path)
        # create dirs
        self.root_dir = '/tmp/arc' if tmp else root_dir
        self.agent = agent

        self.tb_log_path = os.path.join(self.root_dir, 'runs')
        self.tb_log_name = f'{cfg_id}-agent-{agent}-seed-{global_seed}'
        latest_run_id = get_latest_run_id(self.tb_log_path, self.tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        self.cfg_dir = os.path.join(self.root_dir,
                                    'output', f'{cfg_id}-agent-{agent}-seed-{global_seed}_{latest_run_id + 1}')
        self.ckpt_save_path = os.path.join(self.cfg_dir, 'ckpt')
        self.best_model_path = os.path.join(self.cfg_dir, 'best-models')
        self.latest_model_path = os.path.join(self.cfg_dir, 'latest-models')


        # env
        self.env_specs = cfg.get('env_specs', dict())
        self.reward_specs = cfg.get('reward_specs', dict())
        self.obs_specs = cfg.get('obs_specs', dict())
        self.eval_specs = cfg.get('eval_specs', dict())

        # agent config
        self.agent_specs = cfg.get('agent_specs', dict())
        self.mlp_specs = cfg.get('mlp_specs', dict())
        self.gnn_specs = cfg.get('gnn_specs', dict())

        # training config
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.95)
        self.state_encoder_specs = cfg.get('state_encoder_specs', dict())
        self.policy_specs = cfg.get('policy_specs', dict())
        self.value_specs = cfg.get('value_specs', dict())
        self.lr = cfg.get('lr', 4e-4)
        self.weightdecay = cfg.get('weightdecay', 0.0)
        self.eps = cfg.get('eps', 1e-5)
        self.value_pred_coef = cfg.get('value_pred_coef', 0.5)
        self.entropy_coef = cfg.get('entropy_coef', 0.01)
        self.clip_epsilon = cfg.get('clip_epsilon', 0.2)
        self.max_num_iterations = cfg.get('max_num_iterations', 1000)
        self.num_episodes_per_iteration = cfg.get('num_episodes_per_iteration', 1000)
        self.max_sequence_length = cfg.get('max_sequence_length', 100)
        self.num_optim_epoch = cfg.get('num_optim_epoch', 4)
        self.mini_batch_size = cfg.get('mini_batch_size', 1024)
        self.save_model_interval = cfg.get('save_model_interval', 10)

    def log(self, logger, tb_logger):
        """Log cfg to logger and tensorboard."""
        logger.info(f'id: {self.cfg_id}')
        logger.info(f'seed: {self.seed}')
        logger.info(f'env_specs: {self.env_specs}')
        logger.info(f'reward_specs: {self.reward_specs}')
        logger.info(f'obs_specs: {self.obs_specs}')
        logger.info(f'agent_specs: {self.agent_specs}')
        logger.info(f'gamma: {self.gamma}')
        logger.info(f'tau: {self.tau}')
        logger.info(f'state_encoder_specs: {self.state_encoder_specs}')
        logger.info(f'policy_specs: {self.policy_specs}')
        logger.info(f'value_specs: {self.value_specs}')
        logger.info(f'lr: {self.lr}')
        logger.info(f'weightdecay: {self.weightdecay}')
        logger.info(f'eps: {self.eps}')
        logger.info(f'value_pred_coef: {self.value_pred_coef}')
        logger.info(f'entropy_coef: {self.entropy_coef}')
        logger.info(f'clip_epsilon: {self.clip_epsilon}')
        logger.info(f'max_num_iterations: {self.max_num_iterations}')
        logger.info(f'num_episodes_per_iteration: {self.num_episodes_per_iteration}')
        logger.info(f'max_sequence_length: {self.max_sequence_length}')
        logger.info(f'num_optim_epoch: {self.num_optim_epoch}')
        logger.info(f'mini_batch_size: {self.mini_batch_size}')
        logger.info(f'save_model_interval: {self.save_model_interval}')

        if tb_logger is not None:
            tb_logger.add_hparams(
                hparam_dict={
                    'id': self.cfg_id,
                    'seed': self.seed,
                    'env_specs': str(self.env_specs),
                    'reward_specs': str(self.reward_specs),
                    'obs_specs': str(self.obs_specs),
                    'agent_specs': str(self.agent_specs),
                    'gamma': self.gamma,
                    'tau': self.tau,
                    'state_encoder_specs': str(self.state_encoder_specs),
                    'policy_specs': str(self.policy_specs),
                    'value_specs': str(self.value_specs),
                    'lr': self.lr,
                    'weightdecay': self.weightdecay,
                    'eps': self.eps,
                    'value_pred_coef': self.value_pred_coef,
                    'entropy_coef': self.entropy_coef,
                    'clip_epsilon': self.clip_epsilon,
                    'max_num_iterations': self.max_num_iterations,
                    'num_episodes_per_iteration': self.num_episodes_per_iteration,
                    'max_sequence_length': self.max_sequence_length,
                    'num_optim_epoch': self.num_optim_epoch,
                    'mini_batch_size': self.mini_batch_size,
                    'save_model_interval': self.save_model_interval},
                metric_dict={'hparam/placeholder': 0.0})
