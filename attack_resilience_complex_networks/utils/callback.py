from typing import Union

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.logger import HParam


class HParamCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "steps_per_iteration": self.model.n_steps * self.model.n_envs,
            "batch_size": self.model.batch_size,
            "optim_epochs_per_iteration": self.model.n_epochs,
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "ent_coef": self.model.ent_coef,
            "vf_coef": self.model.vf_coef,
        }
        metric_dict = {
            "eval/mean_reward": 0,
            "train/loss": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class UpdateValEnv(BaseCallback):
    def __init__(self, val_env: Union[gym.Env, VecEnv], verbose: int = 0):
        super().__init__(verbose=verbose)
        if not isinstance(val_env, VecEnv):
            val_env = DummyVecEnv([lambda: val_env])

        self.val_env = val_env

    def _on_step(self) -> bool:
        assert self.parent is not None, "``UpdateValEnv`` callback must be used with an ``EvalCallback``"

        self.val_env.env_method('reset_instance_id')
        return True


class UpdateValEnvAndStopTrainingOnNoModelImprovement(StopTrainingOnNoModelImprovement):
    def __init__(self, val_env: Union[gym.Env, VecEnv],
                 max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        super().__init__(max_no_improvement_evals=max_no_improvement_evals, min_evals=min_evals, verbose=verbose)
        if not isinstance(val_env, VecEnv):
            val_env = DummyVecEnv([lambda: val_env])

        self.val_env = val_env

    def _on_step(self) -> bool:
        self.val_env.env_method('reset_instance_id')
        return super()._on_step()
