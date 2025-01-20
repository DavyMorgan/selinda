from functools import partial
from typing import Callable, Tuple, Union
from collections import OrderedDict

import numpy as np
from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.type_aliases import Schedule


def create_mlp(head: str, input_dim: int, hidden_units: Tuple) -> nn.Sequential:
    layers = OrderedDict()
    for i, units in enumerate(hidden_units):
        if i == 0:
            layers[f'{head}_linear_{i}'] = nn.Linear(input_dim, units)
        else:
            layers[f'{head}_linear_{i}'] = nn.Linear(hidden_units[i - 1], units)
        if i != len(hidden_units) - 1:
            layers[f'{head}_tanh_{i}'] = nn.Tanh()
    if head.startswith('policy'):
        layers[f'{head}_flatten'] = nn.Flatten()
    return nn.Sequential(layers)


class MaskedMLPExtractor(nn.Module):

    def __init__(
        self,
        policy_feature_dim: int,
        value_feature_dim: int,
        policy_hidden_units: Tuple = (32, 32, 1),
        value_hidden_units: Tuple = (32, 32, 1),
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)

        # Policy network
        self.policy_net = create_mlp('policy',
                                     policy_feature_dim,
                                     policy_hidden_units).to(device)

        # Value network
        self.value_net = create_mlp('value',
                                    value_feature_dim,
                                    value_hidden_units).to(device)

    def forward(self,
                features: Tuple[th.Tensor, th.Tensor, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: Tuple[th.Tensor, th.Tensor, th.Tensor]) -> th.Tensor:
        state_policy, _, action_mask = features

        logits = self.policy_net(state_policy)  # (batch_size, node_range)
        padding = th.full_like(action_mask, -th.inf, dtype=th.float32)
        masked_logits = th.where(action_mask, logits, padding)

        return masked_logits

    def forward_critic(self, features: Tuple[th.Tensor, th.Tensor, th.Tensor]) -> th.Tensor:
        _, state_value, _ = features
        return self.value_net(state_value)

    def forward_explain(self, features: th.Tensor) -> th.Tensor:
        logits = self.policy_net(features).reshape(-1)  # (batch_size, node_range)
        return logits

    def forward_explain_value(self, features: th.Tensor) -> th.Tensor:
        r = self.value_net(features).reshape(-1)
        return r


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        self.policy_feature_dim = kwargs.pop('policy_feature_dim')
        self.value_feature_dim = kwargs.pop('value_feature_dim')
        self.policy_hidden_units = kwargs.pop('policy_hidden_units')
        self.value_hidden_units = kwargs.pop('value_hidden_units')

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MaskedMLPExtractor(
            self.policy_feature_dim,
            self.value_feature_dim,
            self.policy_hidden_units,
            self.value_hidden_units,
            self.device,
        )


class ExplainPolicy(nn.Module):
    def __init__(self, device: th.device, **kwargs):
        super().__init__()
        self.device = device

        self.features_extractor_class = kwargs.pop('features_extractor_class')
        self.features_extractor_kwargs = kwargs.pop('features_extractor_kwargs')
        self.features_extractor = self.features_extractor_class(self.device, **self.features_extractor_kwargs)

        self.policy_feature_dim = kwargs.pop('policy_feature_dim')
        self.value_feature_dim = kwargs.pop('value_feature_dim')
        self.policy_hidden_units = kwargs.pop('policy_hidden_units')
        self.value_hidden_units = kwargs.pop('value_hidden_units')

        self.mlp_extractor = MaskedMLPExtractor(
            self.policy_feature_dim,
            self.value_feature_dim,
            self.policy_hidden_units,
            self.value_hidden_units,
            self.device,
        )

    def forward(self, x: th.Tensor, edge_index: th.Tensor, edge_attr: th.Tensor) \
            -> th.Tensor:
        features = self.features_extractor(x, edge_index, edge_attr)
        logits = self.mlp_extractor.forward_explain(features)
        return logits

    def get_logits(self, x:th. Tensor, edge_index: th.Tensor, edge_attr: th.Tensor) -> th.Tensor:
        logits = self.forward(x, edge_index, edge_attr)
        return logits

    def predict(self, x: th.Tensor, edge_index: th.Tensor, edge_attr: th.Tensor, deterministic: bool = True) \
            -> Tuple[th.Tensor, th.Tensor]:
        logits = self.get_logits(x, edge_index, edge_attr)
        probs = th.softmax(logits, dim=-1)
        if deterministic:
            action = th.argmax(probs, dim=-1)
        else:
            action = th.multinomial(probs, num_samples=1)
        return action, probs


class ExplainValue(ExplainPolicy):
    def __init__(self, device: th.device, **kwargs):
        super().__init__(device, **kwargs)

    def forward(self, x: th.Tensor, edge_index: th.Tensor, edge_attr: th.Tensor) \
            -> th.Tensor:
        features = self.features_extractor(x, edge_index, edge_attr)
        features = features.mean(dim=0)
        value = self.mlp_extractor.forward_explain_value(features)
        return value

    def get_logits(self, x:th. Tensor, edge_index: th.Tensor, edge_attr: th.Tensor) -> th.Tensor:
        logits = super().forward(x, edge_index, edge_attr)
        return logits

