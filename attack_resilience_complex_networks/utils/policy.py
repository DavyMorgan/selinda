from typing import Dict

from gymnasium import spaces

from attack_resilience_complex_networks.agent.features_extractor import PyGGNNExtractor, ExplainFeatureExtractor
from attack_resilience_complex_networks.utils.config import Config


def get_policy_kwargs(cfg: Config, explain: bool = False, observation_space: spaces.Dict = None) -> Dict:
    num_gnn_layers = cfg.gnn_specs.get('num_gnn_layers', 2)
    node_dim = cfg.gnn_specs.get('node_dim', 32)
    if not explain:
        features_extractor_class = PyGGNNExtractor
        features_extractor_kwargs = dict(
            num_gnn_layers=num_gnn_layers,
            node_dim=node_dim)
    else:
        features_extractor_class = ExplainFeatureExtractor
        features_extractor_kwargs = dict(
            num_gnn_layers=num_gnn_layers,
            node_dim=node_dim,
            observation_space=observation_space)
    policy_feature_dim = features_extractor_class.get_policy_feature_dim(node_dim)
    value_feature_dim = features_extractor_class.get_value_feature_dim(node_dim)
    policy_kwargs = dict(
        policy_feature_dim=policy_feature_dim,
        value_feature_dim=value_feature_dim,
        policy_hidden_units=cfg.agent_specs.get('policy_hidden_units', (32, 32, 1)),
        value_hidden_units=cfg.agent_specs.get('value_hidden_units', (32, 32, 1)),
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,)
    return policy_kwargs
