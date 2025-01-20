from typing import Tuple, Optional

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


def get_node_features_dim(observation_space: spaces.Dict) -> int:
    return observation_space.spaces['node_features'].shape[1]


def create_node_encoder(num_node_features: int, node_dim: int) -> nn.Sequential:
    node_encoder = nn.Sequential(
        nn.Linear(num_node_features, node_dim),
        nn.Tanh())
    return node_encoder


def create_gnn_layers(num_gnn_layers: int, node_dim: Optional[int]) -> nn.ModuleList:
    layers = nn.ModuleList()
    for i in range(num_gnn_layers):
        assert node_dim is not None
        gnn_layer = GCNConv(node_dim, node_dim)
        layers.append(gnn_layer)
    return layers


def batch_graph(
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
        edge_mask: th.Tensor,
        node_mask: th.Tensor,
        node_features: th.Tensor = None
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, int, th.Tensor, th.Tensor, int]:
    """Batch graph data.

    :param edge_index: (B, 2, M)
    :param edge_attr: (B, M)
    :param edge_mask: (B, M)
    :param node_mask: (B, N)
    :param node_features: (B, N, F)

    :return: (optional) node_features: (N_1 + ... + N_B, F)
    :return: node_mask: (B, N)
    :return: num_nodes: (B)
    :return: max_num_nodes: int
    :return: edge_index: (2, M_1 + ... + M_B)
    :return: edge_attr: (M_1 + ... + M_B,)
    :return: batch_size: int
    """
    batch_size = node_mask.shape[0]
    max_num_nodes = node_mask.shape[1]
    num_nodes = node_mask.sum(dim=-1)  # (B,)
    node_mask = node_mask.reshape(-1)  # (B*N,)

    edge_index = edge_index.permute(1, 0, 2).reshape(2, -1)  # (2, B*M)
    edge_attr = edge_attr.reshape(-1)  # (B*M,)
    num_edges = edge_mask.sum(dim=-1)  # (B,)
    edge_mask = edge_mask.reshape(-1)  # (B*M,)
    edge_index = edge_index[:, edge_mask]  # (2, M')
    edge_attr = edge_attr[edge_mask]  # (M',)

    if batch_size > 1:
        num_nodes_cum = th.cumsum(num_nodes, dim=0)  # (B,)
        edge_offset = th.repeat_interleave(num_nodes_cum[:-1], num_edges[1:])
        edge_index[:, -edge_offset.shape[0]:] += edge_offset

    node_features = node_features.reshape(-1, node_features.shape[-1])  # (B*N, F)
    node_features = node_features[node_mask]  # (N', F)

    node_mask = node_mask.reshape(batch_size, max_num_nodes)  # (B, N)
    return node_features, node_mask, num_nodes, max_num_nodes, edge_index, edge_attr, batch_size


def batch_obs(observations: TensorDict
              ) -> Tuple[Optional[th.Tensor], th.Tensor, th.Tensor, int, th.Tensor, th.Tensor, int]:
    edge_index = observations['edge_index'].long()  # (B, 2, M)
    edge_attr = observations['edge_attr']  # (B, M)
    edge_mask = observations['edge_mask'].bool()  # (B, M)
    node_mask = observations['node_mask'].bool()  # (B, N)
    node_features = observations['node_features']  # (B, N, F)
    return batch_graph(edge_index, edge_attr, edge_mask, node_mask, node_features)


def mean_features(h: th.Tensor, mask: th.Tensor):
    float_mask = mask.float()
    mean_h = (h * float_mask.unsqueeze(-1)).sum(dim=1) / float_mask.sum(dim=1, keepdim=True)
    return mean_h


def compute_state(node_mask: th.Tensor, h_nodes: th.Tensor):
    mean_h_nodes = mean_features(h_nodes, node_mask)

    state_policy = h_nodes
    state_value = mean_h_nodes

    return state_policy, state_value


class PyGGNNExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: spaces.Dict,
            num_gnn_layers: int = 2,
            node_dim: Optional[int] = 32,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        self.num_node_features = get_node_features_dim(observation_space)
        if node_dim is not None:
            self.node_encoder = create_node_encoder(self.num_node_features, node_dim)
        self.gnn_layers = create_gnn_layers(num_gnn_layers, node_dim)

    @staticmethod
    def get_policy_feature_dim(node_dim: int) -> int:
        return node_dim

    @staticmethod
    def get_value_feature_dim(node_dim: int) -> int:
        return node_dim

    def gnn_forward(self, node_features: th.Tensor, edge_index: th.Tensor, edge_attr: th.Tensor,
                    node_mask: th.Tensor, num_nodes: th.Tensor, max_num_nodes: int, batch_size: int) -> th.Tensor:
        x = self.node_encoder(node_features)
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)
            x = th.tanh(x)

        batch = th.repeat_interleave(th.arange(batch_size, device=node_features.device), num_nodes)
        h_nodes, node_mask_reconstructed = to_dense_batch(x, batch, max_num_nodes=max_num_nodes)  # (B, N, F)

        assert th.all(node_mask == node_mask_reconstructed)
        return h_nodes

    def forward(self, observations: TensorDict) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        node_features, node_mask, num_nodes, max_num_nodes, edge_index, edge_attr, batch_size = \
            batch_obs(observations)

        h_nodes = self.gnn_forward(node_features, edge_index, edge_attr,
                                   node_mask, num_nodes, max_num_nodes, batch_size)

        state_policy, state_value = compute_state(node_mask, h_nodes)
        action_mask = observations['action_mask'].bool()
        return state_policy, state_value, action_mask


class ExplainFeatureExtractor(nn.Module):
    def __init__(
            self,
            device: th.device,
            observation_space: spaces.Dict,
            num_gnn_layers: int = 2,
            node_dim: Optional[int] = 32,
    ) -> None:
        super().__init__()

        device = get_device(device)

        self.num_node_features = get_node_features_dim(observation_space)
        if node_dim is not None:
            self.node_encoder = create_node_encoder(self.num_node_features, node_dim).to(device)
        self.gnn_layers = create_gnn_layers(num_gnn_layers, node_dim).to(device)

    def forward(self, x: th.Tensor, edge_index: th.Tensor, edge_attr: th.Tensor) -> th.Tensor:
        x = self.node_encoder(x)
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)
            x = th.tanh(x)
        return x

    @staticmethod
    def get_policy_feature_dim(node_dim: int) -> int:
        return node_dim

    @staticmethod
    def get_value_feature_dim(node_dim: int) -> int:
        return node_dim
