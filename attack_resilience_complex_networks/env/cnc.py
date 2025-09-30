import warnings
from functools import partial
from typing import Callable, Tuple, Dict, List, Optional, Union
from tqdm.rich import tqdm

import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import powerlaw

from attack_resilience_complex_networks.utils.config import Config
from attack_resilience_complex_networks.utils.resilience import compute_beta_np


class ComplexNetworkClient:
    def __init__(self, cfg: Config, rng: np.random.Generator, topology: nx.Graph, dynamics: Callable,
                 dynamics_vec_params: str = None, dynamics_params: Dict = None,
                 target_dynamics_params: Dict = None, protected_nodes: List[int] = None,
                 pre_attacked_nodes: List[int] = None,
                 block_feature: List[str] = None) -> None:
        """
        :param topology: adjacency matrix
        :param dynamics: dynamics of the network
        """
        self.cfg = cfg
        self.rng = rng
        self._initial_g = topology.copy()
        self._initial_gcc_size = len(max(nx.connected_components(self._initial_g), key=len))
        if dynamics is not None:
            self.initial_topology = nx.to_numpy_array(topology)

        self.dynamics = dynamics
        if self.dynamics is not None:
            self.dynamics_type = cfg.env_specs['dynamics_type']
        else:
            self.dynamics_type = None
        self.dynamics_vec_params = dynamics_vec_params
        self.dynamics_params_base = dynamics_params
        self.target_params = target_dynamics_params

        self.protected_nodes = protected_nodes if protected_nodes is not None else []
        self.pre_attacked_nodes = pre_attacked_nodes if pre_attacked_nodes is not None else []
        self.block_feature = block_feature if block_feature is not None else []

        self._initialize_system()
        self.reset()

    def _initialize_system(self) -> None:
        """
        Initialize the system
        """
        graph_layout = self.cfg.env_specs['graph_layout']
        if graph_layout == 'kamada_kawai':
            layout_weight = None if self.cfg.env_specs.get('layout_weight', None) != 'weight' else 'weight'
            self._initial_graph_layout = nx.kamada_kawai_layout(self._initial_g,
                                                                weight=layout_weight)
        elif graph_layout == 'circular':
            self._initial_graph_layout = nx.circular_layout(self._initial_g)
        elif graph_layout == 'spring':
            self._initial_graph_layout = nx.spring_layout(self._initial_g)
        elif graph_layout.startswith('shell'):
            n_ring = int(graph_layout.split('-')[1])
            nlist = []
            # divide the nodes into n_ring groups, with number of nodes in each group have ratio 1:2:3:...:n_ring
            num_nodes_per_ring = np.arange(1, n_ring + 1)
            num_nodes_per_ring = num_nodes_per_ring / np.sum(num_nodes_per_ring) * self._initial_g.number_of_nodes()
            num_nodes_per_ring = np.round(num_nodes_per_ring).astype(int)
            num_nodes_per_ring[-1] = self._initial_g.number_of_nodes() - np.sum(num_nodes_per_ring[:-1])
            start = 0
            for num_nodes in num_nodes_per_ring:
                nlist.append(np.arange(start, start + num_nodes).tolist())
                start += num_nodes
            self._initial_graph_layout = nx.shell_layout(self._initial_g, nlist=nlist)
        elif graph_layout == 'degree':
            # divide the nodes into three groups based on their degree, arrange them in three rings, with the nodes
            # with the highest degree in the center
            degree = np.array([d for (n, d) in self._initial_g.degree()])
            degree_indices = np.argsort(degree)[::-1]
            nlist = [
                degree_indices[:len(degree_indices) // 6].tolist(),
                degree_indices[len(degree_indices) // 6: len(degree_indices) // 2].tolist(),
                degree_indices[len(degree_indices) // 2:].tolist()
            ]
            self._initial_graph_layout = nx.shell_layout(self._initial_g, nlist=nlist)
        else:
            raise ValueError(f'Unknown graph layout: {graph_layout}')

        self._initial_dynamics_params = self._get_dynamics_params(self._initial_g)

    def _get_dynamics_params(self, topology: nx.Graph) -> Dict:
        """
        :return: dynamics parameters
        """
        if self.dynamics_type is None:
            return {}

        num_nodes = topology.number_of_nodes()
        degree = np.array([d for (n, d) in topology.degree()])
        dynamics_params = {}

        def power_params(k: str):
            # generate a power-law sequence using scipy.stats.powerlaw
            seed = self.target_params['seed']
            a = self.target_params['powerlaw']['a']
            loc = self.target_params['powerlaw']['loc']
            _scale = self.target_params['powerlaw']['scale']
            powerlaw_sequence = powerlaw.rvs(a, loc=loc, scale=_scale, size=num_nodes, random_state=seed)
            dynamics_params[k] = powerlaw_sequence

        def degree_params(k: str):
            degree_indices = np.argsort(degree)
            start = self.target_params['degree']['start']
            span = self.target_params['degree']['span']
            strengthened_nodes = degree_indices[start:start + span]
            dynamics_params[k][strengthened_nodes] = self.target_params['min'][k]

        for key, value in self.dynamics_params_base.items():
            dynamics_params[key] = np.full(num_nodes, value)
            if self.dynamics_vec_params is None:
                continue
            if self.dynamics_type in ['gene', 'neuron', 'epidemic']:
                if key == 'b':
                    if self.dynamics_vec_params == 'powerlaw':
                        power_params(key)
                    elif self.dynamics_vec_params == 'degree':
                        degree_params(key)
                    else:
                        raise ValueError(f'Unknown dynamics vector parameters: {self.dynamics_vec_params}.')
            else:
                raise ValueError(f'Unknown dynamics type: {self.dynamics_type}.')
        return dynamics_params

    def _update_dynamics_params(self, node_indices: np.ndarray) -> None:
        """
        Update dynamics parameters

        :param node_indices: node indices
        """
        for key, value in self._dynamics_params.items():
            self._dynamics_params[key] = value[node_indices]

    def _get_dynamics_params_finder(self, node_status: np.ndarray) -> Dict:
        """
        :param node_status: node status
        :return: dynamics parameters
        """
        temp = self._initial_dynamics_params.copy()
        dynamics_params = {}
        for key, value in temp.items():
            dynamics_params[key] = value[node_status]
        return dynamics_params

    def _update_action_mask(self, node_indices: np.ndarray) -> None:
        """
        Update action mask

        :param node_indices: node indices
        """
        self._action_mask = self._action_mask[node_indices]

    def reset(self) -> Tuple[Tuple[float, Tuple[float, float, float]], Tuple[float, float, float]]:
        """
        Reset the system

        :return: initial beta
        """
        self._g = self._initial_g.copy()
        if self.dynamics_type is not None:
            self._current_topology_slim = self.initial_topology.copy()
        self._dynamics_params = self._initial_dynamics_params.copy()

        self._action_mask = np.ones(self._initial_g.number_of_nodes(), dtype=bool)
        self._action_mask[self.protected_nodes] = False

        if len(self.pre_attacked_nodes) > 0:
            self.pre_attack(self.pre_attacked_nodes)
        else:
            self._evolve()

        return self.compute_proxy(), self.compute_dynamic_proxy()

    def get_current_num_nodes(self) -> int:
        """
        :return: current number of nodes
        """
        return self._g.number_of_nodes()

    def get_action_mask_pyg(self) -> np.ndarray:
        """
        :return: current action mask
        """
        return self._action_mask

    def get_edge_pyg(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: current topology
        """
        edge_index = np.array(list(self._g.edges)).T.reshape(2, -1)
        edge_attr = nx.get_edge_attributes(self._g, 'weight')
        edge_attr = np.array([edge_attr[(u, v)] for (u, v) in edge_index.T])
        edge_index = np.hstack([edge_index, edge_index[::-1, :]])
        edge_attr = np.concatenate([edge_attr, edge_attr])
        return edge_index, edge_attr

    def get_topology(self) -> nx.Graph:
        """
        :return: current topology
        """
        return self._g

    def compute_robustness(self):
        gcc_size = len(max(nx.connected_components(self._g), key=len))
        connectivity = gcc_size / self._initial_gcc_size
        return connectivity, (gcc_size, self._initial_gcc_size, 0.0)

    def dismantle(self) -> bool:
        _, (gcc_size, _, _) = self.compute_robustness()
        return gcc_size <= 1

    def compute_proxy(self) -> Tuple[float, Tuple[float, float, float]]:
        """
        :return: current proxy
        """
        if self.dynamics_type in ['gene', 'neuron', 'epidemic']:
            return compute_beta_np(self._current_topology_slim)
        elif self.dynamics_type is None:
            return self.compute_robustness()
        else:
            raise ValueError(f'Unknown dynamics type: {self.dynamics_type}.')

    @staticmethod
    def compute_beta(topology: np.ndarray) -> Tuple[float, Tuple[float, float, float]]:
        """
        :return: current beta
        """
        return compute_beta_np(topology)

    def _get_traj_and_stable_state_neuron(self, dynamics: Callable, num_nodes: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_lo = np.zeros(num_nodes, dtype=np.float32)
        result_lo = solve_ivp(
            dynamics,
            (0.0, self.cfg.env_specs['stable_t']),
            x_lo,
            method=self.cfg.env_specs['ode_method'])
        t_eval_lo = result_lo.t
        idx_lo = np.argmax(t_eval_lo > self.cfg.env_specs['stable_t'] - 10)
        trajectory_lo = result_lo.y
        stable_state_lo = trajectory_lo[:, idx_lo:].mean(axis=1)
        x_hi = np.ones(num_nodes, dtype=np.float32) * 10
        result_hi = solve_ivp(
            dynamics,
            (0.0, self.cfg.env_specs['stable_t']),
            x_hi,
            method=self.cfg.env_specs['ode_method'])
        t_eval_hi = result_hi.t
        idx_hi = np.argmax(t_eval_hi > self.cfg.env_specs['stable_t'] - 10)
        trajectory_hi = result_hi.y
        stable_state_hi = trajectory_hi[:, idx_hi:].mean(axis=1)

        return t_eval_lo, t_eval_hi, trajectory_lo, trajectory_hi, stable_state_lo, stable_state_hi

    def _simulate_resilience_gene(self, dynamics: Callable, num_nodes: int) \
            -> Tuple[bool, Tuple[np.ndarray, np.ndarray], np.ndarray]:
        x0 = np.ones(num_nodes) * 2
        result = solve_ivp(
            dynamics,
            (0.0, self.cfg.env_specs['stable_t']),
            x0,
            method=self.cfg.env_specs['ode_method'])
        t_eval = result.t
        idx = np.argmax(t_eval > self.cfg.env_specs['stable_t'] - 10)
        trajectory = result.y
        stable_state = trajectory[:, idx:].mean(axis=1)
        mean_stable_state = stable_state.mean()
        if mean_stable_state < 1e-5:
            resilience = False
        else:
            resilience = True
        return resilience, (t_eval, trajectory), stable_state

    def _simulate_resilience_epidemic(self, dynamics: Callable, num_nodes: int) \
            -> Tuple[bool, Tuple[np.ndarray, np.ndarray], np.ndarray]:
        x0 = np.ones(num_nodes) * 0.5
        result = solve_ivp(
            dynamics,
            (0.0, self.cfg.env_specs['stable_t']),
            x0,
            method=self.cfg.env_specs['ode_method'])
        t_eval = result.t
        idx = np.argmax(t_eval > self.cfg.env_specs['stable_t'] - 10)
        trajectory = result.y
        stable_state = trajectory[:, idx:].mean(axis=1)
        mean_stable_state = stable_state.mean()
        if mean_stable_state < 1e-5:
            resilience = False
        else:
            resilience = True
        return resilience, (t_eval, trajectory), stable_state

    def _simulate_resilience_neuron(self, dynamics: Callable, num_nodes: int) \
            -> Tuple[bool, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        t_eval_lo, t_eval_hi, trajectory_lo, trajectory_hi, stable_state_lo, stable_state_hi = \
            self._get_traj_and_stable_state_neuron(dynamics, num_nodes)
        if np.min(np.abs(stable_state_hi - stable_state_lo)) < 0.1 < stable_state_lo.mean():
            resilience = True
        else:
            resilience = False
        return resilience, (t_eval_lo, t_eval_hi, trajectory_lo, trajectory_hi), (stable_state_lo, stable_state_hi)

    def _simulate_resilience(self, dynamics: Callable, num_nodes: int) \
            -> Tuple[
                bool,
                Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:

        if self.dynamics_type == 'gene':
            return self._simulate_resilience_gene(dynamics, num_nodes)
        elif self.dynamics_type == 'neuron':
            return self._simulate_resilience_neuron(dynamics, num_nodes)
        elif self.dynamics_type == 'epidemic':
            return self._simulate_resilience_epidemic(dynamics, num_nodes)
        else:
            raise NotImplementedError

    def simulate_resilience(self, return_traj: bool = False) \
            -> Union[
                bool,
                Tuple[
                    bool,
                    Union[
                        Tuple[np.ndarray, np.ndarray],
                        Tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
        """
        Simulate resilience of the system

        :return:
        resilience: True if the system is resilient, False otherwise
        trajectory: trajectory of the system
        """
        if return_traj:
            return self._resilience, self._traj
        else:
            return self._resilience

    def simulate_resilience_finder(self, topology: np.ndarray, node_status: np.ndarray,
                                   return_traj: bool = False, return_state: bool = False):
        """
        Simulate resilience of the system

        :return: True if the system is resilient, False otherwise
        """
        num_nodes = topology.shape[0]
        dynamics_params = self._get_dynamics_params_finder(node_status)
        b = dynamics_params['b'].mean()
        dynamics = partial(self.dynamics, a=topology, **dynamics_params)
        resilience, traj, stable_state = self._simulate_resilience(dynamics, num_nodes)
        if isinstance(stable_state, tuple):
            s_lo, s_hi = stable_state[0].mean(), stable_state[1].mean()
            state = (stable_state[0] + stable_state[1]) / 2
        else:
            s_lo = stable_state.mean()
            s_hi = s_lo
            state = stable_state
        if return_state:
            return state
        elif return_traj:
            return resilience, b, s_lo, s_hi, traj
        else:
            return resilience, b, s_lo, s_hi

    def get_mean_state(self) -> Union[float, Tuple[float, float]]:
        """
        :return: mean state of the system
        """
        if self.dynamics_type == 'neuron':
            stable_state_lo, stable_state_hi = self._stable_state
            mean_state = (stable_state_lo.mean(), stable_state_hi.mean())
        elif self.dynamics_type == 'gene':
            stable_state = self._stable_state
            mean_state = stable_state.mean()
        elif self.dynamics_type == 'epidemic':
            stable_state = self._stable_state
            mean_state = stable_state.mean()
        else:
            raise NotImplementedError
        return mean_state

    @property
    def topology_obs_pyg_shape(self) -> int:
        return 10

    def get_topology_obs_pyg(self) -> np.ndarray:
        """
        :return: topology observation of the system, containing the topology and the action mask
        """
        degree = np.array([d for (n, d) in self._g.degree()])

        avg_neighbor_degree = nx.average_neighbor_degree(self._g)
        avg_neighbor_degree = np.array([avg_neighbor_degree[n] for n in self._g.nodes])

        if 'kcore' not in self.block_feature:
            core_number = nx.core_number(self._g)
            kcore = np.array([core_number[n] for n in self._g.nodes])
            neighbor_kcore = np.array([np.sum([core_number[n] for n in self._g.neighbors(node)]) for node in self._g.nodes])
            avg_neighbor_kcore = neighbor_kcore / degree.clip(min=1)
        else:
            kcore = np.zeros_like(degree)
            avg_neighbor_kcore = np.zeros_like(degree)

        if 'cc' not in self.block_feature:
            cc = nx.clustering(self._g)
            clustering_coefficient = np.array([cc[n] for n in self._g.nodes])
            neighbor_clustering_coefficient = np.array(
                [np.sum([cc[n] for n in self._g.neighbors(node)]) for node in self._g.nodes])
            avg_neighbor_clustering_coefficient = neighbor_clustering_coefficient / degree.clip(min=1)
        else:
            clustering_coefficient = np.zeros_like(degree)
            avg_neighbor_clustering_coefficient = np.zeros_like(degree)

        if 'pagerank' not in self.block_feature:
            pagerank = nx.pagerank(self._g)
            pagerank = np.array([pagerank[n] for n in self._g.nodes])
            neighbor_pagerank = np.array(
                [np.sum([pagerank[n] for n in self._g.neighbors(node)]) for node in self._g.nodes])
            avg_neighbor_pagerank = neighbor_pagerank / degree.clip(min=1)
        else:
            pagerank = np.zeros_like(degree)
            avg_neighbor_pagerank = np.zeros_like(degree)

        if 'betweenness' not in self.block_feature:
            betweenness_centrality = nx.betweenness_centrality(self._g)
            betweenness_centrality = np.array([betweenness_centrality[n] for n in self._g.nodes])
            neighbor_betweenness_centrality = np.array(
                [np.sum([betweenness_centrality[n] for n in self._g.neighbors(node)]) for node in self._g.nodes])
            avg_neighbor_betweenness_centrality = neighbor_betweenness_centrality / degree.clip(min=1)
        else:
            betweenness_centrality = np.zeros_like(degree)
            avg_neighbor_betweenness_centrality = np.zeros_like(degree)

        obs_node = np.column_stack([degree, avg_neighbor_degree,
                                    kcore, avg_neighbor_kcore,
                                    clustering_coefficient, avg_neighbor_clustering_coefficient,
                                    pagerank, avg_neighbor_pagerank,
                                    betweenness_centrality, avg_neighbor_betweenness_centrality])
        return obs_node

    @property
    def nil_obs_pyg_shape(self) -> int:
        return 4

    def get_nil_obs_pyg(self) -> np.ndarray:
        """
        :return: nil observation of the system, containing degree and a random noise
        """
        out_degree = np.sum(self._current_topology_slim, axis=1, keepdims=True)
        max_out_weight = np.max(self._current_topology_slim, axis=1, keepdims=True)
        weighted_nearest_neighbor_out_degree = np.dot(self._current_topology_slim, out_degree)
        if 'resilience_centrality' not in self.block_feature:
            beta, _ = compute_beta_np(self._current_topology_slim)
            resilience_centrality_out_degree = 2*weighted_nearest_neighbor_out_degree + out_degree*(out_degree - 2*beta)
        else:
            resilience_centrality_out_degree = np.zeros_like(out_degree)
        obs_node = np.concatenate([
            out_degree,
            max_out_weight,
            weighted_nearest_neighbor_out_degree,
            resilience_centrality_out_degree], axis=1)
        return obs_node

    @property
    def dynamics_param_obs_pyg_shape(self) -> int:
        return len(self.dynamics_params_base)

    def get_dynamics_param_obs_pyg(self) -> np.ndarray:
        """
        :return: dynamics parameters
        """
        obs_node = np.concatenate([value.reshape(-1, 1) for value in self._dynamics_params.values()], axis=1)
        return obs_node

    @property
    def env_obs_pyg_shape(self) -> int:
        return 4

    def get_stable_state(self, return_state: str = 'mean') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dynamics_type == 'neuron':
            t_eval, _, trajectory, _ = self._traj
            stable_state_lo, stable_state_hi = self._stable_state
            if return_state == 'mean':
                stable_state = (stable_state_lo + stable_state_hi) / 2
            elif return_state == 'lo':
                stable_state = stable_state_lo
            else:
                stable_state = stable_state_hi
        elif self.dynamics_type == 'gene':
            t_eval, trajectory = self._traj
            stable_state = self._stable_state
        elif self.dynamics_type == 'epidemic':
            t_eval, trajectory = self._traj
            stable_state = self._stable_state
        else:
            raise NotImplementedError
        return stable_state, trajectory, t_eval

    def get_env_obs(self) -> np.ndarray:
        """
        :return: environment observation of the system, containing the stable state and the derivatives
        """
        stable_state, trajectory, t_eval = self.get_stable_state()
        # get the index of the first element in t_eval that is greater than 1
        idx = np.argmax(t_eval > 1)
        derivatives = self.dynamics(0, trajectory[:, idx], self._current_topology_slim, **self._dynamics_params)
        weighted_nearest_neighbor_stable_state = np.dot(self._current_topology_slim, stable_state.reshape(-1, 1))
        weighted_nearest_neighbor_derivative = np.dot(self._current_topology_slim, derivatives.reshape(-1, 1))
        obs_env = np.concatenate([
            stable_state[:, np.newaxis],
            derivatives[:, np.newaxis],
            weighted_nearest_neighbor_stable_state,
            weighted_nearest_neighbor_derivative], axis=1)
        if 'stable_state' in self.block_feature:
            obs_env[:, 0] = 0
            obs_env[:, 2] = 0
        if 'derivative' in self.block_feature:
            obs_env[:, 1] = 0
            obs_env[:, 3] = 0
        return obs_env

    def _evolve(self):
        if self.dynamics_type is None:
            return
        num_nodes = self._g.number_of_nodes()
        dynamics = partial(self.dynamics, a=self._current_topology_slim, **self._dynamics_params)
        self._resilience, self._traj, self._stable_state = self._simulate_resilience(dynamics, num_nodes)

    def compute_dynamic_proxy(self) -> Tuple[float, float, float]:
        if self.dynamics_type is None:
            return 0.0, 0.0, 0.0
        b = self._dynamics_params['b'].mean()
        if isinstance(self._stable_state, tuple):
            s_lo, s_hi = self._stable_state[0].mean(), self._stable_state[1].mean()
        else:
            s_lo = self._stable_state.mean()
            s_hi = s_lo
        return b, s_lo, s_hi

    def _transform(self, number_of_nodes_before_attack: int) -> None:
        if self._g.number_of_edges() == 0 and self.dynamics_type is not None:
            raise RuntimeError('The graph is empty.')
        if self.dynamics_type is not None:
            gcc = np.array(list(max(nx.connected_components(self._g), key=len)))
            self._g.remove_nodes_from(np.setdiff1d(np.arange(number_of_nodes_before_attack), gcc))
        node_indices = np.array(list(self._g.nodes))
        self._g = nx.convert_node_labels_to_integers(self._g)
        if self.dynamics_type is not None:
            self._current_topology_slim = nx.to_numpy_array(self._g)

        self._update_dynamics_params(node_indices)
        self._update_action_mask(node_indices)

        self._evolve()

    def attack(self, node: int) -> Tuple[float, float, float]:
        """
        Attack a node

        :param node: node to be attacked
        """
        if node >= self._g.number_of_nodes():
            warn_msg = f'Node {node} is already attacked.'
            warnings.warn(warn_msg)
            node = self.rng.choice(self._g.number_of_nodes())
        number_of_nodes_before_attack = self._g.number_of_nodes()
        self._g.remove_node(node)
        self._transform(number_of_nodes_before_attack)
        return self.compute_dynamic_proxy()

    def pre_attack(self, nodes: List[int]) -> None:
        if max(nodes) >= self._g.number_of_nodes():
            warn_msg = f'Node {max(nodes)} is already attacked.'
            warnings.warn(warn_msg)
            nodes = self.rng.choice(self._g.number_of_nodes(), size=len(nodes), replace=False)
        number_of_nodes_before_attack = self._g.number_of_nodes()
        self._g.remove_nodes_from(nodes)
        self._transform(number_of_nodes_before_attack)

    def get_robustness(self, g: nx.Graph) -> Tuple[bool, float]:
        gcc_size = len(max(nx.connected_components(g), key=len))
        return gcc_size > 1, gcc_size / self._initial_gcc_size

    @staticmethod
    def gcc(current_topology, gcc, node_status, action_mask):
        dead_nodes = np.setdiff1d(np.arange(current_topology.shape[0]), gcc)
        current_topology[dead_nodes, :] = 0.0
        current_topology[:, dead_nodes] = 0.0
        node_status[dead_nodes] = False
        action_mask[dead_nodes] = False
        return current_topology, node_status, action_mask

    def gcc_reproject(self,
                      current_topology: np.ndarray,
                      node_status: np.ndarray,
                      action_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = nx.from_numpy_array(current_topology)
        gcc = np.array(list(max(nx.connected_components(g), key=len)))
        current_topology, node_status, action_mask = self.gcc(current_topology, gcc, node_status, action_mask)
        return current_topology, node_status, action_mask

    @staticmethod
    def batch_gcc_np(topology, gcc, num_nodes):
        node_status = np.ones(num_nodes, dtype=bool)
        if len(gcc) == num_nodes:
            return topology, node_status
        dead_nodes = np.setdiff1d(np.arange(num_nodes), gcc)
        node_status[dead_nodes] = False
        return topology[node_status, :][:, node_status], node_status

    def batch_gcc(self, topology, node_status):
        num_nodes = topology.shape[0]
        g = nx.from_numpy_array(topology)
        gcc = np.array(list(max(nx.connected_components(g), key=len)))
        topology, gcc_node_status = self.batch_gcc_np(topology, gcc, num_nodes)
        node_status[node_status] = gcc_node_status
        return topology, node_status

    def prepare_replay(self) -> Tuple[Union[np.ndarray, nx.Graph], np.ndarray, np.ndarray]:
        if self.dynamics_type is None:
            current_topology = self._initial_g.copy()
            num_nodes = current_topology.number_of_nodes()
            node_status = np.ones(num_nodes, dtype=bool)
            action_mask = np.ones(num_nodes, dtype=bool)
            action_mask[self.protected_nodes] = False
            if len(self.pre_attacked_nodes) > 0:
                current_topology.remove_nodes_from(self.pre_attacked_nodes)
                node_status[self.pre_attacked_nodes] = False
                action_mask[self.pre_attacked_nodes] = False
            return current_topology, node_status, action_mask
        else:
            current_topology = self.initial_topology.copy()
            num_nodes = current_topology.shape[0]
            node_status = np.ones(num_nodes, dtype=bool)
            action_mask = np.ones(num_nodes, dtype=bool)
            action_mask[self.protected_nodes] = False
            if len(self.pre_attacked_nodes) > 0:
                current_topology[self.pre_attacked_nodes, :] = 0.0
                current_topology[:, self.pre_attacked_nodes] = 0.0
                node_status[self.pre_attacked_nodes] = False
                action_mask[self.pre_attacked_nodes] = False
                current_topology, node_status, action_mask = \
                    self.gcc_reproject(current_topology, node_status, action_mask)
            return current_topology, node_status, action_mask

    def replay(self, action_history: np.ndarray, return_ori_action_history: bool = False) \
            -> Union[Tuple[np.ndarray, np.ndarray], List[int]]:
        """
        Replay the action history

        :param action_history: action history
        :param return_ori_action_history: whether to return the original action history
        :return: topology, node status or original action history
        """
        current_topology, node_status, action_mask = self.prepare_replay()
        if self.dynamics_type is not None:
            node_indices = np.arange(current_topology.shape[0])
        else:
            node_indices = np.arange(current_topology.number_of_nodes())
        ori_action_history = []
        step = 0
        action = action_history[step]
        while action != -1:
            node = node_indices[node_status][action]
            if not node_status[node] or not action_mask[node]:
                if not node_status[node]:
                    warn_msg = f'Node {node} is already attacked.'
                else:
                    warn_msg = f'Node {node} can not be attacked.'
                warnings.warn(warn_msg)
                node = self.rng.choice(np.where(action_mask)[0])
            node_status[node] = False
            action_mask[node] = False
            if self.dynamics_type is not None:
                current_topology[node, :] = 0.0
                current_topology[:, node] = 0.0
                current_topology, node_status, action_mask = \
                    self.gcc_reproject(current_topology, node_status, action_mask)
            else:
                current_topology.remove_node(node)
            step += 1
            ori_action_history.append(node)
            if step == len(action_history):
                break
            action = action_history[step]
        if return_ori_action_history:
            return ori_action_history
        return current_topology, node_status

    def replay_finder_solution(self,
                               solution: List[int],
                               max_steps: int,
                               return_d_h_b_s_list: bool = False,
                               return_topology_node_status: bool = False):
        """
        Replay the solution from the finder

        :param solution: solution from the finder approach (action id in the original graph)
        :param max_steps: maximum number of steps to replay
        :param return_d_h_b_s_list: whether to return d, h, b, s list
        :param return_topology_node_status: whether to return topology and node status

        :return: objective value, action history
        """
        current_topology, node_status, action_mask = self.prepare_replay()
        action_history = []
        d_history = []
        h_history = []
        b_connectivity_history = []
        s_lo_history = []
        s_hi_history = []
        cursor = 0
        reward = 0.0
        done = False
        if self.dynamics_type is not None:
            num_nodes = current_topology.shape[0]
            beta, beta_factorized = self.compute_beta(current_topology[node_status, :][:, node_status])
            d = beta_factorized[0]
            h = beta_factorized[2]
            resilience_robustness, b_connectivity, s_lo, s_hi = self.simulate_resilience_finder(
                current_topology[node_status, :][:, node_status],
                node_status)
        else:
            num_nodes = current_topology.number_of_nodes()
            beta = None
            d = None
            h = None
            s_lo = None
            s_hi = None
            resilience_robustness, b_connectivity = self.get_robustness(current_topology)
        if not resilience_robustness:
            done = True
        d_history.append(d)
        h_history.append(h)
        b_connectivity_history.append(b_connectivity)
        s_lo_history.append(s_lo)
        s_hi_history.append(s_hi)
        cursor_end = min(len(solution), max_steps)
        with tqdm(total=cursor_end) as pbar:
            while not done:
                node = solution[cursor]
                if node_status[node] and action_mask[node]:
                    node_status[node] = False
                    action_mask[node] = False
                    if self.dynamics_type is not None:
                        current_topology[node, :] = 0.0
                        current_topology[:, node] = 0.0
                        current_topology, node_status, action_mask = \
                            self.gcc_reproject(current_topology, node_status, action_mask)
                        beta, beta_factorized = self.compute_beta(current_topology[node_status, :][:, node_status])
                        d = beta_factorized[0]
                        h = beta_factorized[2]
                        resilience_robustness, b_connectivity, s_lo, s_hi = self.simulate_resilience_finder(
                            current_topology[node_status, :][:, node_status],
                            node_status)
                    else:
                        current_topology.remove_node(node)
                        resilience_robustness, b_connectivity = self.get_robustness(current_topology)
                        beta = None
                        d = None
                        h = None
                        s_lo = None
                        s_hi = None
                    d_history.append(d)
                    h_history.append(h)
                    b_connectivity_history.append(b_connectivity)
                    s_lo_history.append(s_lo)
                    s_hi_history.append(s_hi)
                    if not resilience_robustness:
                        done = True
                    if self.dynamics_type is not None:
                        reward += 1.0
                    else:
                        reward += b_connectivity
                    action_history.append(node)
                cursor += 1
                pbar.update(1)

                if not done and cursor >= cursor_end:
                    if self.dynamics_type is not None:
                        node_degree = current_topology.sum(axis=0)
                    else:
                        node_degree = np.zeros(num_nodes)
                        for node in current_topology.nodes():
                            node_degree[node] = current_topology.degree(node)
                    # get the node id with the maximum degree that is true in node_status and action_mask
                    node = np.nonzero(node_status & action_mask)[0][np.argmax(node_degree[node_status & action_mask])]
                    solution.append(node)

        if self.dynamics_type is None:
            reward = reward / num_nodes

        if not done:
            reward = -1.0

        if return_topology_node_status:
            return current_topology, node_status
        if not return_d_h_b_s_list:
            return reward, beta, action_history, node_status.sum(), d, h, b_connectivity, s_lo, s_hi
        else:
            return (reward, beta, action_history, node_status.sum(), d_history, h_history, b_connectivity_history,
                    s_lo_history, s_hi_history)

    def batch_attack(self,
                     solution: np.ndarray) -> Tuple[bool, float, float, float, float, float, float]:
        """
        Batch attack
        :param solution: node indices
        :param beta_effc: beta critical
        :param epsilon: threshold
        :param simulate_resilience: whether to evaluate resilience by simulation

        :return: reward
        """
        assert self.dynamics_type is not None
        current_topology = self.initial_topology.copy()
        num_nodes = current_topology.shape[0]
        node_status = np.ones(num_nodes, dtype=bool)
        node_status[solution] = False
        topology = current_topology[node_status, :][:, node_status]
        topology, node_status = self.batch_gcc(topology, node_status)
        done = False
        beta, beta_factorized = self.compute_beta(topology)
        d = beta_factorized[0]
        h = beta_factorized[2]
        resilience_robustness, b_connectivity, s_lo, s_hi = self.simulate_resilience_finder(topology, node_status)
        if not resilience_robustness:
            done = True
        return done, beta, d, h, b_connectivity, s_lo, s_hi

    def get_batch_attack_initial_beta(self) -> float:
        current_topology = self.initial_topology.copy()
        beta, _ = self.compute_beta(current_topology)
        return beta

    def get_graph_plot(self,
                       action_history: np.ndarray,
                       edge_thres_pct: int = 0,
                       edge_color: str = None,
                       selected_nodes: Optional[List] = None,
                       custom_node_size: int = 25,
                       node_size_var: str = None,
                       plotly_plot: bool = False,
                       finder: bool = False) -> Tuple[nx.Graph, Dict, List, List, np.ndarray, List, List]:
        """
        :return: graph and node positions
        """
        full_topology = self.initial_topology.copy()
        if not finder:
            current_topology, node_status = self.replay(action_history)
        else:
            current_topology, node_status = self.replay_finder_solution(action_history.tolist(), 1000,
                                                                        return_topology_node_status=True)
        g = nx.from_numpy_array(current_topology)
        edge_thres_value = np.percentile(current_topology[current_topology > 0], edge_thres_pct)
        current_edge_list = np.argwhere(current_topology >= edge_thres_value).tolist()
        full_edge_list = np.argwhere(full_topology >= edge_thres_value).tolist()
        if edge_color is None:
            edge_color = ['black' if edge in current_edge_list else '#D9D9D9' for edge in full_edge_list]
        elif edge_color == 'light':
            edge_color = ['#D9D9D9' for _ in full_edge_list]
        else:
            raise RuntimeError(f'Unknown edge color: {edge_color}')

        if not plotly_plot:
            edge_style = ['solid' if edge in current_edge_list else '-.' for edge in full_edge_list]
        else:
            edge_style = ['solid' if edge in current_edge_list else 'longdash' for edge in full_edge_list]

        node_color = ['black' if node_status[i] else 'red' for i in range(node_status.shape[0])]

        # calculate node size
        node_size_list = [2*custom_node_size if selected_nodes is not None and node in selected_nodes else custom_node_size
                          for node in g.nodes()]
        if node_size_var is not None:
            topology = current_topology
            degree = topology.sum(axis=0)[node_status]
            state = self.simulate_resilience_finder(topology[node_status, :][:, node_status], node_status,
                                                    return_traj=False, return_state=True)
            ds = degree*state
            if node_size_var == 'degree':
                temp = degree.argsort()
            elif node_size_var == 'state':
                temp = state.argsort()
            elif node_size_var == 'ds':
                temp = ds.argsort()
            else:
                raise ValueError(f'Unknown node size variable: {node_size_var}')
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(degree))
            active_node_size_list = 0.3*custom_node_size*np.logspace(0, 1, base=10, num=len(degree))
            active_node_size_list = active_node_size_list[ranks]
            node_size_list = np.array(node_size_list)
            node_size_list[node_status] = active_node_size_list
            node_size_list = node_size_list.tolist()

            grey_scale = np.logspace(0, 1, base=100, num=len(degree))/100
            grey_scale = 255*grey_scale
            grey_scale = [int(grey) for grey in grey_scale]
            grey_scale = [f'rgb({255-grey}, {255-grey}, {255-grey})' for grey in grey_scale]
            node_color = []
            cursor = 0
            for i in range(len(node_status)):
                if node_status[i]:
                    node_color.append(grey_scale[ranks[cursor]])
                    cursor += 1
                else:
                    node_color.append('red')

        return g, self._initial_graph_layout, node_color, node_size_list, full_edge_list, edge_color, edge_style

    def get_3d_graph_layout(self, return_state: str = 'lo', all_zero: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        stable_state, _, _ = self.get_stable_state(return_state=return_state)

        graph_layout = self._initial_graph_layout
        node_pos = np.array([(graph_layout[i][0], graph_layout[i][1], stable_state[i])
                            for i in range(stable_state.shape[0])])
        if node_pos[:, 2].max() > 1.0:
            node_pos[:, 2] = node_pos[:, 2] / node_pos[:, 2].max()
        node_pos[:, 2] = np.clip(node_pos[:, 2], 0, 1)
        node_pos[:, :2] = 1.0 + 0.8 * node_pos[:, :2]
        if all_zero:
            node_pos[:, 2] = 0.0
        edge_list = self._g.edges()

        return node_pos, stable_state, edge_list

    def transform_action_history(self,
                                 action_history: np.ndarray) -> List[int]:
        """
        Transform the action history from the original graph to the projected graph

        :param action_history: the action history in the pyg reduced form
        :return: action history in the original graph
        """
        action_history = self.replay(action_history, return_ori_action_history=True)
        return action_history
