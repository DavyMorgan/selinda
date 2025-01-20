import io
import os
from typing import Tuple, Dict, Optional, List, Union

import gymnasium as gym
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from attack_resilience_complex_networks.env.cnc import ComplexNetworkClient
from attack_resilience_complex_networks.env.obs_extractor import ObsExtractor
from attack_resilience_complex_networks.utils.config import Config
from attack_resilience_complex_networks.utils.dynamics import load_dynamics
from attack_resilience_complex_networks.utils.graph import load_topology


class CN(gym.Env):
    """Complex network environments."""

    def __init__(self, cfg: Config, has_dynamics: bool = True,
                 random_episode: bool = False, num_instances: int = 0,
                 block_feature: List[str] = None, protected_nodes: List[int] = None,
                 pre_attacked_nodes: List[int] = None):
        super(CN, self).__init__()
        self._has_dynamics = has_dynamics
        self.cfg = cfg
        self._seed(cfg.seed)

        self._random_episode = random_episode
        self._graph_idx = 0
        if not self._random_episode:
            if num_instances > 0:
                self._max_graph_idx = num_instances
                graph_type = self.cfg.env_specs['graph_type']
                self.cfg.env_specs['graph_type'] = graph_type.split('seed-')[0] + f'seed-{self._graph_idx}'
                if has_dynamics:
                    self.cfg.env_specs['target_dynamics_params']['seed'] = self._graph_idx
            else:
                self._max_graph_idx = 1

        self._block_feature = block_feature
        self._protected_nodes = protected_nodes
        self._pre_attacked_nodes = pre_attacked_nodes
        self._prepare()
        self._compute_num_node_features()
        self.declare_spaces()

    def _seed(self, seed: int) -> None:
        self._np_random = np.random.default_rng(seed)

    def _prepare(self) -> None:
        if self._has_dynamics:
            dynamics_type = self.cfg.env_specs['dynamics_type']
            dynamics_params = self.cfg.env_specs['dynamics_params']
            dynamics_vec_params = self.cfg.env_specs.get('dynamics_vec_params', None)
            if dynamics_vec_params == 'equal':
                dynamics_vec_params = None
            target_dynamics_params = self.cfg.env_specs.get('target_dynamics_params', None)
            dynamics = load_dynamics(dynamics_type)
            self._dynamics_type = dynamics_type
        else:
            dynamics_type = None
            dynamics = None
            dynamics_vec_params = None
            dynamics_params = None
            target_dynamics_params = None
            self._dynamics_type = None

        if not self._random_episode:
            topology_filename = os.path.join(self.cfg.code_dir, self.cfg.env_specs['topology_filename'])
            graph_type = self.cfg.env_specs['graph_type']
            graph_scale = self.cfg.env_specs.get('graph_scale', 1.0)
            if topology_filename.endswith('.txt'):
                assert not self._has_dynamics
            topology = load_topology(topology_filename, graph_type, dynamics_type, scale=graph_scale)
        else:
            random_graph_specs = self.cfg.env_specs['random_graph_specs']
            topology = self._generate_random_topology(random_graph_specs)
        self._num_nodes = topology.number_of_nodes()
        self._max_num_nodes = self._num_nodes + 1
        if self._has_dynamics:
            self._max_num_edges = self._max_num_nodes**2
        else:
            self._max_num_edges = topology.number_of_edges()*2 + 1
        self._max_weight = max(nx.get_edge_attributes(topology, 'weight').values())
        self._max_steps = min(round(self.cfg.env_specs['budget']*self._num_nodes), self._num_nodes - 1)

        self._cnc = ComplexNetworkClient(self.cfg, self._np_random, topology, dynamics,
                                         dynamics_vec_params=dynamics_vec_params, dynamics_params=dynamics_params,
                                         target_dynamics_params=target_dynamics_params,
                                         protected_nodes=self._protected_nodes,
                                         pre_attacked_nodes=self._pre_attacked_nodes,
                                         block_feature=self._block_feature)
        self._obs_extractor = ObsExtractor(self.cfg, self._cnc, dynamics_type, self._max_num_nodes, self._max_num_edges)

        self._terminated = False
        self._truncated = False

    @property
    def has_dynamics(self) -> bool:
        return self._has_dynamics

    @staticmethod
    def _generate_random_topology(graph_specs: Dict) -> np.ndarray:
        graph_type = graph_specs['graph_type']
        graph_scale = graph_specs['graph_scale']
        assert graph_scale >= 1.0
        num_nodes = graph_specs['num_nodes']
        avg_degree = graph_specs['avg_degree']
        if graph_type == 'ER':
            m = int(num_nodes * avg_degree / 2)
            topology = nx.gnm_random_graph(num_nodes, m, directed=False)
        elif graph_type == 'SW':
            topology = nx.connected_watts_strogatz_graph(num_nodes, avg_degree, 0.4, tries=1000)
        elif graph_type == 'BA':
            m = int(avg_degree / 2)
            topology = nx.barabasi_albert_graph(num_nodes, m)
        elif graph_type == 'RP':
            prob_out = avg_degree / num_nodes / 5
            prob_in = 10 * prob_out
            topology = nx.random_partition_graph([num_nodes // 2, num_nodes - num_nodes // 2],
                                                 prob_in, prob_out)
        else:
            raise ValueError(f'Unknown graph type: {graph_type}.')
        nx.set_edge_attributes(topology, graph_scale, 'weight')
        return topology

    def _compute_num_node_features(self) -> None:
        if self._dynamics_type is not None:
            num_node_features = (self._cnc.nil_obs_pyg_shape
                                 + self._cnc.dynamics_param_obs_pyg_shape
                                 + self._cnc.env_obs_pyg_shape)
        else:
            num_node_features = self._cnc.topology_obs_pyg_shape
        self._num_node_features = num_node_features

    def declare_spaces(self) -> None:
        self.observation_space = gym.spaces.Dict({
            'node_features': gym.spaces.Box(low=-100, high=100,
                                            shape=(self._max_num_nodes, self._num_node_features), dtype=np.float32),
            'edge_index': gym.spaces.Box(low=0, high=self._num_nodes,
                                         shape=(2, self._max_num_edges), dtype=np.int64),
            'edge_attr': gym.spaces.Box(low=0, high=self._max_weight,
                                        shape=(self._max_num_edges,), dtype=np.float32),
            'edge_mask': gym.spaces.Box(low=0, high=1, shape=(self._max_num_edges,), dtype=bool),
            'node_mask': gym.spaces.Box(low=0, high=1, shape=(self._max_num_nodes,), dtype=bool),
            'action_mask': gym.spaces.Box(low=0, high=1, shape=(self._max_num_nodes,), dtype=bool),
        })
        self.action_space = gym.spaces.Discrete(self._max_num_nodes)

    def get_num_nodes(self) -> int:
        return self._num_nodes

    def get_current_num_nodes(self) -> int:
        return self._cnc.get_current_num_nodes()

    def get_num_node_features(self) -> int:
        return self._num_node_features

    def _terminate(self) -> None:
        if self._t >= self._max_steps:
            self._truncated = True
        elif self._has_dynamics:
            if not self._cnc.simulate_resilience():
                self._terminated = True
        else:
            if self._cnc.dismantle():
                self._terminated = True

    def _compute_reward(self, proxy: float) -> float:
        if self._has_dynamics:
            return -1.0
        else:
            return -proxy/self._num_nodes

    @property
    def _failure_reward(self) -> float:
        return -1.0

    def _failure_step(self) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        self._terminated = True
        return self._get_obs(), self._failure_reward, self._terminated, self._truncated, {}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        if self._terminated or self._truncated:
            raise RuntimeError('Action taken after episode is done.')
        try:
            dynamic_proxy = self._cnc.attack(action)
        except Exception as e:
            print(f'Exception raised at step {self._t}: {e}')
            self._t += 1
            return self._failure_step()
        self._action_history[self._t] = action
        self._t += 1
        proxy, proxy_extra = self._cnc.compute_proxy()
        self._proxy_history[self._t] = proxy
        self._proxy_extra_history[self._t] = proxy_extra
        self._dynamic_proxy_history[self._t] = dynamic_proxy
        self._terminate()
        reward = self._compute_reward(proxy)
        return self._get_obs(), reward, self._terminated, self._truncated, {}

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        if not self._random_episode:
            if self._max_graph_idx > 1:
                if self._graph_idx >= self._max_graph_idx - 1:
                    self._graph_idx = 0
                else:
                    self._graph_idx = self._graph_idx + 1
                graph_type = self.cfg.env_specs['graph_type']
                self.cfg.env_specs['graph_type'] = graph_type.split('seed-')[0] + f'seed-{self._graph_idx}'
                if self.has_dynamics:
                    self.cfg.env_specs['target_dynamics_params']['seed'] = self._graph_idx
                self._prepare()
            else:
                self._graph_idx = 0
        else:
            self._prepare()
        self._terminated = False
        self._truncated = False
        self._t = 0
        (initial_proxy, initial_proxy_extra), dynamic_proxy = self._cnc.reset()
        self._action_history = np.full(self._max_steps, -1, dtype=np.int32)
        self._proxy_history = np.full(self._max_steps + 1, initial_proxy, dtype=np.float32)
        self._proxy_extra_history = np.tile(initial_proxy_extra, (self._max_steps + 1, 1))
        self._dynamic_proxy_history = np.tile(dynamic_proxy, (self._max_steps + 1, 1))
        return self._get_obs(), {}

    def get_budget(self) -> int:
        return self._max_steps

    def get_mean_state(self) -> Union[float, Tuple[float, float]]:
        return self._cnc.get_mean_state()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return self._obs_extractor.get_obs()

    def plot_graph(self, ax: plt.Axes, edge_thres_pct: int) -> None:
        graph, graph_layout, node_color, node_size_list, edge_list, edge_color, edge_style = self._cnc.get_graph_plot(
            self._action_history, edge_thres_pct=edge_thres_pct)
        nx.draw(graph, pos=graph_layout,
                node_size=100, node_color=node_color,
                edgelist=edge_list, edge_color=edge_color, edge_cmap=plt.cm.cool, style=edge_style,
                ax=ax)
        ax.set_title("Graph", fontweight="bold")

    def plot_trajectory(self, ax: plt.Axes) -> None:
        if self._dynamics_type == 'neuron':
            _, (t_eval_lo, t_eval_hi, traj_lo, traj_hi) = self._cnc.simulate_resilience(return_traj=True)
            ax.plot(t_eval_lo, traj_lo.T, color='b', linestyle='-.')
            ax.plot(t_eval_hi, traj_hi.T, color='r', linestyle='-')
        elif self._dynamics_type == 'gene':
            _, (t_eval, traj) = self._cnc.simulate_resilience(return_traj=True)
            ax.plot(t_eval, traj.T)
        else:
            raise ValueError(f'Unknown dynamics type: {self._dynamics_type}.')
        ax.set_title("Trajectory", fontweight="bold")

    def render(self,
               mode: str = 'human',
               dpi: int = 300,
               edge_thres_pct: int = 0) -> Optional[np.ndarray]:
        if self._has_dynamics:
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            self.plot_graph(ax[0], edge_thres_pct)
            self.plot_trajectory(ax[1])
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            self.plot_graph(ax, edge_thres_pct)

        fig.tight_layout()

        if mode == 'human':
            plt.show()
        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=dpi)
            io_buf.seek(0)
            img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                 newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            return img_arr

    def visualize_graph_plotly(self,
                               edge_thres_pct: int = 0,
                               selected_nodes: Optional[List] = None,
                               protected_nodes: Optional[List] = None,
                               node_label: bool = True,
                               fig_size: Tuple = (800, 800),
                               custom_node_size: int = 25,
                               node_size_var: str = None,
                               edge_width: float = 2.0,
                               edge_color: str = None,
                               edge_sample_ratio: float = 1.0,
                               save_path: str = None,
                               action_history: List = None) -> None:
        if action_history is None:
            action_history = self._action_history
            finder = False
        else:
            finder = True
        graph, graph_layout, node_color, node_size_list, edge_list, edge_color, edge_style = self._cnc.get_graph_plot(
            action_history,
            edge_thres_pct,
            edge_color=edge_color,
            selected_nodes=selected_nodes,
            custom_node_size=custom_node_size,
            node_size_var=node_size_var,
            plotly_plot=True,
            finder=finder)

        fig = go.Figure()

        assert 0.0 <= edge_sample_ratio <= 1.0
        if edge_sample_ratio < 1.0:
            num_edges = len(edge_list)
            num_edges_to_sample = int(num_edges * edge_sample_ratio)
            # sample edges from the edge list and the corresponding edge color and style
            np_random = np.random.default_rng(17)
            sample_indices = np_random.choice(num_edges, num_edges_to_sample, replace=False)
            edge_list = [edge_list[i] for i in sample_indices]
            edge_color = [edge_color[i] for i in sample_indices]
            edge_style = [edge_style[i] for i in sample_indices]

        edge_traces = []
        for i, edge in enumerate(edge_list):
            x0, y0 = graph_layout[edge[0]]
            x1, y1 = graph_layout[edge[1]]
            edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                                          line=dict(width=edge_width, color=edge_color[i], dash=edge_style[i])))

        # Add all edge traces at once
        fig.add_traces(edge_traces)

        # Add protected nodes
        if protected_nodes is not None:
            for node in protected_nodes:
                x, y = graph_layout[node]
                node_size = node_size_list[node]
                node_size = node_size * 2.0
                fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                         marker=dict(size=node_size, color='lime')))

        # Add nodes to the plot
        for node in graph.nodes():
            x, y = graph_layout[node]
            node_size = node_size_list[node]
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                     marker=dict(size=node_size, color=node_color[node])))

        # Add node labels
        if node_label:
            labels = [str(node) for node in graph.nodes()]
            x_nodes, y_nodes = zip(*graph_layout.values())
            fig.add_trace(go.Scatter(x=x_nodes, y=y_nodes, mode='text', text=labels,
                                     textposition="middle center", textfont=dict(size=int(0.8*custom_node_size),
                                                                                 color='white')))

        # Configure layout
        fig.update_layout(
            showlegend=False,
            width=fig_size[0],
            height=fig_size[1],
            xaxis=dict(showgrid=False, zeroline=False, showline=False,
                       showticklabels=False, tickmode='array', tickvals=[]),
            yaxis=dict(showgrid=False, zeroline=False, showline=False,
                       showticklabels=False, tickmode='array', tickvals=[]),
            template='simple_white',
        )

        # Save or show the plot
        if save_path is not None:
            fig.write_image(save_path)
        fig.show()

    def visualize_graph_3d_plotly(self,
                                  fig_size: Tuple = (1000, 1000),
                                  save_path: str = None,
                                  show_state: bool = False,
                                  edge_width: float = 1.0,
                                  node_size: float = 3.0,
                                  rescale_node_color: float = 1.0,
                                  colorscale: str = 'Turbo',
                                  return_state: str = 'lo',
                                  all_zero: bool = False,
                                  transparent_background: bool = False) -> None:
        node_pos, stable_state, edge_list = self._cnc.get_3d_graph_layout(
            return_state=return_state, all_zero=all_zero)

        edge_x = []
        edge_y = []
        edge_z = []
        for edge in edge_list:
            x0, y0, z0 = node_pos[edge[0]]
            x1, y1, z1 = node_pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        trace_edges = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='lightblue', width=edge_width),
        )

        if not all_zero and stable_state.mean() > 0.1:
            color = stable_state / stable_state.max()
            color = np.maximum(color, 0.0)
            color = 1 - (1 - color) ** rescale_node_color
        else:
            color = 'darkblue'
        trace_nodes = go.Scatter3d(
            x=node_pos[:, 0],
            y=node_pos[:, 1],
            z=node_pos[:, 2],
            mode='markers',
            marker=dict(
                size=node_size,
                color=color,
                colorscale=colorscale,
                line_color='black',
                line_width=0.,
            )
        )

        trace_bounding_box = go.Scatter3d(
            x=[0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0],
            y=[0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2],
            z=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
            mode="lines",
            line=dict(color='black', width=2),
        )

        layout = go.Layout(
            width=fig_size[0],
            height=fig_size[1],
            showlegend=False,
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.6, y=1.6, z=1.0)
            ),
            scene=dict(
                xaxis=dict(
                    showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title='',
                    tickvals=[100,200]
                ),
                yaxis=dict(
                    showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title='',
                    tickvals=[100,200]
                ),
                zaxis=dict(
                    showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title='state' if show_state else '',
                    range=[0,1],
                    tickvals=[100,200]
                ),
            ),
            font=dict(
                family="Arial",
                size=20,
                color="black"
            ),
        )

        if transparent_background:
            layout['plot_bgcolor'] = 'rgba(0,0,0,0)'
            layout['paper_bgcolor'] = 'rgba(0,0,0,0)'
        else:
            layout['template'] = 'simple_white'

        data = [trace_edges, trace_nodes, trace_bounding_box]
        fig = go.Figure(data=data, layout=layout)

        if save_path is not None:
            fig.write_image(save_path)
        fig.show()

    def visualize_trajectory_plotly(self, selected_nodes: Optional[List] = None,
                                    max_t: float = None, show_title: bool = True,
                                    show_legend: bool = True, bolded_lines: List[int] = None,
                                    fig_size: Tuple = (800, 600),
                                    save_path: str = None) -> None:
        _, node_status = self._cnc.replay(self._action_history)
        ori_node_id = np.arange(node_status.size)[node_status].tolist()
        if selected_nodes is None:
            selected_nodes = ori_node_id
        if bolded_lines is not None:
            line_width = [5 if node in bolded_lines else 1 for node in ori_node_id]
        else:
            line_width = 1
        if self._dynamics_type == 'neuron':
            _, (t_eval_lo, t_eval_hi, traj_lo, traj_hi) = self._cnc.simulate_resilience(return_traj=True)
            if max_t is not None:
                traj_lo = traj_lo[:, t_eval_lo <= max_t]
                traj_hi = traj_hi[:, t_eval_hi <= max_t]
                t_eval_lo = t_eval_lo[t_eval_lo <= max_t]
                t_eval_hi = t_eval_hi[t_eval_hi <= max_t]
            df_lo = pd.DataFrame(np.hstack((t_eval_lo.reshape(-1, 1), traj_lo.T)), columns=['time'] + ori_node_id)
            df_lo['initial_state'] = 'lo'
            df_hi = pd.DataFrame(np.hstack((t_eval_hi.reshape(-1, 1), traj_hi.T)), columns=['time'] + ori_node_id)
            df_hi['initial_state'] = 'hi'
            df = pd.concat([df_lo, df_hi])
            # plot lines using plotly.graph_objects, with the specified line width
            fig = go.Figure()
            color_sequence = px.colors.qualitative.Prism
            for i, node in enumerate(selected_nodes):
                width = line_width[i] if isinstance(line_width, list) else line_width
                # plot low state in dashed line
                fig.add_trace(go.Scatter(x=df_lo['time'], y=df_lo[node], mode='lines', name=node,
                                         line=dict(
                                             color=color_sequence[i % len(color_sequence)],
                                             width=width,
                                             dash='dash')))
                # plot high state in solid line
                fig.add_trace(go.Scatter(x=df_hi['time'], y=df_hi[node], mode='lines', name=node,
                                         line=dict(
                                             color=color_sequence[i % len(color_sequence)],
                                             width=width,
                                             dash='solid')))
        elif self._dynamics_type == 'gene':
            _, (t_eval, traj) = self._cnc.simulate_resilience(return_traj=True)
            if max_t is not None:
                traj = traj[:, t_eval <= max_t]
                t_eval = t_eval[t_eval <= max_t]
            df = pd.DataFrame(np.hstack((t_eval.reshape(-1, 1), traj.T)), columns=['time'] + ori_node_id)
            fig = go.Figure()
            color_sequence = px.colors.qualitative.Prism
            for i, node in enumerate(selected_nodes):
                width = line_width[i] if isinstance(line_width, list) else line_width
                fig.add_trace(go.Scatter(x=df['time'], y=df[node], mode='lines', name=node,
                                         line=dict(
                                             color=color_sequence[i % len(color_sequence)],
                                             width=width,
                                             dash='solid')))
        else:
            raise ValueError(f"Unknown dynamics type: {self._dynamics_type}")
        lim_x = max_t if max_t is not None else int(df['time'].max())
        range_x = [0, lim_x]
        step_x = 100 if lim_x > 100 else 20
        fig.update_xaxes(showline=True, linewidth=1., linecolor='black', mirror=True, title='<b>Time</b>',
                         range=range_x, tickvals=np.arange(range_x[0], range_x[1] + 1, step_x))
        fig.update_yaxes(showline=True, linewidth=1., linecolor='black', mirror=True, title='<b>Node state</b>')
        # use simple_white template
        fig.update_layout(template='simple_white')
        # set figure size
        fig.update_layout(width=fig_size[0], height=fig_size[1])
        if show_title:
            fig.update_layout(
                font_family="Arial",
                font_size=20,
                font_color='black',
                title=dict(
                    text="<b>Trajectory</b>",
                    y=0.95,
                    x=0.5,
                )
            )
        else:
            fig.update_layout(font_family="Arial", font_size=20, font_color='black')
        if not show_legend:
            fig.update_layout(showlegend=False)
        if save_path is not None:
            fig.write_image(save_path)
        fig.show()

    def close(self):
        plt.close()

    def get_slim_action_history(self) -> List[int]:
        return self._action_history[:self._t].tolist()

    def get_attacked_nodes(self) -> List[int]:
        return self._cnc.transform_action_history(self._action_history)

    def report_current_proxy(self) -> Tuple[float, List[float]]:
        return self._proxy_history[self._t], self._proxy_extra_history[self._t].tolist()

    def report_connectivity_history(self) -> List[float]:
        assert not self._has_dynamics
        return self._proxy_history[:self._t + 1].tolist()

    def report_proxy_history(self) -> Tuple[List[float], List[float]]:
        d = [proxy_factorized[0] for proxy_factorized in self._proxy_extra_history]
        h = [proxy_factorized[2] for proxy_factorized in self._proxy_extra_history]
        return d[:self._t + 1], h[:self._t + 1]

    def report_dynamic_proxy_history(self) -> Tuple[List[float], List[float], List[float]]:
        b = [dynamic_proxy[0] for dynamic_proxy in self._dynamic_proxy_history]
        s_lo = [dynamic_proxy[1] for dynamic_proxy in self._dynamic_proxy_history]
        s_hi = [dynamic_proxy[2] for dynamic_proxy in self._dynamic_proxy_history]
        return b[:self._t + 1], s_lo[:self._t + 1], s_hi[:self._t + 1]


class EvalCN(CN):
    def __init__(self, cfg: Config, mode: str, has_dynamics: bool = True,
                 random_episode: bool = False, num_instances: int = 0,
                 block_feature: List[str] = None, protected_nodes: List[int] = None,
                 pre_attacked_nodes: List[int] = None):
        if mode == 'val':
            self._reward_coef = -1.0
        elif mode == 'test':
            self._reward_coef = 1.0
        else:
            raise ValueError(f'Unknown mode {mode}.')

        super().__init__(cfg, has_dynamics, random_episode, num_instances, block_feature,
                         protected_nodes, pre_attacked_nodes)

    def get_eval_num_instances(self) -> int:
        return self._max_graph_idx

    def reset_instance_id(self) -> None:
        self._graph_idx = self._max_graph_idx

    @property
    def _failure_reward(self) -> float:
        return self._reward_coef * self._t

    def anc(self) -> float:
        anc = self._proxy_history[1:self._t + 1].sum() / self._num_nodes
        return anc

    def _compute_reward(self, proxy) -> float:
        if self._terminated or self._truncated:
            if self._has_dynamics:
                reward = self._reward_coef * self._t
            else:
                reward = self._reward_coef * self.anc()
        else:
            reward = 0.0
        return reward

    def get_topology(self) -> nx.Graph:
        return self._cnc.get_topology().copy()

    def evaluate_finder_solution(self,
                                 solution: List[int],
                                 return_d_h_b_s_list: bool = False) \
            -> Tuple[
                float, float, List[int], int,
                Union[float, List[float]],
                Union[float, List[float]],
                Union[float, List[float]],
                Union[float, List[float]],
                Union[float, List[float]]]:
        return self._cnc.replay_finder_solution(solution, self._max_steps, return_d_h_b_s_list)

    def batch_attack(self, solutions: np.ndarray) -> Tuple[bool, float, float, float, float, float, float]:
        return self._cnc.batch_attack(solutions)

    def get_batch_attack_initial_beta(self) -> float:
        return self._cnc.get_batch_attack_initial_beta()
